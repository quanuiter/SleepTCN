"""Hop dong du lieu cho huan luyen theo fold va theo ban ghi."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import SleepRecord, load_record, load_split_manifest, paths_for_role
from .features import MANIPULATIONS


ROLES = ("train", "validation", "test")


@dataclass(frozen=True)
class RolePartition:
    role: str
    subject_ids: tuple[str, ...]
    record_keys: tuple[str, ...]
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class FoldPartitions:
    outer_fold: int
    variant: str
    train: RolePartition
    validation: RolePartition
    test: RolePartition

    def for_role(self, role: str) -> RolePartition:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        return getattr(self, role)


@dataclass(frozen=True)
class EpochSampleRef:
    record_key: str
    subject_id: str
    target_epoch_position: int
    source_epoch_position: int
    original_epoch_index: int
    label: int


@dataclass(frozen=True)
class FeatureSequence:
    record_key: str
    subject_id: str
    preprocess_version: str
    extractor_id: str
    features: np.ndarray
    labels: np.ndarray
    original_epoch_index: np.ndarray

    def __post_init__(self) -> None:
        if self.features.dtype != np.float32 or self.features.ndim != 2:
            raise ValueError("features must be float32 with shape (T,F)")
        if self.labels.ndim != 1 or self.labels.shape[0] != len(self.features):
            raise ValueError("labels must have shape (T,)")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("labels must be integer")
        if not np.isin(self.labels, [-1, 0, 1, 2, 3, 4]).all():
            raise ValueError("labels contain unsupported value")
        if (
            self.original_epoch_index.dtype != np.int32
            or self.original_epoch_index.shape != self.labels.shape
        ):
            raise ValueError("original_epoch_index must be int32 with shape (T,)")
        if len(self.original_epoch_index) and not np.all(
            np.diff(self.original_epoch_index) == 1
        ):
            raise ValueError("original_epoch_index must be consecutive")
        if not np.isfinite(self.features).all():
            raise ValueError("features contain NaN or infinity")
        if not self.record_key or self.subject_id != self.record_key[:5]:
            raise ValueError("record/subject metadata mismatch")

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features),
            torch.from_numpy(self.labels.astype(np.int64, copy=False)),
        )


class FeatureSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, sequences: Sequence[FeatureSequence]) -> None:
        if not sequences:
            raise ValueError("sequences must not be empty")
        feature_dims = {sequence.features.shape[1] for sequence in sequences}
        if len(feature_dims) != 1:
            raise ValueError("all sequences must use the same feature dimension")
        keys = [sequence.record_key for sequence in sequences]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate record_key in feature sequences")
        self.sequences = tuple(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index].tensors()


class RecordEpochDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Epoch hop le cua nhieu ban ghi, dich nguon trong bien tung ban ghi."""

    def __init__(
        self, records: Sequence[SleepRecord], manipulation: str = "current"
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        if manipulation not in MANIPULATIONS:
            raise ValueError(f"unknown manipulation: {manipulation}")
        keys = [record.info.record_key for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate record in epoch dataset")
        variants = {record.info.preprocess_version for record in records}
        if len(variants) != 1:
            raise ValueError("epoch dataset cannot mix preprocessing variants")
        self.records = tuple(records)
        self.manipulation = manipulation
        self._valid_positions = tuple(
            np.flatnonzero(record.valid_mask).astype(np.int32) for record in records
        )
        counts = np.array([len(indices) for indices in self._valid_positions])
        if np.any(counts == 0):
            raise ValueError("record without a valid epoch")
        self._ends = np.cumsum(counts, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._ends[-1])

    def sample_ref(self, index: int) -> EpochSampleRef:
        if not isinstance(index, (int, np.integer)) or index < 0 or index >= len(self):
            raise IndexError(index)
        record_position = bisect_right(self._ends, int(index))
        start = 0 if record_position == 0 else int(self._ends[record_position - 1])
        target_position = int(
            self._valid_positions[record_position][int(index) - start]
        )
        record = self.records[record_position]
        if self.manipulation == "current":
            source_position = target_position
        elif self.manipulation == "previous":
            source_position = max(0, target_position - 1)
        else:
            source_position = min(len(record.y) - 1, target_position + 1)
        label = int(record.y[target_position])
        if label not in range(5):
            raise AssertionError("RecordEpochDataset exposed an ignored target")
        return EpochSampleRef(
            record_key=record.info.record_key,
            subject_id=record.info.subject_id,
            target_epoch_position=target_position,
            source_epoch_position=source_position,
            original_epoch_index=int(record.original_epoch_index[target_position]),
            label=label,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.sample_ref(index)
        record_position = bisect_right(self._ends, int(index))
        record = self.records[record_position]
        signal = torch.from_numpy(record.x[reference.source_epoch_position]).unsqueeze(0)
        return signal, torch.tensor(reference.label, dtype=torch.long)


def class_counts_from_records(records: Sequence[SleepRecord]) -> np.ndarray:
    if not records:
        raise ValueError("records must not be empty")
    counts = np.zeros(5, dtype=np.int64)
    for record in records:
        counts += np.bincount(record.y[record.valid_mask], minlength=5)[:5]
    if np.any(counts == 0):
        raise ValueError("at least one class is absent")
    return counts


def _partition_from_run(
    processed_root: Path,
    manifest: dict[str, Any],
    run: dict[str, Any],
    role: str,
    variant: str,
) -> RolePartition:
    role_data = run[role]
    return RolePartition(
        role=role,
        subject_ids=tuple(role_data["subject_ids"]),
        record_keys=tuple(role_data["record_keys"]),
        paths=tuple(
            paths_for_role(processed_root, manifest, run["outer_fold"], role, variant)
        ),
    )


def resolve_fold_partitions(
    processed_root: Path,
    split_manifest_path: Path,
    outer_fold: int,
    variant: str,
) -> FoldPartitions:
    manifest = load_split_manifest(split_manifest_path)
    matching = [
        run for run in manifest["outer_runs"] if int(run["outer_fold"]) == outer_fold
    ]
    if len(matching) != 1:
        raise ValueError(f"outer fold must occur exactly once: {outer_fold}")
    run = matching[0]
    partitions = {
        role: _partition_from_run(processed_root, manifest, run, role, variant)
        for role in ROLES
    }
    for index, left_role in enumerate(ROLES):
        left = partitions[left_role]
        if len(left.subject_ids) != len(set(left.subject_ids)):
            raise ValueError(f"duplicate subject in {left_role}")
        if len(left.record_keys) != len(set(left.record_keys)):
            raise ValueError(f"duplicate record in {left_role}")
        if tuple(path.stem for path in left.paths) != left.record_keys:
            raise ValueError(f"path order differs from manifest in {left_role}")
        for right_role in ROLES[index + 1 :]:
            right = partitions[right_role]
            if set(left.subject_ids) & set(right.subject_ids):
                raise ValueError(f"subject leakage: {left_role}/{right_role}")
            if set(left.record_keys) & set(right.record_keys):
                raise ValueError(f"record leakage: {left_role}/{right_role}")
    return FoldPartitions(
        outer_fold=outer_fold,
        variant=variant,
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
    )


def load_partition_records(
    partition: RolePartition, expected_variant: str
) -> tuple[SleepRecord, ...]:
    records = tuple(load_record(path, expected_variant) for path in partition.paths)
    if tuple(record.info.record_key for record in records) != partition.record_keys:
        raise AssertionError(f"loaded record order mismatch for {partition.role}")
    if {record.info.subject_id for record in records} != set(partition.subject_ids):
        raise AssertionError(f"loaded subject set mismatch for {partition.role}")
    return records
