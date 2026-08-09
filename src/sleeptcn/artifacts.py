"""Luu/nap dac trung va du doan kem ma bam truy nguyen."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dataset import SleepRecord
from .features import extract_15cnn_features
from .metrics import compute_metrics
from .training_data import FeatureSequence


ARTIFACT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(named_hashes: dict[str, str]) -> str:
    for name, value in named_hashes.items():
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid SHA-256 for {name}")
    canonical = json.dumps(named_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@torch.no_grad()
def extract_resnet_sequence(
    record: SleepRecord,
    model: nn.Module,
    *,
    extractor_id: str,
    device: torch.device | str,
    batch_size: int = 256,
) -> FeatureSequence:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    model = model.to(device).eval()
    features: list[np.ndarray] = []
    for start in range(0, len(record.x), batch_size):
        signals = torch.from_numpy(record.x[start : start + batch_size]).unsqueeze(1)
        batch_features = model.extract_features(signals.to(device))
        if batch_features.ndim != 2 or batch_features.shape[0] != len(signals):
            raise ValueError("ResNet extractor returned invalid shape")
        features.append(batch_features.cpu().numpy())
    matrix = np.concatenate(features).astype(np.float32, copy=False)
    return FeatureSequence(
        record_key=record.info.record_key,
        subject_id=record.info.subject_id,
        preprocess_version=record.info.preprocess_version,
        extractor_id=extractor_id,
        features=matrix,
        labels=record.y.copy(),
        original_epoch_index=record.original_epoch_index.copy(),
    )


def extract_15cnn_sequence(
    record: SleepRecord,
    models: dict[str, nn.Module],
    *,
    extractor_id: str,
    device: torch.device | str,
    batch_size: int = 256,
) -> FeatureSequence:
    features = extract_15cnn_features(
        record.x, models, device=device, batch_size=batch_size
    )
    return FeatureSequence(
        record_key=record.info.record_key,
        subject_id=record.info.subject_id,
        preprocess_version=record.info.preprocess_version,
        extractor_id=extractor_id,
        features=features,
        labels=record.y.copy(),
        original_epoch_index=record.original_epoch_index.copy(),
    )


def save_feature_sequence(
    path: Path,
    sequence: FeatureSequence,
    *,
    extractor_sha256: str,
    split_sha256: str,
    outer_fold: int,
    seed: int,
) -> None:
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "feature_sequence",
        "record_key": sequence.record_key,
        "subject_id": sequence.subject_id,
        "preprocess_version": sequence.preprocess_version,
        "extractor_id": sequence.extractor_id,
        "extractor_sha256": extractor_sha256,
        "split_sha256": split_sha256,
        "outer_fold": outer_fold,
        "seed": seed,
    }
    combined_sha256(
        {"extractor_sha256": extractor_sha256, "split_sha256": split_sha256}
    )
    _write_npz_atomic(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        features=sequence.features,
        labels=sequence.labels.astype(np.int8, copy=False),
        original_epoch_index=sequence.original_epoch_index,
    )


def load_feature_sequence(
    path: Path,
    *,
    expected_extractor_sha256: str,
    expected_split_sha256: str,
    expected_outer_fold: int,
    expected_seed: int,
) -> FeatureSequence:
    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"].item()))
        expected = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "feature_sequence",
            "extractor_sha256": expected_extractor_sha256,
            "split_sha256": expected_split_sha256,
            "outer_fold": expected_outer_fold,
            "seed": expected_seed,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"feature artifact metadata mismatch: {mismatches}")
        return FeatureSequence(
            record_key=metadata["record_key"],
            subject_id=metadata["subject_id"],
            preprocess_version=metadata["preprocess_version"],
            extractor_id=metadata["extractor_id"],
            features=npz["features"].copy(),
            labels=npz["labels"].copy(),
            original_epoch_index=npz["original_epoch_index"].copy(),
        )


@dataclass(frozen=True)
class PredictionTable:
    subject_id: np.ndarray
    record_key: np.ndarray
    original_epoch_index: np.ndarray
    true_label: np.ndarray
    predicted_label: np.ndarray
    logits: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.true_label)
        arrays = (
            self.subject_id,
            self.record_key,
            self.original_epoch_index,
            self.predicted_label,
        )
        if any(array.ndim != 1 or len(array) != length for array in arrays):
            raise ValueError("prediction columns must be one-dimensional and aligned")
        if self.logits.shape != (length, 5) or self.logits.dtype != np.float32:
            raise ValueError("logits must be float32 with shape (N,5)")
        if not np.isin(self.true_label, [0, 1, 2, 3, 4]).all():
            raise ValueError("prediction table must contain valid true labels only")
        if not np.isin(self.predicted_label, [0, 1, 2, 3, 4]).all():
            raise ValueError("invalid predicted label")
        if not np.isfinite(self.logits).all():
            raise ValueError("prediction logits contain NaN or infinity")
        if any(subject != record[:5] for subject, record in zip(self.subject_id, self.record_key, strict=True)):
            raise ValueError("prediction subject/record metadata mismatch")
        keys = list(
            zip(
                self.subject_id.tolist(),
                self.record_key.tolist(),
                self.original_epoch_index.tolist(),
                strict=True,
            )
        )
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate subject/record/original_epoch_index prediction")

    def metrics(self) -> dict[str, object]:
        return compute_metrics(self.true_label, self.predicted_label)


def prediction_table_from_parts(
    parts: list[tuple[FeatureSequence, np.ndarray]],
) -> PredictionTable:
    if not parts:
        raise ValueError("prediction parts must not be empty")
    subjects: list[np.ndarray] = []
    records: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    true: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    seen_records: set[str] = set()
    for sequence, logits in parts:
        if sequence.record_key in seen_records:
            raise ValueError(f"duplicate prediction record: {sequence.record_key}")
        seen_records.add(sequence.record_key)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (len(sequence.labels), 5):
            raise ValueError("record logits do not align with full sequence")
        valid = (sequence.labels >= 0) & (sequence.labels < 5)
        count = int(valid.sum())
        subjects.append(np.full(count, sequence.subject_id, dtype="U5"))
        records.append(np.full(count, sequence.record_key, dtype="U7"))
        indices.append(sequence.original_epoch_index[valid])
        true.append(sequence.labels[valid].astype(np.int8, copy=False))
        selected_logits = logits[valid]
        logits_parts.append(selected_logits)
        predicted.append(selected_logits.argmax(axis=1).astype(np.int8))
    return PredictionTable(
        subject_id=np.concatenate(subjects),
        record_key=np.concatenate(records),
        original_epoch_index=np.concatenate(indices),
        true_label=np.concatenate(true),
        predicted_label=np.concatenate(predicted),
        logits=np.concatenate(logits_parts),
    )


def save_prediction_table(
    path: Path, table: PredictionTable, metadata: dict[str, Any]
) -> None:
    required = {
        "experiment_id",
        "outer_fold",
        "seed",
        "split_sha256",
        "checkpoint_sha256",
        "data_variant",
        "role",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"prediction metadata missing: {missing}")
    if metadata["role"] not in {"validation", "test"}:
        raise ValueError("prediction role must be validation or test")
    combined_sha256(
        {
            "split_sha256": metadata["split_sha256"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        }
    )
    _write_npz_atomic(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        subject_id=table.subject_id,
        record_key=table.record_key,
        original_epoch_index=table.original_epoch_index,
        true_label=table.true_label,
        predicted_label=table.predicted_label,
        logits=table.logits,
    )
