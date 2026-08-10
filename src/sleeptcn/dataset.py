"""Bộ nạp NPZ đã kiểm định cho SleepTCN."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = {
    "x",
    "y",
    "valid_mask",
    "original_epoch_index",
    "record_key",
    "subject_id",
    "preprocess_version",
    "source_hashes_verified",
    "sampling_rate_hz",
    "epoch_seconds",
    "samples_per_epoch",
    "trim_anchor_policy",
}


@dataclass(frozen=True)
class RecordInfo:
    path: Path
    record_key: str
    subject_id: str
    preprocess_version: str
    epochs: int
    valid_epochs: int
    ignored_epochs: int
    label_counts: dict[int, int]


@dataclass(frozen=True)
class SleepRecord:
    info: RecordInfo
    x: np.ndarray
    y: np.ndarray
    valid_mask: np.ndarray
    original_epoch_index: np.ndarray


def _scalar(npz: Any, name: str) -> Any:
    value = npz[name]
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def _validate_metadata(
    npz: Any, path: Path, expected_variant: str
) -> tuple[str, str, str]:
    missing = sorted(REQUIRED_FIELDS - set(npz.files))
    if missing:
        raise ValueError(f"{path.name}: missing fields {missing}")
    record_key = str(_scalar(npz, "record_key"))
    subject_id = str(_scalar(npz, "subject_id"))
    variant = str(_scalar(npz, "preprocess_version"))
    if record_key != path.stem:
        raise ValueError(f"{path.name}: record_key metadata mismatch")
    if subject_id != record_key[:5]:
        raise ValueError(f"{path.name}: subject_id metadata mismatch")
    if variant != expected_variant:
        raise ValueError(
            f"{path.name}: expected {expected_variant}, found {variant}"
        )
    if not bool(_scalar(npz, "source_hashes_verified")):
        raise ValueError(f"{path.name}: source hashes were not verified")
    if float(_scalar(npz, "sampling_rate_hz")) != 100.0:
        raise ValueError(f"{path.name}: sampling rate is not 100 Hz")
    if int(_scalar(npz, "epoch_seconds")) != 30:
        raise ValueError(f"{path.name}: epoch length is not 30 seconds")
    if int(_scalar(npz, "samples_per_epoch")) != 3000:
        raise ValueError(f"{path.name}: samples_per_epoch is not 3000")
    if str(_scalar(npz, "trim_anchor_policy")) != "true_sleep_n1_to_rem":
        raise ValueError(f"{path.name}: unexpected trim policy")
    return record_key, subject_id, variant


def _validate_label_arrays(
    y: np.ndarray, valid_mask: np.ndarray, indices: np.ndarray, path: Path
) -> None:
    if y.dtype != np.int8 or y.ndim != 1:
        raise ValueError(f"{path.name}: y must be one-dimensional int8")
    if valid_mask.dtype != np.bool_ or valid_mask.shape != y.shape:
        raise ValueError(f"{path.name}: invalid valid_mask")
    if not np.array_equal(valid_mask, y >= 0):
        raise ValueError(f"{path.name}: valid_mask differs from y >= 0")
    if indices.dtype != np.int32 or indices.shape != y.shape:
        raise ValueError(f"{path.name}: invalid original_epoch_index")
    if len(indices) and not np.all(np.diff(indices) == 1):
        raise ValueError(f"{path.name}: epoch indices are not consecutive")
    if not np.isin(y, [-1, 0, 1, 2, 3, 4]).all():
        raise ValueError(f"{path.name}: invalid label value")


def inspect_record(path: Path, expected_variant: str) -> RecordInfo:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as npz:
        record_key, subject_id, variant = _validate_metadata(
            npz, path, expected_variant
        )
        y = npz["y"]
        valid_mask = npz["valid_mask"]
        indices = npz["original_epoch_index"]
        _validate_label_arrays(y, valid_mask, indices, path)
        label_counts = {
            label: int(np.sum(y == label)) for label in (-1, 0, 1, 2, 3, 4)
        }
        return RecordInfo(
            path=path,
            record_key=record_key,
            subject_id=subject_id,
            preprocess_version=variant,
            epochs=len(y),
            valid_epochs=int(valid_mask.sum()),
            ignored_epochs=int((y == -1).sum()),
            label_counts=label_counts,
        )


def load_record(path: Path, expected_variant: str) -> SleepRecord:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as npz:
        record_key, subject_id, variant = _validate_metadata(
            npz, path, expected_variant
        )
        x = npz["x"]
        y = npz["y"]
        valid_mask = npz["valid_mask"]
        indices = npz["original_epoch_index"]
        _validate_label_arrays(y, valid_mask, indices, path)
        if x.dtype != np.float32 or x.shape != (len(y), 3000):
            raise ValueError(f"{path.name}: x must have shape (epochs, 3000) float32")
        if not np.isfinite(x).all():
            raise ValueError(f"{path.name}: x contains NaN or infinity")
        label_counts = {
            label: int(np.sum(y == label)) for label in (-1, 0, 1, 2, 3, 4)
        }
        info = RecordInfo(
            path=path,
            record_key=record_key,
            subject_id=subject_id,
            preprocess_version=variant,
            epochs=len(y),
            valid_epochs=int(valid_mask.sum()),
            ignored_epochs=int((y == -1).sum()),
            label_counts=label_counts,
        )
        return SleepRecord(
            info=info,
            x=x.copy(),
            y=y.copy(),
            valid_mask=valid_mask.copy(),
            original_epoch_index=indices.copy(),
        )


def load_split_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported split manifest schema")
    return manifest


def paths_for_role(
    processed_root: Path,
    split_manifest: dict[str, Any],
    outer_fold: int,
    role: str,
    expected_variant: str,
) -> list[Path]:
    if role not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid role: {role}")
    if expected_variant not in split_manifest["compatible_variants"]:
        raise ValueError(f"Variant not permitted by split manifest: {expected_variant}")
    runs = {
        int(run["outer_fold"]): run for run in split_manifest["outer_runs"]
    }
    if outer_fold not in runs:
        raise ValueError(f"Unknown outer fold: {outer_fold}")
    record_keys = runs[outer_fold][role]["record_keys"]
    paths = [processed_root / expected_variant / f"{key}.npz" for key in record_keys]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split NPZ files: {missing}")
    return paths
