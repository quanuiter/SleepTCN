"""Torch-free prediction table schema and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..io.hashing import combined_sha256
from ..io.serialization import atomic_savez
from ..metrics import compute_metrics


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
        if any(
            subject != record[:5]
            for subject, record in zip(
                self.subject_id, self.record_key, strict=True
            )
        ):
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


def load_prediction_table(path: Path) -> tuple[PredictionTable, dict[str, Any]]:
    """Load a prediction table and its JSON metadata from a locked NPZ."""

    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"].item()))
        table = PredictionTable(
            subject_id=npz["subject_id"].copy(),
            record_key=npz["record_key"].copy(),
            original_epoch_index=npz["original_epoch_index"].copy(),
            true_label=npz["true_label"].copy(),
            predicted_label=npz["predicted_label"].copy(),
            logits=npz["logits"].copy(),
        )
    return table, metadata


def save_prediction_table(
    path: Path, table: PredictionTable, metadata: dict[str, Any]
) -> None:
    """Persist a prediction table with the locked provenance contract."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        path,
        {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            "subject_id": table.subject_id,
            "record_key": table.record_key,
            "original_epoch_index": table.original_epoch_index,
            "true_label": table.true_label,
            "predicted_label": table.predicted_label,
            "logits": table.logits,
        },
    )
