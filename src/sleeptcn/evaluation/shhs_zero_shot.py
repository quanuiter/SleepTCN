"""Torch-free artifact and summary helpers for SHHS zero-shot evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_prediction_artifact(
    path: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one fold prediction and enforce its provenance/alignment contract."""

    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"].item()))
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{path}: metadata mismatch {mismatches}")
        probabilities = npz["probabilities"].copy()
        y = npz["y"].copy()
        valid = npz["valid_mask"].copy()
        indices = npz["original_epoch_index"].copy()
    if probabilities.shape != (len(y), 5) or not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: invalid probabilities")
    if (
        y.dtype != np.int8
        or valid.dtype != np.bool_
        or indices.dtype != np.int32
        or valid.shape != y.shape
        or indices.shape != y.shape
        or not np.array_equal(valid, y >= 0)
    ):
        raise ValueError(f"{path}: invalid valid mask")
    return metadata, probabilities, y, valid, indices


def ensemble_probabilities(parts: Iterable[np.ndarray]) -> np.ndarray:
    """Average exactly ten fold probabilities using a float64 accumulator."""

    matrices = list(parts)
    if len(matrices) != 10:
        raise ValueError("Locked ensemble requires exactly ten folds")
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices):
        raise ValueError("Fold probability shapes differ")
    accumulator = np.zeros(shape, dtype=np.float64)
    for matrix in matrices:
        accumulator += matrix.astype(np.float64, copy=False)
    mean = (accumulator / 10.0).astype(np.float32)
    if not np.allclose(mean.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Ensemble probabilities do not sum to one")
    return mean


def confusion_matrix(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.int64)
    for truth, predicted in zip(y, prediction, strict=True):
        matrix[int(truth), int(predicted)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    """Return the compact metric schema used by SHHS result manifests."""

    total = int(matrix.sum())
    per_class_f1 = []
    per_class_recall = []
    for index in range(5):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        per_class_f1.append(float(f1))
        per_class_recall.append(float(recall))
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    expected = (
        float(np.dot(matrix.sum(axis=1), matrix.sum(axis=0)) / (total * total))
        if total
        else 0.0
    )
    kappa = 0.0 if math.isclose(expected, 1.0) else (accuracy - expected) / (1.0 - expected)
    return {
        "macro_f1": float(np.mean(per_class_f1)),
        "accuracy": accuracy,
        "cohen_kappa": float(kappa),
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": matrix.tolist(),
        "valid_epochs": total,
    }


__all__ = [
    "confusion_matrix",
    "ensemble_probabilities",
    "load_prediction_artifact",
    "metrics_from_confusion",
]
