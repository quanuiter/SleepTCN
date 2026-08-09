"""Chi so danh gia 5 lop, khong ngam bo qua lop hiem."""

from __future__ import annotations

import numpy as np


N_CLASSES = 5
STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")


def confusion_matrix_5(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    true = np.asarray(true)
    predicted = np.asarray(predicted)
    if true.shape != predicted.shape or true.ndim != 1:
        raise ValueError("true and predicted must be one-dimensional with same shape")
    valid = (true >= 0) & (true < N_CLASSES)
    unsupported_true = ~(valid | (true == -1) | (true == -100))
    if np.any(unsupported_true):
        raise ValueError("true contains unsupported label")
    selected_true = true[valid].astype(np.int64, copy=False)
    selected_predicted = predicted[valid].astype(np.int64, copy=False)
    if len(selected_true) == 0:
        raise ValueError("no valid true labels")
    if np.any((selected_predicted < 0) | (selected_predicted >= N_CLASSES)):
        raise ValueError("predicted contains invalid class for a valid target")
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(matrix, (selected_true, selected_predicted), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(matrix)
    if matrix.shape != (N_CLASSES, N_CLASSES) or np.any(matrix < 0):
        raise ValueError("confusion matrix must be nonnegative with shape (5,5)")
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("empty confusion matrix")
    diagonal = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        diagonal, predicted_count, out=np.zeros(5), where=predicted_count != 0
    )
    recall = np.divide(diagonal, support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(5),
        where=(precision + recall) != 0,
    )
    observed = float(diagonal.sum() / total)
    expected = float(np.dot(support, predicted_count) / (total * total))
    kappa = 0.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "n_valid_epochs": total,
        "accuracy": observed,
        "macro_f1": float(f1.mean()),
        "cohen_kappa": float(kappa),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            stage: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, stage in enumerate(STAGE_NAMES)
        },
    }


def compute_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    return metrics_from_confusion(confusion_matrix_5(true, predicted))
