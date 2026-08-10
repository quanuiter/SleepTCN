"""Paired subject-level analysis for completed cross-validation predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from .metrics import compute_metrics


@dataclass(frozen=True)
class PredictionArrays:
    subject_id: np.ndarray
    record_key: np.ndarray
    original_epoch_index: np.ndarray
    true_label: np.ndarray
    predicted_label: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.true_label)
        arrays = (
            self.subject_id,
            self.record_key,
            self.original_epoch_index,
            self.predicted_label,
        )
        if any(array.ndim != 1 or len(array) != length for array in arrays):
            raise ValueError("prediction arrays are not aligned")

    def sorted(self) -> "PredictionArrays":
        order = np.lexsort(
            (self.original_epoch_index, self.record_key, self.subject_id)
        )
        return PredictionArrays(
            subject_id=self.subject_id[order],
            record_key=self.record_key[order],
            original_epoch_index=self.original_epoch_index[order],
            true_label=self.true_label[order],
            predicted_label=self.predicted_label[order],
        )


def assert_paired(left: PredictionArrays, right: PredictionArrays) -> None:
    left_sorted, right_sorted = left.sorted(), right.sorted()
    for name in ("subject_id", "record_key", "original_epoch_index", "true_label"):
        if not np.array_equal(getattr(left_sorted, name), getattr(right_sorted, name)):
            raise ValueError(f"paired predictions differ in {name}")


def subject_metric_values(
    predictions: PredictionArrays, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    if metric not in {"macro_f1", "accuracy", "cohen_kappa"}:
        raise ValueError(f"unsupported subject metric: {metric}")
    subjects = np.unique(predictions.subject_id)
    values = []
    for subject in subjects:
        selected = predictions.subject_id == subject
        values.append(
            float(
                compute_metrics(
                    predictions.true_label[selected],
                    predictions.predicted_label[selected],
                )[metric]
            )
        )
    return subjects, np.asarray(values, dtype=np.float64)


def paired_cluster_bootstrap(
    proposed: PredictionArrays,
    reference: PredictionArrays,
    *,
    metric: str = "macro_f1",
    resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, float]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    assert_paired(proposed, reference)
    proposed, reference = proposed.sorted(), reference.sorted()
    subjects = np.unique(proposed.subject_id)
    blocks = [np.flatnonzero(proposed.subject_id == subject) for subject in subjects]
    observed = float(
        compute_metrics(proposed.true_label, proposed.predicted_label)[metric]
        - compute_metrics(reference.true_label, reference.predicted_label)[metric]
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        positions = np.concatenate([blocks[item] for item in sampled])
        differences[index] = float(
            compute_metrics(
                proposed.true_label[positions], proposed.predicted_label[positions]
            )[metric]
            - compute_metrics(
                reference.true_label[positions], reference.predicted_label[positions]
            )[metric]
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "observed_difference": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def paired_subject_wilcoxon(
    proposed: PredictionArrays,
    reference: PredictionArrays,
    *,
    metric: str = "macro_f1",
) -> dict[str, object]:
    assert_paired(proposed, reference)
    subjects_left, left = subject_metric_values(proposed, metric)
    subjects_right, right = subject_metric_values(reference, metric)
    if not np.array_equal(subjects_left, subjects_right):
        raise ValueError("subject membership differs")
    difference = left - right
    if np.allclose(difference, 0.0, rtol=0.0, atol=0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            left,
            right,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "metric": metric,
        "subjects": int(len(subjects_left)),
        "statistic": statistic,
        "p_value": p_value,
        "median_subject_difference": float(np.median(difference)),
        "wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "losses": int(np.sum(difference < 0)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values or any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be a non-empty list in [0,1]")
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(original_index)])
        running = max(running, candidate)
        adjusted[int(original_index)] = running
    return adjusted.tolist()
