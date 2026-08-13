"""Tien ich phan tich bat cap va vung chuyen pha cho Gate 8."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from .metrics import compute_metrics, confusion_matrix_5, metrics_from_confusion
from .statistics import PredictionArrays, assert_paired


MetricFunction = Callable[[np.ndarray, np.ndarray], float]


def transition_mask(
    predictions: PredictionArrays,
    *,
    radius: int,
    stage_pair: tuple[int, int] | None = None,
) -> np.ndarray:
    if radius < 0:
        raise ValueError("transition radius must be non-negative")
    if stage_pair is not None and (
        len(stage_pair) != 2
        or stage_pair[0] == stage_pair[1]
        or any(stage not in range(5) for stage in stage_pair)
    ):
        raise ValueError("invalid transition stage pair")
    result = np.zeros(len(predictions.true_label), dtype=bool)
    for record in np.unique(predictions.record_key):
        positions = np.flatnonzero(predictions.record_key == record)
        order = np.argsort(predictions.original_epoch_index[positions])
        positions = positions[order]
        indices = predictions.original_epoch_index[positions]
        labels = predictions.true_label[positions]
        if len(positions) < 2:
            continue
        segment = np.cumsum(np.r_[False, np.diff(indices) != 1])
        anchors = np.flatnonzero(
            (np.diff(indices) == 1) & (labels[:-1] != labels[1:])
        ) + 1
        for anchor_position in anchors:
            if stage_pair is not None and set(
                (int(labels[anchor_position - 1]), int(labels[anchor_position]))
            ) != set(stage_pair):
                continue
            selected = (
                (segment == segment[anchor_position])
                & (np.abs(indices - indices[anchor_position]) <= radius)
            )
            result[positions[selected]] = True
    return result


def subset_predictions(
    predictions: PredictionArrays, selected: np.ndarray
) -> PredictionArrays:
    if selected.dtype != bool or selected.shape != predictions.true_label.shape:
        raise ValueError("invalid prediction subset mask")
    if not selected.any():
        raise ValueError("prediction subset is empty")
    return PredictionArrays(
        subject_id=predictions.subject_id[selected],
        record_key=predictions.record_key[selected],
        original_epoch_index=predictions.original_epoch_index[selected],
        true_label=predictions.true_label[selected],
        predicted_label=predictions.predicted_label[selected],
    )


def n1_recall(true: np.ndarray, predicted: np.ndarray) -> float:
    selected = true == 1
    if not selected.any():
        return float("nan")
    return float(np.mean(predicted[selected] == 1))


def _macro_f1(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(compute_metrics(true, predicted)["macro_f1"])


def paired_cluster_bootstrap_subset(
    proposed: PredictionArrays,
    reference: PredictionArrays,
    selected: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    assert_paired(proposed, reference)
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    proposed = subset_predictions(proposed, selected).sorted()
    reference = subset_predictions(reference, selected).sorted()
    assert_paired(proposed, reference)
    subjects = np.unique(proposed.subject_id)
    proposed_cm = np.stack([
        confusion_matrix_5(
            proposed.true_label[proposed.subject_id == subject],
            proposed.predicted_label[proposed.subject_id == subject],
        )
        for subject in subjects
    ])
    reference_cm = np.stack([
        confusion_matrix_5(
            reference.true_label[reference.subject_id == subject],
            reference.predicted_label[reference.subject_id == subject],
        )
        for subject in subjects
    ])
    observed = (
        metrics_from_confusion(proposed_cm.sum(axis=0))["macro_f1"]
        - metrics_from_confusion(reference_cm.sum(axis=0))["macro_f1"]
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(subjects), size=len(subjects))
        differences[index] = (
            metrics_from_confusion(proposed_cm[sampled].sum(axis=0))["macro_f1"]
            - metrics_from_confusion(reference_cm[sampled].sum(axis=0))["macro_f1"]
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "observed_difference": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "subjects": int(len(subjects)),
        "selected_epochs": int(selected.sum()),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def paired_subject_subset_test(
    proposed: PredictionArrays,
    reference: PredictionArrays,
    selected: np.ndarray,
    *,
    metric: MetricFunction = _macro_f1,
) -> dict[str, Any]:
    assert_paired(proposed, reference)
    subjects = np.unique(proposed.subject_id[selected])
    left, right, included = [], [], []
    for subject in subjects:
        positions = selected & (proposed.subject_id == subject)
        left_value = metric(proposed.true_label[positions], proposed.predicted_label[positions])
        right_value = metric(reference.true_label[positions], reference.predicted_label[positions])
        if np.isfinite(left_value) and np.isfinite(right_value):
            included.append(subject)
            left.append(left_value)
            right.append(right_value)
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    difference = left_array - right_array
    if not len(difference):
        raise ValueError("no subjects support the selected metric")
    if np.all(difference == 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(
            left_array,
            right_array,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "subjects": len(included),
        "statistic": statistic,
        "p_value": p_value,
        "median_subject_difference": float(np.median(difference)),
        "wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "losses": int(np.sum(difference < 0)),
    }


def descriptive_views(
    predictions: PredictionArrays, transition_radius_1: np.ndarray
) -> dict[str, Any]:
    overall = compute_metrics(predictions.true_label, predictions.predicted_label)
    transition = compute_metrics(
        predictions.true_label[transition_radius_1],
        predictions.predicted_label[transition_radius_1],
    )
    n1_transition = transition_radius_1 & (predictions.true_label == 1)
    n1_stable = (~transition_radius_1) & (predictions.true_label == 1)
    return {
        "overall": {
            "accuracy": overall["accuracy"],
            "macro_f1": overall["macro_f1"],
            "n1_f1": overall["per_class"]["N1"]["f1"],
            "n1_recall": overall["per_class"]["N1"]["recall"],
        },
        "transition_radius_1": {
            "epochs": int(transition_radius_1.sum()),
            "macro_f1": transition["macro_f1"],
            "n1_recall": n1_recall(
                predictions.true_label[n1_transition],
                predictions.predicted_label[n1_transition],
            ),
        },
        "stable_n1": {
            "epochs": int(n1_stable.sum()),
            "recall": n1_recall(
                predictions.true_label[n1_stable],
                predictions.predicted_label[n1_stable],
            ),
        },
    }
