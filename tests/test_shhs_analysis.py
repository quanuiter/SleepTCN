from __future__ import annotations

import numpy as np

from sleeptcn.shhs_analysis import (
    _mean_paired_bootstrap,
    _pooled_metric_bootstrap,
    _wilcoxon_summary,
)


def test_mean_paired_bootstrap_uses_subject_mean_estimand() -> None:
    left = np.asarray([0.8, 0.7, 0.6], dtype=np.float64)
    right = np.asarray([0.7, 0.5, 0.6], dtype=np.float64)
    samples = np.asarray([[0, 1, 2], [0, 0, 0], [1, 1, 1]], dtype=np.int32)
    result = _mean_paired_bootstrap(left, right, samples)
    assert np.isclose(result["observed_difference"], 0.1)
    assert result["ci95_low"] <= result["observed_difference"]
    assert result["ci95_high"] >= result["observed_difference"]


def test_wilcoxon_summary_counts_direction_without_subject_identifiers() -> None:
    result = _wilcoxon_summary(
        np.asarray([0.8, 0.4, 0.5]), np.asarray([0.7, 0.5, 0.5])
    )
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 1)
    assert "subject_id" not in result


def test_pooled_bootstrap_uses_paired_confusion_matrices() -> None:
    left = np.zeros((2, 5, 5), dtype=np.int64)
    right = np.zeros((2, 5, 5), dtype=np.int64)
    for subject in range(2):
        left[subject] = np.eye(5, dtype=np.int64) * 3
        right[subject] = np.eye(5, dtype=np.int64) * 2
        right[subject, 0, 0] = 1
        right[subject, 0, 1] = 1
    samples = np.asarray([[0, 1], [0, 0], [1, 1]], dtype=np.int32)
    result = _pooled_metric_bootstrap(left, right, samples, "macro_f1")
    assert result["observed_difference"] > 0
    assert result["ci95_low"] > 0
