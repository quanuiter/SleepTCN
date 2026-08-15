from __future__ import annotations

import numpy as np

from sleeptcn.shhs_analysis import _pooled_metric_bootstrap
from sleeptcn.shhs_e3_e2_analysis import (
    _pooled_metrics_bootstrap,
    supporting_evidence_decision,
)


def test_supporting_evidence_requires_positive_interval_and_p_below_005() -> None:
    assert (
        supporting_evidence_decision(0.001, 0.049)
        == "supporting_evidence_on_opened_shhs_sample"
    )
    assert (
        supporting_evidence_decision(0.0, 0.001)
        == "not_supported_on_opened_shhs_sample"
    )
    assert (
        supporting_evidence_decision(0.001, 0.05)
        == "not_supported_on_opened_shhs_sample"
    )


def test_grouped_bootstrap_matches_existing_single_metric_implementation() -> None:
    left = np.stack(
        [np.eye(5, dtype=np.int64) * value for value in (2, 3, 4)]
    )
    right = left.copy()
    right[:, 1, 1] -= 1
    right[:, 1, 2] += 1
    samples = np.asarray([[0, 1, 2], [0, 0, 1], [2, 2, 1]], dtype=np.int32)

    grouped = _pooled_metrics_bootstrap(
        left,
        right,
        samples,
        metrics=("macro_f1", "accuracy", "cohen_kappa"),
    )
    for metric in ("macro_f1", "accuracy", "cohen_kappa"):
        expected = _pooled_metric_bootstrap(left, right, samples, metric)
        assert grouped[metric] == expected
