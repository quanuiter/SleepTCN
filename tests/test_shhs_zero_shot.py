from __future__ import annotations

import numpy as np

from sleeptcn.shhs_zero_shot import ensemble_probabilities, metrics_from_confusion


def test_ensemble_is_float64_accumulated_mean_of_ten_probabilities() -> None:
    parts = []
    for fold in range(10):
        matrix = np.asarray(
            [[0.6 - fold * 0.001, 0.1 + fold * 0.001, 0.1, 0.1, 0.1]],
            dtype=np.float32,
        )
        parts.append(matrix)
    observed = ensemble_probabilities(parts)
    expected = (sum(part.astype(np.float64) for part in parts) / 10.0).astype(np.float32)
    np.testing.assert_array_equal(observed, expected)


def test_metrics_from_perfect_confusion_are_one() -> None:
    matrix = np.diag([2, 3, 4, 5, 6]).astype(np.int64)
    metrics = metrics_from_confusion(matrix)
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["valid_epochs"] == 20
