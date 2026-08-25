from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sleeptcn.evaluation.shhs_zero_shot import (
    ensemble_probabilities,
    load_prediction_artifact,
    metrics_from_confusion,
)
from sleeptcn.io.serialization import atomic_savez


def test_ensemble_uses_float64_accumulator() -> None:
    parts = [
        np.asarray([[0.6 - fold * 0.001, 0.1 + fold * 0.001, 0.1, 0.1, 0.1]], dtype=np.float32)
        for fold in range(10)
    ]
    observed = ensemble_probabilities(parts)
    expected = (sum(part.astype(np.float64) for part in parts) / 10.0).astype(np.float32)
    np.testing.assert_array_equal(observed, expected)


def test_load_prediction_artifact_validates_metadata_and_alignment(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npz"
    y = np.asarray([0, 1], dtype=np.int8)
    atomic_savez(
        path,
        {
            "metadata_json": np.asarray(json.dumps({"role": "validation"})),
            "probabilities": np.asarray([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32),
            "y": y,
            "valid_mask": np.asarray([True, True], dtype=np.bool_),
            "original_epoch_index": np.asarray([3, 4], dtype=np.int32),
        },
    )
    metadata, probabilities, observed_y, valid, indices = load_prediction_artifact(
        path, {"role": "validation"}
    )
    assert metadata["role"] == "validation"
    assert probabilities.shape == (2, 5)
    np.testing.assert_array_equal(observed_y, y)
    np.testing.assert_array_equal(valid, [True, True])
    np.testing.assert_array_equal(indices, [3, 4])


def test_shhs_metrics_from_perfect_confusion_are_one() -> None:
    metrics = metrics_from_confusion(np.diag([2, 3, 4, 5, 6]).astype(np.int64))
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["valid_epochs"] == 20

