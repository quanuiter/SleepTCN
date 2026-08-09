import unittest

import numpy as np
import torch
from torch import nn

from sleeptcn.features import (
    class_specific_weights,
    expected_15cnn_keys,
    extract_15cnn_features,
    shift_within_record,
)
from sleeptcn.metrics import compute_metrics, confusion_matrix_5


class ConstantLogits(nn.Module):
    def __init__(self, selected: int) -> None:
        super().__init__()
        values = torch.zeros(5)
        values[selected] = 2.0
        self.register_buffer("values", values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(len(x), -1)


class ManipulationTests(unittest.TestCase):
    def test_shift_duplicates_only_own_record_boundary(self) -> None:
        signals = np.repeat(np.arange(3, dtype=np.float32)[:, None], 3000, axis=1)
        previous = shift_within_record(signals, "previous")
        following = shift_within_record(signals, "next")
        np.testing.assert_array_equal(previous[:, 0], [0, 0, 1])
        np.testing.assert_array_equal(following[:, 0], [1, 2, 2])

    def test_class_specific_weights_balance_target_vs_rest(self) -> None:
        counts = np.array([50, 10, 20, 10, 10])
        weights = class_specific_weights(1, counts)
        self.assertAlmostEqual(float(weights[1] * counts[1]), 50.0, places=5)
        self.assertAlmostEqual(float(weights[0] * counts[[0, 2, 3, 4]].sum()), 50.0, places=5)

    def test_feature_order_and_probability_blocks(self) -> None:
        signals = np.zeros((2, 3000), dtype=np.float32)
        models = {
            key: ConstantLogits(index % 5)
            for index, key in enumerate(expected_15cnn_keys())
        }
        features = extract_15cnn_features(signals, models, batch_size=1)
        self.assertEqual(features.shape, (2, 75))
        for index in range(15):
            block = features[0, index * 5 : (index + 1) * 5]
            self.assertEqual(int(block.argmax()), index % 5)
            self.assertAlmostEqual(float(block.sum()), 1.0, places=6)


class MetricTests(unittest.TestCase):
    def test_ignored_and_padding_targets_are_excluded(self) -> None:
        true = np.array([0, 1, 2, 3, 4, -1, -100])
        predicted = np.array([0, 1, 2, 0, 4, 3, 2])
        matrix = confusion_matrix_5(true, predicted)
        self.assertEqual(int(matrix.sum()), 5)
        self.assertEqual(int(matrix[3, 0]), 1)
        metrics = compute_metrics(true, predicted)
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertEqual(metrics["n_valid_epochs"], 5)

    def test_all_five_classes_always_reported(self) -> None:
        metrics = compute_metrics(np.array([0, 0]), np.array([0, 0]))
        self.assertEqual(list(metrics["per_class"]), ["W", "N1", "N2", "N3", "REM"])
        self.assertEqual(metrics["per_class"]["N1"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
