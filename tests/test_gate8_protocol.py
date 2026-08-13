import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleeptcn.gate8 import (
    context_groups,
    load_protocol,
    mask_feature_sequences,
    train_replacement_mean,
)
from sleeptcn.gate8_analysis import transition_mask
from sleeptcn.statistics import PredictionArrays
from sleeptcn.training_data import FeatureSequence


def sequence(key: str, features: np.ndarray, labels: list[int]) -> FeatureSequence:
    return FeatureSequence(
        record_key=key,
        subject_id=key[:5],
        preprocess_version="paper_raw_v1",
        extractor_id="cnn15_test",
        features=features.astype(np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        original_epoch_index=np.arange(len(labels), dtype=np.int32),
    )


class Gate8MaskingTests(unittest.TestCase):
    def test_replacement_mean_uses_valid_train_labels_only(self) -> None:
        features = np.vstack(
            [
                np.zeros(75, dtype=np.float32),
                np.full(75, 2.0, dtype=np.float32),
                np.full(75, 1000.0, dtype=np.float32),
            ]
        )
        mean, count = train_replacement_mean(
            [sequence("SC4001E", features, [0, 1, -1])]
        )
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(mean, np.ones(75, dtype=np.float32))

    def test_mask_preserves_75_dimensions_and_retained_groups(self) -> None:
        features = np.arange(150, dtype=np.float32).reshape(2, 75)
        original = sequence("SC4001E", features, [0, 1])
        mean = np.arange(75, dtype=np.float32) + 500
        masked = mask_feature_sequences([original], "CP", mean)[0]
        self.assertEqual(masked.features.shape, (2, 75))
        np.testing.assert_array_equal(masked.features[:, :50], features[:, :50])
        np.testing.assert_array_equal(
            masked.features[:, 50:], np.broadcast_to(mean[50:], (2, 25))
        )
        np.testing.assert_array_equal(original.features, features)

    def test_condition_groups_are_frozen(self) -> None:
        self.assertEqual(context_groups("CP"), ("C", "P"))
        self.assertEqual(context_groups("CN"), ("C", "N"))
        self.assertEqual(context_groups("C"), ("C",))


class Gate8TransitionTests(unittest.TestCase):
    def predictions(self) -> PredictionArrays:
        return PredictionArrays(
            subject_id=np.array(["SC400"] * 7),
            record_key=np.array(["SC4001E"] * 7),
            original_epoch_index=np.array([0, 1, 2, 3, 6, 7, 8], dtype=np.int32),
            true_label=np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int8),
            predicted_label=np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int8),
        )

    def test_radius_one_uses_first_new_stage_epoch_as_anchor(self) -> None:
        selected = transition_mask(self.predictions(), radius=1)
        np.testing.assert_array_equal(
            selected, np.array([False, True, True, True, True, True, True])
        )

    def test_gap_breaks_sequence_and_pair_filter_is_unordered(self) -> None:
        predictions = self.predictions()
        w_n1 = transition_mask(predictions, radius=1, stage_pair=(1, 0))
        np.testing.assert_array_equal(
            w_n1, np.array([False, True, True, True, False, False, False])
        )
        n1_n2 = transition_mask(predictions, radius=1, stage_pair=(1, 2))
        np.testing.assert_array_equal(
            n1_n2, np.array([False, False, False, False, True, True, True])
        )


class Gate8ConfigTests(unittest.TestCase):
    def test_repository_protocol_is_frozen(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol, digest = load_protocol(root)
        self.assertEqual(protocol["analysis"]["primary_comparison"], "FULL_CPN-C")
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
