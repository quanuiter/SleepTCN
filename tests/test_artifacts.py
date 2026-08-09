import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleeptcn.artifacts import (
    combined_sha256,
    load_feature_sequence,
    prediction_table_from_parts,
    save_feature_sequence,
    save_prediction_table,
)
from sleeptcn.training_data import FeatureSequence


HASH_A = "a" * 64
HASH_B = "b" * 64


def sequence(record_key: str = "SC4001E") -> FeatureSequence:
    return FeatureSequence(
        record_key=record_key,
        subject_id=record_key[:5],
        preprocess_version="paper_raw_v1",
        extractor_id="cnn15_fold0_seed42",
        features=np.arange(12, dtype=np.float32).reshape(3, 4),
        labels=np.array([0, -1, 2], dtype=np.int8),
        original_epoch_index=np.array([100, 101, 102], dtype=np.int32),
    )


class FeatureArtifactTests(unittest.TestCase):
    def test_round_trip_and_metadata_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "feature.npz"
            save_feature_sequence(
                path,
                sequence(),
                extractor_sha256=HASH_A,
                split_sha256=HASH_B,
                outer_fold=0,
                seed=42,
            )
            loaded = load_feature_sequence(
                path,
                expected_extractor_sha256=HASH_A,
                expected_split_sha256=HASH_B,
                expected_outer_fold=0,
                expected_seed=42,
            )
            np.testing.assert_array_equal(loaded.features, sequence().features)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_feature_sequence(
                    path,
                    expected_extractor_sha256="c" * 64,
                    expected_split_sha256=HASH_B,
                    expected_outer_fold=0,
                    expected_seed=42,
                )
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_combined_hash_is_order_independent(self) -> None:
        first = combined_sha256({"one": HASH_A, "two": HASH_B})
        second = combined_sha256({"two": HASH_B, "one": HASH_A})
        self.assertEqual(first, second)


class PredictionArtifactTests(unittest.TestCase):
    def test_ignored_epoch_is_not_exported(self) -> None:
        logits = np.array(
            [[4, 0, 0, 0, 0], [0, 4, 0, 0, 0], [0, 0, 4, 0, 0]],
            dtype=np.float32,
        )
        table = prediction_table_from_parts([(sequence(), logits)])
        self.assertEqual(table.true_label.tolist(), [0, 2])
        self.assertEqual(table.original_epoch_index.tolist(), [100, 102])
        self.assertEqual(table.metrics()["accuracy"], 1.0)

    def test_duplicate_record_is_rejected(self) -> None:
        logits = np.zeros((3, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "duplicate prediction record"):
            prediction_table_from_parts(
                [(sequence(), logits), (sequence(), logits)]
            )

    def test_prediction_file_requires_role_and_hashes(self) -> None:
        logits = np.zeros((3, 5), dtype=np.float32)
        table = prediction_table_from_parts([(sequence(), logits)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npz"
            metadata = {
                "experiment_id": "E0",
                "outer_fold": 0,
                "seed": 42,
                "split_sha256": HASH_B,
                "checkpoint_sha256": HASH_A,
                "data_variant": "paper_raw_v1",
                "role": "test",
            }
            save_prediction_table(path, table, metadata)
            with np.load(path, allow_pickle=False) as npz:
                self.assertEqual(npz["true_label"].tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
