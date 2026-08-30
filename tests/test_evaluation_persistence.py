import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleeptcn.evaluation import PredictionTable, save_role_artifacts


HASH_A = "a" * 64
HASH_B = "b" * 64


def table() -> PredictionTable:
    return PredictionTable(
        subject_id=np.array(["SC400"], dtype="U5"),
        record_key=np.array(["SC4001E"], dtype="U7"),
        original_epoch_index=np.array([0], dtype=np.int32),
        true_label=np.array([0], dtype=np.int8),
        predicted_label=np.array([0], dtype=np.int8),
        logits=np.array([[4, 0, 0, 0, 0]], dtype=np.float32),
    )


class EvaluationPersistenceTests(unittest.TestCase):
    def test_role_writer_keeps_prediction_and_metric_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = save_role_artifacts(
                root,
                table(),
                "validation",
                prediction_metadata={
                    "experiment_id": "E0",
                    "outer_fold": 0,
                    "seed": 42,
                    "split_sha256": HASH_A,
                    "checkpoint_sha256": HASH_B,
                    "data_variant": "paper_raw_v1",
                    "role": "validation",
                },
                metrics_metadata={
                    "experiment_id": "E0",
                    "outer_fold": 0,
                    "seed": 42,
                    "role": "validation",
                    "checkpoint_sha256": HASH_B,
                },
            )
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertTrue((root / "predictions" / "validation.npz").is_file())
            payload = json.loads(
                (root / "metrics" / "validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["metrics"], metrics)
            self.assertEqual(payload["metadata"]["role"], "validation")

    def test_role_writer_rejects_mismatched_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "does not match"):
                save_role_artifacts(
                    Path(temporary),
                    table(),
                    "test",
                    prediction_metadata={"role": "validation"},
                    metrics_metadata={},
                )


if __name__ == "__main__":
    unittest.main()
