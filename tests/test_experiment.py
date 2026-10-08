import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sleeptcn.experiment import build_context, train_sequence_model
from sleeptcn.training_data import FeatureSequence


class ContextTests(unittest.TestCase):
    def test_smoke_cannot_unlock_test_role(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "must not evaluate"):
            build_context(
                root,
                "E2",
                0,
                42,
                "cpu",
                smoke=True,
                allow_test_evaluation=True,
                num_workers=0,
            )

    def test_full_context_hashes_config_and_split(self) -> None:
        root = Path(__file__).resolve().parents[1]
        context = build_context(
            root,
            "E3",
            9,
            123,
            "cpu",
            smoke=False,
            allow_test_evaluation=False,
            num_workers=0,
        )
        self.assertEqual(context.data_variant, "filtered_v2")
        self.assertEqual(len(context.config_sha256), 64)
        self.assertEqual(len(context.split_sha256), 64)
        self.assertIn("full", context.run_root.parts)

    def test_sequence_initialization_is_controlled_by_campaign_seed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        context = build_context(root, "E3", 0, 42, "cpu", smoke=True,
                                allow_test_evaluation=False, num_workers=0)
        sequence = FeatureSequence(
            "SC4001E", "SC400", "filtered_v2", "synthetic-test",
            np.zeros((3, 128), dtype=np.float32), np.array([0, 2, 3]),
            np.arange(3, dtype=np.int32),
        )

        def initial_weights(ambient_seed: int, campaign_seed: int) -> dict:
            torch.manual_seed(ambient_seed)
            torch.rand(37)  # Simulate an arbitrary preceding stage consuming the RNG.
            with patch("sleeptcn.experiment._loader"), \
                    patch("sleeptcn.experiment._fit_kwargs", return_value={}), \
                    patch("sleeptcn.experiment.fit_model"), \
                    patch("sleeptcn.experiment._mark_stage_complete"), \
                    patch("sleeptcn.experiment.load_sequence_checkpoint", return_value=Path("unused.pt")):
                model, _ = train_sequence_model(replace(context, seed=campaign_seed),
                                                [sequence], [sequence], "tcn")
            return {key: value.detach().clone() for key, value in model.state_dict().items()}

        first = initial_weights(1, 42)
        second = initial_weights(999, 42)
        other = initial_weights(1, 123)
        self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))
        self.assertTrue(any(not torch.equal(first[key], other[key]) for key in first))


if __name__ == "__main__":
    unittest.main()
