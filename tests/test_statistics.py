import unittest

import numpy as np

from sleeptcn.statistics import (
    PredictionArrays,
    assert_paired,
    holm_adjust,
    paired_cluster_bootstrap,
    paired_subject_wilcoxon,
)
from sleeptcn.metrics import compute_metrics


def predictions(predicted: list[int]) -> PredictionArrays:
    return PredictionArrays(
        subject_id=np.array(["S1", "S1", "S2", "S2"]),
        record_key=np.array(["R1", "R1", "R2", "R2"]),
        original_epoch_index=np.array([0, 1, 0, 1], dtype=np.int32),
        true_label=np.array([0, 1, 0, 1], dtype=np.int8),
        predicted_label=np.array(predicted, dtype=np.int8),
    )


class StatisticsTests(unittest.TestCase):
    def test_pairing_rejects_different_truth(self) -> None:
        left, right = predictions([0, 1, 0, 1]), predictions([0, 1, 0, 1])
        right.true_label[0] = 2
        with self.assertRaisesRegex(ValueError, "true_label"):
            assert_paired(left, right)

    def test_bootstrap_and_wilcoxon_are_deterministic(self) -> None:
        proposed = predictions([0, 1, 0, 1])
        reference = predictions([0, 0, 0, 0])
        first = paired_cluster_bootstrap(proposed, reference, resamples=50, seed=7)
        second = paired_cluster_bootstrap(proposed, reference, resamples=50, seed=7)
        self.assertEqual(first, second)
        test = paired_subject_wilcoxon(proposed, reference)
        self.assertEqual(test["subjects"], 2)
        self.assertEqual(test["wins"], 2)

    def test_cluster_bootstrap_matches_direct_epoch_resampling(self) -> None:
        proposed = predictions([0, 1, 0, 1])
        reference = predictions([0, 0, 0, 0])
        resamples, seed = 25, 11
        optimized = paired_cluster_bootstrap(
            proposed, reference, resamples=resamples, seed=seed
        )
        rng = np.random.default_rng(seed)
        subjects = np.unique(proposed.subject_id)
        blocks = [np.flatnonzero(proposed.subject_id == subject) for subject in subjects]
        direct = []
        for _ in range(resamples):
            sampled = rng.integers(0, len(blocks), size=len(blocks))
            positions = np.concatenate([blocks[index] for index in sampled])
            direct.append(
                compute_metrics(
                    proposed.true_label[positions], proposed.predicted_label[positions]
                )["macro_f1"]
                - compute_metrics(
                    reference.true_label[positions], reference.predicted_label[positions]
                )["macro_f1"]
            )
        low, high = np.quantile(np.asarray(direct), [0.025, 0.975])
        self.assertAlmostEqual(optimized["ci95_low"], low)
        self.assertAlmostEqual(optimized["ci95_high"], high)

    def test_holm_adjustment_preserves_original_order(self) -> None:
        adjusted = holm_adjust([0.03, 0.01, 0.04])
        np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])


if __name__ == "__main__":
    unittest.main()
