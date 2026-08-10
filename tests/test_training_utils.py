import unittest

import torch
import torch.nn.functional as F

from sleeptcn.training import (
    IGNORED_LABEL,
    PAD_LABEL,
    collate_feature_sequences,
    masked_cross_entropy,
)


class CollateTests(unittest.TestCase):
    def test_preserves_ignored_and_separates_padding(self) -> None:
        first_x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        first_y = torch.tensor([0, IGNORED_LABEL, 2], dtype=torch.long)
        second_x = torch.ones((2, 4), dtype=torch.float32)
        second_y = torch.tensor([4, 1], dtype=torch.long)
        batch = collate_feature_sequences([(first_x, first_y), (second_x, second_y)])
        self.assertEqual(tuple(batch.features.shape), (2, 3, 4))
        self.assertEqual(batch.targets.tolist(), [[0, -1, 2], [4, 1, -100]])
        self.assertEqual(batch.lengths.tolist(), [3, 2])
        self.assertTrue(batch.ignored_epoch_mask[0, 1])
        self.assertTrue(batch.padding_mask[1, 2])
        self.assertFalse(batch.padding_mask[0, 1])
        self.assertFalse(batch.ignored_epoch_mask[1, 2])


class MaskedLossTests(unittest.TestCase):
    def test_matches_manual_valid_only_loss(self) -> None:
        logits = torch.tensor(
            [
                [[2.0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0], [0, 0, 2.0, 0, 0]],
                [[0, 0, 0, 0, 2.0], [0, 2.0, 0, 0, 0], [1.0, 0, 0, 0, 0]],
            ],
            requires_grad=True,
        )
        targets = torch.tensor([[0, -1, 2], [4, 1, -100]], dtype=torch.long)
        loss = masked_cross_entropy(logits, targets)
        valid = (targets >= 0) & (targets < 5)
        expected = F.cross_entropy(logits[valid], targets[valid])
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[1, 2].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0.0)

    def test_rejects_batch_without_valid_target(self) -> None:
        logits = torch.zeros((1, 2, 5), dtype=torch.float32)
        targets = torch.tensor([[IGNORED_LABEL, PAD_LABEL]], dtype=torch.long)
        with self.assertRaises(ValueError):
            masked_cross_entropy(logits, targets)


if __name__ == "__main__":
    unittest.main()
