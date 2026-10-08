import unittest
from copy import deepcopy

import torch

from sleeptcn.models import BiLSTMSleepNet, EEGResNet1D, SleepCNN, SleepTCN


class SleepCNNTests(unittest.TestCase):
    def test_shape_parameters_and_probabilities(self) -> None:
        model = SleepCNN().eval()
        x = torch.randn(2, 1, 3000)
        with torch.no_grad():
            logits = model(x)
            probabilities = model.probabilities(x)
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2495)
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(2)))


class BiLSTMTests(unittest.TestCase):
    def test_padding_is_zero_and_shape_preserved(self) -> None:
        model = BiLSTMSleepNet().eval()
        x = torch.randn(2, 7, 75)
        lengths = torch.tensor([7, 4])
        with torch.no_grad():
            logits = model(x, lengths)
        self.assertEqual(tuple(logits.shape), (2, 7, 5))
        self.assertEqual(float(logits[1, 4:].abs().sum()), 0.0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 211205)


class ResNetTests(unittest.TestCase):
    def test_fixed_input_scale_can_be_compensated_at_bias_free_stem(self) -> None:
        torch.manual_seed(19)
        model = EEGResNet1D().eval()
        # Non-default running statistics also remain unchanged by the compensation.
        model.stem[1].running_mean.copy_(torch.randn(32))
        model.stem[1].running_var.copy_(torch.rand(32) + 0.5)
        compensated = deepcopy(model)
        with torch.no_grad():
            compensated.stem[0].weight.mul_(100)
            x = 40 * torch.randn(2, 1, 3000)
            original = model(x)
            rescaled = compensated(x / 100)
        torch.testing.assert_close(rescaled, original, rtol=3e-5, atol=3e-6)

    def test_feature_and_logit_shapes(self) -> None:
        model = EEGResNet1D().eval()
        x = torch.randn(2, 1, 3000)
        with torch.no_grad():
            features = model.extract_features(x)
            logits = model(x)
        self.assertEqual(tuple(features.shape), (2, 128))
        self.assertEqual(tuple(logits.shape), (2, 5))


class TCNTests(unittest.TestCase):
    def test_shape_and_receptive_field(self) -> None:
        model = SleepTCN(input_dim=75, dropout=0.0).eval()
        x = torch.randn(2, 20, 75)
        mask = torch.zeros((2, 20), dtype=torch.bool)
        mask[1, 13:] = True
        with torch.no_grad():
            logits = model(x, mask)
        self.assertEqual(tuple(logits.shape), (2, 20, 5))
        self.assertEqual(model.receptive_field, 253)
        self.assertEqual(float(logits[1, 13:].abs().sum()), 0.0)

    def test_valid_tail_is_invariant_to_extra_padding(self) -> None:
        torch.manual_seed(7)
        model = SleepTCN(input_dim=8, hidden_dim=16, n_blocks=3, dropout=0.0).eval()
        short = torch.randn(1, 9, 8)
        padded = torch.zeros(1, 15, 8)
        padded[:, :9] = short
        mask = torch.zeros((1, 15), dtype=torch.bool)
        mask[:, 9:] = True
        with torch.no_grad():
            alone = model(short)
            in_batch = model(padded, mask)[:, :9]
        self.assertTrue(torch.allclose(alone, in_batch, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
