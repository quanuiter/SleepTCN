import unittest

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
    def test_feature_and_logit_shapes(self) -> None:
        model = EEGResNet1D().eval()
        x = torch.randn(2, 1, 3000)
        with torch.no_grad():
            features = model.extract_features(x)
            logits = model(x)
        self.assertEqual(tuple(features.shape), (2, 128))
        self.assertEqual(tuple(logits.shape), (2, 5))

    def test_explicit_config_changes_architecture(self) -> None:
        config = {
            "input_channels": 1,
            "stem": {
                "channels": 16,
                "kernel_size": 50,
                "stride": 2,
                "padding": 25,
                "max_pool_kernel": 3,
                "max_pool_stride": 2,
                "max_pool_padding": 1,
            },
            "residual_blocks": [
                {"in_channels": 16, "out_channels": 32, "stride": 1, "kernel_size": 5},
                {"in_channels": 32, "out_channels": 64, "stride": 2, "kernel_size": 5},
            ],
            "feature_dim": 64,
            "classifier_dropout": 0.2,
        }
        model = EEGResNet1D.from_config(config).eval()
        x = torch.randn(2, 1, 3000)
        with torch.no_grad():
            features = model.extract_features(x)
            logits = model(x)
        self.assertEqual(tuple(features.shape), (2, 64))
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(model.resolved_config["stem"]["stride"], 2)


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
