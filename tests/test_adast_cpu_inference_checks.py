"""Exact forward-only equivalence tests; no optimizer, gradients or training."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from adast_cpu_inference_checks import predict_both_attention
from run_colab_adast_training import predict, state_digest


class CountingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 4), torch.nn.BatchNorm1d(4))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.layers(x)


def models():
    torch.manual_seed(17)
    return {'encoder': CountingEncoder(), 'head1': torch.nn.Linear(4, 5), 'head2': torch.nn.Linear(4, 5),
            'source_attention': torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.5)),
            'target_attention': torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.5))}


@pytest.mark.parametrize('size', [1, 128, 129, 257])
def test_joint_forward_exact_logits_state_and_partial_batch(size):
    torch.set_num_threads(4)
    torch.set_flush_denormal(False)
    value = models()
    x = np.random.default_rng(2).normal(size=(size, 2)).astype(np.float32)
    before = state_digest(value)
    independent = {d: predict(value, x, d, 'cpu') for d in ('source', 'target')}
    value['encoder'].calls = 0
    joint = predict_both_attention(value, x)
    for domain in joint:
        np.testing.assert_array_equal(joint[domain], independent[domain])
    assert value['encoder'].calls == (size + 127) // 128
    assert state_digest(value) == before
    assert all(p.grad is None for m in value.values() for p in m.parameters())
    assert all(not m.training for m in value.values())


def test_joint_forward_keeps_stop_guard():
    def stop():
        raise TimeoutError('test resource stop')
    with pytest.raises(TimeoutError, match='resource stop'):
        predict_both_attention(models(), np.zeros((128, 2), dtype=np.float32), stop)
