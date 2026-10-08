import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


new = module('adast_cuda_test', ROOT / 'scripts/run_colab_adast_training.py')
old = module('adast_cpu_original_test', ROOT / 'scripts/run_adast_cpu_pilot.py')
UPSTREAM = ROOT / 'runs/teacher_revision_cpu_20261001/adast_upstream'


@pytest.mark.parametrize('arm', ['source_only', 'adast'])
def test_cuda_adapter_update_matches_original_on_cpu(arm):
    torch.set_num_threads(1)
    models = module('adast_test_models', UPSTREAM / 'models/models.py')
    cfg_base = module('adast_test_cfg', UPSTREAM / 'config_files/configs.py').Config()
    utils = module('adast_test_utils', UPSTREAM / 'utils.py')
    cfg = json.loads(old.PROTOCOL.read_bytes())
    left = old.build_models(models, cfg_base, 123)
    right = new.build_models(models, cfg_base, 123, 'cpu')
    assert old.model_digest(left) == new.state_digest(right)
    params = cfg['optimizer']
    names = ['encoder', 'head1', 'head2', 'source_attention', 'target_attention']
    def optim(m):
        return (torch.optim.Adam([p for n in names for p in m[n].parameters()], lr=params['lr'],
                                 betas=tuple(params['betas']), weight_decay=params['weight_decay']),
                torch.optim.Adam(m['discriminator'].parameters(), lr=params['lr'],
                                 betas=tuple(params['betas']), weight_decay=params['weight_decay']))
    lo, ld = optim(left)
    ro, rd = optim(right)
    sx = torch.randn(4, 1, 3000)
    sy = torch.arange(4)
    tx = torch.randn(4, 1, 3000)
    py = torch.arange(4)
    torch.manual_seed(88)
    a = old.update(left, lo, ld, utils, sx, sy, tx, py, 1, cfg, arm)
    torch.manual_seed(88)
    b = new.update(right, ro, rd, utils, sx, sy, tx, py, 1, cfg, arm)
    assert a == b
    assert old.model_digest(left) == new.state_digest(right)
    np.testing.assert_array_equal(old.predict(left, sx.squeeze(1).numpy(), 'source'),
                                  new.predict(right, sx.squeeze(1).numpy(), 'source', 'cpu'))


def test_resume_pair_budget_and_source_only_no_target_forward(tmp_path):
    torch.set_num_threads(1)
    models = module('adast_resume_models', UPSTREAM / 'models/models.py')
    cfg_base = module('adast_resume_cfg', UPSTREAM / 'config_files/configs.py').Config()
    utils = module('adast_resume_utils', UPSTREAM / 'utils.py')
    cfg = json.loads(old.PROTOCOL.read_bytes())
    cfg.update(rounds=2, epochs_per_round=1, steps_per_epoch=1, batch_size=4)
    x = np.random.default_rng(1).normal(size=(8, 3000)).astype(np.float32)
    y = np.arange(8, dtype=np.int64) % 5
    target = x[::-1].copy()
    guard = new.Guard(tmp_path, 300)
    results = {}
    for arm in cfg['arms']:
        m = new.build_models(models, cfg_base, 123, 'cpu')
        if arm == 'source_only':
            def forbidden(*args):
                raise AssertionError('Source-only target forward forbidden')
            m['target_attention'].register_forward_pre_hook(forbidden)
        identity = {'fold': 0, 'arm': arm}
        selected = new.train_arm(arm, m, utils, x, y, np.arange(8), target, cfg,
                                 tmp_path / arm, identity, guard, 'cpu')
        fresh = new.build_models(models, cfg_base, 123, 'cpu')
        replay = new.train_arm(arm, fresh, utils, x, y, np.arange(8), target, cfg,
                              tmp_path / arm, identity, guard, 'cpu')
        assert selected == replay
        assert selected['updates'] == 2
        assert new.state_digest(m) == new.state_digest(fresh)
        results[arm] = selected
    assert results['source_only']['initial_state_sha256'] == results['adast']['initial_state_sha256']
    assert [r['source_order_sha256'] for r in results['source_only']['history']] == [
        r['source_order_sha256'] for r in results['adast']['history']]


def test_budget_does_not_silently_restart(tmp_path):
    guard = new.Guard(tmp_path, 0)
    with pytest.raises(new.BudgetStop):
        guard()
