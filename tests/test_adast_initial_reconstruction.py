import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('archive_verifier', ROOT / 'scripts/verify_colab_adast_results.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_native_exact_and_wrong_hash_rejected(monkeypatch):
    def build(_module, _cfg, seed, _device):
        torch.manual_seed(seed)
        return {'tiny': torch.nn.Linear(7, 3)}
    monkeypatch.setattr(verifier.helper, 'build_models', build)
    expected = verifier.helper.state_digest(build(None, None, 123, 'cpu'))
    original = torch.Tensor.uniform_
    models, audit = verifier.reconstruct_initial(None, None, 123, expected)
    assert verifier.helper.state_digest(models) == expected
    assert audit['method'] == 'native_exact'
    with pytest.raises(ValueError, match='matches initial SHA256'):
        verifier.reconstruct_initial(None, None, 123, '0' * 64)
    assert torch.Tensor.uniform_ is original


def test_observed_colab_initial_bytes_reconstructed():
    work = ROOT / 'runs/colab_adast_20261004/payload'
    if not work.exists():
        pytest.skip('Private local CUDA campaign input is not in source distribution')
    module = verifier.helper.load_module('test_cloud_models', work / 'upstream/models.py')
    cfg = verifier.helper.load_module('test_cloud_config', work / 'upstream/configs.py').Config()
    original = torch.Tensor.uniform_
    expected = '0d43c546206e330bb48068a97e2970f2cb0770b6f8efc5037aa5e1fc9770ecd2'
    models, audit = verifier.reconstruct_initial(module, cfg, 123, expected)
    assert verifier.helper.state_digest(models) == expected
    assert audit['method'] in {'native_exact', 'fused_uniform_float32_exact'}
    assert torch.Tensor.uniform_ is original
