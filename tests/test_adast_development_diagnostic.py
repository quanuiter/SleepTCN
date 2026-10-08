import importlib.util
from pathlib import Path
import hashlib
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('adast_development', ROOT / 'scripts/diagnose_adast_development.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_head_rules_are_explicit_and_not_probabilities():
    first = np.array([[2, 3, 0, 0, 0]], dtype=np.float32)
    second = np.array([[5, 2, 0, 0, 0]], dtype=np.float32)
    values = diagnostic.combine(first, second)
    assert values['head1'].argmax(1).item() == 1
    assert values['maximum'].argmax(1).item() == 0
    assert values['mean'].argmax(1).item() == 0
    with pytest.raises(ValueError):
        diagnostic.combine(first, np.full_like(first, np.nan))


def test_unlabelled_summary_has_no_accuracy_or_reference_labels():
    first = np.zeros((3, 5), dtype=np.float32)
    second = np.zeros_like(first)
    result = diagnostic.summaries(first, second)
    assert all('metrics' not in v for v in result.values())
    labelled = diagnostic.summaries(first, second, np.array([0, 1, 2]))
    assert labelled['maximum']['metrics']['n_valid_epochs'] == 3


def test_exposure_checks_linux_int64_order_hash_and_no_full_pass_claim():
    indices = np.arange(10)
    labels = indices % 5
    cfg = {'seed': 123, 'steps_per_epoch': 1, 'batch_size': 3}
    order = np.random.default_rng(123).permutation(10)[:3].astype(np.int64)
    history = [{'source_order_sha256': hashlib.sha256(order.tobytes()).hexdigest()}]
    result = diagnostic.source_exposure(indices, labels, history, cfg)
    assert result['presentations'] == 3
    assert result['equivalent_full_source_passes'] == .3
    history[0]['source_order_sha256'] = 'incorrect'
    with pytest.raises(ValueError):
        diagnostic.source_exposure(indices, labels, history, cfg)
