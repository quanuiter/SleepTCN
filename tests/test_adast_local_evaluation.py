import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('adast_local_eval', ROOT / 'scripts/run_adast_10fold_local_evaluation.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_softmax_is_applied_before_ensemble_mean():
    logits = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    p = evaluation.softmax_logits(logits)
    np.testing.assert_allclose(p.sum(-1), 1., atol=1e-6)
    assert p.argmax(1).item() == 4
    with pytest.raises(ValueError):
        evaluation.softmax_logits(np.array([[np.nan] * 5], dtype=np.float32))


def test_prediction_audit_requires_all_ten_and_exact_mean(tmp_path):
    rng = np.random.default_rng(1)
    parts = rng.random((10, 3, 5)).astype(np.float32)
    parts /= parts.sum(-1, keepdims=True)
    mean = parts.astype(np.float64).mean(0).astype(np.float32)
    entry = {'subject_id': 'synthetic', 'source_edf_sha256': 'synthetic_hash', 'epochs': 3}
    metadata = {**{k: entry[k] for k in ['subject_id', 'source_edf_sha256']},
                'frozen_checkpoints_sha256': 'frozen', 'specification_sha256': 'spec'}
    values = {'fold_source_only': parts, 'fold_adast': parts, 'source_only': mean, 'adast': mean,
              'original_epoch_index': np.arange(3), 'metadata_json': np.array(json.dumps(metadata))}
    path = tmp_path / 'prediction.npz'
    np.savez(path, **values)
    result = evaluation.audit_prediction(path, entry, 'frozen', 'spec')
    np.testing.assert_array_equal(result['adast'], mean)
    values['fold_adast'] = parts[:1]
    np.savez(path, **values)
    with pytest.raises(ValueError, match='all ten folds'):
        evaluation.audit_prediction(path, entry, 'frozen', 'spec')
    values['fold_adast'] = parts
    values['adast'] = mean + .0001
    np.savez(path, **values)
    with pytest.raises(AssertionError):
        evaluation.audit_prediction(path, entry, 'frozen', 'spec')


def test_source_order_hash_uses_colab_int64_bytes():
    import hashlib
    order = np.random.default_rng(123).permutation(157200)[:4864].astype(np.int64)
    assert hashlib.sha256(order.tobytes()).hexdigest() == '98632cabe93b5b289f6f2ef8873957010ba22c11c14d27d149d55a31c1f2c592'
