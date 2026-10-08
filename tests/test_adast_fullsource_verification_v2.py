"""Pure verification tests; deliberately no CPU training even for tiny fixtures."""
import copy
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_adast_fullsource_completion_results_v2 as verifier


def config():
    return json.loads((ROOT / 'configs/adast_fullsource_completion_v1_20261006.json').read_bytes())


@pytest.mark.parametrize('field,value', [('seed', 124), ('folds_to_train', [2]),
                                       ('arms', ['adast_full_source']), ('batch_size', 64),
                                       ('source_loss_weights_by_round', [1, 1]),
                                       ('target_loss_weights_by_round', [0, 0]),
                                       ('cpu_training_fallback', True), ('adaptation_subjects', 180)])
def test_changed_protocol_rejected(field, value):
    cfg = config()
    verifier.validate_protocol(cfg)
    cfg[field] = value
    with pytest.raises(ValueError, match='protocol differs'):
        verifier.validate_protocol(cfg)


def test_wrong_fold_support_or_budget_rejected():
    cfg = config()
    cfg['remaining_fold_sizes'][0]['updates_per_model'] -= 1
    with pytest.raises(ValueError, match='budget differs'):
        verifier.validate_protocol(cfg)


@pytest.mark.parametrize('bad', [np.array([1, 3]), np.array([2, 2]), np.array([-1, 3]),
                               np.array([2, 6]), np.array([2., 3.])])
def test_source_overlap_duplicates_bounds_dtype_rejected(bad):
    roles = {'train': np.array([0, 1]), 'validation': np.array([2, 3]), 'test': np.array([4, 5])}
    verifier.validate_roles(roles, 6)
    roles['validation'] = bad
    with pytest.raises(ValueError):
        verifier.validate_roles(roles, 6)


def test_role_array_reads_only_ordered_validation_subset():
    data = np.arange(20).reshape(10, 2)
    value = verifier.RoleArray(data, np.array([9, 2, 4]))
    assert len(value) == 3 and value.shape == (3, 2)
    np.testing.assert_array_equal(value[:2], data[[9, 2]])
    np.testing.assert_array_equal(value[np.array([2, 0])], data[[4, 9]])


@pytest.mark.parametrize('name', ['../a.pt', '/a.pt', 'C:/a.pt', 'a\\b.pt', 'a//b.pt', 'a/', ''])
def test_unsafe_members_rejected(name):
    with pytest.raises(ValueError, match='Unsafe'):
        verifier.safe_name(name)


def test_wrong_archive_hash_fails_before_any_payload_access(tmp_path):
    path = tmp_path / 'bad.zip'
    path.write_bytes(b'not the observed result')
    args = type('Args', (), {'archive': path, 'expected_sha256': '0' * 64, 'max_seconds': 30})()
    with pytest.raises(ValueError, match='observed Colab SHA'):
        verifier.verify(args)


def test_archive_tampering_rejected(tmp_path):
    path = tmp_path / 'result.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('a.json', '{}')
        z.writestr('export_manifest.json', json.dumps({'a.json': '0' * 64}))
    with zipfile.ZipFile(path) as z, pytest.raises(ValueError, match='hash differs'):
        verifier.verify_archive_members(z, lambda: None)


@pytest.mark.parametrize('epoch,adaptive', [(0, False), (9, True), (10, False), (15, True), (29, True)])
def test_sampling_partial_batch_loss_lr_reconstruct(epoch, adaptive):
    cfg = config()
    labels = np.arange(131, dtype=np.int64) % 5
    source, target = verifier.shared.replay_sampling({**cfg, 'source_train_epochs': len(labels)}, epoch, True)
    import hashlib
    target_seen = np.zeros(4989, dtype=bool)
    seen = np.zeros(4989, dtype=bool)
    if adaptive:
        seen[target] = True
    coef = {'source_ce': 1 if epoch < 15 else .1, 'similarity': .001,
            'adversarial': 1 if adaptive else 0, 'target_pseudo_ce': .01 if adaptive and epoch >= 15 else 0}
    loss = {'source_ce': 2., 'similarity': 1., 'adversarial': .2 if adaptive else 0.,
            'target_pseudo_ce': 3. if adaptive else 0., 'discriminator': .4 if adaptive else 0.}
    loss['total'] = sum(coef[k] * loss[k] for k in coef)
    row = {'global_epoch': epoch + 1, 'round': epoch // 15, 'epoch': epoch % 15 + 1,
           'updates': 2, 'source_presentations': 131, 'source_unique_seen': 131,
           'target_unique_seen': int(seen.sum()), 'target_training_presentations': 131 if adaptive else 0,
           'source_order_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
           'target_order_sha256': hashlib.sha256(target.tobytes()).hexdigest() if adaptive else None,
           'source_label_counts': np.bincount(labels[source], minlength=5).tolist(),
           'optimizer_lr_before_epoch': .001 * (.1 if epoch >= 10 else 1),
           'optimizer_lr_after_epoch': .001 * (.1 if epoch >= 9 else 1),
           'loss_coefficients': coef, 'loss_batch_means': loss, 'seconds': 1.}
    verifier.check_epoch(row, cfg, labels, epoch, adaptive, target_seen)
    np.testing.assert_array_equal(seen, target_seen)
    for field in ('updates', 'source_presentations', 'target_training_presentations'):
        bad = copy.deepcopy(row)
        bad[field] += 1
        with pytest.raises(ValueError, match=field):
            verifier.check_epoch(bad, cfg, labels, epoch, adaptive, np.zeros(4989, dtype=bool))


def test_optimizer_steps_lr_and_source_only_discriminator():
    cfg = config()
    def opt(lr, trained):
        return {'param_groups': [{'lr': lr, 'betas': (.5, .99), 'weight_decay': .0003}],
                'state': {0: {'step': torch.tensor(60.), 'exp_avg': torch.ones(1), 'exp_avg_sq': torch.ones(1)}} if trained else {}}
    value = {'optimizer': opt(.0001, True), 'disc_optimizer': opt(.001, False)}
    verifier.check_optimizer(value, cfg, 30, 2, False)
    bad = copy.deepcopy(value)
    bad['disc_optimizer'] = opt(.001, True)
    with pytest.raises(ValueError, match='Source-only'):
        verifier.check_optimizer(bad, cfg, 30, 2, False)
    verifier.check_optimizer(bad, cfg, 30, 2, True)
    bad['optimizer']['state'][0]['step'] = torch.tensor(59.)
    with pytest.raises(ValueError, match='update count'):
        verifier.check_optimizer(bad, cfg, 30, 2, True)


def test_diagnostic_including_target_attention_is_source_validation(tmp_path):
    labels = np.arange(5, dtype=np.int64)
    logits = np.eye(5, dtype=np.float32)
    stream = io.BytesIO()
    np.savez_compressed(stream, source=logits, target=logits)
    path = tmp_path / 'diagnostic.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('validation.npz', stream.getvalue())
    metric = verifier.shared.metrics_from_confusion(verifier.shared.confusion_matrix_5(labels, logits.argmax(1)))
    with zipfile.ZipFile(path) as z:
        verifier.shared.verify_logits(z, 'validation.npz', labels, {'source': metric, 'target': metric})
        with pytest.raises(ValueError, match='reconstruct'):
            verifier.shared.verify_logits(z, 'validation.npz', labels,
                                          {'source': metric, 'target': {**metric, 'macro_f1': 0.}})
