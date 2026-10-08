"""Pure evaluation/analysis checks; no model training or real cohort scoring."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_adast_fullsource_local_evaluation as runner
import analyze_adast_fullsource_budget as analysis
import verify_adast_fullsource_local_results_recovery as recovery


def test_planned_checkpoint_rule_rejects_unknown_selection():
    with pytest.raises(ValueError, match='planned final'):
        runner.frozen_models('target_best')


def test_missing_new_training_proof_cannot_launch(monkeypatch):
    monkeypatch.setattr(runner, 'read', lambda path: {'status': 'running', 'results_root': '.', 'archive_sha256': 'bad'})
    with pytest.raises(ValueError, match='provenance differs'):
        runner.frozen_models('final')


def test_evaluation_hash_accepts_json_string_and_path(tmp_path):
    path = tmp_path / 'input.dat'
    path.write_bytes(b'synthetic verification input')
    assert runner.sha(str(path)) == runner.sha(path)


def test_recovery_allows_only_path_cast_and_preserves_original_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path / 'repo')
    relative = recovery.RELATIVE
    snapshot = tmp_path / 'output/code_snapshot' / relative
    current = recovery.ROOT / relative
    snapshot.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    original = recovery.OLD_ALIAS + '\n# unchanged inference\n'
    snapshot.write_text(original)
    current.write_text(original.replace(recovery.OLD_ALIAS, recovery.NEW_ALIAS))
    spec = {'code_sha256': {relative: runner.sha(snapshot)}}
    proof = recovery.check_code_provenance(tmp_path / 'output', spec)
    assert proof['original_sha256'] == runner.sha(snapshot)
    assert proof['repaired_sha256'] == runner.sha(current)
    current.write_text(current.read_text() + '# unrelated change\n')
    with pytest.raises(ValueError, match='exactly the Path conversion'):
        recovery.check_code_provenance(tmp_path / 'output', spec)


def test_recovery_rejects_changed_original_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path / 'repo')
    snapshot = tmp_path / 'output/code_snapshot' / recovery.RELATIVE
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(recovery.OLD_ALIAS)
    digest = runner.sha(snapshot)
    snapshot.write_text('tampered')
    with pytest.raises(ValueError, match='Original executed code snapshot changed'):
        recovery.check_code_provenance(tmp_path / 'output', {'code_sha256': {recovery.RELATIVE: digest}})


def test_budget_analysis_hashes_json_bound_path_as_path(tmp_path, monkeypatch):
    bound = tmp_path / 'bound.dat'
    bound.write_bytes(b'synthetic input')
    calls = []
    def checked_hash(path):
        assert isinstance(path, Path)
        calls.append(path)
        return 'expected'
    monkeypatch.setattr(analysis, 'read', lambda path: {str(bound): 'expected'})
    monkeypatch.setattr(analysis, 'sha256_file', checked_hash)
    monkeypatch.setattr(analysis, 'load_verified_comparison', lambda: (_ for _ in ()).throw(RuntimeError('passed hash boundary')))
    with pytest.raises(RuntimeError, match='passed hash boundary'):
        analysis.verify(type('Args', (), {'output': tmp_path})())
    assert calls == [bound]


def test_vector_metrics_agree_with_individual_confusion_formula():
    rng = np.random.default_rng(5)
    matrices = rng.integers(0, 20, (7, 5, 5))
    actual = analysis.vector_metrics(matrices)
    for i, cm in enumerate(matrices):
        expected = analysis.metrics_from_confusion(cm)
        assert actual['accuracy'][i] == pytest.approx(expected['accuracy'])
        assert actual['macro_f1'][i] == pytest.approx(expected['macro_f1'])
        for stage in analysis.STAGE_NAMES:
            for metric in ('precision', 'recall', 'f1'):
                assert actual[stage + '_' + metric][i] == pytest.approx(expected['per_class'][stage][metric])


def test_equal_adaptation_effect_cancels_even_when_budgets_raise_both_systems():
    baseline = np.stack([np.eye(5, dtype=np.int64) * 5 + 1 for _ in range(4)])
    cms = {k: baseline.copy() for k in ('limited_source_only', 'limited_adast', 'full_source_only', 'full_adast')}
    for key in ('full_source_only', 'full_adast'):
        cms[key][:, 0, 1] -= 1
        cms[key][:, 0, 0] += 1
    report = analysis.paired_budget_contrast(cms, 20, 4)
    for contrast in report['contrasts'].values():
        assert contrast['subject_mean_macro_f1']['difference'] == 0.
        assert contrast['subject_mean_macro_f1']['ci95'] == [0., 0.]
        for value in contrast['pooled'].values():
            assert value['difference'] == 0.


def test_difference_of_differences_uses_same_people_exactly():
    rng = np.random.default_rng(7)
    reference = rng.integers(2, 10, (6, 5, 5))
    cms = {k: reference.copy() for k in ('limited_source_only', 'limited_adast', 'full_source_only', 'full_adast')}
    cms['full_adast'][:, 3, 2] -= 1
    cms['full_adast'][:, 3, 3] += 1
    report = analysis.paired_budget_contrast(cms, 50, 8)
    effects = report['contrasts']
    assert report['same_resampling_across_all_four_systems']
    assert effects['difference_of_differences'] == effects['full']
    assert effects['full']['pooled']['N3_to_N2_rate']['difference'] < 0
    assert report == analysis.paired_budget_contrast(cms, 50, 8)
    cms['limited_adast'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='supports/people ordering'):
        analysis.paired_budget_contrast(cms, 50, 8)


def test_prediction_audit_keeps_ten_folds_and_provenance(tmp_path):
    parts = np.full((10, 2, 5), .2, dtype=np.float32)
    entry = {'subject_id': 'synthetic', 'source_edf_sha256': 'synthetic_hash', 'epochs': 2}
    metadata = {'subject_id': 'synthetic', 'source_edf_sha256': 'synthetic_hash',
                'frozen_checkpoints_sha256': 'frozen', 'specification_sha256': 'spec'}
    path = tmp_path / 'predictions.npz'
    mean = parts.astype(np.float64).mean(0).astype(np.float32)
    values = {'source_only': mean, 'adast': mean, 'fold_source_only': parts, 'fold_adast': parts,
              'original_epoch_index': np.arange(2), 'metadata_json': np.array(json.dumps(metadata))}
    np.savez(path, **values)
    runner.common.audit_prediction(path, entry, 'frozen', 'spec')
    values['fold_adast'] = parts[:9]
    np.savez(path, **values)
    with pytest.raises(ValueError, match='all ten folds'):
        runner.common.audit_prediction(path, entry, 'frozen', 'spec')


def test_budget_contrast_matches_manual_paired_bootstrap():
    rng = np.random.default_rng(13)
    base = rng.integers(3, 15, (8, 5, 5))
    keys = ('limited_source_only', 'limited_adast', 'full_source_only', 'full_adast')
    cms = {key: base.copy() for key in keys}
    cms['limited_adast'][::2, 1, 2] -= 2
    cms['limited_adast'][::2, 1, 1] += 2
    cms['full_adast'][1::2, 3, 2] -= 1
    cms['full_adast'][1::2, 3, 3] += 1
    result = analysis.paired_budget_contrast(cms, 31, 19)
    draws = np.random.default_rng(19).integers(0, 8, (31, 8), dtype=np.int32)
    manual_pooled, manual_subject = [], []
    effects = {}
    for key, cm in cms.items():
        effects[key] = np.array([analysis.metrics_from_confusion(c)['macro_f1'] for c in cm])
    for indices in draws:
        scores = {key: analysis.metrics_from_confusion(cm[indices].sum(0))['macro_f1'] for key, cm in cms.items()}
        manual_pooled.append((scores['full_adast'] - scores['full_source_only']) -
                             (scores['limited_adast'] - scores['limited_source_only']))
        manual_subject.append(((effects['full_adast'] - effects['full_source_only']) -
                               (effects['limited_adast'] - effects['limited_source_only']))[indices].mean())
    actual = result['contrasts']['difference_of_differences']
    np.testing.assert_allclose(actual['pooled']['macro_f1']['ci95'], np.quantile(manual_pooled, [.025, .975]), atol=1e-15)
    np.testing.assert_allclose(actual['subject_mean_macro_f1']['ci95'], np.quantile(manual_subject, [.025, .975]), atol=1e-15)


def test_analysis_verifier_rejects_changed_bound_input_before_recomputation(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'read', lambda path: {'synthetic': 'expected'})
    monkeypatch.setattr(analysis, 'sha256_file', lambda path: 'changed')
    with pytest.raises(ValueError, match='Bound evaluation/code changed'):
        analysis.verify(type('Args', (), {'output': tmp_path})())
