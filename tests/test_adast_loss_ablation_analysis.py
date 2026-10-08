import copy
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'ablation_analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze_adast_loss_ablation_results.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixtures():
    def metrics(score):
        return {'n_valid_epochs': 18763, 'accuracy': .7, 'macro_f1': score, 'cohen_kappa': .6,
                'per_class': {c: {'precision': .5, 'recall': .4, 'f1': .45, 'support': 1}
                              for c in analysis.CLASSES},
                'confusion_matrix': [[1 if i == j else 0 for j in range(5)] for i in range(5)]}

    arms = []
    for name in analysis.ARMS:
        scores = [.6] * 30
        scores[6] = .69
        if name != analysis.ARMS[2]:
            scores[21] = .70 if name != analysis.ARMS[3] else .701
        history = [{'global_epoch': e + 1, 'loss_coefficients': {},
                    'validation': {'source': metrics(s), 'target': metrics(s)}}
                   for e, s in enumerate(scores)]
        best = max(range(30), key=lambda e: scores[e])
        arms.append({'arm': name, 'epochs_completed': 30, 'total_updates': 36870,
                     'best_epoch': best + 1, 'history': history,
                     'best_source_validation': history[best]['validation'],
                     'final_source_validation': history[-1]['validation']})
    prior_ada = copy.deepcopy(arms[0])
    prior_ada['arm'] = 'adast_full_source'
    source_only = copy.deepcopy(arms[0])
    source_only['arm'] = 'source_only_full_source'
    return {'arms': arms, 'elapsed_seconds': 100}, {'arms': [source_only, prior_ada]}


def test_selected_and_final_kept_separate_and_primary_ranking_preserved():
    result, earlier = fixtures()
    report = analysis.summarize(result, earlier)
    assert report['no_pseudo_selected_before_intervention_activation']
    assert report['selected'][analysis.ARMS[2]]['epoch'] == 7
    assert report['final'][analysis.ARMS[2]]['epoch'] == 30
    assert report['candidate_order_by_locked_primary_macro_f1'][0] == analysis.ARMS[3]
    assert report['first_15_validation_metrics_shared_exactly']
    assert report['fresh_reference_all_30_validation_metrics_match_prior_adast']
    assert not report['target_test_access']


def test_changed_round_one_prefix_is_rejected():
    result, earlier = fixtures()
    result['arms'][1]['history'][0]['validation']['source']['macro_f1'] = .9
    with pytest.raises(ValueError, match='shared prefix'):
        analysis.summarize(result, earlier)


def test_previous_reference_mismatch_is_rejected():
    result, earlier = fixtures()
    earlier['arms'][1]['history'][-1]['validation']['source']['macro_f1'] = .9
    with pytest.raises(ValueError, match='prior reference'):
        analysis.summarize(result, earlier)


def test_partial_budget_is_rejected():
    result, earlier = fixtures()
    result['arms'][2]['total_updates'] = 1140
    with pytest.raises(ValueError, match='learning budget'):
        analysis.summarize(result, earlier)


def test_independent_gate_binds_aggregate_bytes(tmp_path):
    aggregate = b'{"arms": []}'
    (tmp_path / 'aggregate_results.json').write_bytes(aggregate)
    sha = hashlib.sha256(aggregate).hexdigest()
    proof = {'status': 'passed', 'aggregate_results_sha256': sha,
             'all_124_validation_diagnostics_recomputed': True,
             'all_eight_best_final_checkpoints_CPU_validation_replayed': True,
             'sampling_coverage_and_budget_replayed': True,
             'source_outer_test_access': False, 'target_test_access': False}
    (tmp_path / 'independent_verification.json').write_text(json.dumps(proof))
    (tmp_path / 'verification.json').write_text(json.dumps(
        {'status': 'passed', 'aggregate_results_sha256': sha}))
    assert analysis.verified_result(tmp_path)[2] == sha
    (tmp_path / 'aggregate_results.json').write_bytes(b'{"arms": [1]}')
    with pytest.raises(ValueError, match='independently verified'):
        analysis.verified_result(tmp_path)


def test_failed_internal_gate_is_rejected(tmp_path):
    (tmp_path / 'aggregate_results.json').write_bytes(b'{}')
    (tmp_path / 'independent_verification.json').write_text(json.dumps({'status': 'failed'}))
    (tmp_path / 'verification.json').write_text(json.dumps({'status': 'passed'}))
    with pytest.raises(ValueError, match='independently verified'):
        analysis.verified_result(tmp_path)
