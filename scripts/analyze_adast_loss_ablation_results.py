"""Summarize independently verified ADAST loss ablations; no model/data loading or training."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('W', 'N1', 'N2', 'N3', 'REM')
ARMS = ('adast_reference_full_source', 'adast_keep_source_ce_full_source',
        'adast_no_pseudo_full_source', 'adast_no_adversarial_full_source')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_result(directory):
    proof = json.loads((directory / 'independent_verification.json').read_bytes())
    internal = json.loads((directory / 'verification.json').read_bytes())
    actual = digest(directory / 'aggregate_results.json')
    if (proof['status'] != 'passed' or internal['status'] != 'passed'
            or proof['aggregate_results_sha256'] != actual
            or internal['aggregate_results_sha256'] != actual
            or not proof['all_124_validation_diagnostics_recomputed']
            or not proof['all_eight_best_final_checkpoints_CPU_validation_replayed']
            or not proof['sampling_coverage_and_budget_replayed']
            or proof['source_outer_test_access'] or proof['target_test_access']):
        raise ValueError('Require independently verified, source-validation-only results')
    return json.loads((directory / 'aggregate_results.json').read_bytes()), proof, actual


def view(metrics):
    return {
        'validation_epochs': metrics['n_valid_epochs'],
        'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
        'cohen_kappa': metrics['cohen_kappa'], 'per_class': metrics['per_class'],
        'confusion_matrix': metrics['confusion_matrix'],
    }


def difference(current, reference):
    return {
        'macro_f1': current['macro_f1'] - reference['macro_f1'],
        'accuracy': current['accuracy'] - reference['accuracy'],
        'per_class': {name: {metric: current['per_class'][name][metric]
                            - reference['per_class'][name][metric]
                            for metric in ('precision', 'recall', 'f1')}
                      for name in CLASSES},
    }


def summarize(result, earlier):
    if tuple(a['arm'] for a in result['arms']) != ARMS:
        raise ValueError('Unexpected ablation arms/order')
    source_only = next(a for a in earlier['arms'] if a['arm'] == 'source_only_full_source')
    prior_adast = next(a for a in earlier['arms'] if a['arm'] == 'adast_full_source')
    reference = result['arms'][0]
    for arm in result['arms']:
        if arm['epochs_completed'] != 30 or arm['total_updates'] != 36870:
            raise ValueError('Incomplete or unequal learning budget')
    selected = {}
    final = {}
    trajectories = {}
    for arm in result['arms']:
        name = arm['arm']
        selected[name] = {
            'epoch': arm['best_epoch'],
            'source_attention': view(arm['best_source_validation']['source']),
            'target_attention_on_same_source_validation': view(arm['best_source_validation']['target']),
            'delta_vs_fresh_reference_selected': difference(
                arm['best_source_validation']['source'], reference['best_source_validation']['source']),
            'delta_vs_previous_source_only_selected': difference(
                arm['best_source_validation']['source'], source_only['best_source_validation']['source']),
        }
        final[name] = {
            'epoch': 30,
            'source_attention': view(arm['final_source_validation']['source']),
            'target_attention_on_same_source_validation': view(arm['final_source_validation']['target']),
            'delta_vs_fresh_reference_final': difference(
                arm['final_source_validation']['source'], reference['final_source_validation']['source']),
        }
        rows = []
        for h in arm['history']:
            rows.append({
                'epoch': h['global_epoch'], 'loss_coefficients': h['loss_coefficients'],
                'source_attention': view(h['validation']['source']),
                'target_attention_on_same_source_validation': view(h['validation']['target']),
            })
        mean = lambda key: sum(r['source_attention']['per_class'][key]['recall']
                               for r in rows[15:]) / 15
        trajectories[name] = {
            'epochs': rows,
            'round_two_mean_epoch_N1_recall_descriptive_only': mean('N1'),
            'round_two_mean_epoch_N3_recall_descriptive_only': mean('N3'),
            'minimum_N1_recall': min(r['source_attention']['per_class']['N1']['recall'] for r in rows),
            'epochs_with_zero_N1_predictions': [r['epoch'] for r in rows
                if sum(cm[1] for cm in r['source_attention']['confusion_matrix']) == 0],
        }
    shared_prefix = all(
        a['history'][e]['validation'] == reference['history'][e]['validation']
        for a in result['arms'][1:3] for e in range(15))
    replay_matches = all(
        reference['history'][e]['validation'] == prior_adast['history'][e]['validation']
        for e in range(30))
    if not shared_prefix or not replay_matches:
        raise ValueError('Expected shared prefix or prior reference replay differs')
    return {
        'scope': 'pooled_epoch_source_validation_fold_0_seed_123',
        'attention_for_primary_comparison': 'source',
        'checkpoint_rule': 'source-validation macro-F1 maximum; first tie; final reported separately',
        'validation_epochs': 18763,
        'class_supports': {c: selected[ARMS[0]]['source_attention']['per_class'][c]['support']
                           for c in CLASSES},
        'training_seconds': result['elapsed_seconds'],
        'updates_per_arm': 36870, 'arms': list(ARMS),
        'selected': selected, 'final': final, 'trajectories': trajectories,
        'first_15_validation_metrics_shared_exactly': shared_prefix,
        'fresh_reference_all_30_validation_metrics_match_prior_adast': replay_matches,
        'no_pseudo_selected_before_intervention_activation': selected[ARMS[2]]['epoch'] <= 15,
        'candidate_order_by_locked_primary_macro_f1': sorted(
            ARMS, key=lambda a: selected[a]['source_attention']['macro_f1'], reverse=True),
        'previous_source_only_selected': {
            'epoch': source_only['best_epoch'],
            'source_attention': view(source_only['best_source_validation']['source'])},
        'source_outer_test_access': False, 'target_test_access': False,
        'training_variability_estimated': False,
    }


def analyze(args):
    output = args.output.resolve()
    report_root = (ROOT / 'Reports' / 'analysis').resolve()
    if not output.is_relative_to(report_root) or output == report_root or output.exists():
        raise ValueError('Require a new Reports/analysis output subdirectory')
    result, proof, aggregate_sha = verified_result(args.results)
    earlier, _, earlier_sha = verified_result(args.reference_results)
    if not all(proof.get(k) for k in ('single_factor_overrides_verified',
                                    'removed_discriminator_unchanged', 'round_one_shared_prefix_verified')):
        raise ValueError('Ablation verification gates missing')
    spec = json.loads((args.results / 'execution_specification.json').read_bytes())
    prior_spec = json.loads((args.reference_results / 'execution_specification.json').read_bytes())
    for key in ('fold', 'seed', 'source_train_epochs', 'source_validation_epochs', 'adaptation_epochs'):
        if spec['protocol'][key] != prior_spec['protocol'][key]:
            raise ValueError('Previous source-only context does not match')
    report = summarize(result, earlier)
    report['provenance'] = {
        'aggregate_results_sha256': aggregate_sha,
        'reference_aggregate_results_sha256': earlier_sha,
        'independent_verification_sha256': digest(args.results / 'independent_verification.json'),
        'analysis_script_sha256': digest(Path(__file__).resolve()),
        'result_archive_sha256': proof['archive_sha256'],
        'verification_seconds': proof['elapsed_seconds'],
    }
    table = io.StringIO(newline='')
    writer = csv.writer(table, lineterminator='\n')
    writer.writerow(['arm', 'selection', 'epoch', 'macro_f1', 'accuracy', 'class',
                     'precision', 'recall', 'f1', 'support'])
    for selection in ('selected', 'final'):
        for arm in ARMS:
            entry = report[selection][arm]
            metrics = entry['source_attention']
            for name in CLASSES:
                values = metrics['per_class'][name]
                writer.writerow([arm, selection, entry['epoch'], metrics['macro_f1'], metrics['accuracy'],
                                 name, *(values[k] for k in ('precision', 'recall', 'f1', 'support'))])
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'all_class_metrics.csv').write_text(table.getvalue(), encoding='utf-8')
    manifest = {name: digest(output / name) for name in ('summary.json', 'all_class_metrics.csv')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'training_seconds': report['training_seconds'],
                      'verification_seconds': proof['elapsed_seconds'],
                      'ranking': report['candidate_order_by_locked_primary_macro_f1'],
                      'no_pseudo_selected_before_activation': report['no_pseudo_selected_before_intervention_activation'],
                      'files_sha256': manifest}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--reference-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    analyze(parser.parse_args())
