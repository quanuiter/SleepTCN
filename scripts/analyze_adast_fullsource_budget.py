"""Paired participant bootstrap for the planned two-budget ADAST contrast; no inference/training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sleeptcn.metrics import metrics_from_confusion, STAGE_NAMES
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import summarize_pair, write_once_json

OLD = ROOT / 'runs/teacher_revision_gpu_20261004/adast_10fold_local'
NEW = ROOT / 'runs/teacher_revision_gpu_20261007/adast_fullsource_local/final'
OUTPUT = ROOT / 'runs/teacher_revision_gpu_20261007/adast_fullsource_budget_contrast'


def vector_metrics(cm):
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape[-2:] != (5, 5) or (cm < 0).any() or not np.isfinite(cm).all():
        raise ValueError('Require nonnegative finite five-class confusion matrices')
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=support + predicted > 0)
    total = cm.sum(axis=(-2, -1))
    if (total <= 0).any():
        raise ValueError('Empty scored subject/bootstrap sample')
    result = {'accuracy': tp.sum(-1) / total, 'macro_f1': f1.mean(-1)}
    for i, stage in enumerate(STAGE_NAMES):
        for name, value in [('precision', precision), ('recall', recall), ('f1', f1)]:
            result[stage + '_' + name] = value[..., i]
    for true, predicted, key in [(3, 2, 'N3_to_N2_rate'), (2, 3, 'N2_to_N3_rate')]:
        result[key] = np.divide(cm[..., true, predicted], support[..., true],
                                out=np.full_like(support[..., true], np.nan), where=support[..., true] > 0)
    return result


def paired_budget_contrast(cms, resamples=10000, seed=2031):
    keys = ('limited_source_only', 'limited_adast', 'full_source_only', 'full_adast')
    if set(cms) != set(keys) or resamples < 1:
        raise ValueError('Require four planned paired systems and positive resamples')
    values = {k: np.asarray(cms[k]) for k in keys}
    base = values[keys[0]]
    if base.ndim != 3 or base.shape[1:] != (5, 5) or not len(base):
        raise ValueError('Require paired (people,5,5) confusions')
    for cm in values.values():
        if cm.shape != base.shape or not np.array_equal(cm.sum(-1), base.sum(-1)):
            raise ValueError('Paired reference supports/people ordering differ')
        vector_metrics(cm)
    # The SAME people indices are used for all four systems and all estimands.
    draws = np.random.default_rng(seed).integers(0, len(base), (resamples, len(base)), dtype=np.int32)
    observed = {k: vector_metrics(cm.sum(0)) for k, cm in values.items()}
    distribution = {name: {key: [] for key in observed[keys[0]]} for name in ('limited', 'full', 'difference_of_differences')}
    means = {k: vector_metrics(cm)['macro_f1'] for k, cm in values.items()}
    mean_effects = {'limited': means['limited_adast'] - means['limited_source_only'],
                    'full': means['full_adast'] - means['full_source_only']}
    mean_effects['difference_of_differences'] = mean_effects['full'] - mean_effects['limited']
    for start in range(0, resamples, 256):
        selected = draws[start:start + 256]
        metrics = {k: vector_metrics(cm[selected].sum(1)) for k, cm in values.items()}
        for key in observed[keys[0]]:
            limited = metrics['limited_adast'][key] - metrics['limited_source_only'][key]
            full = metrics['full_adast'][key] - metrics['full_source_only'][key]
            for name, delta in [('limited', limited), ('full', full), ('difference_of_differences', full - limited)]:
                distribution[name][key].extend(delta)
    contrasts = {}
    for name in distribution:
        means_delta = mean_effects[name]
        item = {'subject_mean_macro_f1': {'difference': float(means_delta.mean()),
                 'ci95': np.quantile(means_delta[draws].mean(1), [.025, .975]).tolist()}, 'pooled': {}}
        for key, dist in distribution[name].items():
            limited = observed['limited_adast'][key] - observed['limited_source_only'][key]
            full = observed['full_adast'][key] - observed['full_source_only'][key]
            delta = {'limited': limited, 'full': full, 'difference_of_differences': full - limited}[name]
            dist = np.asarray(dist)
            defined = np.isfinite(dist)
            item['pooled'][key] = {'difference': float(delta) if np.isfinite(delta) else None,
                                   'ci95': np.quantile(dist[defined], [.025, .975]).tolist() if defined.any() else None,
                                   'defined_bootstrap_draws': int(defined.sum())}
        contrasts[name] = item
    return {'unit': 'participant', 'subjects': len(base), 'resamples': resamples, 'seed': seed,
            'same_resampling_across_all_four_systems': True, 'primary_checkpoint': 'final_epoch30',
            'contrast_definition': '(ADAST_full-source_only_full)-(ADAST_limited-source_only_limited)',
            'contrasts': contrasts}


def read(path):
    return json.loads(Path(path).read_bytes())


def load_verified_comparison():
    result_sets, matrices, bindings = {}, {}, {}
    for name, folder in [('limited', OLD), ('full', NEW)]:
        result = read(folder / 'aggregate_results.json')
        digest = sha256_file(folder / 'aggregate_results.json')
        for proof_name in ('verification.json', 'independent_verification.json'):
            proof = read(folder / proof_name)
            if proof['status'] != 'passed' or proof['aggregate_results_sha256'] != digest:
                raise ValueError('Require matching independently verified evaluation')
            bindings[(folder / proof_name).resolve().as_posix()] = sha256_file(folder / proof_name)
        if result['status'] != ('complete_ten_fold_harmonized_ADAST_evaluation' if name == 'limited' else 'complete_fullsource_final'):
            raise ValueError('Only completed final-epoch matched evaluations allowed')
        if name == 'full' and result['selection'] != 'final':
            raise ValueError('Do not substitute secondary best in the primary budget contrast')
        with np.load(folder / 'private_target_confusions.npz', allow_pickle=False) as z:
            if set(z.files) != {'source_only', 'adast'}:
                raise ValueError('Confusion arm schema differs')
            cm = {k: z[k].copy() for k in z.files}
        if (any(c.shape != (180, 5, 5) for c in cm.values()) or
                summarize_pair(cm['adast'], cm['source_only'], 'adast', 'source_only') != result['target']):
            raise ValueError('Confusions do not reconstruct verified target results')
        for arm, values in cm.items():
            matrices[name + '_' + arm] = values
        entries = read(folder / 'private_inference_manifest.json')['records']
        result_sets[name] = [{k: e[k] for k in ('subject_id', 'record_key', 'epochs', 'source_edf_sha256')} for e in entries]
        for filename in ('aggregate_results.json', 'private_target_confusions.npz', 'private_inference_manifest.json',
                         'all_checkpoints_frozen.json', 'execution_specification.json'):
            bindings[(folder / filename).resolve().as_posix()] = sha256_file(folder / filename)
    if result_sets['limited'] != result_sets['full']:
        raise ValueError('Participant identities/order differ across budgets')
    bindings[Path(__file__).resolve().as_posix()] = sha256_file(Path(__file__))
    for filename in ('metrics.py', 'preprocessing.py', 'revision_campaign.py', 'cpu_followups.py'):
        path = ROOT / 'src/sleeptcn' / filename
        bindings[path.resolve().as_posix()] = sha256_file(path)
    return matrices, bindings


def build_report(matrices):
    report = paired_budget_contrast(matrices)
    report['systems'] = {key: {'subject_mean_macro_f1': float(vector_metrics(cm)['macro_f1'].mean()),
                               'pooled': metrics_from_confusion(cm.sum(0))} for key, cm in matrices.items()}
    report['status'] = 'complete_verified_final_epoch_budget_contrast'
    report['valid_epochs'] = int(matrices['full_adast'].sum())
    if report['valid_epochs'] != 169012:
        raise ValueError('Scored cohort support differs')
    report['interpretation_scope'] = 'Budget changes source updates, target exposure, and scheduler position in update units; not isolated source coverage.'
    report['training_variability_included_in_intervals'] = False
    return report


def verify(args):
    bindings = read(args.output / 'bound_provenance.private.json')
    for path, digest in bindings.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('Bound evaluation/code changed during analysis')
    matrices, actual_bindings = load_verified_comparison()
    if actual_bindings != bindings or build_report(matrices) != read(args.output / 'aggregate_results.json'):
        raise ValueError('Paired bootstrap/report does not replay exactly in separate process')
    write_once_json(args.output / 'independent_verification.json', {'status': 'passed', 'separate_process_recomputation': True,
        'all_four_systems_same_people_checked': True,
        'all_pooled_stage_and_subject_mean_contrasts_recomputed': True, 'resamples': 10000,
        'aggregate_results_sha256': sha256_file(args.output / 'aggregate_results.json')})


def analyze(args):
    if args.output.exists():
        raise FileExistsError('Preserve completed/partial analysis; no automatic overwrite')
    matrices, bindings = load_verified_comparison()
    report = build_report(matrices)
    args.output.mkdir(parents=True)
    write_once_json(args.output / 'bound_provenance.private.json', bindings)
    write_once_json(args.output / 'aggregate_results.json', report)
    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--verify-only',
                    '--output', args.output.resolve().as_posix()], check=True, timeout=1800)
    print(json.dumps({'status': report['status'], 'subjects': 180,
                      'subject_mean_macro_f1_contrasts': {k: v['subject_mean_macro_f1'] for k, v in report['contrasts'].items()}}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise ValueError('Use the planned private contrast directory')
    if args.verify_only:
        verify(args)
    else:
        analyze(args)
