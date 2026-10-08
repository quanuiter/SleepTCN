"""Operational v2 of the historical local evaluator; no training or automatic replay.

Original runner and proofs remain unchanged. v2 binds its own code and runtime_io
in a new output directory; use the original runner for historical verification.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from sleeptcn.runtime_io import ImmutableHashCache, ReportingBudget
import run_adast_10fold_local_evaluation as common
import verify_adast_fullsource_completion_results as training_check

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'runs/adast_fullsource_retry_20261007'
PREFLIGHT = ROOT / 'runs/adast_fullsource_preflight_20261006'
EVALUATION = ROOT / 'runs/adast_fullsource_local_v2'
ARMS = common.ARMS
sha = lambda path: common.sha256_file(Path(path))
read = lambda path: json.loads(Path(path).read_bytes())
write = common.write_once_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_proof(folder, expected_aggregate):
    require(sha(folder / 'aggregate_results.json') == expected_aggregate, 'Training aggregate changed')
    for name in ('verification.json', 'independent_verification.json'):
        proof = read(folder / name)
        require(proof['status'] == 'passed' and proof['aggregate_results_sha256'] == expected_aggregate,
                'Require matching completed training proofs')


def frozen_models(selection, guard=lambda: None):
    require(selection in ('final', 'best'), 'Only planned final and source-validation best allowed')
    pointer = read(RECORD / 'verified_results_pointer.json')
    fresh = Path(pointer['results_root'])
    require(pointer['status'] == 'passed' and pointer['archive_sha256'] ==
            '59cd81bca2a1bb67149b8d3348415d63c8524642240a32b608723ce56c0dd295', 'Fresh result provenance differs')
    check_proof(fresh, pointer['aggregate_results_sha256'])
    require(sha(fresh / 'independent_verification.json') == pointer['independent_verification_sha256'], 'Fresh proof changed')
    proof = read(fresh / 'independent_verification.json')
    require(proof['all_496_validation_diagnostics_recomputed'] and
            proof['all_32_best_final_checkpoints_CPU_validation_replayed'], 'Full validation replay absent')
    storage = read(ROOT / 'runs/adast_fullsource_completion_20261006/storage.json')
    input_manifest = read(Path(storage['payload_root']) / 'manifest.json')
    audit = read(PREFLIGHT / 'audit.json')
    require(sha(PREFLIGHT / 'audit.json') == input_manifest['reuse_audit_sha256'] and
            audit['status'] == 'passed' and audit['reusable_models'] == 4, 'Reuse audit changed')
    cache = ImmutableHashCache(sha)
    ledger = cache.ledger

    def checked(path, expected=None):
        guard()
        return cache.check(path, expected)

    for name in ('aggregate_results.json', 'verification.json', 'independent_verification.json', 'execution_specification.json'):
        checked(fresh / name)
    checked(RECORD / 'verified_results_pointer.json')
    checked(PREFLIGHT / 'audit.json', input_manifest['reuse_audit_sha256'])
    checked(PREFLIGHT / 'checkpoint_inventory.private.json')
    cfg = read(ROOT / 'configs/adast_fullsource_completion_v1_20261006.json')
    training_check.validate_protocol(cfg)
    result = read(fresh / 'aggregate_results.json')
    require(result['status'] == training_check.STATUS and result['new_models_completed'] == 16,
            'Require all fresh matched models')
    inventory = read(PREFLIGHT / 'checkpoint_inventory.private.json')
    require(len(inventory) == 8, 'Require eight reused best/final files')
    records, previous = [], []
    for fold in (0, 1):
        subset = [row for row in inventory if row['fold'] == fold]
        require(len(subset) == 4, 'Reuse checkpoint inventory differs')
        source = next(row for row in subset if row['arm'] == 'source_only_full_source' and row['kind'] == selection)
        adaptive_key = 'adast_full_source' if fold == 0 else 'adast_reference_full_source'
        adaptive = next(row for row in subset if row['arm'] == adaptive_key and row['kind'] == selection)
        folder = Path(source['path']).parent.parent
        expected = audit['folds'][fold]['aggregate_sha256']
        require(expected == ('ff5ef23108fa3116ae4b8ebae3ebe945e2c6ee026f2c3ee615429ed8e52c9a99' if fold == 0 else
                             '579ce087700074764044f13823829803003f75ab1986cc58dd0689b67785bf53'), 'Reuse aggregate identity differs')
        check_proof(folder, expected)
        for name in ('aggregate_results.json', 'verification.json', 'independent_verification.json', 'execution_specification.json'):
            checked(folder / name)
        spec = read(folder / 'execution_specification.json')
        payload = ROOT / 'runs/adast_development_20261004/payload' if fold == 0 else folder.parent / 'payload'
        checked(payload / 'manifest.json', spec['manifest_sha256'])
        manifest = read(payload / 'manifest.json')
        for name, digest in manifest['files'].items():
            checked(payload / name, digest)
        for key in ('seed', 'batch_size', 'rounds', 'epochs_per_round', 'optimizer',
                    'source_loss_weights_by_round', 'target_loss_weights_by_round', 'similarity_weight', 'adversarial_weight'):
            require(spec['protocol'][key] == cfg[key], 'Reuse training setting differs: ' + key)
        require(spec['backend'] == cfg['backend'] and spec['torch'] == cfg['required_torch'] and
                not any(spec[k] for k in ('source_outer_test_access', 'target_test_access', 'target_true_label_access')),
                'Reuse backend/data roles differ')
        checkpoint_rows = {}
        for arm, row in [('source_only', source), ('adast', adaptive)]:
            expected_cp = next(c for c in audit['folds'][fold]['checkpoints'] if c['arm'] == row['arm'] and c['kind'] == selection)
            checked(row['path'], expected_cp['sha256'])
            require(row['sha256'] == expected_cp['sha256'], 'Inventory checkpoint differs from audit')
            checkpoint_rows[arm] = {'checkpoint_path': row['path'], 'checkpoint_sha256': row['sha256'],
                                    'selected_epoch': expected_cp['epoch'], 'original_arm': row['arm'], 'reuse': True}
        records.append({'fold': fold, 'arms': checkpoint_rows})
        previous.append({'fold': fold, 'aggregate_sha256': expected, 'results_root': folder.as_posix()})
    for fold_row in result['folds']:
        fold, pair = fold_row['fold'], fold_row['arms']
        require([a['arm'] for a in pair] == training_check.ARMS, 'Fresh pair schema differs')
        checkpoint_rows = {}
        for arm, row in zip(ARMS, pair):
            path = fresh / f'fold_{fold:02d}' / row['arm'] / (selection + '.pt')
            checked(path, row[selection + '_checkpoint_sha256'])
            checkpoint_rows[arm] = {'checkpoint_path': path.as_posix(), 'checkpoint_sha256': row[selection + '_checkpoint_sha256'],
                                    'selected_epoch': 30 if selection == 'final' else row['best_epoch'],
                                    'original_arm': row['arm'], 'reuse': False}
        records.append({'fold': fold, 'arms': checkpoint_rows})
    require([r['fold'] for r in records] == list(range(10)), 'Require all ten pairs in original order')
    return {'selection': selection, 'records': records}, ledger, previous


def freeze(output, args, budget):
    frozen, ledger, reuse = frozen_models(args.selection, budget)
    private = read(common.BASE / 'private_local_provenance.json')
    target_path = common.PILOT / 'private_inference_manifest.json'
    split_path = ROOT / 'data/splits/sleepedf_sc_10fold_seed42_v2.json'
    adaptation_path = ROOT / 'data/cache/shhs_label_free_adaptation_v1_20261001/private_manifest.json'
    for path, expected in [(target_path, private['target_test_manifest_sha256']), (split_path, private['source_split_sha256']),
                           (adaptation_path, private['adaptation_manifest_sha256'])]:
        require(sha(path) == expected, 'Local data-role provenance differs')
        ledger[path.resolve().as_posix()] = expected
    ledger[(common.BASE / 'private_local_provenance.json').resolve().as_posix()] = sha(common.BASE / 'private_local_provenance.json')
    manifest = read(common.WORK / 'manifest.json')
    binding = read(common.BASE / 'bundle_verification.json')
    require(sha(common.WORK / 'manifest.json') == binding['manifest_sha256'], 'Original source payload manifest differs')
    ledger[(common.WORK / 'manifest.json').resolve().as_posix()] = binding['manifest_sha256']
    for name, digest in manifest['files'].items():
        budget()
        require(sha(common.WORK / name) == digest, 'Original source payload differs')
        ledger[(common.WORK / name).resolve().as_posix()] = digest
    entries = read(target_path)['records']
    locked = read(args.shhs_root / 'manifests/shhs1_subject_manifest_seed42.json')['subjects']
    target_ids, adaptation_ids = {e['subject_id'] for e in entries}, {e['subject_id'] for e in read(adaptation_path)['records']}
    require(len(entries) == len(target_ids) == 180 and len(adaptation_ids) == 5 and not target_ids & adaptation_ids
            and target_ids == {s['subject_id'] for s in locked if s['role'] == 'test'}
            and adaptation_ids == {s['subject_id'] for s in locked if s['role'] == 'adaptation'}
            and sum(e['epochs'] for e in entries) == 183528, 'Target/adaptation roles/support differ')
    paths = [Path(__file__), ROOT / 'src/sleeptcn/runtime_io.py', ROOT / 'scripts/run_adast_10fold_local_evaluation.py',
             ROOT / 'scripts/run_colab_adast_training.py', ROOT / 'scripts/run_recovered_e3_cpu_pilot.py',
             ROOT / 'scripts/verify_adast_fullsource_completion_results.py', ROOT / 'scripts/verify_adast_development_results.py',
             ROOT / 'scripts/verify_adast_fullsource_completion_results_v2.py', ROOT / 'scripts/adast_cpu_inference_checks.py',
             ROOT / 'scripts/verify_colab_adast_results.py', *[ROOT / 'src/sleeptcn' / (n + '.py') for n in
             ['metrics', 'cpu_followups', 'preprocessing', 'shhs_preprocessing', 'revision_campaign', 'revision_weighted_campaign']]]
    spec = {'selection': args.selection, 'source_selection_rule': 'final_epoch30_primary_best_source_validation_first_tie_secondary',
            'ensemble': 'maximum_dual_head_logits_softmax_per_fold_then_float64_mean_cast_float32',
            'target_attention': 'source_for_source_only_target_for_ADAST', 'source_attention': 'source_for_both_arms_OOF',
            'torch': str(torch.__version__), 'inference_backend': 'local_CPU_float32', 'cpu_threads': 4, 'batch_size': 128,
            'cpu_flush_denormal': True, 'weights_or_input_scaling_changed_for_inference': False,
            'max_seconds': args.max_seconds, 'target_score_based_selection': False, 'source_OOF': True,
            'new_models': 16, 'reused_models': 4, 'reuse_provenance': reuse, 'bound_file_sha256': ledger,
            'code_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    write(output / 'execution_specification.json', spec)
    write(output / 'all_checkpoints_frozen.json', frozen)
    for path in paths:
        destination = output / 'code_snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return spec, frozen, entries


def verify(args):
    output = args.output
    spec, frozen, result = [read(output / name) for name in
                            ('execution_specification.json', 'all_checkpoints_frozen.json', 'aggregate_results.json')]
    for path, digest in spec['bound_file_sha256'].items():
        require(sha(path) == digest, 'Bound checkpoint/proof/input changed')
    for relative, digest in spec['code_sha256'].items():
        require(sha(ROOT / relative) == digest and sha(output / 'code_snapshot' / relative) == digest, 'Executed code changed')
    require(result['status'] == 'complete_fullsource_' + spec['selection'] and
            result['selection'] == spec['selection'] and result['frozen_checkpoints_sha256'] == sha(output / 'all_checkpoints_frozen.json')
            and result['execution_specification_sha256'] == sha(output / 'execution_specification.json')
            and result['inference_manifest_sha256'] == sha(output / 'private_inference_manifest.json')
            and result['source_manifest_sha256'] == sha(output / 'private_source_manifest.json'), 'Result/provenance differs')
    require([p['fold'] for p in frozen['records']] == list(range(10)), 'Fold coverage differs')
    entries = read(output / 'private_inference_manifest.json')['records']
    original = read(common.PILOT / 'private_inference_manifest.json')['records']
    keys = ('subject_id', 'record_key', 'epochs', 'source_edf_sha256')
    require([{k: e[k] for k in keys} for e in entries] == [{k: e[k] for k in keys} for e in original], 'Target identity/order differs')
    target_cms, refs = common.target_confusions(args, entries, sha(output / 'all_checkpoints_frozen.json'), sha(output / 'execution_specification.json'))
    source_cms, source_entries, source_refs = common.source_confusions(output)
    require(read(output / 'private_source_manifest.json') == {'records': source_entries, 'reference_sha256': source_refs}
            and read(output / 'private_reference_hashes.json') == refs, 'Prediction/reference manifest differs')
    for name, matrices in [('target', target_cms), ('source', source_cms)]:
        require(common.summarize_pair(matrices['adast'], matrices['source_only'], 'adast', 'source_only') == result[name],
                'Metrics/bootstrap do not recompute')
        with np.load(output / ('private_' + name + '_confusions.npz'), allow_pickle=False) as values:
            for arm, cm in matrices.items():
                np.testing.assert_array_equal(values[arm], cm)
    networks = common.load_networks(frozen)
    signal = common.load_module('fullsource_verification_signal', ROOT / 'scripts/run_recovered_e3_cpu_pilot.py')
    for entry in (entries[0], entries[-1]):
        x, _ = signal.read_full_record(args.shhs_root / 'shhs/polysomnography/edfs/shhs1' / (entry['record_key'] + '.edf'), entry['source_edf_sha256'])
        with np.load(entry['path'], allow_pickle=False) as values:
            for (fold, arm), models in networks.items():
                actual = common.softmax_logits(common.predict(models, x, 'source' if arm == 'source_only' else 'target', 'cpu'))
                np.testing.assert_array_equal(actual, values['fold_' + arm][fold])
    x = np.load(common.WORK / 'data/source_x.npy', mmap_mode='r', allow_pickle=False)
    for fold in range(10):
        with np.load(output / 'source_predictions' / f'fold_{fold:02d}.npz', allow_pickle=False) as values:
            for arm in ARMS:
                actual = common.softmax_logits(common.predict(networks[fold, arm], x[values['indices'][:128]], 'source', 'cpu'))
                np.testing.assert_array_equal(actual, values[arm][:128])
    proof = {'status': 'passed', 'selection': spec['selection'], 'checkpoint_hashes_verified': 20,
             'target_prediction_files_verified': 180, 'source_OOF_folds_verified': 10,
             'target_model_record_replays': 40, 'source_model_batch_replays': 20,
             'all_confusions_and_bootstrap_recomputed': True, 'aggregate_results_sha256': sha(output / 'aggregate_results.json')}
    write(output / 'independent_verification.json', proof)
    print(json.dumps(proof, indent=2), flush=True)


def run(args):
    output = args.output
    require(output == EVALUATION / args.selection, 'Require the planned private selection directory')
    if output.exists():
        raise FileExistsError('Preserve existing attempt; no automatic restart')
    output.mkdir(parents=True)
    write(output / 'process.json', {'python_pid': os.getpid(), 'started_local': datetime.now().astimezone().isoformat(),
                                  'executable': sys.executable, 'script': str(Path(__file__).resolve()), 'arguments': sys.argv[1:], 'training': False})
    budget = ReportingBudget(args.max_seconds, output / 'progress.json')
    try:
        budget.publish(status='running', phase='verify_and_freeze', target_records_completed=0)
        spec, frozen, entries = freeze(output, args, budget)
        if args.freeze_only:
            budget.publish(status='frozen_only', phase='complete')
            return 0
        networks = common.load_networks(frozen)
        frozen_hash, spec_hash = sha(output / 'all_checkpoints_frozen.json'), sha(output / 'execution_specification.json')
        signal = common.load_module('fullsource_inference_signal', ROOT / 'scripts/run_recovered_e3_cpu_pilot.py')
        targets = []
        for ordinal, entry in enumerate(entries, 1):
            budget()
            budget.publish(phase='target_inference', target_records_completed=ordinal - 1)
            x, _ = signal.read_full_record(args.shhs_root / 'shhs/polysomnography/edfs/shhs1' / (entry['record_key'] + '.edf'), entry['source_edf_sha256'])
            require(x.shape == (entry['epochs'], 3000), 'Full recording support differs')
            parts = {a: [] for a in ARMS}
            for (fold, arm), models in networks.items():
                budget.publish(fold=fold, arm=arm)
                parts[arm].append(common.softmax_logits(common.predict(models, x, 'source' if arm == 'source_only' else 'target', 'cpu', budget)))
            parts = {a: np.stack(v) for a, v in parts.items()}
            path = output / 'target_predictions' / (entry['record_key'] + '.npz')
            path.parent.mkdir(exist_ok=True)
            metadata = {'subject_id': entry['subject_id'], 'source_edf_sha256': entry['source_edf_sha256'],
                        'frozen_checkpoints_sha256': frozen_hash, 'specification_sha256': spec_hash}
            with path.open('xb') as stream:
                np.savez_compressed(stream, **{a: p.astype(np.float64).mean(0).astype(np.float32) for a, p in parts.items()},
                                    **{'fold_' + a: p for a, p in parts.items()}, original_epoch_index=np.arange(len(x)), metadata_json=np.array(json.dumps(metadata)))
            common.audit_prediction(path, entry, frozen_hash, spec_hash)
            targets.append({**entry, 'path': path.resolve().as_posix(), 'sha256': sha(path)})
            if ordinal == 1 or ordinal % 10 == 0:
                print(f'Full-source {args.selection}: target {ordinal}/180; {args.max_seconds - budget.remaining():.1f}s', flush=True)
        write(output / 'private_inference_manifest.json', {'records': targets})
        budget.publish(phase='target_scoring', target_records_completed=180, fold=None, arm=None)
        target_cms, refs = common.target_confusions(args, targets, frozen_hash, spec_hash, budget)
        write(output / 'private_reference_hashes.json', refs)
        budget.publish(phase='source_OOF_inference')
        source_cms, source_entries, source_refs = common.source_confusions(output, networks, budget)
        write(output / 'private_source_manifest.json', {'records': source_entries, 'reference_sha256': source_refs})
        summaries = {}
        for name, matrices in [('target', target_cms), ('source', source_cms)]:
            budget()
            summaries[name] = common.summarize_pair(matrices['adast'], matrices['source_only'], 'adast', 'source_only')
            with (output / ('private_' + name + '_confusions.npz')).open('xb') as stream:
                np.savez_compressed(stream, **matrices)
        result = {'status': 'complete_fullsource_' + args.selection, 'selection': args.selection, **summaries,
                  'execution_specification_sha256': spec_hash, 'frozen_checkpoints_sha256': frozen_hash,
                  'inference_manifest_sha256': sha(output / 'private_inference_manifest.json'),
                  'source_manifest_sha256': sha(output / 'private_source_manifest.json'),
                  'elapsed_inference_and_scoring_seconds': args.max_seconds - budget.remaining(),
                  'training_backend': 'CUDA_all_twenty_models_sixteen_new_four_reused', 'inference_backend': 'local_CPU',
                  'target_labels_used_for_selection': False, 'previously_examined_cohort_post_hoc': True,
                  'training_variance_not_in_bootstrap': True, 'published_ADAST_exact_reproduction': False}
        write(output / 'aggregate_results.json', result)
        write(output / 'verification.json', {'status': 'passed', 'checkpoint_count': 20, 'target_predictions_audited': 180,
                                            'source_OOF_folds': 10, 'aggregate_results_sha256': sha(output / 'aggregate_results.json')})
        budget.publish(phase='independent_verification')
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--verify-only', '--selection', args.selection,
                        '--output', output.as_posix(), '--shhs-root', args.shhs_root.as_posix()], timeout=budget.remaining(), check=True)
        budget.publish(status=result['status'], phase='complete', target_records_completed=180)
        print('FULLSOURCE LOCAL EVALUATION AND INDEPENDENT VERIFICATION COMPLETE', flush=True)
        return 0
    except (common.CampaignStop, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        budget.publish(status='stopped_resource_budget_or_interruption', reason=str(error), artifacts_retained=True)
        return 2
    except Exception as error:
        budget.publish(status='failed', reason=f'{type(error).__name__}: {error}', artifacts_retained=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', choices=['final', 'best'], default='final')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--shhs-root', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=18000)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.output = (args.output or EVALUATION / args.selection).resolve()
    args.shhs_root = args.shhs_root.resolve()
    require(0 < args.max_seconds <= 18000, 'Require at most five hours for inference/scoring/verification')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    require(torch.set_flush_denormal(True), 'Require profiled CPU flush-denormal support for inference')
    if args.verify_only:
        verify(args)
    else:
        sys.exit(run(args))
