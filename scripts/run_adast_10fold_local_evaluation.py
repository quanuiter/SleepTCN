"""Locked local inference/scoring of verified CUDA ADAST pairs; never trains."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sleeptcn.metrics import confusion_matrix_5
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import summarize_pair, write_once_json
from sleeptcn.revision_weighted_campaign import CampaignBudget, CampaignStop, audit_probabilities
from run_colab_adast_training import build_models, load_module, predict

BASE = ROOT / 'runs/colab_adast_20261004'
TRAIN = BASE / 'verified_gpu_results'
WORK = BASE / 'payload'
PILOT = ROOT / 'runs/teacher_revision_cpu_20261001/recovered_e3_fold0'
ARMS = ('source_only', 'adast')
COMPLETE = 'complete_ten_fold_harmonized_ADAST_evaluation'


def softmax_logits(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != 5 or not torch.isfinite(values).all():
        raise ValueError('Require finite five-class maximum logits')
    return torch.softmax(values, -1).numpy()


def audit_prediction(path, entry, frozen_hash, spec_hash):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'source_only', 'adast', 'fold_source_only', 'fold_adast',
                            'original_epoch_index', 'metadata_json'}:
            raise ValueError('Unexpected prediction fields')
        expected = {'subject_id': entry['subject_id'], 'source_edf_sha256': entry['source_edf_sha256'],
                    'frozen_checkpoints_sha256': frozen_hash, 'specification_sha256': spec_hash}
        if json.loads(str(z['metadata_json'])) != expected:
            raise ValueError('Prediction provenance differs')
        np.testing.assert_array_equal(z['original_epoch_index'], np.arange(entry['epochs']))
        result = {}
        for arm in ARMS:
            parts = z['fold_' + arm]
            audit_probabilities(parts, entry['epochs'])
            mean = parts.astype(np.float64).mean(0).astype(np.float32)
            np.testing.assert_array_equal(z[arm], mean)
            result[arm] = mean
        return result


def load_networks(frozen):
    module = load_module('locked_ADAST_model', WORK / 'upstream/models.py')
    cfg = load_module('locked_ADAST_config', WORK / 'upstream/configs.py').Config()
    networks = {}
    for pair in frozen['records']:
        if set(pair['arms']) != set(ARMS):
            raise ValueError('Matched pair missing')
        for arm, selected in pair['arms'].items():
            path = Path(selected['checkpoint_path'])
            if sha256_file(path) != selected['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint changed')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            models = build_models(module, cfg, 123, 'cpu')
            for name, model in models.items():
                model.load_state_dict(payload['models'][name], strict=True)
                model.eval()
            networks[pair['fold'], arm] = models
    return networks


def freeze(output, args):
    result = json.loads((TRAIN / 'aggregate_results.json').read_bytes())
    for name in ('verification.json', 'independent_verification.json'):
        evidence = json.loads((TRAIN / name).read_bytes())
        if evidence['status'] != 'passed' or evidence['aggregate_results_sha256'] != sha256_file(TRAIN / 'aggregate_results.json'):
            raise ValueError('Require verified CUDA training')
    if result['selected_checkpoints'] != 20 or result['status'] != 'complete_ten_fold_matched_ADAST_source_training':
        raise ValueError('Require all twenty completed checkpoints')
    private = json.loads((BASE / 'private_local_provenance.json').read_bytes())
    target_path = PILOT / 'private_inference_manifest.json'
    split_path = ROOT / 'data/splits/sleepedf_sc_10fold_seed42_v2.json'
    adaptation_path = ROOT / 'data/cache/shhs_label_free_adaptation_v1_20261001/private_manifest.json'
    if (sha256_file(target_path) != private['target_test_manifest_sha256']
            or sha256_file(split_path) != private['source_split_sha256']
            or sha256_file(adaptation_path) != private['adaptation_manifest_sha256']):
        raise ValueError('Locked local input manifests changed')
    entries = json.loads(target_path.read_bytes())['records']
    target_ids = {e['subject_id'] for e in entries}
    adaptation = json.loads(adaptation_path.read_bytes())['records']
    locked = json.loads((args.shhs_root / 'manifests/shhs1_subject_manifest_seed42.json').read_bytes())['subjects']
    expected_test = {s['subject_id'] for s in locked if s['role'] == 'test'}
    expected_adaptation = {s['subject_id'] for s in locked if s['role'] == 'adaptation'}
    if (len(entries) != 180 or len(target_ids) != 180 or target_ids != expected_test
            or {s['subject_id'] for s in adaptation} != expected_adaptation
            or len(expected_adaptation) != 5 or expected_adaptation & target_ids
            or sum(e['epochs'] for e in entries) != 183528):
        raise ValueError('Locked subject roles/full-record support differ')
    manifest = json.loads((WORK / 'manifest.json').read_bytes())
    for name, digest in manifest['files'].items():
        if sha256_file(WORK / name) != digest:
            raise ValueError('Frozen training payload changed')
    paths = [Path(__file__), ROOT / 'scripts/run_colab_adast_training.py',
             ROOT / 'scripts/run_recovered_e3_cpu_pilot.py', ROOT / 'scripts/verify_colab_adast_results.py',
             *[ROOT / 'src/sleeptcn' / (n + '.py') for n in
               ['metrics', 'cpu_followups', 'preprocessing', 'shhs_preprocessing',
                'revision_campaign', 'revision_weighted_campaign']]]
    spec = {'training_aggregate_sha256': sha256_file(TRAIN / 'aggregate_results.json'),
            'training_independent_verification_sha256': sha256_file(TRAIN / 'independent_verification.json'),
            'training_protocol': json.loads((WORK / 'protocol.json').read_bytes()),
            'target_manifest_sha256': sha256_file(target_path),
            'private_source_mapping_sha256': sha256_file(BASE / 'private_local_provenance.json'),
            'source_split_sha256': sha256_file(split_path),
            'adaptation_manifest_sha256': sha256_file(adaptation_path),
            'torch': str(torch.__version__), 'inference_backend': 'local_CPU_float32',
            'cpu_threads': 4, 'batch_size': 128, 'max_seconds': args.max_seconds,
            'selection': 'all_twenty_final_fixed_budget_checkpoints_locked_no_new_selection',
            'ensemble': 'maximum_dual_head_logits_softmax_per_fold_then_float64_mean_cast_float32',
            'source_test_attention': 'source_for_both_arms_OOF_not_ensemble',
            'target_attention': 'source_for_source_only_target_for_ADAST',
            'target_test_labels_for_selection': False,
            'code_sha256': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}}
    write_once_json(output / 'execution_specification.json', spec)
    for path in paths:
        destination = output / 'code_snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != sha256_file(path):
            raise ValueError('Code snapshot changed')
        if not destination.exists():
            shutil.copyfile(path, destination)
    records = []
    for pair in result['folds']:
        records.append({'fold': pair['fold'], 'arms': {arm: {
            'checkpoint_path': str((TRAIN / ('fold_%02d' % pair['fold']) / arm / 'final.pt').resolve()),
            'checkpoint_sha256': selection['checkpoint_sha256']}
            for arm, selection in pair['arms'].items()}})
    if [p['fold'] for p in records] != list(range(10)):
        raise ValueError('Must lock all ten folds in order')
    frozen = {'status': 'twenty_verified_CUDA_checkpoints_locked_before_local_inference', 'records': records}
    write_once_json(output / 'all_checkpoints_frozen.json', frozen)
    return spec, frozen, entries


def source_confusions(output, networks=None, budget=lambda: None):
    private = json.loads((BASE / 'private_local_provenance.json').read_bytes())
    splits = json.loads((ROOT / 'data/splits/sleepedf_sc_10fold_seed42_v2.json').read_bytes())['outer_runs']
    x = np.load(WORK / 'data/source_x.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(WORK / 'data/source_y.npy', mmap_mode='r', allow_pickle=False)
    offset, source_records, reference_hashes = 0, {}, {}
    for mapping in private['source_mapping']:
        path = ROOT / 'data/processed/filtered_v2' / (mapping['record_key'] + '.npz')
        if sha256_file(path) != mapping['sha256']:
            raise ValueError('Source reference changed')
        with np.load(path, allow_pickle=False) as z:
            valid_y = z['y'][z['y'] >= 0]
            np.testing.assert_array_equal(valid_y, y[offset:offset + mapping['epochs']])
            source_records[mapping['record_key']] = (offset, offset + len(valid_y), str(z['subject_id']))
        offset += len(valid_y)
        reference_hashes[mapping['record_key']] = mapping['sha256']
    if offset != 195469:
        raise ValueError('Source epoch support differs')
    subject_cms = {a: {} for a in ARMS}
    entries = []
    seen = []
    for fold, split in enumerate(splits):
        budget()
        with np.load(WORK / ('data/fold_%02d_roles.npz' % fold), allow_pickle=False) as roles:
            indices = roles['test'].astype(np.int64)
        expected = np.concatenate([np.arange(*source_records[k][:2]) for k in sorted(split['test']['record_keys'])])
        np.testing.assert_array_equal(indices, expected)
        path = output / 'source_predictions' / ('fold_%02d.npz' % fold)
        if networks is not None:
            if path.exists():
                raise FileExistsError('Source prediction exists; no automatic resume')
            probs = {arm: softmax_logits(predict(networks[fold, arm], x[indices], 'source', 'cpu', budget)) for arm in ARMS}
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as stream:
                np.savez_compressed(stream, **probs, indices=indices)
            print('Source OOF fold %d/10 complete' % (fold + 1), flush=True)
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {*ARMS, 'indices'}:
                raise ValueError('Source prediction schema differs')
            np.testing.assert_array_equal(z['indices'], indices)
            for key in sorted(split['test']['record_keys']):
                start, end, subject = source_records[key]
                positions = np.searchsorted(indices, np.arange(start, end))
                np.testing.assert_array_equal(indices[positions], np.arange(start, end))
                for arm in ARMS:
                    p = z[arm]
                    if p.shape != (len(indices), 5) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                        raise ValueError('Invalid source probabilities')
                    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6, rtol=0)
                    subject_cms[arm].setdefault(subject, np.zeros((5, 5), dtype=np.int64))
                    subject_cms[arm][subject] += confusion_matrix_5(y[start:end], p[positions].argmax(1))
        entries.append({'fold': fold, 'path': str(path.resolve()), 'sha256': sha256_file(path)})
        seen.extend(split['test']['subject_ids'])
    if len(seen) != 78 or len(set(seen)) != 78:
        raise ValueError('Source OOF subject overlap/support differs')
    cms = {a: np.stack([v[k] for k in sorted(v)]) for a, v in subject_cms.items()}
    if any(cm.sum() != 195469 for cm in cms.values()):
        raise ValueError('Source OOF score support differs')
    return cms, entries, reference_hashes


def target_confusions(args, entries, frozen_hash, spec_hash, budget=lambda: None):
    signal = load_module('ADAST_scoring_alignment', ROOT / 'scripts/run_recovered_e3_cpu_pilot.py')
    cms = {a: [] for a in ARMS}
    refs = {}
    for entry in entries:
        budget()
        prediction = audit_prediction(entry['path'], entry, frozen_hash, spec_hash)
        if sha256_file(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Target prediction hash changed')
        reference = args.shhs_root / 'processed_v1/filtered_v2' / (entry['record_key'] + '.npz')
        refs[entry['record_key']] = sha256_file(reference)
        with np.load(reference, allow_pickle=False) as z:
            if (str(z['subject_id']) != entry['subject_id'] or str(z['role']) != 'test'
                    or str(z['source_edf_sha256']) != entry['source_edf_sha256']):
                raise ValueError('Target reference identity differs')
            positions = signal.benchmark_positions(z['original_epoch_index'], entry['epochs'])
            for arm in ARMS:
                cms[arm].append(confusion_matrix_5(z['y'], prediction[arm][positions].argmax(1)))
    cms = {a: np.stack(cm) for a, cm in cms.items()}
    if any(cm.sum() != 169012 for cm in cms.values()):
        raise ValueError('Target scored support differs')
    return cms, refs


def verify(args):
    output = args.output
    result = json.loads((output / 'aggregate_results.json').read_bytes())
    spec = json.loads((output / 'execution_specification.json').read_bytes())
    frozen = json.loads((output / 'all_checkpoints_frozen.json').read_bytes())
    if [p['fold'] for p in frozen['records']] != list(range(10)):
        raise ValueError('Frozen fold coverage differs')
    for path, expected in [(TRAIN / 'aggregate_results.json', spec['training_aggregate_sha256']),
                           (TRAIN / 'independent_verification.json', spec['training_independent_verification_sha256']),
                           (BASE / 'private_local_provenance.json', spec['private_source_mapping_sha256']),
                           (ROOT / 'data/splits/sleepedf_sc_10fold_seed42_v2.json', spec['source_split_sha256']),
                           (ROOT / 'data/cache/shhs_label_free_adaptation_v1_20261001/private_manifest.json', spec['adaptation_manifest_sha256'])]:
        if sha256_file(path) != expected:
            raise ValueError('Frozen training/input evidence changed')
    manifest = json.loads((WORK / 'manifest.json').read_bytes())
    for name, digest in manifest['files'].items():
        if sha256_file(WORK / name) != digest:
            raise ValueError('Source/upstream payload changed')
    for relative, digest in spec['code_sha256'].items():
        if sha256_file(ROOT / relative) != digest or sha256_file(output / 'code_snapshot' / relative) != digest:
            raise ValueError('Executed code changed')
    if (result['status'] != COMPLETE or result['execution_specification_sha256'] != sha256_file(output / 'execution_specification.json')
            or result['frozen_checkpoints_sha256'] != sha256_file(output / 'all_checkpoints_frozen.json')
            or result['inference_manifest_sha256'] != sha256_file(output / 'private_inference_manifest.json')
            or result['source_manifest_sha256'] != sha256_file(output / 'private_source_manifest.json')):
        raise ValueError('Result/manifests mismatch')
    entries = json.loads((output / 'private_inference_manifest.json').read_bytes())['records']
    original = json.loads((PILOT / 'private_inference_manifest.json').read_bytes())['records']
    if sha256_file(PILOT / 'private_inference_manifest.json') != spec['target_manifest_sha256']:
        raise ValueError('Original locked target manifest changed')
    if [{k: e[k] for k in ('subject_id', 'record_key', 'epochs', 'source_edf_sha256')} for e in entries] != [
            {k: e[k] for k in ('subject_id', 'record_key', 'epochs', 'source_edf_sha256')} for e in original]:
        raise ValueError('Target identity/order differs from locked input')
    target_cms, refs = target_confusions(args, entries, sha256_file(output / 'all_checkpoints_frozen.json'),
                                         sha256_file(output / 'execution_specification.json'))
    source_cms, source_entries, source_refs = source_confusions(output)
    if json.loads((output / 'private_source_manifest.json').read_bytes()) != {'records': source_entries, 'reference_sha256': source_refs}:
        raise ValueError('Source prediction/reference manifest mismatch')
    if json.loads((output / 'private_reference_hashes.json').read_bytes()) != refs:
        raise ValueError('Target reference changed')
    for name, matrices in [('target', target_cms), ('source', source_cms)]:
        expected = summarize_pair(matrices['adast'], matrices['source_only'], 'adast', 'source_only')
        if expected != result[name]:
            raise ValueError('Metrics/bootstrap failed independent recomputation')
        with np.load(output / ('private_' + name + '_confusions.npz'), allow_pickle=False) as stored:
            for arm, cm in matrices.items():
                np.testing.assert_array_equal(stored[arm], cm)
    networks = load_networks(frozen)
    signal = load_module('ADAST_replay_signal', ROOT / 'scripts/run_recovered_e3_cpu_pilot.py')
    replay_count = 0
    # Separate-process replay of all 20 models on both endpoint recordings.
    for entry in (entries[0], entries[-1]):
        x, _ = signal.read_full_record(args.shhs_root / 'shhs/polysomnography/edfs/shhs1' / (entry['record_key'] + '.edf'), entry['source_edf_sha256'])
        with np.load(entry['path'], allow_pickle=False) as z:
            for (fold, arm), models in networks.items():
                actual = softmax_logits(predict(models, x, 'source' if arm == 'source_only' else 'target', 'cpu'))
                np.testing.assert_array_equal(actual, z['fold_' + arm][fold])
                replay_count += 1
    source_replays = 0
    source_x = np.load(WORK / 'data/source_x.npy', mmap_mode='r', allow_pickle=False)
    for fold in range(10):
        with np.load(output / 'source_predictions' / ('fold_%02d.npz' % fold), allow_pickle=False) as z:
            indices = z['indices'][:128]
            for arm in ARMS:
                actual = softmax_logits(predict(networks[fold, arm], source_x[indices], 'source', 'cpu'))
                np.testing.assert_array_equal(actual, z[arm][:128])
                source_replays += 1
    evidence = {'status': 'passed', 'checkpoint_hashes_verified': 20, 'target_prediction_files_verified': 180,
                'target_fold_arm_probabilities_verified': 3600, 'target_model_record_replays': replay_count,
                'source_OOF_folds_verified': 10, 'all_confusions_and_bootstrap_recomputed': True,
                'source_model_batch_replays': source_replays,
                'aggregate_results_sha256': sha256_file(output / 'aggregate_results.json')}
    write_once_json(output / 'independent_verification.json', evidence)
    print(json.dumps(evidence, indent=2), flush=True)


def run(args):
    output = args.output
    if not output.is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('Private outputs must be in runs')
    if (output / 'execution_specification.json').exists():
        raise FileExistsError('Already launched: inspect status, do not automatically restart')
    output.mkdir(parents=True, exist_ok=True)
    write_once_json(output / 'process.json', {'python_pid': os.getpid(),
        'started_local': datetime.now().astimezone().isoformat(),
        'executable': sys.executable, 'script': str(Path(__file__).resolve()),
        'arguments': sys.argv[1:], 'training': False})
    budget = CampaignBudget(args.max_seconds, output / 'progress.json')
    try:
        budget.publish(status='running', phase='verify_and_freeze', target_records_completed=0)
        spec, frozen, entries = freeze(output, args)
        if args.freeze_only:
            budget.publish(status='frozen_only', phase='complete')
            return 0
        networks = load_networks(frozen)
        frozen_hash, spec_hash = [sha256_file(output / p) for p in ('all_checkpoints_frozen.json', 'execution_specification.json')]
        helper = load_module('ADAST_full_record_signal', ROOT / 'scripts/run_recovered_e3_cpu_pilot.py')
        targets = []
        for ordinal, entry in enumerate(entries, 1):
            budget()
            budget.publish(phase='target_inference', target_records_completed=ordinal - 1)
            x, _ = helper.read_full_record(args.shhs_root / 'shhs/polysomnography/edfs/shhs1' / (entry['record_key'] + '.edf'), entry['source_edf_sha256'])
            if x.shape != (entry['epochs'], 3000):
                raise ValueError('Full-record signal support differs')
            parts = {a: [] for a in ARMS}
            for (fold, arm), models in networks.items():
                budget.publish(fold=fold, arm=arm)
                p = softmax_logits(predict(models, x, 'source' if arm == 'source_only' else 'target', 'cpu', budget))
                parts[arm].append(p)
            parts = {a: np.stack(v) for a, v in parts.items()}
            path = output / 'target_predictions' / (entry['record_key'] + '.npz')
            path.parent.mkdir(exist_ok=True)
            meta = {'subject_id': entry['subject_id'], 'source_edf_sha256': entry['source_edf_sha256'],
                    'frozen_checkpoints_sha256': frozen_hash, 'specification_sha256': spec_hash}
            with path.open('xb') as stream:
                np.savez_compressed(stream, **{a: p.astype(np.float64).mean(0).astype(np.float32) for a, p in parts.items()},
                    **{'fold_' + a: p for a, p in parts.items()}, original_epoch_index=np.arange(len(x)),
                    metadata_json=np.array(json.dumps(meta)))
            audit_prediction(path, entry, frozen_hash, spec_hash)
            targets.append({**entry, 'path': str(path.resolve()), 'sha256': sha256_file(path)})
            if ordinal == 1 or ordinal % 10 == 0:
                print('ADAST matched ten-fold target inference %d/180; %.1fs elapsed' % (ordinal, args.max_seconds - budget.remaining()), flush=True)
        write_once_json(output / 'private_inference_manifest.json', {'records': targets})
        # No target reference labels have been opened before this point.
        budget.publish(phase='target_scoring', target_records_completed=180, fold=None, arm=None)
        target_cms, refs = target_confusions(args, targets, frozen_hash, spec_hash, budget)
        write_once_json(output / 'private_reference_hashes.json', refs)
        budget.publish(phase='source_OOF_inference')
        source_cms, source_entries, source_refs = source_confusions(output, networks, budget)
        write_once_json(output / 'private_source_manifest.json', {'records': source_entries, 'reference_sha256': source_refs})
        summaries = {}
        for name, cms in [('target', target_cms), ('source', source_cms)]:
            budget()
            summaries[name] = summarize_pair(cms['adast'], cms['source_only'], 'adast', 'source_only')
            with (output / ('private_' + name + '_confusions.npz')).open('xb') as stream:
                np.savez_compressed(stream, **cms)
        result = {'status': COMPLETE, **summaries, 'execution_specification_sha256': spec_hash,
                  'frozen_checkpoints_sha256': frozen_hash,
                  'inference_manifest_sha256': sha256_file(output / 'private_inference_manifest.json'),
                  'source_manifest_sha256': sha256_file(output / 'private_source_manifest.json'),
                  'elapsed_inference_and_scoring_seconds': args.max_seconds - budget.remaining(),
                  'training_backend': 'all_twenty_fresh_CUDA', 'inference_backend': 'local_CPU',
                  'target_labels_used_for_selection': False, 'previously_examined_cohort_post_hoc': True,
                  'training_variance_not_in_bootstrap': True, 'published_ADAST_exact_reproduction': False}
        write_once_json(output / 'aggregate_results.json', result)
        write_once_json(output / 'verification.json', {'status': 'passed', 'checkpoint_count': 20,
            'target_predictions_audited': 180, 'source_OOF_folds': 10,
            'aggregate_results_sha256': sha256_file(output / 'aggregate_results.json')})
        budget.publish(phase='independent_verification')
        budget()
        completed = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--verify-only',
                                   '--output', str(output), '--shhs-root', str(args.shhs_root)],
                                  timeout=budget.remaining(), check=True)
        budget.publish(status=COMPLETE, phase='complete', target_records_completed=180)
        print('LOCAL ADAST EVALUATION AND INDEPENDENT VERIFICATION COMPLETE', flush=True)
        return completed.returncode
    except (CampaignStop, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        budget.publish(status='stopped_resource_budget_or_interruption', reason=str(error), artifacts_retained=True)
        return 2
    except Exception as error:
        budget.publish(status='failed', reason='%s: %s' % (type(error).__name__, error), artifacts_retained=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/teacher_revision_gpu_20261004/adast_10fold_local')
    parser.add_argument('--shhs-root', type=Path, default=Path('E:/research/Dataset/SHHS_v1'))
    parser.add_argument('--max-seconds', type=float, default=18000)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.output, args.shhs_root = args.output.resolve(), args.shhs_root.resolve()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.verify_only:
        verify(args)
    else:
        sys.exit(run(args))
