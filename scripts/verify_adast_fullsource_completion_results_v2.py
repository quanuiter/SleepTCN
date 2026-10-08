"""Verify eight full-source pairs and replay best/final validation on CPU; never train."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile

import numpy as np
import torch
import verify_adast_development_results as shared
from adast_cpu_inference_checks import predict_both_attention

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'runs/adast_fullsource_completion_20261006'
CANONICAL = ROOT / 'runs/adast_fullsource_retry_20261007'
RECORD = CANONICAL / 'verification_v2'
STATUS = 'complete_eight_new_fullsource_pairs'
FOLDS = list(range(2, 10))
ARMS = ['source_only_full_source', 'adast_full_source']
INPUT_SHA = '03182361c85375593ada052cab6f55380313e21337144e855ab886e25ebd58f9'
MANIFEST_SHA = '3b4fe50bf8823df7e7915e531a86dbc1f25616a225524b71e24589ca07fd3ae8'
BASE_SHA = 'fd6071fa77fc463e25890f34350ad1a22fc946bd9e37a95fe2d801a984a3b204'
WRAPPER_SHA = '8c7e15f8ecd92ebfef3f989731bffcbdc07470f736e5abade7217a8897b33f93'
SIZES = [(2, 152192, 19396), (3, 156616, 19457), (4, 158698, 17314),
         (5, 156997, 21158), (6, 153349, 20962), (7, 155344, 19163),
         (8, 160437, 15869), (9, 160094, 19506)]


def validate_protocol(cfg):
    expected = {
        'seed': 123, 'completed_folds_reused': [0, 1], 'folds_to_train': FOLDS,
        'arms': ARMS, 'rounds': 2, 'epochs_per_round': 15, 'batch_size': 128,
        'limited_steps_per_epoch': 38, 'source_loss_weights_by_round': [1, .1],
        'target_loss_weights_by_round': [0, .01], 'adversarial_weight': 1,
        'similarity_weight': .001, 'adaptation_subjects': 5, 'adaptation_epochs': 4989,
        'optimizer': {'name': 'Adam', 'lr': .001, 'betas': [.5, .99], 'weight_decay': .0003},
        'base_runner_sha256': BASE_SHA, 'required_torch': '2.11.0+cu130',
        'backend': 'CUDA_float32_deterministic_no_tf32', 'max_seconds': 18000,
        'cpu_training_fallback': False, 'automatic_restart': False,
        'automatic_continuation': False, 'target_test_data_upload': False,
        'target_labels_upload': False, 'checkpoint_primary': 'final_epoch30_for_both_budgets',
        'checkpoint_secondary': 'strict_maximum_source_attention_source_validation_macro_f1_first_tie',
        'no_target_score_based_checkpoint_or_configuration_choice': True,
    }
    for field, value in expected.items():
        if cfg.get(field) != value:
            raise ValueError('Full-source protocol differs: ' + field)
    expected_sizes = [{'fold': f, 'train_epochs': n, 'validation_epochs': v,
                       'updates_per_model': 30 * math.ceil(n / 128)} for f, n, v in SIZES]
    if cfg.get('remaining_fold_sizes') != expected_sizes:
        raise ValueError('Full-source fold support/update budget differs')


def validate_roles(roles, total):
    if set(roles) != {'train', 'validation', 'test'}:
        raise ValueError('Source role schema differs')
    for indices in roles.values():
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or not len(indices) or indices.min() < 0 or indices.max() >= total
                or len(np.unique(indices)) != len(indices)):
            raise ValueError('Invalid or duplicate source role index')
    joined = np.concatenate(list(roles.values()))
    if len(joined) != total or len(np.unique(joined)) != total:
        raise ValueError('Source roles overlap or fail coverage')


class RoleArray:
    """Only ordered validation indices are read during checkpoint inference."""
    def __init__(self, full, indices):
        self.full, self.indices = full, indices
        self.shape, self.dtype = (len(indices), *full.shape[1:]), full.dtype

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        return self.full[self.indices[key]]


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
            or str(path) != name or name.endswith('/')):
        raise ValueError('Unsafe archive/payload member')


def verify_archive_members(zipped, guard):
    hashes = shared.read_json(zipped, 'export_manifest.json')
    names = zipped.namelist()
    if len(names) != len(set(names)) or set(names) != set(hashes) | {'export_manifest.json'}:
        raise ValueError('Result allowlist or duplicate entries differ')
    for entry in zipped.infolist():
        guard()
        safe_name(entry.orig_filename)
        if entry.file_size > 100_000_000:
            raise ValueError('Unexpectedly large result member')
        if entry.filename != 'export_manifest.json':
            if hashlib.sha256(zipped.read(entry)).hexdigest() != hashes[entry.filename]:
                raise ValueError('Result member hash differs: ' + entry.filename)
    return hashes


def check_epoch(row, cfg, labels, epoch, adaptive, target_seen):
    source, target = shared.replay_sampling({**cfg, 'source_train_epochs': len(labels)}, epoch, True)
    if adaptive:
        target_seen[target] = True
    expected = {
        'global_epoch': epoch + 1, 'round': epoch // 15, 'epoch': epoch % 15 + 1,
        'updates': math.ceil(len(labels) / 128), 'source_presentations': len(labels),
        'source_unique_seen': len(labels), 'target_unique_seen': int(target_seen.sum()),
        'target_training_presentations': len(labels) if adaptive else 0,
        'source_order_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
        'target_order_sha256': hashlib.sha256(target.tobytes()).hexdigest() if adaptive else None,
        'source_label_counts': np.bincount(labels[source], minlength=5).tolist(),
        'optimizer_lr_before_epoch': .001 * (.1 if epoch >= 10 else 1),
        'optimizer_lr_after_epoch': .001 * (.1 if epoch >= 9 else 1),
        'loss_coefficients': {'source_ce': 1 if epoch < 15 else .1, 'similarity': .001,
                              'adversarial': 1 if adaptive else 0,
                              'target_pseudo_ce': .01 if adaptive and epoch >= 15 else 0},
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError('Epoch sampling/loss/budget differs: ' + key)
    if not np.isfinite(list(row['loss_batch_means'].values())).all():
        raise ValueError('Nonfinite training diagnostic')
    shared.verify_loss_components(row['loss_batch_means'], expected['loss_coefficients'], adaptive)
    if not np.isfinite(row['seconds']) or row['seconds'] <= 0:
        raise ValueError('Invalid measured epoch time')


def check_optimizer(payload, cfg, epoch, updates_per_epoch, adaptive):
    for key, lr in [('optimizer', .001 * (.1 if epoch >= 10 else 1)), ('disc_optimizer', .001)]:
        optimizer = payload[key]
        if len(optimizer['param_groups']) != 1:
            raise ValueError('Optimizer group count differs')
        group = optimizer['param_groups'][0]
        if (group['lr'] != lr or tuple(group['betas']) != tuple(cfg['optimizer']['betas'])
                or group['weight_decay'] != cfg['optimizer']['weight_decay']):
            raise ValueError('Checkpoint optimizer hyperparameters differ')
        state = optimizer['state']
        if key == 'disc_optimizer' and not adaptive:
            if state:
                raise ValueError('Source-only discriminator optimizer has updates')
            continue
        if not state or any(float(s['step']) != epoch * updates_per_epoch for s in state.values()):
            raise ValueError('Optimizer update count differs')
        for state_row in state.values():
            for name in ('exp_avg', 'exp_avg_sq'):
                if not torch.isfinite(state_row[name]).all():
                    raise ValueError('Nonfinite optimizer state')


def verify(args):
    started = time.monotonic()
    reference = shared.reference
    if not 0 < args.max_seconds <= 18000:
        raise ValueError('Verification limit must be at most five hours; no training')
    if reference.digest(args.archive) != args.expected_sha256:
        raise ValueError('Downloaded result archive differs from observed Colab SHA')
    progress_path = RECORD / 'independent_verification_progress.json'
    if progress_path.exists():
        raise FileExistsError('Preserve existing verification attempt; inspect before another run')
    progress = {'status': 'running', 'archive_sha256': args.expected_sha256,
                'phase': 'checking_inputs', 'models_completed': 0, 'training': False}

    def publish(**values):
        progress.update(values, elapsed_seconds=time.monotonic() - started)
        temp = progress_path.with_suffix('.tmp')
        temp.write_text(json.dumps(progress, indent=2), encoding='utf8')
        temp.replace(progress_path)

    def guard():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError('Verification limit reached; no training or automatic restart')

    RECORD.mkdir(parents=True, exist_ok=True)
    torch.set_flush_denormal(False)
    publish()
    try:
        pointer = shared.read_local_json(INPUT / 'storage.json')
        storage, work = Path(pointer['storage_root']).resolve(), Path(pointer['payload_root']).resolve()
        if work != storage / 'payload':
            raise ValueError('Unexpected input storage location')
        output = (args.output or storage / 'retry_results_20261007_v2' / 'verified_results').resolve()
        if (not output.is_relative_to(storage) or output == storage or output.exists()
                or (CANONICAL / 'verified_results_pointer.json').exists()):
            raise ValueError('Require new private output; preserve existing result/proof')
        bundle = shared.read_local_json(INPUT / 'bundle_verification.json')
        if (bundle['status'] != 'passed' or bundle['archive_sha256'] != INPUT_SHA
                or bundle['manifest_sha256'] != MANIFEST_SHA
                or reference.digest(Path(bundle['archive_path'])) != INPUT_SHA
                or reference.digest(work / 'manifest.json') != MANIFEST_SHA):
            raise ValueError('Frozen input archive/manifest differs')
        manifest = shared.read_local_json(work / 'manifest.json')
        if (any(manifest[k] for k in ('target_test_data_included', 'target_labels_included',
                                    'participant_ids_included', 'raw_edf_included', 'reused_models_in_this_payload'))
                or not manifest['source_all_records_included'] or not manifest['source_role_training_restricted']
                or manifest['folds_to_train'] != FOLDS):
            raise ValueError('Input data-role scope differs')
        for name, digest in manifest['files'].items():
            guard()
            safe_name(name)
            if reference.digest(work / name) != digest:
                raise ValueError('Frozen payload differs: ' + name)
        if (reference.digest(ROOT / 'src/sleeptcn/metrics.py') != manifest['files']['development_metrics.py']
                or manifest['files']['run_adast_development_cuda.py'] != BASE_SHA
                or manifest['files']['run_adast_fullsource_completion_cuda.py'] != WRAPPER_SHA
                or reference.digest(ROOT / 'configs/adast_fullsource_completion_v1_20261006.json')
                != manifest['files']['protocol.json']):
            raise ValueError('Frozen protocol/core/metric identity differs')
        cfg = shared.read_local_json(work / 'protocol.json')
        validate_protocol(cfg)
        x = np.load(work / 'data/source_x.npy', mmap_mode='r', allow_pickle=False)
        y = np.load(work / 'data/source_y.npy', mmap_mode='r', allow_pickle=False)
        target = np.load(work / 'data/adaptation_x.npy', mmap_mode='r', allow_pickle=False)
        if (x.shape != (195469, 3000) or x.dtype != np.float32 or y.shape != (195469,)
                or y.dtype != np.int64 or not np.isin(y, range(5)).all()
                or target.shape != (4989, 3000) or target.dtype != np.float32
                or not np.isfinite(target).all()):
            raise ValueError('Input signal/label schema differs')
        for start in range(0, len(x), 2048):
            guard()
            if not np.isfinite(x[start:start + 2048]).all():
                raise ValueError('Nonfinite source signal')
        roles_by_fold = {}
        for fold, train_n, val_n in SIZES:
            with np.load(work / f'data/fold_{fold:02d}_roles.npz', allow_pickle=False) as values:
                roles = {name: values[name].copy() for name in values.files}
            validate_roles(roles, len(x))
            if len(roles['train']) != train_n or len(roles['validation']) != val_n:
                raise ValueError('Fold role support differs')
            roles_by_fold[fold] = roles
        torch.set_num_threads(4)
        upstream = reference.load_module('fullsource_verified_models', work / 'upstream/models.py')
        model_cfg = reference.load_module('fullsource_verified_cfg', work / 'upstream/configs.py').Config()
        publish(phase='checking_result_archive')
        with zipfile.ZipFile(args.archive) as z:
            hashes = verify_archive_members(z, guard)
            result, spec = shared.read_json(z, 'aggregate_results.json'), shared.read_json(z, 'execution_specification.json')
            internal, terminal = shared.read_json(z, 'verification.json'), shared.read_json(z, 'progress.json')
            aggregate_sha = hashlib.sha256(z.read('aggregate_results.json')).hexdigest()
            spec_sha = hashlib.sha256(z.read('execution_specification.json')).hexdigest()
            if (result['status'] != STATUS or terminal['status'] != STATUS
                    or terminal['completed_models'] != 16 or terminal['completed_pairs'] != 8
                    or result['new_models_completed'] != 16 or result['reused_folds_separate'] != [0, 1]
                    or result['source_outer_test_access_during_training'] or result['target_test_access']
                    or [r['fold'] for r in result['folds']] != FOLDS
                    or internal != {'status': 'passed', 'complete_matched_pairs': 8,
                                    'aggregate_results_sha256': aggregate_sha}):
                raise ValueError('Require complete eight matched pairs and matching aggregate proof')
            if (spec['protocol'] != cfg or spec['manifest_sha256'] != MANIFEST_SHA
                    or spec['runner_sha256'] != WRAPPER_SHA or spec['base_runner_sha256'] != BASE_SHA
                    or spec['torch'] != cfg['required_torch'] or spec['backend'] != cfg['backend']
                    or any(spec[k] for k in ('source_outer_test_access_during_training', 'target_test_access',
                                            'target_true_label_access', 'reused_models_present_in_this_archive'))
                    or not 0 < spec['max_seconds'] <= 18000
                    or not 0 < result['elapsed_seconds'] <= spec['max_seconds']):
                raise ValueError('Frozen execution identity/budget differs')
            expected_initial = result['folds'][0]['arms'][0]['initial_state_sha256']
            initial, initial_audit = shared.initial_helpers.reconstruct_initial(upstream, model_cfg, 123, expected_initial)
            rows = []
            for fold_row in result['folds']:
                fold, arms = fold_row['fold'], fold_row['arms']
                if [a['arm'] for a in arms] != ARMS or len(arms) != 2:
                    raise ValueError('Pair arm schema differs')
                roles = roles_by_fold[fold]
                labels, vy = y[roles['train']], y[roles['validation']]
                vx = np.ascontiguousarray(x[roles['validation']])
                steps = math.ceil(len(labels) / 128)
                if shared.read_json(z, f'fold_{fold:02d}/pair_verification.json') != {
                        'status': 'passed', 'matched_initialization_and_source_orders': True}:
                    raise ValueError('Internal pair proof differs')
                for arm in arms:
                    key, adaptive = arm['arm'], arm['arm'].startswith('adast_')
                    prefix = f'fold_{fold:02d}/{key}'
                    identity = {'arm': key, 'fold': fold, 'execution_specification_sha256': spec_sha}
                    history = arm['history']
                    publish(phase='checking_model_diagnostics', fold=fold, arm=key)
                    if (arm['initial_state_sha256'] != expected_initial or arm['epochs_completed'] != 30
                            or len(history) != 30 or shared.read_json(z, prefix + '/selection.json') != arm
                            or shared.read_json(z, prefix + '/identity.json') != {
                                'identity': identity, 'initial_state_sha256': expected_initial}):
                        raise ValueError('Arm identity/history differs')
                    init = torch.load(io.BytesIO(z.read(prefix + '/initial.pt')), map_location='cpu', weights_only=True)
                    if init['identity'] != identity or init['initial_state_sha256'] != expected_initial:
                        raise ValueError('Saved initialization identity differs')
                    for name, model in initial.items():
                        for k, v in model.state_dict().items():
                            if not torch.equal(v, init['models'][name][k]):
                                raise ValueError('Saved initialization does not replay exactly')
                    target_seen = np.zeros(4989, dtype=bool)
                    for epoch, row in enumerate(history):
                        guard()
                        check_epoch(row, cfg, labels, epoch, adaptive, target_seen)
                        if adaptive:
                            pseudo = np.load(io.BytesIO(z.read(prefix + f'/pseudo_round_{epoch // 15}.npy')), allow_pickle=False)
                            if (pseudo.shape != (4989,) or pseudo.dtype != np.int64 or not np.isin(pseudo, range(5)).all()
                                    or row['pseudo_label_counts_at_round_start'] != np.bincount(pseudo, minlength=5).tolist()):
                                raise ValueError('Pseudo-label schema/support differs')
                        elif row['pseudo_label_counts_at_round_start'] is not None:
                            raise ValueError('Source-only has target training diagnostics')
                        diagnostic = prefix + f'/validation_epoch_{epoch + 1:03d}.npz'
                        if hashes[diagnostic] != row['validation_logits_sha256']:
                            raise ValueError('Validation logits identity differs')
                        shared.verify_logits(z, diagnostic, vy, row['validation'])
                    shared.verify_logits(z, prefix + '/validation_epoch_000.npz', vy,
                                         shared.read_json(z, prefix + '/validation_before_training.json'))
                    best_epoch = int(np.argmax([h['validation']['source']['macro_f1'] for h in history])) + 1
                    if (arm['best_epoch'] != best_epoch or arm['total_updates'] != 30 * steps
                            or arm['best_source_validation'] != history[best_epoch - 1]['validation']
                            or arm['final_source_validation'] != history[-1]['validation']
                            or hashes[prefix + '/latest.pt'] != arm['final_checkpoint_sha256']
                            or arm['training_and_validation_seconds'] != sum(h['seconds'] for h in history)):
                        raise ValueError('Checkpoint selection/final/time differs')
                    replays, checkpoints = {}, {}
                    for selection, epoch in [('best', best_epoch), ('final', 30)]:
                        guard()
                        publish(phase='CPU_validation_inference', selection=selection)
                        data = z.read(prefix + '/' + selection + '.pt')
                        if hashlib.sha256(data).hexdigest() != arm[selection + '_checkpoint_sha256']:
                            raise ValueError('Checkpoint hash differs')
                        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
                        checkpoints[selection] = payload
                        if (payload['identity'] != identity or payload['initial_state_sha256'] != expected_initial
                                or payload['epochs_completed'] != epoch or payload['history'] != history[:epoch]
                                or not payload['cuda_rng_state'] or payload['rng_state'].dtype != torch.uint8
                                or (selection == 'best' and payload['best_epoch'] != epoch)):
                            raise ValueError('Checkpoint identity/history/RNG differs')
                        prefix_best = int(np.argmax([h['validation']['source']['macro_f1'] for h in history[:epoch]])) + 1
                        if payload['best_epoch'] != prefix_best or payload['best_score'] != history[prefix_best - 1]['validation']['source']['macro_f1']:
                            raise ValueError('Checkpoint retained selection differs')
                        check_optimizer(payload, cfg, epoch, steps, adaptive)
                        models = reference.build_models(upstream, model_cfg, 123, 'cpu')
                        for name, model in models.items():
                            model.load_state_dict(payload['models'][name], strict=True)
                        state_before = reference.state_digest(models)
                        if not torch.set_flush_denormal(True):
                            raise RuntimeError('CPU does not support profiled flush-denormal inference')
                        try:
                            outputs = predict_both_attention(models, vx, guard, batch_size=128)
                        finally:
                            torch.set_flush_denormal(False)
                        if reference.state_digest(models) != state_before:
                            raise ValueError('Validation inference changed checkpoint model state')
                        replay = {}
                        for domain in ('source', 'target'):
                            logits = outputs[domain]
                            replay[domain] = shared.metrics_from_confusion(shared.confusion_matrix_5(vy, logits.argmax(1)))
                            if replay[domain] != history[epoch - 1]['validation'][domain]:
                                raise ValueError('CPU checkpoint confusion/metrics differ; inspect numerical disagreement')
                        replays[selection] = replay
                        if not adaptive:
                            for name in ('discriminator', 'target_attention'):
                                if any(not torch.equal(v, init['models'][name][k]) for k, v in payload['models'][name].items()):
                                    raise ValueError('Source-only unused module changed')
                        del models, logits
                    for name, state in checkpoints['best']['models'].items():
                        if any(not torch.equal(v, checkpoints['final']['best_models'][name][k]) for k, v in state.items()):
                            raise ValueError('Retained best state differs')
                    for kind, expected in [('source', np.ones(len(labels), dtype=bool)), ('target', target_seen)]:
                        packed = np.frombuffer(bytes.fromhex(checkpoints['final'][kind + '_seen_packed_hex']), dtype=np.uint8)
                        np.testing.assert_array_equal(np.unpackbits(packed)[:len(expected)], expected)
                    rows.append({'fold': fold, 'arm': key, 'best_epoch': best_epoch, 'total_updates': 30 * steps,
                                 'CPU_checkpoint_validation_replays': replays})
                    reference.write_once(RECORD / f'model_fold_{fold:02d}_{key}_verified.json', rows[-1])
                    publish(models_completed=len(rows), phase='model_verified')
                    del checkpoints, payload, init
                if [h['source_order_sha256'] for h in arms[0]['history']] != [h['source_order_sha256'] for h in arms[1]['history']]:
                    raise ValueError('Matched source orders differ')
            guard()
            if len(rows) != 16 or shutil.disk_usage(storage).free < sum(e.file_size for e in z.infolist()) + 100_000_000:
                raise ValueError('Model count or extraction disk space insufficient')
            publish(phase='extracting_verified_results')
            output.mkdir(parents=True)
            for name in z.namelist():
                guard()
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(z.read(name))
        proof = {'status': 'passed', 'archive_sha256': args.expected_sha256,
                 'aggregate_results_sha256': aggregate_sha, 'initialization_replay': initial_audit,
                 'all_496_validation_diagnostics_recomputed': True,
                 'all_32_best_final_checkpoints_CPU_validation_replayed': True,
                 'sampling_coverage_and_budget_replayed': True, 'optimizer_hyperparameters_and_steps_checked': True,
                 'all_16_initializations_replayed_and_paired_source_orders_matched': True,
                 'source_outer_test_access_during_training': False, 'target_test_access': False,
                 'reuse_models_not_in_this_archive': True, 'arms': rows,
                 'elapsed_seconds': time.monotonic() - started,
                 'CPU_inference': {'threads': 4, 'batch_size': 128, 'flush_denormal': True,
                                   'encoder_shared_across_attention_paths': True, 'all_model_states_unchanged': True},
                 'verifier_sha256': reference.digest(Path(__file__)),
                 'inference_helper_sha256': reference.digest(ROOT / 'scripts/adast_cpu_inference_checks.py'),
                 'previous_timeout_attempt_preserved': True}
        reference.write_once(output / 'independent_verification.json', proof)
        reference.write_once(CANONICAL / 'verified_results_pointer.json', {
            'status': 'passed', 'results_root': output.as_posix(), 'archive_sha256': args.expected_sha256,
            'aggregate_results_sha256': aggregate_sha,
            'independent_verification_sha256': reference.digest(output / 'independent_verification.json')})
        publish(status='passed', phase='complete', results_root=output.as_posix())
        print(json.dumps({'status': 'passed', 'models': 16, 'aggregate_results_sha256': aggregate_sha,
                          'elapsed_seconds': proof['elapsed_seconds']}, indent=2), flush=True)
    except BaseException as error:
        publish(status='failed', reason=str(error), no_training_performed=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=float, default=18000)
    verify(parser.parse_args())
