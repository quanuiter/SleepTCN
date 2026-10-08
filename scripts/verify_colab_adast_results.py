"""Verify downloaded twenty-checkpoint ADAST campaign, without retraining."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/colab_adast_20261004'
loader = importlib.util.spec_from_file_location('adast_archive_helpers', ROOT / 'scripts/run_colab_adast_training.py')
helper = importlib.util.module_from_spec(loader)
loader.loader.exec_module(helper)


def reconstruct_initial(models_module, model_cfg, seed, expected_digest):
    """Reproduce initial bytes, not a tolerance-based substitute for the hash.

    Linux float32 uniform initialization can use a fused multiply-add where
    Windows uses two roundings. Replaying the same random uniforms with one
    final float32 rounding reproduces the observed Colab initialization exactly.
    This is verification only: never changes a training or inference checkpoint.
    """
    native = helper.build_models(models_module, model_cfg, seed, 'cpu')
    native_digest = helper.state_digest(native)
    if native_digest == expected_digest:
        return native, {'method': 'native_exact', 'native_initial_sha256': native_digest}
    original = torch.Tensor.uniform_

    def fused_uniform(tensor, a=0, b=1, *, generator=None):
        if tensor.device.type != 'cpu' or tensor.dtype != torch.float32:
            raise ValueError('Uniform reconstruction supports CPU float32 only')
        original(tensor, 0, 1, generator=generator)
        lower = np.float32(a)
        width = np.float32(np.float32(b) - lower)
        values = (tensor.numpy().astype(np.float64) * float(width) + float(lower)).astype(np.float32)
        tensor.copy_(torch.from_numpy(values))
        return tensor

    try:
        torch.Tensor.uniform_ = fused_uniform
        rebuilt = helper.build_models(models_module, model_cfg, seed, 'cpu')
    finally:
        torch.Tensor.uniform_ = original
    rebuilt_digest = helper.state_digest(rebuilt)
    if rebuilt_digest != expected_digest:
        raise ValueError('Neither native nor fused-rounding reconstruction matches initial SHA256')
    max_difference = max(float((value - rebuilt[name].state_dict()[key]).abs().max())
                         for name, model in native.items() for key, value in model.state_dict().items())
    return rebuilt, {'method': 'fused_uniform_float32_exact', 'native_initial_sha256': native_digest,
                     'reconstructed_initial_sha256': rebuilt_digest,
                     'native_vs_reconstructed_max_abs': max_difference,
                     'exact_sha256_required': True}


def verify(args):
    if helper.digest(args.archive) != args.expected_sha256:
        raise ValueError('Downloaded archive SHA differs from Colab printed SHA')
    bundle = json.loads((BASE / 'bundle_verification.json').read_bytes())
    if helper.digest(BASE / 'SleepTCN_ADAST_10Fold_20261004.zip') != bundle['archive_sha256']:
        raise ValueError('Original input bundle changed')
    work = BASE / 'payload'
    manifest = json.loads((work / 'manifest.json').read_bytes())
    if helper.digest(work / 'manifest.json') != bundle['manifest_sha256']:
        raise ValueError('Input manifest changed')
    for name, sha in manifest['files'].items():
        if helper.digest(work / name) != sha:
            raise ValueError('Original input payload changed')
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        hashes = json.loads(z.read('export_manifest.json'))
        if len(names) != len(set(names)) or set(names) != set(hashes) | {'export_manifest.json'} or z.testzip() is not None:
            raise ValueError('Duplicate entries, allowlist or CRC differs')
        import hashlib
        output = BASE / 'verified_gpu_results'
        output.mkdir(exist_ok=True)
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
                raise ValueError('Unsafe export entry')
            if z.getinfo(name).file_size > 50_000_000:
                raise ValueError('Unexpectedly large result entry')
            data = z.read(name)
            if name != 'export_manifest.json' and hashlib.sha256(data).hexdigest() != hashes[name]:
                raise ValueError('Result payload hash differs')
            path = output / relative
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('Existing verified result differs')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    spec = json.loads((output / 'execution_specification.json').read_bytes())
    cfg = json.loads((work / 'protocol.json').read_bytes())
    if (spec['protocol'] != cfg or spec['target_test_access'] or spec['target_true_label_access']
            or spec['bundle_manifest_sha256'] != bundle['manifest_sha256']
            or spec['runner_sha256'] != helper.digest(work / 'run_colab_adast_training.py')
            or spec['upstream_files_sha256'] != manifest['upstream_files_sha256']):
        raise ValueError('Executed identity or scope differs')
    result = json.loads((output / 'aggregate_results.json').read_bytes())
    internal = json.loads((output / 'verification.json').read_bytes())
    progress = json.loads((output / 'progress.json').read_bytes())
    if (result['status'] != 'complete_ten_fold_matched_ADAST_source_training'
            or result['selected_checkpoints'] != 20 or result['target_test_access'] or result['target_true_label_access']
            or internal['status'] != 'passed' or not internal['all_fixed_budget_updates_verified']
            or internal['aggregate_results_sha256'] != helper.digest(output / 'aggregate_results.json')
            or result['execution_specification_sha256'] != helper.digest(output / 'execution_specification.json')
            or progress['status'] != result['status'] or progress['completed_pairs'] != 10
            or [p['fold'] for p in result['folds']] != list(range(10))):
        raise ValueError('Require completed, matching ten-fold campaign')
    torch.set_num_threads(1)
    models_module = helper.load_module('adast_original_model_verification', work / 'upstream/models.py')
    model_cfg = helper.load_module('adast_original_config_verification', work / 'upstream/configs.py').Config()
    initial_hashes = {selection['initial_state_sha256'] for pair in result['folds']
                      for selection in pair['arms'].values()}
    if len(initial_hashes) != 1 or any(set(pair['arms']) != {'source_only', 'adast'} for pair in result['folds']):
        raise ValueError('Require twenty selections with one matched initial state')
    initial, initial_audit = reconstruct_initial(models_module, model_cfg, cfg['seed'], initial_hashes.pop())
    initial_digest = helper.state_digest(initial)
    rows = []
    for pair in result['folds']:
        fold = pair['fold']
        roles_path = work / f'data/fold_{fold:02d}_roles.npz'
        with np.load(roles_path, allow_pickle=False) as roles:
            n = len(roles['train'])
        selected_file = json.loads((output / f'fold_{fold:02d}/both_arms_selected.json').read_bytes())
        if selected_file != pair['arms']:
            raise ValueError('Frozen pair differs')
        for arm, selected in pair['arms'].items():
            folder = output / f'fold_{fold:02d}' / arm
            checkpoint_path = folder / 'final.pt'
            if (helper.digest(checkpoint_path) != selected['checkpoint_sha256']
                    or helper.digest(folder / 'latest.pt') != selected['checkpoint_sha256']
                    or json.loads((folder / 'selection.json').read_bytes()) != selected
                    or selected['initial_state_sha256'] != initial_digest
                    or selected['updates'] != 1140 or selected['epochs_completed'] != 30
                    or selected['selection'] != cfg['selection'] or len(selected['history']) != 30):
                raise ValueError('Final selection/update budget differs')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            identity = {'fold': fold, 'arm': arm,
                        'execution_specification_sha256': helper.digest(output / 'execution_specification.json'),
                        'source_roles_sha256': helper.digest(roles_path)}
            if (checkpoint['identity'] != identity or json.loads((folder / 'identity.json').read_bytes()) != identity
                    or checkpoint['history'] != selected['history'] or checkpoint['epochs_completed'] != 30
                    or checkpoint['initial_state_sha256'] != initial_digest):
                raise ValueError('Checkpoint identity/history differs')
            for epoch, row in enumerate(selected['history']):
                # Colab/Linux np.int_ is 64-bit; Windows NumPy 1.x defaults to
                # 32-bit. Replay identical values with the original byte dtype.
                order = np.random.default_rng(cfg['seed'] + epoch).permutation(n)[:4864].astype(np.int64)
                expected_lr = cfg['optimizer']['lr'] * (.1 if epoch >= 9 else 1)
                if (row['source_order_sha256'] != hashlib.sha256(order.tobytes()).hexdigest()
                        or row['updates'] != 38 or row['round'] != epoch // 15 or row['epoch'] != epoch % 15 + 1
                        or not np.isfinite(row['loss_mean']) or row['optimizer_lr_after_epoch'] != expected_lr):
                    raise ValueError('Sample order, loss, schedule or learning rate differs')
            actual = helper.build_models(models_module, model_cfg, cfg['seed'], 'cpu')
            for name, model in actual.items():
                model.load_state_dict(checkpoint['models'][name], strict=True)
                if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
                    raise ValueError('Nonfinite model parameter or buffer')
                if arm == 'source_only' and name in ['discriminator', 'target_attention']:
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, initial[name].state_dict()[key], rtol=0, atol=0)
            if arm == 'adast':
                for round_index in range(2):
                    pseudo = np.load(folder / f'pseudo_round_{round_index}.npy', allow_pickle=False)
                    if pseudo.shape != (4989,) or not np.isin(pseudo, range(5)).all():
                        raise ValueError('Pseudo-label schema differs')
            rows.append({'fold': fold, 'arm': arm, 'training_seconds': selected['training_seconds'], 'updates': selected['updates']})
        if [r['source_order_sha256'] for r in pair['arms']['adast']['history']] != [
                r['source_order_sha256'] for r in pair['arms']['source_only']['history']]:
            raise ValueError('Paired source exposure differs')
    verification = {'status': 'passed', 'selected_checkpoints_verified': 20,
                    'same_initialization_and_source_sample_orders_verified': True,
                    'fixed_update_and_lr_schedule_verified': True,
                    'source_only_target_modules_unchanged': True,
                    'pseudo_labels_audited': 20, 'target_test_access': False,
                    'initialization_reconstruction': initial_audit,
                    'source_order_hash_dtype': 'int64_matches_Colab_Linux',
                    'archive_sha256': args.expected_sha256,
                    'aggregate_results_sha256': helper.digest(output / 'aggregate_results.json'), 'rows': rows,
                    'verifier_sha256': helper.digest(Path(__file__))}
    helper.write_once(output / 'independent_verification.json', verification)
    archive_copy = BASE / args.archive.name
    if not archive_copy.exists():
        shutil.copyfile(args.archive, archive_copy)
    elif helper.digest(archive_copy) != args.expected_sha256:
        raise ValueError('Archive copy differs')
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    verify(parser.parse_args())
