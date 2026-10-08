"""Ten matched CUDA ADAST/source-only pairs; no target-test data or labels."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch
from torch import nn


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError('Frozen JSON differs')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding='utf-8')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_digest(models):
    h = hashlib.sha256()
    for name, model in models.items():
        h.update(name.encode())
        for key, value in model.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_models(module, cfg, seed, device):
    torch.manual_seed(seed)
    models = {'encoder': module.cnn_feature_extractor(cfg.base_model),
              'head1': module.Classifier(cfg.base_model),
              'head2': module.Classifier(cfg.base_model),
              'discriminator': module.Discriminator(cfg.base_model),
              'source_attention': module.Self_Attn(cfg.base_model.final_out_channels),
              'target_attention': module.Self_Attn(cfg.base_model.final_out_channels)}
    return {name: model.to(device) for name, model in models.items()}


def logits(models, x, domain):
    f = models[domain + '_attention'](models['encoder'](x))
    return f, models['head1'](f), models['head2'](f)


@torch.inference_mode()
def predict(models, x, domain, device, guard=lambda: None, pseudo=False, batch_size=128):
    for model in models.values():
        model.eval()
    output = []
    for start in range(0, len(x), batch_size):
        guard()
        batch = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32).copy()).unsqueeze(1).to(device)
        _, first, second = logits(models, batch, domain)
        value = (first + second) / 2 if pseudo else torch.maximum(first, second)
        output.append((value.argmax(1) if pseudo else value).cpu().numpy())
    return np.concatenate(output)


def update(models, optimizer, disc_optimizer, utils, sx, sy, tx, pseudo, round_index, cfg, arm):
    for model in models.values():
        model.train()
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    sf, sl1, sl2 = logits(models, sx, 'source')
    source_loss = ce(sl1, sy) + ce(sl2, sy)
    loss = (cfg['source_loss_weights_by_round'][round_index] * source_loss
            + cfg['similarity_weight'] * utils.calc_similiar_penalty(models['head1'], models['head2']))
    if arm == 'adast':
        tf, tl1, tl2 = logits(models, tx, 'target')
        discriminator = models['discriminator']
        for param in discriminator.parameters():
            param.requires_grad = True
        disc_output = discriminator(torch.cat([sf, tf]).detach()).squeeze(-1)
        labels = torch.cat([torch.ones(len(sf), device=sx.device), torch.zeros(len(tf), device=sx.device)])
        disc_loss = bce(disc_output, labels)
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        for param in discriminator.parameters():
            param.requires_grad = False
        fake_output = torch.cat([discriminator(tf).squeeze(-1), discriminator(sf).squeeze(-1)])
        fake_labels = torch.cat([torch.ones(len(tf), device=sx.device), torch.zeros(len(sf), device=sx.device)])
        loss = (loss + cfg['adversarial_weight'] * bce(fake_output, fake_labels)
                + cfg['target_loss_weights_by_round'][round_index] * (ce(tl1, pseudo) + ce(tl2, pseudo)))
    optimizer.zero_grad()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                     for model in models.values() for p in model.parameters()):
        raise ValueError('Nonfinite loss/gradient')
    optimizer.step()
    return float(loss.detach())


class BudgetStop(RuntimeError):
    pass


class Guard:
    def __init__(self, output, seconds):
        self.started = time.monotonic()
        self.seconds = seconds
        self.output = Path(output)
        self.state = {'status': 'running', 'phase': 'training', 'completed_pairs': 0}

    def publish(self, **values):
        self.state.update(values)
        self.state['elapsed_attempt_seconds'] = time.monotonic() - self.started
        self.state['remaining_attempt_seconds'] = max(0, self.seconds - self.state['elapsed_attempt_seconds'])
        temporary = self.output / 'progress.tmp'
        temporary.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
        temporary.replace(self.output / 'progress.json')

    def __call__(self):
        if time.monotonic() - self.started >= self.seconds:
            raise BudgetStop('Five-hour attempt budget reached; epoch-boundary checkpoints retained')


def train_arm(arm, models, utils, x, y, source_indices, target, cfg, folder, identity, guard, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_once(folder / 'identity.json', identity)
    initial = state_digest(models)
    selection_path, final_path = folder / 'selection.json', folder / 'final.pt'
    if selection_path.exists():
        selected = json.loads(selection_path.read_bytes())
        if digest(final_path) != selected['checkpoint_sha256'] or initial != selected['initial_state_sha256']:
            raise ValueError('Completed checkpoint changed')
        payload = torch.load(final_path, map_location=device, weights_only=True)
        for name, model in models.items():
            model.load_state_dict(payload['models'][name])
        return selected
    names = ['encoder', 'head1', 'head2', 'source_attention', 'target_attention']
    opt = cfg['optimizer']
    optimizer = torch.optim.Adam([p for name in names for p in models[name].parameters()],
                                 lr=opt['lr'], betas=tuple(opt['betas']), weight_decay=opt['weight_decay'])
    disc_optimizer = torch.optim.Adam(models['discriminator'].parameters(), lr=opt['lr'],
                                      betas=tuple(opt['betas']), weight_decay=opt['weight_decay'])
    done, elapsed, history = 0, 0., []
    latest = folder / 'latest.pt'
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
        if checkpoint['identity'] != identity or checkpoint['initial_state_sha256'] != initial:
            raise ValueError('Resume identity differs')
        for name, model in models.items():
            model.load_state_dict(checkpoint['models'][name])
        optimizer.load_state_dict(checkpoint['optimizer'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng_state']])
        done, elapsed, history = checkpoint['epochs_completed'], checkpoint['elapsed_seconds'], checkpoint['history']
    for round_index in range(cfg['rounds']):
        pseudo_path = folder / f'pseudo_round_{round_index}.npy'
        if arm == 'adast' and done < (round_index + 1) * cfg['epochs_per_round']:
            if pseudo_path.exists():
                pseudo = np.load(pseudo_path, allow_pickle=False)
            else:
                if done != round_index * cfg['epochs_per_round']:
                    raise ValueError('Missing frozen pseudo-labels')
                pseudo = predict(models, target, 'target', device, guard, pseudo=True)
                with pseudo_path.open('xb') as stream:
                    np.save(stream, pseudo)
            if pseudo.shape != (len(target),) or not np.isin(pseudo, range(5)).all():
                raise ValueError('Invalid pseudo-labels')
        for epoch in range(cfg['epochs_per_round']):
            global_epoch = round_index * cfg['epochs_per_round'] + epoch
            if global_epoch < done:
                continue
            guard()
            guard.publish(fold=identity['fold'], arm=arm, round=round_index, epochs_completed=done,
                          updates_completed=done * cfg['steps_per_epoch'])
            tick = time.perf_counter()
            order = np.random.default_rng(cfg['seed'] + global_epoch).permutation(len(source_indices))[:cfg['steps_per_epoch'] * cfg['batch_size']]
            target_order = np.random.default_rng(cfg['seed'] + 10000 + global_epoch).permutation(len(target))[:len(order)]
            losses = []
            for start in range(0, len(order), cfg['batch_size']):
                guard()
                si = source_indices[order[start:start + cfg['batch_size']]]
                ti = target_order[start:start + cfg['batch_size']]
                sx = torch.from_numpy(np.asarray(x[si], dtype=np.float32).copy()).unsqueeze(1).to(device)
                sy = torch.from_numpy(y[si].copy()).to(device)
                tx = torch.from_numpy(target[ti].copy()).unsqueeze(1).to(device) if arm == 'adast' else None
                py = torch.from_numpy(pseudo[ti].astype(np.int64)).to(device) if arm == 'adast' else None
                losses.append(update(models, optimizer, disc_optimizer, utils, sx, sy, tx, py, round_index, cfg, arm))
            if round_index == 0 and (epoch + 1) % 10 == 0:
                for group in optimizer.param_groups:
                    group['lr'] *= .1
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            elapsed += seconds
            row = {'round': round_index, 'epoch': epoch + 1, 'updates': len(losses),
                   'source_order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                   'loss_mean': float(np.mean(losses)), 'seconds': seconds,
                   'optimizer_lr_after_epoch': optimizer.param_groups[0]['lr']}
            history.append(row)
            checkpoint = {'models': {name: model.state_dict() for name, model in models.items()},
                          'optimizer': optimizer.state_dict(), 'disc_optimizer': disc_optimizer.state_dict(),
                          'rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                          'epochs_completed': global_epoch + 1, 'elapsed_seconds': elapsed,
                          'history': history, 'initial_state_sha256': initial, 'identity': identity}
            torch.save(checkpoint, folder / 'latest.tmp')
            (folder / 'latest.tmp').replace(latest)
            done = global_epoch + 1
            guard.publish(epochs_completed=done, updates_completed=done * cfg['steps_per_epoch'])
            print(f"fold {identity['fold']} {arm}: epoch {done}/30, loss={row['loss_mean']:.4f}, {seconds:.2f}s", flush=True)
    if done != cfg['rounds'] * cfg['epochs_per_round']:
        raise ValueError('Incomplete schedule')
    shutil.copyfile(latest, final_path)
    selection = {'initial_state_sha256': initial, 'checkpoint_sha256': digest(final_path),
                 'selection': cfg['selection'], 'epochs_completed': done,
                 'updates': sum(r['updates'] for r in history), 'training_seconds': elapsed, 'history': history}
    write_once(selection_path, selection)
    return selection


def export_results(output):
    paths = sorted(p for p in output.rglob('*') if p.is_file() and p.suffix in {'.pt', '.json', '.npy', '.log'})
    manifest = {p.relative_to(output).as_posix(): digest(p) for p in paths}
    archive = output.parent / 'SleepTCN_ADAST_CUDA_10Fold_Results_20261004.zip'
    temporary = archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths:
            z.write(p, arcname=p.relative_to(output).as_posix())
        z.writestr('export_manifest.json', json.dumps(manifest, indent=2))
    temporary.replace(archive)
    print('RESULT_ARCHIVE', archive, 'SHA256', digest(archive), flush=True)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not fall back to CPU training')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    work = Path(__file__).resolve().parent
    manifest_path = work / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['target_test_data_included'] or manifest['target_labels_included'] or manifest['participant_ids_included']:
        raise ValueError('Forbidden target-test data, labels or identifiers')
    for name, sha in manifest['files'].items():
        if digest(work / name) != sha:
            raise ValueError('Bundle payload changed')
    cfg = json.loads((work / 'protocol.json').read_bytes())
    x, y, target = [np.load(work / f'data/{n}.npy', mmap_mode='r', allow_pickle=False)
                    for n in ['source_x', 'source_y', 'adaptation_x']]
    if x.shape != (195469, 3000) or y.shape != (195469,) or target.shape != (4989, 3000):
        raise ValueError('Data support differs')
    module = load_module('adast_pinned_models', work / 'upstream/models.py')
    upstream_cfg = load_module('adast_pinned_config', work / 'upstream/configs.py').Config()
    utils = load_module('adast_pinned_utils', work / 'upstream/utils.py')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = {'protocol': cfg, 'bundle_manifest_sha256': digest(manifest_path),
            'runner_sha256': digest(Path(__file__)), 'upstream_files_sha256': manifest['upstream_files_sha256'],
            'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'training_backend': 'cuda_float32_deterministic_no_tf32', 'cpu_threads': 2,
            'target_test_access': False, 'target_true_label_access': False,
            'target_adaptation_subjects': 5, 'max_attempt_seconds': args.max_seconds}
    write_once(output / 'execution_specification.json', spec)
    guard = Guard(output, args.max_seconds)
    selections = []
    try:
        for fold in range(10):
            roles = np.load(work / f'data/fold_{fold:02d}_roles.npz', allow_pickle=False)
            source_indices = roles['train']
            if any(np.intersect1d(roles[a], roles[b]).size for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
                raise ValueError('Source role overlap')
            guard.publish(completed_pairs=fold)
            pair = {}
            for arm in cfg['arms']:
                models = build_models(module, upstream_cfg, cfg['seed'], 'cuda')
                identity = {'fold': fold, 'arm': arm, 'execution_specification_sha256': digest(output / 'execution_specification.json'),
                            'source_roles_sha256': digest(work / f'data/fold_{fold:02d}_roles.npz')}
                pair[arm] = train_arm(arm, models, utils, x, y, source_indices, target, cfg,
                                      output / f'fold_{fold:02d}' / arm, identity, guard, 'cuda')
                del models
                torch.cuda.empty_cache()
            if pair['source_only']['initial_state_sha256'] != pair['adast']['initial_state_sha256'] or [r['source_order_sha256'] for r in pair['source_only']['history']] != [r['source_order_sha256'] for r in pair['adast']['history']]:
                raise ValueError('Matched initialization/sample order differs')
            write_once(output / f'fold_{fold:02d}/both_arms_selected.json', pair)
            selections.append({'fold': fold, 'arms': pair})
            print(f'FOLD {fold} MATCHED ADAST PAIR COMPLETE', flush=True)
        result = {'status': 'complete_ten_fold_matched_ADAST_source_training', 'selected_checkpoints': 20,
                  'target_test_access': False, 'target_true_label_access': False,
                  'execution_specification_sha256': digest(output / 'execution_specification.json'), 'folds': selections}
        write_once(output / 'aggregate_results.json', result)
        write_once(output / 'verification.json', {'status': 'passed', 'matched_pairs': 10,
                   'all_fixed_budget_updates_verified': all(s['updates'] == 1140 for p in selections for s in p['arms'].values()),
                   'aggregate_results_sha256': digest(output / 'aggregate_results.json')})
        guard.publish(status='complete_ten_fold_matched_ADAST_source_training', phase='complete', completed_pairs=10)
    except BudgetStop as error:
        guard.publish(status='stopped_resource_budget', reason=str(error), checkpoints_retained=True)
        raise
    except Exception as error:
        guard.publish(status='failed', reason=f'{type(error).__name__}: {error}', checkpoints_retained=True)
        raise
    finally:
        export_results(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=18000)
    main(parser.parse_args())
