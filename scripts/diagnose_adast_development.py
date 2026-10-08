"""Read-only checkpoint diagnostics on source validation and unlabelled adaptation.

Never loads SHHS test signals/labels, never updates model parameters, and never
selects checkpoints or alternative head rules from target-test performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import write_once_json
from sleeptcn.revision_weighted_campaign import atomic_json
from run_colab_adast_training import build_models, load_module

BASE = ROOT / 'runs/colab_adast_20261004'
WORK = BASE / 'payload'
TRAIN = BASE / 'verified_gpu_results'
PROTOCOL = ROOT / 'configs/teacher_revision_development_v2.json'
VIEWS = ('source_only_source_attention', 'adast_source_attention', 'adast_target_attention')
RULES = ('head1', 'head2', 'maximum', 'mean')


def combine(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 5:
        raise ValueError('Require matched five-class logits')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Nonfinite logits')
    return {'head1': first, 'head2': second, 'maximum': np.maximum(first, second),
            'mean': (first + second) / 2}


def summaries(first, second, labels=None):
    result = {}
    for rule, values in combine(first, second).items():
        predicted = values.argmax(1)
        probability = torch.softmax(torch.from_numpy(values), -1).numpy()
        others = values[:, [0, 2, 3, 4]].max(1)
        margin = values[:, 1] - others
        summary = {'epochs': len(values), 'prediction_counts': np.bincount(predicted, minlength=5).tolist(),
                   'mean_probabilities': probability.astype(np.float64).mean(0).tolist(),
                   'N1_margin_quantiles': np.quantile(margin, [0, .25, .5, .75, 1]).tolist(),
                   'mean_entropy_nats': float(-(probability * np.log(np.maximum(probability, 1e-30))).sum(1).mean())}
        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape != predicted.shape or not np.isin(labels, range(5)).all():
                raise ValueError('Source validation labels invalid')
            summary['metrics'] = metrics_from_confusion(confusion_matrix_5(labels, predicted))
            summary['true_N1_margin_quantiles'] = (np.quantile(margin[labels == 1], [0, .25, .5, .75, 1]).tolist()
                                                   if np.any(labels == 1) else None)
        result[rule] = summary
    return result


def source_exposure(train_indices, labels, history, cfg):
    seen = np.zeros(len(train_indices), dtype=bool)
    presented_counts = np.zeros(5, dtype=np.int64)
    for epoch, row in enumerate(history):
        order = np.random.default_rng(cfg['seed'] + epoch).permutation(len(train_indices))
        order = order[:cfg['steps_per_epoch'] * cfg['batch_size']].astype(np.int64)
        if hashlib.sha256(order.tobytes()).hexdigest() != row['source_order_sha256']:
            raise ValueError('Source order hash does not replay')
        seen[order] = True
        presented_counts += np.bincount(labels[train_indices[order]], minlength=5)
    train_y = labels[train_indices]
    return {'train_epochs': len(train_indices), 'presentations': int(presented_counts.sum()),
            'equivalent_full_source_passes': float(presented_counts.sum() / len(train_indices)),
            'unique_epochs_seen': int(seen.sum()), 'unique_fraction': float(seen.mean()),
            'train_class_counts': np.bincount(train_y, minlength=5).tolist(),
            'seen_unique_class_counts': np.bincount(train_y[seen], minlength=5).tolist(),
            'presentation_class_counts': presented_counts.tolist()}


@torch.inference_mode()
def infer_views(source_models, adast_models, data, indices, guard, batch_size=128):
    parts = {name: [[], []] for name in VIEWS}
    for start in range(0, len(indices), batch_size):
        guard()
        x = torch.from_numpy(np.asarray(data[indices[start:start + batch_size]], dtype=np.float32).copy()).unsqueeze(1)
        for arm, models in [('source_only', source_models), ('adast', adast_models)]:
            features = models['encoder'](x)
            domains = ('source',) if arm == 'source_only' else ('source', 'target')
            for domain in domains:
                represented = models[domain + '_attention'](features)
                key = arm + '_' + domain + '_attention'
                for head in range(2):
                    parts[key][head].append(models['head' + str(head + 1)](represented).numpy())
    return {name: tuple(np.concatenate(p) for p in pair) for name, pair in parts.items()}


def make_models(module, cfg, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    models = build_models(module, cfg, 123, 'cpu')
    for name, model in models.items():
        model.load_state_dict(payload['models'][name], strict=True)
        model.eval()
    return models


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output == (ROOT / 'runs').resolve() or output.exists():
        raise ValueError('Require a new private runs directory')
    output.mkdir(parents=True)
    development = json.loads(PROTOCOL.read_bytes())
    plan = development['adast_diagnostic']
    if not 0 < args.max_seconds <= plan['max_seconds']:
        raise ValueError('Diagnostic budget must be at most 1800 seconds')
    started = time.monotonic()
    context = {'status': 'running', 'phase': 'input_verification', 'training': False}
    def publish(**values):
        context.update(values)
        atomic_json(output / 'progress.json', {**context, 'elapsed_seconds': time.monotonic() - started})
    def guard():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError('Diagnostic budget exhausted; completed fold outputs retained')
    publish()
    evidence_files = [TRAIN / name for name in ('aggregate_results.json', 'verification.json', 'independent_verification.json')]
    aggregate_sha = sha256_file(evidence_files[0])
    for path in evidence_files[1:]:
        evidence = json.loads(path.read_bytes())
        if evidence['status'] != 'passed' or evidence['aggregate_results_sha256'] != aggregate_sha:
            raise ValueError('Require verified training aggregate')
    manifest = json.loads((WORK / 'manifest.json').read_bytes())
    input_paths = [WORK / name for name in manifest['files']]
    for name, digest in manifest['files'].items():
        guard()
        if sha256_file(WORK / name) != digest:
            raise ValueError('Frozen training input changed')
    train_cfg = json.loads((WORK / 'protocol.json').read_bytes())
    train_result = json.loads(evidence_files[0].read_bytes())
    frozen = {str(p.relative_to(ROOT)): sha256_file(p) for p in evidence_files + input_paths +
              [WORK / 'manifest.json', PROTOCOL, Path(__file__)]}
    for pair in train_result['folds']:
        for arm in ('source_only', 'adast'):
            p = TRAIN / f"fold_{pair['fold']:02d}" / arm / 'final.pt'
            if sha256_file(p) != pair['arms'][arm]['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            frozen[str(p.relative_to(ROOT))] = sha256_file(p)
    write_once_json(output / 'execution_specification.json', {'protocol': development, 'frozen_sha256': frozen,
                     'torch': str(torch.__version__), 'source_role': 'validation', 'target_test_access': False,
                     'target_true_label_access': False, 'optimizer_created': False, 'max_seconds': args.max_seconds})
    module = load_module('development_adast_models', WORK / 'upstream/models.py')
    cfg = load_module('development_adast_cfg', WORK / 'upstream/configs.py').Config()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    x = np.load(WORK / 'data/source_x.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(WORK / 'data/source_y.npy', mmap_mode='r', allow_pickle=False)
    adaptation = np.load(WORK / 'data/adaptation_x.npy', mmap_mode='r', allow_pickle=False)
    folds = []
    try:
        for pair in train_result['folds']:
            fold = pair['fold']
            guard()
            publish(phase='validation_inference', fold=fold, completed_folds=len(folds))
            with np.load(WORK / f'data/fold_{fold:02d}_roles.npz', allow_pickle=False) as roles:
                train_indices, validation_indices = roles['train'], roles['validation']
            if np.intersect1d(train_indices, validation_indices).size:
                raise ValueError('Train/validation overlap')
            arm_models = {arm: make_models(module, cfg, TRAIN / f'fold_{fold:02d}' / arm / 'final.pt')
                          for arm in ('source_only', 'adast')}
            validation = infer_views(arm_models['source_only'], arm_models['adast'], x, validation_indices, guard)
            unlabelled = infer_views(arm_models['source_only'], arm_models['adast'], adaptation, np.arange(len(adaptation)), guard)
            summaries_val = {name: summaries(*values, y[validation_indices]) for name, values in validation.items()}
            summaries_adapt = {name: summaries(*values) for name, values in unlabelled.items()}
            arrays = {'validation_indices': validation_indices.astype(np.int64), 'validation_labels': y[validation_indices].copy()}
            for role, values in [('validation', validation), ('adaptation', unlabelled)]:
                for name, pair_logits in values.items():
                    for h, values_h in enumerate(pair_logits):
                        arrays[f'{role}_{name}_head{h+1}'] = values_h
            array_path = output / f'fold_{fold:02d}_logits.npz'
            with array_path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            pseudo = {}
            for r in range(train_cfg['rounds']):
                p = TRAIN / f'fold_{fold:02d}/adast/pseudo_round_{r}.npy'
                labels = np.load(p, allow_pickle=False)
                if labels.shape != (len(adaptation),) or not np.isin(labels, range(5)).all():
                    raise ValueError('Pseudo-label schema invalid')
                pseudo[str(r)] = {'counts': np.bincount(labels, minlength=5).tolist(), 'sha256': sha256_file(p),
                                 'target_loss_weight': train_cfg['target_loss_weights_by_round'][r]}
            histories = [pair['arms'][arm]['history'] for arm in ('source_only', 'adast')]
            if [r['source_order_sha256'] for r in histories[0]] != [r['source_order_sha256'] for r in histories[1]]:
                raise ValueError('Matched source orders differ')
            row = {'fold': fold, 'validation': summaries_val, 'adaptation_unlabelled': summaries_adapt,
                   'source_exposure': source_exposure(train_indices, y, histories[0], train_cfg), 'pseudo_labels': pseudo,
                   'logits_sha256': sha256_file(array_path),
                   'loss_history': {arm: [{'round': h['round'], 'epoch': h['epoch'], 'loss_mean': h['loss_mean'],
                                         'optimizer_lr_after_epoch': h['optimizer_lr_after_epoch']} for h in pair['arms'][arm]['history']]
                                   for arm in ('source_only', 'adast')}}
            write_once_json(output / f'fold_{fold:02d}_summary.json', row)
            folds.append(row)
            print(f'DIAGNOSTIC fold {fold}: source validation {len(validation_indices)}, unlabelled adaptation {len(adaptation)} COMPLETE', flush=True)
        pooled = {name: {rule: metrics_from_confusion(sum(np.asarray(f['validation'][name][rule]['metrics']['confusion_matrix'])
                                                        for f in folds)) for rule in RULES} for name in VIEWS}
        result = {'status': 'complete_validation_and_unlabelled_diagnostic_no_training',
                  'class_order': ['W','N1','N2','N3','REM'], 'folds': folds, 'pooled_validation': pooled,
                  'target_test_access': False, 'target_true_label_access': False,
                  'scope': 'Validation diagnostics, not independent test results or causal ablations. Alternative heads/attention are diagnostic, not a replacement inference rule.',
                  'elapsed_seconds': time.monotonic() - started}
        write_once_json(output / 'aggregate_results.json', result)
        # Independent reconstruction from saved logits, with direct first/last
        # batch replay on the final fold; no parameters or checkpoints updated.
        for row in folds:
            path = output / f"fold_{row['fold']:02d}_logits.npz"
            if sha256_file(path) != row['logits_sha256']:
                raise ValueError('Saved logits hash changed')
            with np.load(path, allow_pickle=False) as z:
                for name in VIEWS:
                    for role, expected, labels in [('validation', row['validation'][name], z['validation_labels']),
                                                   ('adaptation', row['adaptation_unlabelled'][name], None)]:
                        actual = summaries(z[f'{role}_{name}_head1'], z[f'{role}_{name}_head2'], labels)
                        if actual != expected:
                            raise ValueError('Saved diagnostic summaries do not replay')
        last = folds[-1]['fold']
        with np.load(output / f'fold_{last:02d}_logits.npz', allow_pickle=False) as z:
            for begin in (0, max(0, len(z['validation_indices']) - 128)):
                stop = min(begin + 128, len(z['validation_indices']))
                actual = infer_views(arm_models['source_only'], arm_models['adast'], x, z['validation_indices'][begin:stop], guard)
                for name, heads in actual.items():
                    for h, head in enumerate(heads):
                        np.testing.assert_allclose(head, z[f'validation_{name}_head{h+1}'][begin:stop], rtol=0, atol=3e-6)
        for name, digest in frozen.items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError('Frozen input/code/checkpoint changed during diagnostic')
        write_once_json(output / 'verification.json', {'status':'passed', 'all_saved_logits_summaries_recomputed':True,
                         'final_fold_first_last_validation_batches_replayed':True, 'frozen_hashes_unchanged':True,
                         'aggregate_results_sha256':sha256_file(output / 'aggregate_results.json')})
        publish(status='complete', phase='complete', completed_folds=len(folds))
    except Exception as error:
        publish(status='stopped_resource_budget' if isinstance(error, TimeoutError) else 'failed', reason=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800)
    run(parser.parse_args())
