"""Four fresh CUDA TCNs on source train/validation only; no final-test scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from sleeptcn.development import softened_class_weights, require_development_roles
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import write_once_json
from sleeptcn.revision_weighted import train_arm, evaluate_source
from sleeptcn.revision_weighted_campaign import atomic_json


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required: refusing CPU training fallback')
    protocol_path = ROOT / 'configs/teacher_revision_development_v2.json'
    cfg = json.loads(protocol_path.read_bytes())['weighted_development']
    if not 0 < args.max_seconds <= cfg['max_seconds']:
        raise ValueError('Require a positive attempt budget at most five hours')
    manifest_path = ROOT / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    for field in ('outer_test_included', 'target_data_included', 'participant_ids_included'):
        if manifest[field]:
            raise ValueError('Forbidden development input')
    require_development_roles(manifest['records'])
    for name, digest in manifest['files'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('Frozen bundle changed')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    roles = {'train': [], 'validation': []}
    counts = np.zeros(5, dtype=np.int64)
    for entry in manifest['records']:
        with np.load(ROOT / entry['path'], allow_pickle=False) as z:
            if set(z.files) != {'features', 'labels'}:
                raise ValueError('Unexpected fields or participant identifiers')
            x, y = z['features'].copy(), z['labels'].astype(np.int64)
        if x.shape != (entry['epochs'], 128) or not np.isfinite(x).all() or not np.isin(y, [-1,0,1,2,3,4]).all():
            raise ValueError('Feature/label schema differs')
        if int((y >= 0).sum()) != entry['valid_epochs']:
            raise ValueError('Valid epoch count differs')
        roles[entry['role']].append((torch.from_numpy(x), torch.from_numpy(y)))
        if entry['role'] == 'train':
            counts += np.bincount(y[y >= 0], minlength=5)
    np.testing.assert_array_equal(counts, manifest['train_class_counts'])
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('New development attempt only: do not overwrite/restart automatically')
    output.mkdir(parents=True)
    identity = {'fold':cfg['fold'], 'encoder_sha256':manifest['encoder_sha256'],
                'protocol_sha256':sha256_file(protocol_path), 'bundle_manifest_sha256':sha256_file(manifest_path),
                'runner_sha256':sha256_file(Path(__file__)), 'torch':str(torch.__version__), 'backend':'CUDA_float32'}
    write_once_json(output / 'execution_specification.json', {'identity':identity, 'protocol':cfg,
                     'target_access':False, 'outer_test_access':False, 'max_seconds':args.max_seconds,
                     'gpu':torch.cuda.get_device_name(0), 'fresh_classifiers_only_encoder_frozen':True})
    started = time.monotonic()
    context = {'status':'running','phase':'training'}
    def guard(event=None):
        if event and event['phase'] == 'epoch_saved':
            row = event['history'][-1]
            context.update(arm=event['arm'], epoch=row['epoch'], selected_epoch=row['best_epoch'],
                           validation_macro_f1=row['validation_macro_f1'], epoch_seconds=row['seconds'])
            atomic_json(output / 'progress.json', {**context, 'elapsed_seconds':time.monotonic()-started})
        if time.monotonic()-started >= args.max_seconds:
            raise TimeoutError('Five-hour budget reached; saved epochs retained, no automatic continuation')
    arms = []
    try:
        for power in cfg['powers']:
            guard()
            key = 'power_' + str(power).replace('.', '_')
            weights = softened_class_weights(counts, power)
            model, selection = train_arm(roles['train'], roles['validation'], weights, cfg['training'],
                                         output / key, {**identity,'power':power}, 'cuda', guard)
            validation = evaluate_source(model, roles['validation'], cfg['training']['batch_size_records'], 'cuda')
            if validation['macro_f1'] != selection['selected_validation_macro_f1']:
                raise ValueError('Selected validation score does not replay')
            arms.append({'power':power,'weights':weights.tolist(),'selection':selection,'validation':validation})
            del model
            torch.cuda.empty_cache()
        if len({a['selection']['initial_state_sha256'] for a in arms}) != 1:
            raise ValueError('Initial states are not matched')
        common = min(a['selection']['epochs_completed'] for a in arms)
        orders = [[h['batch_order_sha256'] for h in a['selection']['history'][:common]] for a in arms]
        if any(order != orders[0] for order in orders):
            raise ValueError('Source batch order differs')
        result = {'status':'complete_source_validation_development_only','arms':arms,
                  'target_access':False,'outer_test_access':False,'elapsed_seconds':time.monotonic()-started,
                  'scope':'One development fold; no final target winner selected, no generalization claim.'}
        write_once_json(output / 'aggregate_results.json', result)
        write_once_json(output / 'verification.json', {'status':'passed','initialization_and_source_order_matched':True,
                         'validation_replayed':True,'aggregate_results_sha256':sha256_file(output / 'aggregate_results.json')})
        context.update(status='complete',phase='complete')
    except Exception as error:
        context.update(status='stopped_resource_budget' if isinstance(error,TimeoutError) else 'failed',reason=str(error))
        raise
    finally:
        atomic_json(output / 'progress.json', {**context,'elapsed_seconds':time.monotonic()-started})
        paths = sorted(p for p in output.rglob('*') if p.is_file() and p.suffix in {'.pt','.json','.log'})
        archive = output.parent / 'SleepTCN_Weighted_Development_Results.zip'
        if archive.exists():
            raise FileExistsError('Result archive already exists; not overwriting')
        hashes = {p.relative_to(output).as_posix():sha256_file(p) for p in paths}
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                z.write(p,p.relative_to(output).as_posix())
            z.writestr('export_manifest.json',json.dumps(hashes,indent=2))
        print('RESULT_ARCHIVE',archive,'SHA256',sha256_file(archive),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=18000)
    run(parser.parse_args())
