"""Independent download/hash/selection validation and CPU inference replay; no training."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sleeptcn.models import SleepTCN
from sleeptcn.development import softened_class_weights, require_development_roles
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_weighted import evaluate_source, state_digest
from sleeptcn.revision_campaign import write_once_json


def checked_archive(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Archive SHA differs from observed/locked value')
    z = zipfile.ZipFile(path)
    names = z.namelist()
    if len(names) != len(set(names)) or z.testzip() is not None:
        raise ValueError('Duplicate member or CRC failure')
    for member in z.infolist():
        # orig_filename preserves separators before Windows ZipInfo normalizes.
        name = member.orig_filename
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe archive member')
    return z


def verify_initialization_proof(path, expected_sha, result_sha, version, seed, expected_digest):
    """Require hash-bound same-version replay, never silently ignore a mismatch."""
    with checked_archive(path, expected_sha) as z:
        if set(z.namelist()) != {'proof.json', 'initial_cpu.pt'}:
            raise ValueError('Initialization proof allowlist differs')
        proof = json.loads(z.read('proof.json'))
        if (proof['result_archive_sha256'] != result_sha or proof['torch'] != version
                or proof['seed'] != seed or proof['training'] or proof['additional_data_access']
                or not proof['all_four_recorded_initial_states_match']):
            raise ValueError('Initialization proof scope/identity differs')
        if proof['CPU_initial_state_sha256'] != expected_digest or proof['CUDA_initial_state_sha256'] != expected_digest:
            raise ValueError('Initialization proof digest differs')
        state = torch.load(io.BytesIO(z.read('initial_cpu.pt')), map_location='cpu', weights_only=True)
    model = SleepTCN(input_dim=128)
    model.load_state_dict(state, strict=True)
    if state_digest(model) != expected_digest:
        raise ValueError('Exported initialization tensor digest differs')
    return state


def run(args):
    bundle_verification = json.loads((args.bundle.parent / 'bundle_verification.json').read_bytes())
    with checked_archive(args.bundle, bundle_verification['archive_sha256']) as inputs:
        manifest_bytes = inputs.read('manifest.json')
        manifest = json.loads(manifest_bytes)
        require_development_roles(manifest['records'])
        if any(manifest[k] for k in ('outer_test_included','target_data_included','participant_ids_included','raw_eeg_included')):
            raise ValueError('Development input contains forbidden data')
        if set(inputs.namelist()) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Bundle allowlist differs')
        for name, digest in manifest['files'].items():
            if hashlib.sha256(inputs.read(name)).hexdigest() != digest:
                raise ValueError('Bundle payload hash differs')
        cfg_bytes = inputs.read('configs/teacher_revision_development_v2.json')
        cfg = json.loads(cfg_bytes)['weighted_development']
        for name in ('src/sleeptcn/models.py','src/sleeptcn/training.py','src/sleeptcn/metrics.py','src/sleeptcn/revision_weighted.py'):
            if sha256_file(ROOT / name) != manifest['files'][name]:
                raise ValueError('Local replay code differs from training bundle')
        roles = {'train':[], 'validation':[]}
        counts = np.zeros(5, dtype=np.int64)
        for entry in manifest['records']:
            with np.load(io.BytesIO(inputs.read(entry['path'])), allow_pickle=False) as z:
                if set(z.files) != {'features','labels'}:
                    raise ValueError('Feature schema differs')
                x, y = z['features'].copy(), z['labels'].astype(np.int64)
            if len(y) != entry['epochs'] or int((y >= 0).sum()) != entry['valid_epochs']:
                raise ValueError('Feature support differs')
            roles[entry['role']].append((torch.from_numpy(x), torch.from_numpy(y)))
            if entry['role'] == 'train':
                counts += np.bincount(y[y >= 0], minlength=5)
    np.testing.assert_array_equal(counts, manifest['train_class_counts'])
    with checked_archive(args.archive, args.expected_sha256) as z:
        export = json.loads(z.read('export_manifest.json'))
        if set(z.namelist()) != set(export) | {'export_manifest.json'}:
            raise ValueError('Result allowlist differs')
        for name, digest in export.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('Result payload hash differs')
        result = json.loads(z.read('aggregate_results.json'))
        verification = json.loads(z.read('verification.json'))
        spec = json.loads(z.read('execution_specification.json'))
        aggregate_sha = hashlib.sha256(z.read('aggregate_results.json')).hexdigest()
        if verification['status'] != 'passed' or verification['aggregate_results_sha256'] != aggregate_sha:
            raise ValueError('Training verification does not bind aggregate')
        if result['status'] != 'complete_source_validation_development_only' or result['target_access'] or result['outer_test_access']:
            raise ValueError('Unexpected evaluation scope')
        identity = spec['identity']
        if spec['protocol'] != cfg or spec['target_access'] or spec['outer_test_access'] or identity['backend'] != 'CUDA_float32':
            raise ValueError('Frozen protocol differs')
        if identity['bundle_manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest() or identity['protocol_sha256'] != hashlib.sha256(cfg_bytes).hexdigest():
            raise ValueError('Frozen input identity differs')
        if identity['runner_sha256'] != manifest['files']['scripts/run_weighted_development_cuda.py'] or identity['encoder_sha256'] != manifest['encoder_sha256']:
            raise ValueError('Runner/encoder identity differs')
        if not 0 < spec['max_seconds'] <= 18000 or result['elapsed_seconds'] > spec['max_seconds']:
            raise ValueError('Attempt exceeded budget')
        if [a['power'] for a in result['arms']] != cfg['powers']:
            raise ValueError('Four-arm grid differs')
        torch.set_num_threads(4)
        replay_rows = []
        initial = []
        initialization_replay = []
        for arm in result['arms']:
            key = 'power_' + str(arm['power']).replace('.', '_')
            selection = arm['selection']
            np.testing.assert_array_equal(arm['weights'],softened_class_weights(counts,arm['power']))
            arm_identity = json.loads(z.read(key + '/identity.json'))
            if arm_identity['training_config'] != cfg['training'] or arm_identity['device'] != 'cuda':
                raise ValueError('Arm training configuration/backend differs')
            if json.loads(z.read(key + '/selection.json')) != selection:
                raise ValueError('Arm selection differs')
            np.testing.assert_allclose(selection['loss_weights'], softened_class_weights(counts, arm['power']).astype(np.float32), rtol=0, atol=0)
            if hashlib.sha256(z.read(key + '/best.pt')).hexdigest() != selection['checkpoint_sha256']:
                raise ValueError('Selected checkpoint hash differs')
            best = torch.load(io.BytesIO(z.read(key + '/best.pt')), map_location='cpu', weights_only=True)
            latest = torch.load(io.BytesIO(z.read(key + '/latest.pt')), map_location='cpu', weights_only=True)
            if best['identity'] != {**identity,'power':arm['power']} or latest['identity'] != best['identity']:
                raise ValueError('Checkpoint identity differs')
            recorded_initial = selection['initial_state_sha256']
            if arm_identity['initial_state_sha256'] != recorded_initial or best['initial_state_sha256'] != recorded_initial:
                raise ValueError('Recorded initialization hashes differ')
            history = selection['history']
            if latest['history'] != history or len(history) != selection['epochs_completed']:
                raise ValueError('Latest training history differs')
            if latest['epoch'] != len(history) or not latest['cuda_rng_state']:
                raise ValueError('Latest saved epoch/CUDA RNG state differs')
            if [h['epoch'] for h in history] != list(range(1,len(history)+1)):
                raise ValueError('Non-contiguous epoch history')
            scores = [h['validation_macro_f1'] for h in history]
            selected_epoch = int(np.argmax(scores)) + 1
            if selected_epoch != selection['selected_epoch'] or best['epoch'] != selected_epoch or best['validation_macro_f1'] != max(scores):
                raise ValueError('Strict validation selection does not replay')
            if selection['selected_validation_macro_f1'] != max(scores):
                raise ValueError('Reported selection score differs')
            generator = torch.Generator().manual_seed(cfg['training']['seed'])
            for row in history:
                order = torch.randperm(len(roles['train']), generator=generator).numpy().astype(np.int64)
                if hashlib.sha256(order.tobytes()).hexdigest() != row['batch_order_sha256']:
                    raise ValueError('Source batch order does not replay')
            for name, value in best['model_state'].items():
                if not torch.equal(value, latest['best_model_state'][name]):
                    raise ValueError('Latest/best selected state differs')
            torch.manual_seed(cfg['training']['seed'])
            model = SleepTCN(input_dim=128)
            local_initial = state_digest(model)
            initial.append(recorded_initial)
            replay_detail = {'power':arm['power'], 'local_torch':str(torch.__version__),
                             'training_torch':identity['torch'], 'local_initial_sha256':local_initial,
                             'recorded_initial_sha256':recorded_initial,
                             'local_initialization_bitwise_replayed':local_initial == recorded_initial}
            if local_initial != recorded_initial:
                if str(torch.__version__) == identity['torch'] or args.initialization_proof is None or args.initialization_proof_sha256 is None:
                    raise ValueError('Initial classifier state does not replay; require same-version initialization proof')
                same_version_state = verify_initialization_proof(args.initialization_proof,
                    args.initialization_proof_sha256,args.expected_sha256,identity['torch'],
                    cfg['training']['seed'],recorded_initial)
                differences = [(model.state_dict()[k] - v).abs() for k,v in same_version_state.items()]
                replay_detail.update(same_version_exported_state_verified=True,
                    differing_elements=sum(int((d != 0).sum()) for d in differences),
                    max_absolute_initialization_difference=max(float(d.max()) for d in differences),
                    proof_archive_sha256=args.initialization_proof_sha256)
            initialization_replay.append(replay_detail)
            model.load_state_dict(best['model_state'], strict=True)
            metrics = evaluate_source(model, roles['validation'], cfg['training']['batch_size_records'], 'cpu')
            if metrics['confusion_matrix'] != arm['validation']['confusion_matrix']:
                raise ValueError('CPU replay predictions differ: inspect numerics before claiming verification')
            replay_rows.append({'power':arm['power'],'selected_epoch':selected_epoch,'validation':metrics})
        if len(set(initial)) != 1:
            raise ValueError('Initializations differ')
        output = args.output.resolve()
        if not output.is_relative_to((ROOT/'runs').resolve()) or output == (ROOT/'runs').resolve():
            raise ValueError('Require a new private runs subdirectory')
        if output.exists():
            raise FileExistsError('Verified output already exists; no overwrite')
        output.mkdir(parents=True)
        for name in z.namelist():
            destination = output / name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(z.read(name))
    proof = {'status':'passed','archive_sha256':args.expected_sha256,'aggregate_results_sha256':aggregate_sha,
             'four_selected_checkpoints_verified':True,'source_validation_CPU_inference_replayed':True,
             'source_order_and_initialization_replayed':True,'target_access':False,'outer_test_access':False,
             'initialization_replay_details':initialization_replay,
             'initialization_note':'Same-version replay required if local PyTorch initialization is not bitwise identical; no cross-version retraining equivalence claimed.',
             'arms':replay_rows}
    write_once_json(output / 'independent_verification.json', proof)
    print(json.dumps(proof,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--bundle',type=Path,default=ROOT/'runs/development_20261004/weighted_bundle/SleepTCN_Weighted_Development_20261004.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/development_20261004/weighted_verified')
    parser.add_argument('--initialization-proof',type=Path)
    parser.add_argument('--initialization-proof-sha256')
    run(parser.parse_args())
