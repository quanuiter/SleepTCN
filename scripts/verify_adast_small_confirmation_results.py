"""Independent fold-1 confirmation checks: all diagnostics and six CPU replays."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile
import numpy as np
import torch
import verify_adast_development_results as shared

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'runs/adast_small_confirmation_20261004'
STATUS = 'complete_three_arm_new_fold_confirmation'

def validate_protocol(cfg):
    expected = {'source_only_full_source':{},'adast_reference_full_source':{},
                'adast_keep_source_ce_full_source':{'source_loss_weights_by_round':[1.,1.]}}
    if (cfg['campaign_kind'] != 'three_arm_new_fold_confirmation' or cfg['fold'] != 1 or cfg['seed'] != 123
        or cfg['arms'] != list(expected) or cfg['arm_overrides'] != expected
        or cfg['source_train_epochs'] != 152825 or cfg['source_validation_epochs'] != 23881
        or cfg['batch_size'] != 128 or cfg['rounds'] != 2 or cfg['epochs_per_round'] != 15
        or cfg['updates_per_arm'] != 35820 or cfg['adaptation_subjects'] != 5 or cfg['adaptation_epochs'] != 4989):
        raise ValueError('Locked three-arm/new-fold protocol differs')

def verify(args):
    started = time.monotonic()
    def guard():
        if time.monotonic()-started > args.max_seconds:
            raise TimeoutError('Verification time limit; no training or automatic restart')
    if not 0 < args.max_seconds <= 1800:
        raise ValueError('Require at most 30 minutes for independent verification')
    reference = shared.reference
    if reference.digest(args.archive) != args.expected_sha256:
        raise ValueError('Downloaded archive differs from observed Colab SHA')
    pointer = shared.read_local_json(RECORD/'storage.json')
    storage = Path(pointer['storage_root']).resolve()
    work = Path(pointer['payload_root']).resolve()
    if work != storage/'payload':
        raise ValueError('Unexpected input storage location')
    bundle = shared.read_local_json(RECORD/'bundle_verification.json')
    if reference.digest(Path(bundle['archive_path'])) != bundle['archive_sha256']:
        raise ValueError('Frozen input archive changed')
    manifest = shared.read_local_json(work/'manifest.json')
    if reference.digest(work/'manifest.json') != bundle['manifest_sha256'] or bundle['status'] != 'passed':
        raise ValueError('Frozen input manifest differs')
    if any(manifest[k] for k in ('source_outer_test_included','target_test_data_included','target_labels_included','participant_ids_included')):
        raise ValueError('Forbidden input role')
    for name, digest in manifest['files'].items():
        guard()
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name or reference.digest(work/name) != digest:
            raise ValueError('Frozen input member differs')
    if reference.digest(ROOT/'src/sleeptcn/metrics.py') != manifest['files']['development_metrics.py']:
        raise ValueError('Independent metric code differs')
    cfg = shared.read_local_json(work/'protocol.json')
    validate_protocol(cfg)
    if reference.digest(ROOT/'configs/adast_small_confirmation_v1_20261004.json') != manifest['files']['protocol.json']:
        raise ValueError('Locked confirmation configuration changed')
    y = np.load(work/'data/train_y.npy',mmap_mode='r',allow_pickle=False)
    vx = np.load(work/'data/validation_x.npy',mmap_mode='r',allow_pickle=False)
    vy = np.load(work/'data/validation_y.npy',mmap_mode='r',allow_pickle=False)
    if len(y) != cfg['source_train_epochs'] or len(vy) != cfg['source_validation_epochs'] or vx.shape != (len(vy),3000):
        raise ValueError('Input support differs')
    torch.set_num_threads(2)
    upstream = reference.load_module('confirmation_verified_models',work/'upstream/models.py')
    model_cfg = reference.load_module('confirmation_verified_cfg',work/'upstream/configs.py').Config()
    with zipfile.ZipFile(args.archive) as z:
        hashes = shared.read_json(z,'export_manifest.json')
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(hashes)|{'export_manifest.json'}:
            raise ValueError('Result allowlist/duplicates differ')
        for entry in z.infolist():
            guard()
            name = entry.orig_filename
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name or entry.file_size > 100_000_000:
                raise ValueError('Unsafe result archive member')
            if name != 'export_manifest.json' and hashlib.sha256(z.read(entry)).hexdigest() != hashes[name]:
                raise ValueError('Result member hash differs')
        result = shared.read_json(z,'aggregate_results.json')
        internal = shared.read_json(z,'verification.json')
        spec = shared.read_json(z,'execution_specification.json')
        progress = shared.read_json(z,'progress.json')
        aggregate_sha = hashlib.sha256(z.read('aggregate_results.json')).hexdigest()
        spec_sha = hashlib.sha256(z.read('execution_specification.json')).hexdigest()
        if (result['status'] != STATUS or progress['status'] != STATUS or progress['completed_arms'] != 3
            or internal['status'] != 'passed' or internal['aggregate_results_sha256'] != aggregate_sha
            or result['source_outer_test_access'] or result['target_test_access']
            or [r['arm'] for r in result['arms']] != cfg['arms']):
            raise ValueError('Require complete three-arm confirmation results')
        if (spec['protocol'] != cfg or spec['manifest_sha256'] != bundle['manifest_sha256']
            or spec['runner_sha256'] != manifest['files']['run_adast_small_confirmation_cuda.py']
            or spec['base_runner_sha256'] != cfg['previous_development_runner_sha256']
            or spec['backend'] != 'CUDA_float32_deterministic_no_tf32'
            or any(spec[k] for k in ('source_outer_test_access','target_test_access','target_true_label_access'))
            or not 0 < spec['max_seconds'] <= 18000 or result['elapsed_seconds'] > spec['max_seconds']):
            raise ValueError('Frozen execution identity/budget differs')
        expected_initial = result['arms'][0]['initial_state_sha256']
        initial, initial_audit = shared.initial_helpers.reconstruct_initial(upstream,model_cfg,cfg['seed'],expected_initial)
        rows = []
        for arm in result['arms']:
            guard()
            key = arm['arm']
            arm_cfg = {**cfg,**cfg['arm_overrides'][key]}
            adaptive = key.startswith('adast_')
            identity = {'arm':key,'fold':cfg['fold'],'execution_specification_sha256':spec_sha}
            history = arm['history']
            if (arm['initial_state_sha256'] != expected_initial or len(history) != 30 or arm['epochs_completed'] != 30
                or shared.read_json(z,key+'/identity.json') != {'identity':identity,'initial_state_sha256':expected_initial}
                or shared.read_json(z,key+'/selection.json') != arm):
                raise ValueError('Arm identity/history differs')
            init = torch.load(io.BytesIO(z.read(key+'/initial.pt')),map_location='cpu',weights_only=True)
            if init['identity'] != identity or init['initial_state_sha256'] != expected_initial:
                raise ValueError('Saved initialization identity differs')
            for name,model in initial.items():
                for k,v in model.state_dict().items():
                    if not torch.equal(v,init['models'][name][k]):
                        raise ValueError('Saved initialization does not replay exactly')
            seen = np.zeros(len(y),dtype=bool)
            target_seen = np.zeros(cfg['adaptation_epochs'],dtype=bool)
            for epoch,row in enumerate(history):
                guard()
                source,target = shared.replay_sampling(cfg,epoch,True)
                seen[source] = True
                if adaptive:
                    target_seen[target] = True
                if (row['global_epoch'] != epoch+1 or row['round'] != epoch//15 or row['epoch'] != epoch%15+1
                    or row['updates'] != 1194 or row['source_presentations'] != len(source)
                    or row['source_unique_seen'] != int(seen.sum()) or row['target_unique_seen'] != int(target_seen.sum())
                    or row['target_training_presentations'] != (len(target) if adaptive else 0)
                    or row['source_order_sha256'] != hashlib.sha256(source.tobytes()).hexdigest()
                    or row['target_order_sha256'] != (hashlib.sha256(target.tobytes()).hexdigest() if adaptive else None)
                    or row['source_label_counts'] != np.bincount(y[source],minlength=5).tolist()):
                    raise ValueError('Epoch sampling/coverage/budget does not replay')
                lr_before = cfg['optimizer']['lr']*(.1 if epoch >= 10 else 1.)
                lr_after = cfg['optimizer']['lr']*(.1 if epoch >= 9 else 1.)
                if row['optimizer_lr_before_epoch'] != lr_before or row['optimizer_lr_after_epoch'] != lr_after:
                    raise ValueError('Learning-rate schedule differs')
                coefficients = {'source_ce':arm_cfg['source_loss_weights_by_round'][epoch//15],
                    'similarity':cfg['similarity_weight'],'adversarial':cfg['adversarial_weight'] if adaptive else 0.,
                    'target_pseudo_ce':cfg['target_loss_weights_by_round'][epoch//15] if adaptive else 0.}
                if row['loss_coefficients'] != coefficients or not np.isfinite(list(row['loss_batch_means'].values())).all():
                    raise ValueError('Loss coefficients differ')
                shared.verify_loss_components(row['loss_batch_means'],coefficients,adaptive)
                if adaptive:
                    pseudo = np.load(io.BytesIO(z.read(key+f'/pseudo_round_{epoch//15}.npy')),allow_pickle=False)
                    if pseudo.shape != (4989,) or not np.isin(pseudo,range(5)).all() or row['pseudo_label_counts_at_round_start'] != np.bincount(pseudo,minlength=5).tolist():
                        raise ValueError('Pseudo-label support differs')
                elif row['pseudo_label_counts_at_round_start'] is not None:
                    raise ValueError('Source-only has target training diagnostics')
                name = key+f'/validation_epoch_{epoch+1:03d}.npz'
                if hashlib.sha256(z.read(name)).hexdigest() != row['validation_logits_sha256']:
                    raise ValueError('Validation logits identity differs')
                shared.verify_logits(z,name,vy,row['validation'])
            shared.verify_logits(z,key+'/validation_epoch_000.npz',vy,shared.read_json(z,key+'/validation_before_training.json'))
            scores = [h['validation']['source']['macro_f1'] for h in history]
            best_epoch = int(np.argmax(scores))+1
            if (arm['best_epoch'] != best_epoch or arm['total_updates'] != 35820
                or arm['best_source_validation'] != history[best_epoch-1]['validation']
                or arm['final_source_validation'] != history[-1]['validation']
                or hashlib.sha256(z.read(key+'/latest.pt')).hexdigest() != arm['final_checkpoint_sha256']):
                raise ValueError('Checkpoint selection/final differs')
            replays, checkpoints = {}, {}
            for selection,epoch,digest in [('best',best_epoch,arm['best_checkpoint_sha256']),('final',30,arm['final_checkpoint_sha256'])]:
                data = z.read(key+'/'+selection+'.pt')
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError('Checkpoint hash differs')
                payload = torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
                checkpoints[selection] = payload
                if (payload['identity'] != identity or payload['initial_state_sha256'] != expected_initial
                    or payload['epochs_completed'] != epoch or payload['history'] != history[:epoch]
                    or not payload['cuda_rng_state'] or 'optimizer' not in payload or 'disc_optimizer' not in payload):
                    raise ValueError('Checkpoint identity/history/RNG differs')
                models = reference.build_models(upstream,model_cfg,cfg['seed'],'cpu')
                for name,model in models.items():
                    model.load_state_dict(payload['models'][name],strict=True)
                replay = {}
                for domain in ('source','target'):
                    logits = reference.predict(models,vx,domain,'cpu',guard,batch_size=128)
                    replay[domain] = shared.metrics_from_confusion(shared.confusion_matrix_5(vy,logits.argmax(1)))
                    if replay[domain] != history[epoch-1]['validation'][domain]:
                        raise ValueError('CPU checkpoint confusion/metrics differ')
                replays[selection] = replay
                if not adaptive:
                    for name in ('discriminator','target_attention'):
                        if any(not torch.equal(v,init['models'][name][k]) for k,v in payload['models'][name].items()):
                            raise ValueError('Source-only unused module changed')
            for name,state in checkpoints['best']['models'].items():
                if any(not torch.equal(v,checkpoints['final']['best_models'][name][k]) for k,v in state.items()):
                    raise ValueError('Best checkpoint differs from retained final best state')
            for kind, expected in [('source',seen),('target',target_seen)]:
                packed = np.frombuffer(bytes.fromhex(checkpoints['final'][kind+'_seen_packed_hex']),dtype=np.uint8)
                np.testing.assert_array_equal(np.unpackbits(packed)[:len(expected)],expected)
            rows.append({'arm':key,'best_epoch':best_epoch,'total_updates':35820,'CPU_checkpoint_validation_replays':replays})
        orders = [[h['source_order_sha256'] for h in arm['history']] for arm in result['arms']]
        if not all(order == orders[0] for order in orders):
            raise ValueError('All three source orders must match')
        for epoch in range(16):
            suffix = f'/validation_epoch_{epoch:03d}.npz'
            with np.load(io.BytesIO(z.read(cfg['arms'][1]+suffix))) as left, np.load(io.BytesIO(z.read(cfg['arms'][2]+suffix))) as right:
                for domain in ('source','target'):
                    np.testing.assert_array_equal(left[domain],right[domain])
        output = (args.output or storage/'verified_results').resolve()
        if not output.is_relative_to(storage) or output == storage or output.exists():
            raise ValueError('Require new result subdirectory in the recorded private temporary storage')
        output.mkdir(parents=True)
        for name in z.namelist():
            target = output/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(name))
    proof = {'status':'passed','archive_sha256':args.expected_sha256,'aggregate_results_sha256':aggregate_sha,
        'initialization_replay':initial_audit,'all_93_validation_diagnostics_recomputed':True,
        'all_six_best_final_checkpoints_CPU_validation_replayed':True,'sampling_coverage_and_budget_replayed':True,
        'all_three_initializations_and_source_orders_match':True,'round_one_reference_keep_prefix_verified':True,
        'source_outer_test_access':False,'target_test_access':False,'arms':rows,'elapsed_seconds':time.monotonic()-started}
    reference.write_once(output/'independent_verification.json',proof)
    reference.write_once(RECORD/'verified_results_pointer.json',{'status':'passed','results_root':output.as_posix(),
        'aggregate_results_sha256':aggregate_sha,'independent_verification_sha256':reference.digest(output/'independent_verification.json')})
    print(json.dumps({'status':'passed','results_root':output.as_posix(),'aggregate_results_sha256':aggregate_sha,
                     'elapsed_seconds':proof['elapsed_seconds']},indent=2),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-seconds',type=float,default=1800)
    verify(parser.parse_args())
