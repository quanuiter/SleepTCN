"""Independently verify all epoch diagnostics, sampling and CPU checkpoint inference; no training."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/adast_development_20261004'
sys.path.insert(0,str(ROOT/'src'))
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


reference=load('adast_development_reference_verifier',ROOT/'scripts/run_colab_adast_training.py')
initial_helpers=load('adast_development_initial_replay_helpers',ROOT/'scripts/verify_colab_adast_results.py')


def read_json(z,name):
    return json.loads(z.read(name))


def replay_sampling(cfg,epoch,full):
    # Independent implementation rather than importing the development sampler.
    source=np.random.default_rng(cfg['seed']+epoch).permutation(cfg['source_train_epochs']).astype(np.int64)
    size=cfg['limited_steps_per_epoch']*cfg['batch_size']
    if not full:
        source=source[:size]
    target=np.random.default_rng(cfg['seed']+10000+epoch).permutation(cfg['adaptation_epochs']).astype(np.int64)[:size]
    target=np.tile(target,int(np.ceil(len(source)/len(target))))[:len(source)]
    return source,target


def verify(args):
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>args.max_seconds:
            raise TimeoutError('CPU verification budget reached; no training or automatic restart')
    if reference.digest(args.archive)!=args.expected_sha256:
        raise ValueError('Downloaded result archive differs from observed Colab SHA')
    input_base=getattr(args,'input_base',BASE).resolve()
    bundle=read_local_json(input_base/'bundle_verification.json')
    work=input_base/'payload'
    if reference.digest(input_base/bundle.get('archive_filename','SleepTCN_ADAST_Development_20261004.zip'))!=bundle['archive_sha256']:
        raise ValueError('Locked input archive differs')
    manifest=read_local_json(work/'manifest.json')
    if reference.digest(work/'manifest.json')!=bundle['manifest_sha256']:
        raise ValueError('Locked manifest differs')
    if any(manifest[k] for k in ('source_outer_test_included','target_test_data_included','target_labels_included','participant_ids_included')):
        raise ValueError('Forbidden data role')
    for name,digest in manifest['files'].items():
        guard()
        if reference.digest(work/name)!=digest:
            raise ValueError('Locked payload changed')
    if reference.digest(ROOT/'src/sleeptcn/metrics.py')!=manifest['files']['development_metrics.py']:
        raise ValueError('Independent metric code differs from training bundle')
    cfg=read_local_json(work/'protocol.json')
    loss_ablation=cfg.get('campaign_kind')=='full_source_single_factor_loss_ablation'
    if loss_ablation:
        validate_ablation_overrides(cfg)
        if bundle['base_manifest_sha256']!=reference.digest(BASE/'payload/manifest.json'):
            raise ValueError('Ablation does not bind the verified development inputs')
        original=read_local_json(BASE/'payload/manifest.json')
        if manifest['reused_data_files']!={k:v for k,v in original['files'].items() if k.startswith('data/')}:
            raise ValueError('Reused development data differs')
    y=np.load(work/'data/train_y.npy',mmap_mode='r',allow_pickle=False)
    vx=np.load(work/'data/validation_x.npy',mmap_mode='r',allow_pickle=False)
    vy=np.load(work/'data/validation_y.npy',mmap_mode='r',allow_pickle=False)
    torch.set_num_threads(2)
    upstream=reference.load_module('adast_development_verified_models',work/'upstream/models.py')
    model_cfg=reference.load_module('adast_development_verified_cfg',work/'upstream/configs.py').Config()
    with zipfile.ZipFile(args.archive) as z:
        hashes=read_json(z,'export_manifest.json')
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(hashes)|{'export_manifest.json'}:
            raise ValueError('Result allowlist/duplicate differs')
        for entry in z.infolist():
            guard()
            name=entry.orig_filename
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or entry.file_size>100_000_000:
                raise ValueError('Unsafe result archive member')
            if name!='export_manifest.json' and hashlib.sha256(z.read(entry)).hexdigest()!=hashes[name]:
                raise ValueError('Result member hash differs')
        result=read_json(z,'aggregate_results.json')
        internal=read_json(z,'verification.json')
        spec=read_json(z,'execution_specification.json')
        progress=read_json(z,'progress.json')
        aggregate_sha=hashlib.sha256(z.read('aggregate_results.json')).hexdigest()
        spec_sha=hashlib.sha256(z.read('execution_specification.json')).hexdigest()
        if (result['status']!='complete_four_arm_source_validation_development' or internal['status']!='passed'
            or internal['aggregate_results_sha256']!=aggregate_sha or progress['status']!=result['status']
            or progress['completed_arms']!=4 or result['source_outer_test_access'] or result['target_test_access']
            or [r['arm'] for r in result['arms']]!=cfg['arms']):
            raise ValueError('Require complete four-arm development results')
        if (spec['protocol']!=cfg or spec['manifest_sha256']!=bundle['manifest_sha256']
            or spec['runner_sha256']!=manifest['files']['run_adast_development_cuda.py']
            or any(spec[k] for k in ('source_outer_test_access','target_test_access','target_true_label_access'))
            or spec['backend']!='CUDA_float32_deterministic_no_tf32'
            or not 0<spec['max_seconds']<=18000 or result['elapsed_seconds']>spec['max_seconds']):
            raise ValueError('Frozen execution identity/budget differs')
        expected_initial=result['arms'][0]['initial_state_sha256']
        initial,initial_audit=initial_helpers.reconstruct_initial(upstream,model_cfg,cfg['seed'],expected_initial)
        rows=[]
        for arm in result['arms']:
            guard()
            key=arm['arm']
            arm_cfg={**cfg,**cfg['arm_overrides'][key]} if loss_ablation else cfg
            adaptive=key.startswith('adast_')
            full=key.endswith('full_source')
            identity={'arm':key,'fold':cfg['fold'],'execution_specification_sha256':spec_sha}
            history=arm['history']
            if (arm['initial_state_sha256']!=expected_initial or len(history)!=30 or arm['epochs_completed']!=30
                or read_json(z,key+'/identity.json')!={'identity':identity,'initial_state_sha256':expected_initial}
                or read_json(z,key+'/selection.json')!=arm):
                raise ValueError('Arm identity/history differs')
            init=torch.load(io.BytesIO(z.read(key+'/initial.pt')),map_location='cpu',weights_only=True)
            if init['identity']!=identity or init['initial_state_sha256']!=expected_initial:
                raise ValueError('Saved initialization identity differs')
            for name,model in initial.items():
                for k,v in model.state_dict().items():
                    if not torch.equal(v,init['models'][name][k]):
                        raise ValueError('Saved initialization does not replay exactly')
            seen=np.zeros(len(y),dtype=bool)
            target_seen=np.zeros(cfg['adaptation_epochs'],dtype=bool)
            for epoch,row in enumerate(history):
                guard()
                source,target=replay_sampling(cfg,epoch,full)
                seen[source]=True
                if adaptive:
                    target_seen[target]=True
                actual_updates=int(np.ceil(len(source)/cfg['batch_size']))
                if (row['global_epoch']!=epoch+1 or row['round']!=epoch//15 or row['epoch']!=epoch%15+1
                    or row['updates']!=actual_updates or row['source_presentations']!=len(source)
                    or row['source_unique_seen']!=int(seen.sum()) or row['target_unique_seen']!=int(target_seen.sum())
                    or row['target_training_presentations']!=(len(target) if adaptive else 0)
                    or row['source_order_sha256']!=hashlib.sha256(source.tobytes()).hexdigest()
                    or row['target_order_sha256']!=(hashlib.sha256(target.tobytes()).hexdigest() if adaptive else None)
                    or row['source_label_counts']!=np.bincount(y[source],minlength=5).tolist()):
                    raise ValueError('Epoch sampling/coverage/budget does not replay')
                lr=cfg['optimizer']['lr']*(.1 if epoch>=10 else 1.)
                if row['optimizer_lr_before_epoch']!=lr or row['optimizer_lr_after_epoch']!=cfg['optimizer']['lr']*(.1 if epoch>=9 else 1.):
                    raise ValueError('Learning-rate schedule differs')
                coefficients={'source_ce':arm_cfg['source_loss_weights_by_round'][epoch//15],
                    'similarity':arm_cfg['similarity_weight'],'adversarial':arm_cfg['adversarial_weight'] if adaptive else 0.,
                    'target_pseudo_ce':arm_cfg['target_loss_weights_by_round'][epoch//15] if adaptive else 0.}
                if row['loss_coefficients']!=coefficients or not np.isfinite(list(row['loss_batch_means'].values())).all():
                    raise ValueError('Loss coefficients/components differ')
                verify_loss_components(row['loss_batch_means'],coefficients,adaptive)
                if loss_ablation and arm_cfg['adversarial_weight']==0. and any(row['loss_batch_means'][n]!=0. for n in ('adversarial','discriminator')):
                    raise ValueError('Removed adversarial branch still has alignment loss')
                if adaptive:
                    pseudo=np.load(io.BytesIO(z.read(key+f'/pseudo_round_{epoch//15}.npy')),allow_pickle=False)
                    if pseudo.shape!=(cfg['adaptation_epochs'],) or not np.isin(pseudo,range(5)).all() or row['pseudo_label_counts_at_round_start']!=np.bincount(pseudo,minlength=5).tolist():
                        raise ValueError('Pseudo-label support differs')
                elif row['pseudo_label_counts_at_round_start'] is not None:
                    raise ValueError('Source-only has target training diagnostics')
                name=key+f'/validation_epoch_{epoch+1:03d}.npz'
                if hashlib.sha256(z.read(name)).hexdigest()!=row['validation_logits_sha256']:
                    raise ValueError('Validation logits identity differs')
                verify_logits(z,name,vy,row['validation'])
            verify_logits(z,key+'/validation_epoch_000.npz',vy,read_json(z,key+'/validation_before_training.json'))
            scores=[h['validation']['source']['macro_f1'] for h in history]
            best_epoch=int(np.argmax(scores))+1
            if (arm['best_epoch']!=best_epoch or arm['total_updates']!=sum(h['updates'] for h in history)
                or arm['best_source_validation']!=history[best_epoch-1]['validation']
                or arm['final_source_validation']!=history[-1]['validation']):
                raise ValueError('Source-validation checkpoint selection differs')
            if hashlib.sha256(z.read(key+'/latest.pt')).hexdigest()!=arm['final_checkpoint_sha256']:
                raise ValueError('Latest/final checkpoint hash differs')
            replays={}
            checkpoints={}
            for selection,epoch,sha in [('best',best_epoch,arm['best_checkpoint_sha256']),('final',30,arm['final_checkpoint_sha256'])]:
                data=z.read(key+'/'+selection+'.pt')
                if hashlib.sha256(data).hexdigest()!=sha:
                    raise ValueError('Checkpoint hash differs')
                payload=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
                checkpoints[selection]=payload
                if (payload['identity']!=identity or payload['initial_state_sha256']!=expected_initial
                    or payload['epochs_completed']!=epoch or payload['history']!=history[:epoch]
                    or not payload['cuda_rng_state'] or 'optimizer' not in payload or 'disc_optimizer' not in payload):
                    raise ValueError('Checkpoint identity/history/RNG differs')
                models=reference.build_models(upstream,model_cfg,cfg['seed'],'cpu')
                for n,m in models.items():
                    m.load_state_dict(payload['models'][n],strict=True)
                expected=history[epoch-1]['validation']
                replay={}
                for domain in ('source','target'):
                    guard()
                    logits=reference.predict(models,vx,domain,'cpu',guard,batch_size=cfg['batch_size'])
                    replay[domain]=metrics_from_confusion(confusion_matrix_5(vy,logits.argmax(1)))
                    if replay[domain]!=expected[domain]:
                        raise ValueError('CPU checkpoint confusion/metrics differ; inspect numerical disagreement')
                replays[selection]=replay
                if not adaptive:
                    for n in ('discriminator','target_attention'):
                        if any(not torch.equal(v,init['models'][n][k]) for k,v in payload['models'][n].items()):
                            raise ValueError('Source-only unused module changed')
                if loss_ablation and arm_cfg['adversarial_weight']==0.:
                    if payload['disc_optimizer']['state'] or any(not torch.equal(v,init['models']['discriminator'][k]) for k,v in payload['models']['discriminator'].items()):
                        raise ValueError('Removed adversarial branch still trains discriminator')
            for n,state in checkpoints['best']['models'].items():
                if any(not torch.equal(v,checkpoints['final']['best_models'][n][k]) for k,v in state.items()):
                    raise ValueError('Best checkpoint differs from retained final best state')
            saved_seen=np.unpackbits(np.frombuffer(bytes.fromhex(checkpoints['final']['source_seen_packed_hex']),dtype=np.uint8))[:len(y)]
            np.testing.assert_array_equal(saved_seen,seen)
            saved_target_seen=np.unpackbits(np.frombuffer(bytes.fromhex(checkpoints['final']['target_seen_packed_hex']),dtype=np.uint8))[:cfg['adaptation_epochs']]
            np.testing.assert_array_equal(saved_target_seen,target_seen)
            rows.append({'arm':key,'best_epoch':best_epoch,'total_updates':arm['total_updates'],
                         'CPU_checkpoint_validation_replays':replays})
        for a,b in ((0,1),(2,3)):
            if [h['source_order_sha256'] for h in result['arms'][a]['history']]!=[h['source_order_sha256'] for h in result['arms'][b]['history']]:
                raise ValueError('Paired sample orders differ')
        if loss_ablation:
            # These two changes activate only in round two, so the saved logits must match the fresh reference up to epoch 15.
            baseline=result['arms'][0]
            for other in result['arms'][1:3]:
                for epoch in range(16):
                    name=f'/validation_epoch_{epoch:03d}.npz'
                    if z.read(baseline['arm']+name)!=z.read(other['arm']+name):
                        raise ValueError('Round-one reference/keep-source/no-pseudo prefix differs')
        output=args.output.resolve()
        if not output.is_relative_to((ROOT/'runs').resolve()) or output==(ROOT/'runs').resolve() or output.exists():
            raise ValueError('Require new private runs subdirectory')
        output.mkdir(parents=True)
        for name in z.namelist():
            target=output/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(name))
    proof={'status':'passed','archive_sha256':args.expected_sha256,'aggregate_results_sha256':aggregate_sha,
           'initialization_replay':initial_audit,'all_124_validation_diagnostics_recomputed':True,
           'all_eight_best_final_checkpoints_CPU_validation_replayed':True,'sampling_coverage_and_budget_replayed':True,
           'source_outer_test_access':False,'target_test_access':False,'arms':rows,'elapsed_seconds':time.monotonic()-started}
    if loss_ablation:
        proof.update(single_factor_overrides_verified=True,removed_discriminator_unchanged=True,round_one_shared_prefix_verified=True)
    reference.write_once(output/'independent_verification.json',proof)
    print(json.dumps(proof,indent=2),flush=True)


def verify_loss_components(values,coefficients,adaptive):
    if set(values)!={'total','source_ce','similarity','adversarial','target_pseudo_ce','discriminator'}:
        raise ValueError('Loss component schema differs')
    reconstructed=sum(coefficients[k]*values[k] for k in coefficients)
    if not np.isclose(values['total'],reconstructed,rtol=1e-5,atol=1e-6):
        raise ValueError('Total loss does not reconstruct from components')
    if not adaptive and any(values[k]!=0. for k in ('adversarial','target_pseudo_ce','discriminator')):
        raise ValueError('Source-only has nonzero adaptation loss')


def verify_logits(z,name,labels,expected):
    with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as values:
        if set(values.files)!={'source','target'}:
            raise ValueError('Validation logits schema differs')
        for domain in values.files:
            logits=values[domain]
            if logits.shape!=(len(labels),5) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
                raise ValueError('Validation logits support differs')
            actual=metrics_from_confusion(confusion_matrix_5(labels,logits.argmax(1)))
            if actual!=expected[domain]:
                raise ValueError('Validation metrics do not reconstruct from logits')


def read_local_json(path):
    return json.loads(path.read_bytes())


def validate_ablation_overrides(cfg):
    # Deliberately independent of the training module's configuration registry.
    expected={
        'adast_reference_full_source':{},
        'adast_keep_source_ce_full_source':{'source_loss_weights_by_round':[1.,1.]},
        'adast_no_pseudo_full_source':{'target_loss_weights_by_round':[0.,0.]},
        'adast_no_adversarial_full_source':{'adversarial_weight':0.},
    }
    if cfg['arms']!=list(expected) or cfg['arm_overrides']!=expected:
        raise ValueError('Single-factor protocol differs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--input-base',type=Path,default=BASE)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--output',type=Path,default=BASE/'verified_results')
    parser.add_argument('--max-seconds',type=float,default=1800)
    verify(parser.parse_args())
