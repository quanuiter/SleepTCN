"""Four CUDA budget-development arms, source validation only; no automatic extension."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
import torch
from torch import nn

# The builder copies these exact historical files without editing them.
import adast_reference_adapter as reference
from development_metrics import confusion_matrix_5, metrics_from_confusion


def digest_array(value):
    return hashlib.sha256(np.asarray(value, dtype=np.int64).tobytes()).hexdigest()


def epoch_orders(train_size, target_size, seed, global_epoch, batch_size, limited_steps, budget):
    if budget not in {'limited', 'full_source'} or min(train_size,target_size,batch_size,limited_steps) <= 0:
        raise ValueError('Invalid sampling parameters')
    source = np.random.default_rng(seed + global_epoch).permutation(train_size).astype(np.int64)
    target_base = np.random.default_rng(seed + 10000 + global_epoch).permutation(target_size).astype(np.int64)
    target_base = target_base[:limited_steps * batch_size]
    if len(target_base) != limited_steps * batch_size:
        raise ValueError('Target support cannot satisfy historical limited schedule')
    if budget == 'limited':
        source = source[:limited_steps * batch_size]
        if len(source) != limited_steps * batch_size:
            raise ValueError('Source support cannot satisfy limited schedule')
    target = np.resize(target_base, len(source))
    return source, target


def measured_update(models, optimizer, disc_optimizer, utils, sx, sy, tx, pseudo, round_index, cfg, arm):
    """Same computation/RNG order as the frozen adapter; expose individual losses."""
    for model in models.values():
        model.train()
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    sf, sl1, sl2 = reference.logits(models, sx, 'source')
    source_loss = ce(sl1, sy) + ce(sl2, sy)
    similarity = utils.calc_similiar_penalty(models['head1'], models['head2'])
    loss = cfg['source_loss_weights_by_round'][round_index] * source_loss + cfg['similarity_weight'] * similarity
    adversarial = target_loss = disc_loss = torch.zeros((),device=sx.device)
    if arm == 'adast':
        tf, tl1, tl2 = reference.logits(models, tx, 'target')
        discriminator = models['discriminator']
        for param in discriminator.parameters():
            param.requires_grad = True
        disc_output = discriminator(torch.cat([sf,tf]).detach()).squeeze(-1)
        labels = torch.cat([torch.ones(len(sf),device=sx.device),torch.zeros(len(tf),device=sx.device)])
        disc_loss = bce(disc_output,labels)
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        for param in discriminator.parameters():
            param.requires_grad = False
        fake_output = torch.cat([discriminator(tf).squeeze(-1),discriminator(sf).squeeze(-1)])
        fake_labels = torch.cat([torch.ones(len(tf),device=sx.device),torch.zeros(len(sf),device=sx.device)])
        adversarial = bce(fake_output,fake_labels)
        target_loss = ce(tl1,pseudo) + ce(tl2,pseudo)
        loss = loss + cfg['adversarial_weight'] * adversarial + cfg['target_loss_weights_by_round'][round_index] * target_loss
    optimizer.zero_grad()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                      for model in models.values() for p in model.parameters()):
        raise ValueError('Nonfinite loss/gradient')
    optimizer.step()
    return {name:float(value.detach()) for name,value in
            {'total':loss,'source_ce':source_loss,'similarity':similarity,'adversarial':adversarial,
             'target_pseudo_ce':target_loss,'discriminator':disc_loss}.items()}


@torch.inference_mode()
def validation(models, x, y, device, guard, batch_size=128):
    for model in models.values():
        model.eval()
    outputs = {'source':[], 'target':[]}
    for start in range(0,len(x),batch_size):
        guard()
        batch = torch.from_numpy(np.asarray(x[start:start+batch_size],dtype=np.float32).copy()).unsqueeze(1).to(device)
        features = models['encoder'](batch)
        for domain in outputs:
            f = models[domain+'_attention'](features)
            value = torch.maximum(models['head1'](f),models['head2'](f))
            outputs[domain].append(value.cpu().numpy())
    outputs = {k:np.concatenate(v) for k,v in outputs.items()}
    metrics = {k:metrics_from_confusion(confusion_matrix_5(np.asarray(y),v.argmax(1))) for k,v in outputs.items()}
    return metrics, outputs


def save_checkpoint(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload,temp)
    temp.replace(path)


def train_development_arm(key, models, utils, arrays, cfg, folder, identity, guard, device):
    if folder.exists():
        raise FileExistsError('Fresh development only; preserve old/partial arm, no automatic resume')
    folder.mkdir()
    arm = 'adast' if key.startswith('adast_') else 'source_only'
    budget = 'full_source' if key.endswith('full_source') else 'limited'
    x,y,vx,vy,target = arrays
    initial = reference.state_digest(models)
    reference.write_once(folder/'identity.json',{'identity':identity,'initial_state_sha256':initial})
    save_checkpoint(folder/'initial.pt',{'models':{n:m.state_dict() for n,m in models.items()},
                                       'identity':identity,'initial_state_sha256':initial})
    opt = cfg['optimizer']
    names = ['encoder','head1','head2','source_attention','target_attention']
    optimizer = torch.optim.Adam([p for n in names for p in models[n].parameters()],lr=opt['lr'],
                                 betas=tuple(opt['betas']),weight_decay=opt['weight_decay'])
    disc_optimizer = torch.optim.Adam(models['discriminator'].parameters(),lr=opt['lr'],
                                      betas=tuple(opt['betas']),weight_decay=opt['weight_decay'])
    history, best, best_epoch, best_state = [], -float('inf'), None, None
    source_seen, target_seen = np.zeros(len(x),dtype=bool),np.zeros(len(target),dtype=bool)
    before, before_logits = validation(models,vx,vy,device,guard,cfg['batch_size'])
    np.savez_compressed(folder/'validation_epoch_000.npz',**before_logits)
    reference.write_once(folder/'validation_before_training.json',before)
    for round_index in range(cfg['rounds']):
        if arm == 'adast':
            pseudo = reference.predict(models,target,'target',device,guard,pseudo=True,batch_size=cfg['batch_size'])
            np.save(folder/f'pseudo_round_{round_index}.npy',pseudo,allow_pickle=False)
            pseudo_counts = np.bincount(pseudo,minlength=5).tolist()
        else:
            pseudo_counts = None
        for epoch in range(cfg['epochs_per_round']):
            global_epoch = round_index*cfg['epochs_per_round']+epoch
            guard()
            guard.publish(arm=key,phase='training',epochs_completed=len(history))
            tick = time.perf_counter()
            source_order,target_order = epoch_orders(len(x),len(target),cfg['seed'],global_epoch,
                cfg['batch_size'],cfg['limited_steps_per_epoch'],budget)
            lr_before = optimizer.param_groups[0]['lr']
            losses = []
            for start in range(0,len(source_order),cfg['batch_size']):
                guard()
                si,ti = source_order[start:start+cfg['batch_size']],target_order[start:start+cfg['batch_size']]
                sx = torch.from_numpy(np.asarray(x[si],dtype=np.float32).copy()).unsqueeze(1).to(device)
                sy = torch.from_numpy(np.asarray(y[si],dtype=np.int64).copy()).to(device)
                tx = torch.from_numpy(np.asarray(target[ti],dtype=np.float32).copy()).unsqueeze(1).to(device) if arm=='adast' else None
                py = torch.from_numpy(pseudo[ti].astype(np.int64)).to(device) if arm=='adast' else None
                losses.append(measured_update(models,optimizer,disc_optimizer,utils,sx,sy,tx,py,round_index,cfg,arm))
            if round_index == 0 and (epoch+1)%10 == 0:
                for group in optimizer.param_groups:
                    group['lr'] *= .1
            source_seen[source_order] = True
            if arm=='adast':
                target_seen[target_order] = True
            guard.publish(phase='source_validation')
            metrics, outputs = validation(models,vx,vy,device,guard,cfg['batch_size'])
            logits_path = folder/f'validation_epoch_{global_epoch+1:03d}.npz'
            np.savez_compressed(logits_path,**outputs)
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            row = {'global_epoch':global_epoch+1,'round':round_index,'epoch':epoch+1,'updates':len(losses),
                   'source_presentations':len(source_order),'source_unique_seen':int(source_seen.sum()),
                   'target_training_presentations':len(target_order) if arm=='adast' else 0,
                   'target_unique_seen':int(target_seen.sum()),'source_order_sha256':digest_array(source_order),
                   'target_order_sha256':digest_array(target_order) if arm=='adast' else None,
                   'source_label_counts':np.bincount(y[source_order],minlength=5).tolist(),
                   'pseudo_label_counts_at_round_start':pseudo_counts,
                   'loss_batch_means':{k:float(np.mean([v[k] for v in losses])) for k in losses[0]},
                   'loss_coefficients':{'source_ce':cfg['source_loss_weights_by_round'][round_index],
                       'similarity':cfg['similarity_weight'],'adversarial':cfg['adversarial_weight'] if arm=='adast' else 0.,
                       'target_pseudo_ce':cfg['target_loss_weights_by_round'][round_index] if arm=='adast' else 0.},
                   'optimizer_lr_before_epoch':lr_before,'optimizer_lr_after_epoch':optimizer.param_groups[0]['lr'],
                   'validation':metrics,'validation_logits_sha256':reference.digest(logits_path),
                   'seconds':time.perf_counter()-tick}
            history.append(row)
            if metrics['source']['macro_f1'] > best:
                best, best_epoch = metrics['source']['macro_f1'],global_epoch+1
                best_state = copy.deepcopy({n:m.state_dict() for n,m in models.items()})
            payload = {'models':{n:m.state_dict() for n,m in models.items()},'optimizer':optimizer.state_dict(),
                       'disc_optimizer':disc_optimizer.state_dict(),'rng_state':torch.get_rng_state(),
                       'cuda_rng_state':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                       'identity':identity,'initial_state_sha256':initial,'epochs_completed':len(history),
                       'history':history,'best_epoch':best_epoch,'best_score':best,'best_models':best_state,
                       'source_seen_packed_hex':np.packbits(source_seen).tobytes().hex(),
                       'target_seen_packed_hex':np.packbits(target_seen).tobytes().hex()}
            save_checkpoint(folder/'latest.pt',payload)
            if best_epoch == global_epoch+1:
                save_checkpoint(folder/'best.pt',payload)
            recall = metrics['source']['per_class']['N1']['recall']
            guard.publish(phase='epoch_saved',epochs_completed=len(history),source_validation_macro_f1=metrics['source']['macro_f1'],
                          source_validation_N1_recall=recall,epoch_seconds=row['seconds'])
            print(f"{key}: epoch {len(history)}/30 val={metrics['source']['macro_f1']:.5f} N1={recall:.4f} N3={metrics['source']['per_class']['N3']['recall']:.4f} updates={len(losses)} {row['seconds']:.2f}s",flush=True)
    shutil.copyfile(folder/'latest.pt',folder/'final.pt')
    result = {'arm':key,'initial_state_sha256':initial,'epochs_completed':len(history),'best_epoch':best_epoch,
              'best_source_validation':history[best_epoch-1]['validation'],'final_source_validation':history[-1]['validation'],
              'best_checkpoint_sha256':reference.digest(folder/'best.pt'),
              'final_checkpoint_sha256':reference.digest(folder/'final.pt'),'history':history,
              'total_updates':sum(h['updates'] for h in history),'training_and_validation_seconds':sum(h['seconds'] for h in history)}
    reference.write_once(folder/'selection.json',result)
    return result


def export(output):
    paths = sorted(p for p in output.rglob('*') if p.is_file() and p.suffix in {'.pt','.json','.npz','.npy','.log'})
    archive = output.parent/'SleepTCN_ADAST_Development_Results_20261004.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in paths:
            z.write(p,p.relative_to(output).as_posix())
        z.writestr('export_manifest.json',json.dumps({p.relative_to(output).as_posix():reference.digest(p) for p in paths},indent=2))
    print('RESULT_ARCHIVE',archive,'SHA256',reference.digest(archive),flush=True)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refuse CPU training fallback')
    work = Path(__file__).resolve().parent
    manifest = json.loads((work/'manifest.json').read_bytes())
    if any(manifest[k] for k in ('source_outer_test_included','target_test_data_included','target_labels_included','participant_ids_included')):
        raise ValueError('Forbidden data role')
    for name,sha in manifest['files'].items():
        if reference.digest(work/name) != sha:
            raise ValueError('Frozen payload changed')
    cfg = json.loads((work/'protocol.json').read_bytes())
    if not 0 < args.max_seconds <= cfg['max_seconds']:
        raise ValueError('Require at most five-hour total attempt')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    arrays = [np.load(work/'data'/f'{n}.npy',mmap_mode='r',allow_pickle=False)
              for n in ('train_x','train_y','validation_x','validation_y','adaptation_x')]
    for array,size,label in zip(arrays,[cfg['source_train_epochs']]*2+[cfg['source_validation_epochs']]*2+[cfg['adaptation_epochs']],
                                [False,True,False,True,False]):
        if array.shape != ((size,) if label else (size,3000)) or array.dtype != (np.int64 if label else np.float32):
            raise ValueError('Input schema/support differs')
        if not (np.isin(array,range(5)).all() if label else np.isfinite(array).all()):
            raise ValueError('Invalid labels or signal')
    for n,y in [('train',arrays[1]),('validation',arrays[3])]:
        if np.bincount(y,minlength=5).tolist()!=manifest[n+'_class_counts']:
            raise ValueError('Source class support differs')
    if int(np.ceil(len(arrays[0])/cfg['batch_size'])) != cfg['full_source_steps_per_epoch']:
        raise ValueError('Full-source step count differs')
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve existing attempt; no automatic overwrite/resume')
    output.mkdir(parents=True)
    module=reference.load_module('adast_dev_models',work/'upstream/models.py')
    base_cfg=reference.load_module('adast_dev_config',work/'upstream/configs.py').Config()
    utils=reference.load_module('adast_dev_utils',work/'upstream/utils.py')
    spec={'protocol':cfg,'manifest_sha256':reference.digest(work/'manifest.json'),
          'runner_sha256':reference.digest(Path(__file__)),'torch':str(torch.__version__),
          'gpu':torch.cuda.get_device_name(0),'backend':'CUDA_float32_deterministic_no_tf32',
          'max_seconds':args.max_seconds,'source_outer_test_access':False,'target_test_access':False,'target_true_label_access':False}
    reference.write_once(output/'execution_specification.json',spec)
    guard=reference.Guard(output,args.max_seconds)
    rows=[]
    try:
        for key in cfg['arms']:
            models=reference.build_models(module,base_cfg,cfg['seed'],'cuda')
            identity={'arm':key,'fold':cfg['fold'],'execution_specification_sha256':reference.digest(output/'execution_specification.json')}
            rows.append(train_development_arm(key,models,utils,arrays,cfg,output/key,identity,guard,'cuda'))
            del models
            torch.cuda.empty_cache()
            guard.publish(completed_arms=len(rows))
        if len({r['initial_state_sha256'] for r in rows})!=1:
            raise ValueError('All four initial states must match')
        for left,right in ((0,1),(2,3)):
            if [h['source_order_sha256'] for h in rows[left]['history']] != [h['source_order_sha256'] for h in rows[right]['history']]:
                raise ValueError('Paired source orders differ')
        result={'status':'complete_four_arm_source_validation_development','arms':rows,
                'elapsed_seconds':time.monotonic()-guard.started,'source_outer_test_access':False,'target_test_access':False}
        reference.write_once(output/'aggregate_results.json',result)
        reference.write_once(output/'verification.json',{'status':'passed','matched_initial_states_and_paired_orders':True,
            'aggregate_results_sha256':reference.digest(output/'aggregate_results.json')})
        guard.publish(status='complete_four_arm_source_validation_development',phase='complete')
    except Exception as error:
        guard.publish(status='stopped_resource_budget' if isinstance(error,reference.BudgetStop) else 'failed',
                      reason=str(error),checkpoints_retained=True)
        raise
    finally:
        export(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=18000)
    main(parser.parse_args())
