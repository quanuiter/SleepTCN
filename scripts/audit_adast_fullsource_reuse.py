"""Read-only preflight of existing full-source pairs; never train or upload."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'runs/adast_fullsource_preflight_20261006'
OLD = ROOT / 'runs/colab_adast_20261004/payload'
KEYS = ('seed','batch_size','rounds','epochs_per_round',
        'source_loss_weights_by_round','target_loss_weights_by_round',
        'adversarial_weight','similarity_weight','optimizer')

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def roles_valid(roles, size):
    require(set(roles) == {'train','validation','test'}, 'Missing/extra role')
    for indices in roles.values():
        require(indices.ndim == 1 and np.issubdtype(indices.dtype,np.integer), 'Role index type')
        require(len(indices)>0 and indices.min()>=0 and indices.max()<size, 'Role index range')
        require(len(np.unique(indices))==len(indices), 'Duplicate role index')
    joined=np.concatenate(list(roles.values()))
    require(len(joined)==size and len(np.unique(joined))==size, 'Roles overlap or fail coverage')

def compare_subset(full, indices, subset):
    require(subset.shape==(len(indices),*full.shape[1:]) and subset.dtype==full.dtype,
            'Subset schema differs')
    for start in range(0,len(indices),2048):
        require(np.array_equal(full[indices[start:start+2048]],subset[start:start+2048]),
                'Sanitized subset differs from original fold data')

def compatible_configs(left,right):
    for key in KEYS:
        require(left[key]==right[key], 'Training setting differs: '+key)

def validate_pair(rows, cfg):
    require(len(rows)==2, 'Two arms required')
    require(rows[0]['initial_state_sha256']==rows[1]['initial_state_sha256'], 'Initialization differs')
    require([h['source_order_sha256'] for h in rows[0]['history']]==
            [h['source_order_sha256'] for h in rows[1]['history']], 'Source order differs')
    steps=math.ceil(cfg['source_train_epochs']/cfg['batch_size'])
    for row in rows:
        require(row['epochs_completed']==30 and len(row['history'])==30, 'Incomplete training')
        require(row['total_updates']==30*steps, 'Update budget differs')
        best=max(range(30),key=lambda i:row['history'][i]['validation']['source']['macro_f1'])+1
        require(row['best_epoch']==best, 'Checkpoint selection differs')
        for i,h in enumerate(row['history']):
            require(h['global_epoch']==i+1 and h['updates']==steps, 'Epoch/update mismatch')
            require(h['source_presentations']==cfg['source_train_epochs'], 'Partial source pass')
            require(h['source_unique_seen']==cfg['source_train_epochs'], 'Source coverage differs')
            expected_lr=.001 if i<10 else .0001
            require(math.isclose(h['optimizer_lr_before_epoch'],expected_lr), 'LR schedule differs')
            require(h['loss_coefficients']['source_ce']==(1. if i<15 else .1), 'Source loss differs')
            if row is rows[0]:
                require(h['target_training_presentations']==0, 'Source-only consumes target')

def main():
    started=time.monotonic()
    require(not (RECORD/'audit.json').exists(), 'Preserve existing completed audit')
    RECORD.mkdir(parents=True,exist_ok=True)
    old_manifest=read(OLD/'manifest.json')
    old_bundle=read(OLD.parent/'bundle_verification.json')
    require(sha(OLD/'manifest.json')==old_bundle['manifest_sha256'], 'Old manifest identity')
    old_cfg=read(OLD/'protocol.json')
    # Hash only relevant input/code files, not unrelated historical runs.
    checked={}
    def checked_file(path,expected):
        if path not in checked:
            checked[path]=sha(path)
        require(checked[path]==expected, 'File identity differs: '+path.name)
    for name in ['data/source_x.npy','data/source_y.npy','data/adaptation_x.npy']+[
            f'data/fold_{f:02d}_roles.npz' for f in range(10)]:
        checked_file(OLD/name,old_manifest['files'][name])
    x=np.load(OLD/'data/source_x.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(OLD/'data/source_y.npy',mmap_mode='r',allow_pickle=False)
    require(x.shape==(195469,3000) and x.dtype==np.float32 and y.dtype==np.int64,'Source schema')
    folds={}
    test_indices=[]
    for fold in range(10):
        with np.load(OLD/f'data/fold_{fold:02d}_roles.npz',allow_pickle=False) as z:
            roles={key:z[key] for key in ('train','validation','test')}
        roles_valid(roles,len(y))
        folds[fold]=roles
        test_indices.append(roles['test'])
    outer=np.concatenate(test_indices)
    require(len(outer)==len(y) and len(np.unique(outer))==len(y),'Outer-test coverage differs')
    pointer=read(ROOT/'runs/adast_small_confirmation_20261004/verified_results_pointer.json')
    confirmation=Path(pointer['results_root'])
    sources=[
        (0,ROOT/'runs/adast_development_20261004/verified_results',
         ROOT/'runs/adast_development_20261004/payload',
         ['source_only_full_source','adast_full_source'],
         'ff5ef23108fa3116ae4b8ebae3ebe945e2c6ee026f2c3ee615429ed8e52c9a99'),
        (1,confirmation,confirmation.parent/'payload',
         ['source_only_full_source','adast_reference_full_source'],
         '579ce087700074764044f13823829803003f75ab1986cc58dd0689b67785bf53')]
    reports=[]
    private=[]
    runners=[]
    for fold,result,payload,arms,digest in sources:
        require(sha(result/'aggregate_results.json')==digest,'Aggregate identity differs')
        for name in ('verification.json','independent_verification.json'):
            proof=read(result/name)
            require(proof['status']=='passed' and proof['aggregate_results_sha256']==digest,
                    'Previous verification differs')
        require(proof['sampling_coverage_and_budget_replayed'],'Sampling replay absent')
        spec=read(result/'execution_specification.json')
        cfg=spec['protocol']
        compatible_configs(old_cfg,cfg)
        require(spec['backend']=='CUDA_float32_deterministic_no_tf32','Backend differs')
        require(spec['torch']=='2.11.0+cu130','Torch version differs')
        require(not any(spec[k] for k in ('source_outer_test_access','target_test_access',
                                          'target_true_label_access')),'Forbidden training data role')
        require(sha(payload/'manifest.json')==spec['manifest_sha256'],'Payload identity differs')
        manifest=read(payload/'manifest.json')
        require(not any(manifest[k] for k in ('source_outer_test_included','target_test_data_included',
                                              'target_labels_included','participant_ids_included')),
                'Forbidden payload data role')
        for name,d in manifest['files'].items():
            checked_file(payload/name,d)
        require(read(payload/'protocol.json')==cfg,'Executed protocol differs')
        runners.append(manifest['files']['run_adast_development_cuda.py'])
        require(manifest['files']['data/adaptation_x.npy']==old_manifest['files']['data/adaptation_x.npy'],
                'Adaptation input differs')
        for key in ('models.py','configs.py','utils.py'):
            name='upstream/'+key
            require(manifest['files'][name]==old_manifest['files'][name],'Upstream implementation differs')
        for role in ('train','validation'):
            for suffix,full in [('x',x),('y',y)]:
                subset=np.load(payload/f'data/{role}_{suffix}.npy',mmap_mode='r',allow_pickle=False)
                compare_subset(full,folds[fold][role],subset)
        aggregate=read(result/'aggregate_results.json')
        rows=[next(a for a in aggregate['arms'] if a['arm']==name) for name in arms]
        validate_pair(rows,cfg)
        cps=[]
        for row in rows:
            for kind in ('best','final'):
                path=result/row['arm']/(kind+'.pt')
                digest_cp=row[kind+'_checkpoint_sha256']
                checked_file(path,digest_cp)
                cps.append({'arm':row['arm'],'kind':kind,'sha256':digest_cp,
                            'epoch':row['best_epoch'] if kind=='best' else 30})
                private.append({'fold':fold,'path':path.as_posix(),'sha256':digest_cp,
                                'arm':row['arm'],'kind':kind})
        reports.append({'fold':fold,'eligible_for_fullsource_pair_reuse':True,
                        'source_train_epochs':cfg['source_train_epochs'],
                        'source_validation_epochs':cfg['source_validation_epochs'],
                        'updates_per_model':rows[0]['total_updates'],
                        'seconds_by_arm':{r['arm']:r['training_and_validation_seconds'] for r in rows},
                        'best_epochs':{r['arm']:r['best_epoch'] for r in rows},
                        'checkpoints':cps,'aggregate_sha256':digest})
        print(f'Fold {fold}: verified two existing full-source models and four best/final files.',flush=True)
    require(len(set(runners))==1,'Base runner differs across folds')
    require(sha(ROOT/'scripts/run_adast_development_cuda.py')==runners[0],'Current base runner changed')
    # Use the slower observed pair rate per update; include an explicit 20% scheduling reserve.
    rates=[sum(r['seconds_by_arm'].values())/r['updates_per_model'] for r in reports]
    remaining=[{'fold':f,'train_epochs':len(folds[f]['train']),
                'validation_epochs':len(folds[f]['validation']),
                'updates_per_model':30*math.ceil(len(folds[f]['train'])/128)} for f in range(2,10)]
    projected=max(rates)*sum(f['updates_per_model'] for f in remaining)
    result={'status':'passed','action':'preflight_only_no_training_inference_or_upload',
            'reusable_models':4,'reusable_best_final_files':8,'new_models_needed':16,
            'folds':reports,'remaining_folds':remaining,'common_base_runner_sha256':runners[0],
            'source_roles_and_sanitized_arrays_match':True,'hash_checked_files':len(checked),
            'projected_training_validation_seconds':projected,
            'projected_seconds_with_20_percent_reserve':projected*1.2,
            'training_limit_seconds':18000,'projection_fits_limit':projected*1.2<=18000,
            'storage_free_bytes':{d:shutil.disk_usage(d+':/').free for d in ('C','D')},
            'pending':['runner_for_remaining_folds_and_tests','target_account_upload_consent',
                       'runtime_GPU_and_payload_storage_preflight'],
            'elapsed_seconds':time.monotonic()-started}
    (RECORD/'checkpoint_inventory.private.json').write_text(json.dumps(private,indent=2)+'\n',encoding='utf8')
    (RECORD/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('folds','remaining_folds')},indent=2))

if __name__=='__main__':
    main()
