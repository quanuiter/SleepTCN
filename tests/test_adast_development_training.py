import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


reference=load('adast_reference_adapter',ROOT/'scripts/run_colab_adast_training.py')
load('development_metrics',ROOT/'src/sleeptcn/metrics.py')
new=load('adast_development_training_test',ROOT/'scripts/run_adast_development_cuda.py')


def test_epoch_sampling_replays_limited_and_covers_full_source():
    a,t=new.epoch_orders(157200,4989,123,0,128,38,'limited')
    expected=np.random.default_rng(123).permutation(157200)[:4864]
    np.testing.assert_array_equal(a,expected)
    np.testing.assert_array_equal(t,np.random.default_rng(10123).permutation(4989)[:4864])
    full,cycled=new.epoch_orders(157200,4989,123,0,128,38,'full_source')
    np.testing.assert_array_equal(full[:len(a)],a)
    np.testing.assert_array_equal(np.sort(full),np.arange(157200))
    assert len(full)==len(cycled) and len(full)%128==16
    np.testing.assert_array_equal(cycled[:4864],t)
    np.testing.assert_array_equal(cycled[4864:9728],t)


@pytest.mark.parametrize('arm',['source_only','adast'])
def test_instrumented_update_exactly_matches_historical_adapter(arm):
    torch.set_num_threads(1)
    upstream=ROOT/'runs/colab_adast_20261004/payload/upstream'
    module=load('dev_test_upstream_models',upstream/'models.py')
    cfg_base=load('dev_test_upstream_cfg',upstream/'configs.py').Config()
    utils=load('dev_test_upstream_utils',upstream/'utils.py')
    cfg=json.loads((ROOT/'configs/adast_development_budget_v1_20261004.json').read_bytes())
    left=reference.build_models(module,cfg_base,123,'cpu')
    right=reference.build_models(module,cfg_base,123,'cpu')
    params=cfg['optimizer']
    def optim(models):
        names=['encoder','head1','head2','source_attention','target_attention']
        return (torch.optim.Adam([p for n in names for p in models[n].parameters()],lr=params['lr'],betas=tuple(params['betas']),weight_decay=params['weight_decay']),
                torch.optim.Adam(models['discriminator'].parameters(),lr=params['lr'],betas=tuple(params['betas']),weight_decay=params['weight_decay']))
    lo,ld=optim(left)
    ro,rd=optim(right)
    sx,tx=torch.randn(4,1,3000),torch.randn(4,1,3000)
    sy,py=torch.arange(4),torch.arange(4)
    if arm=='source_only':
        def forbidden(*args):
            raise AssertionError('No target forward during source-only training')
        right['target_attention'].register_forward_pre_hook(forbidden)
    torch.manual_seed(88)
    old_loss=reference.update(left,lo,ld,utils,sx,sy,tx,py,1,cfg,arm)
    torch.manual_seed(88)
    measured=new.measured_update(right,ro,rd,utils,sx,sy,tx,py,1,cfg,arm)
    assert old_loss==measured['total']
    assert reference.state_digest(left)==reference.state_digest(right)
    assert np.isfinite(list(measured.values())).all()


def test_refuses_cpu_training_before_output_created(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA required'):
        new.main(type('Args',(),{'output':tmp_path/'out','max_seconds':18000})())
    assert not (tmp_path/'out').exists()


def test_tiny_train_saves_epoch_metrics_best_final_rng_and_never_overwrites(tmp_path):
    torch.set_num_threads(1)
    upstream=ROOT/'runs/colab_adast_20261004/payload/upstream'
    module=load('tiny_dev_models',upstream/'models.py')
    cfg_base=load('tiny_dev_cfg',upstream/'configs.py').Config()
    utils=load('tiny_dev_utils',upstream/'utils.py')
    cfg=json.loads((ROOT/'configs/adast_development_budget_v1_20261004.json').read_bytes())
    cfg.update(rounds=2,epochs_per_round=1,batch_size=4,limited_steps_per_epoch=1)
    x=np.random.default_rng(1).normal(size=(8,3000)).astype(np.float32)
    y=np.arange(8,dtype=np.int64)%5
    arrays=[x,y,x,y,x[:4]]
    guard=reference.Guard(tmp_path,30)
    for key in ('source_only_limited','adast_limited','source_only_full_source','adast_full_source'):
        models=reference.build_models(module,cfg_base,123,'cpu')
        result=new.train_development_arm(key,models,utils,arrays,cfg,tmp_path/key,{'arm':key},guard,'cpu')
        assert result['epochs_completed']==2
        assert result['total_updates']==(4 if key.endswith('full_source') else 2)
        latest=torch.load(tmp_path/key/'latest.pt',weights_only=True)
        assert latest['epochs_completed']==2 and latest['rng_state'].numel()>0
        assert latest['best_epoch']==result['best_epoch']
        assert new.reference.digest(tmp_path/key/'final.pt')==new.reference.digest(tmp_path/key/'latest.pt')
        assert len(latest['history'][0]['validation']['source']['per_class'])==5
        assert latest['history'][-1]['source_unique_seen']==(8 if key.endswith('full_source') else int(np.unpackbits(np.frombuffer(bytes.fromhex(latest['source_seen_packed_hex']),dtype=np.uint8))[:8].sum()))
        with pytest.raises(FileExistsError):
            new.train_development_arm(key,models,utils,arrays,cfg,tmp_path/key,{'arm':key},guard,'cpu')


def test_main_cuda_preflight_reaches_training_boundary_and_exports_safe_stop(tmp_path,monkeypatch):
    work=tmp_path/'work'
    work.mkdir()
    (work/'data').mkdir()
    upstream=ROOT/'runs/colab_adast_20261004/payload/upstream'
    import shutil
    shutil.copytree(upstream,work/'upstream')
    cfg=json.loads((ROOT/'configs/adast_development_budget_v1_20261004.json').read_bytes())
    cfg.update(source_train_epochs=8,source_validation_epochs=8,adaptation_epochs=8,batch_size=4,full_source_steps_per_epoch=2)
    (work/'protocol.json').write_text(json.dumps(cfg))
    (work/'run_adast_development_cuda.py').write_text('preflight fixture')
    x=np.ones((8,3000),dtype=np.float32)
    y=np.arange(8,dtype=np.int64)%5
    for name,value in [('train_x',x),('train_y',y),('validation_x',x),('validation_y',y),('adaptation_x',x)]:
        np.save(work/'data'/f'{name}.npy',value)
    manifest={k:False for k in ('source_outer_test_included','target_test_data_included','target_labels_included','participant_ids_included')}
    manifest.update(train_class_counts=np.bincount(y,minlength=5).tolist(),validation_class_counts=np.bincount(y,minlength=5).tolist(),
                    files={p.relative_to(work).as_posix():reference.digest(p) for p in work.rglob('*') if p.is_file()})
    (work/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(new,'__file__',str(work/'run_adast_development_cuda.py'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'mock GPU preflight only')
    monkeypatch.setattr(reference,'build_models',lambda *args: {})
    reached=[]
    def boundary(*args):
        reached.append(args[0])
        raise reference.BudgetStop('test stops before training')
    monkeypatch.setattr(new,'train_development_arm',boundary)
    output=tmp_path/'out'
    with pytest.raises(reference.BudgetStop,match='before training'):
        new.main(type('Args',(),{'output':output,'max_seconds':30})())
    assert reached==['source_only_limited']
    assert json.loads((output/'progress.json').read_bytes())['status']=='stopped_resource_budget'
    assert not list(output.rglob('*.pt'))
    assert (output.parent/'SleepTCN_ADAST_Development_Results_20261004.zip').exists()
