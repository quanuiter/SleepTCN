import copy
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


load('adast_reference_adapter',ROOT/'scripts/run_colab_adast_training.py')
load('development_metrics',ROOT/'src/sleeptcn/metrics.py')
base=load('run_adast_development_cuda',ROOT/'scripts/run_adast_development_cuda.py')
ablation=load('adast_loss_ablation_test',ROOT/'scripts/run_adast_loss_ablation_cuda.py')
verifier=load('adast_ablation_verifier_test',ROOT/'scripts/verify_adast_development_results.py')


def config():
    cfg=json.loads((ROOT/'configs/adast_development_budget_v1_20261004.json').read_bytes())
    plan=json.loads((ROOT/'configs/adast_loss_ablation_v1_20261004.json').read_bytes())
    cfg.update(arms=plan['arms'],arm_overrides=plan['arm_overrides'],campaign_kind=plan['campaign_kind'])
    return cfg


def fixture():
    torch.set_num_threads(1)
    upstream=ROOT/'runs/adast_development_20261004/payload/upstream'
    module=load('ablation_upstream_models_test',upstream/'models.py')
    cfg=load('ablation_upstream_cfg_test',upstream/'configs.py').Config()
    utils=load('ablation_upstream_utils_test',upstream/'utils.py')
    return module,cfg,utils


def optim(models,cfg):
    params=cfg['optimizer']
    names=['encoder','head1','head2','source_attention','target_attention']
    return (torch.optim.Adam([p for n in names for p in models[n].parameters()],lr=params['lr'],betas=tuple(params['betas']),weight_decay=params['weight_decay']),
            torch.optim.Adam(models['discriminator'].parameters(),lr=params['lr'],betas=tuple(params['betas']),weight_decay=params['weight_decay']))


def test_one_factor_only_and_independent_registry_rejects_extra_factor():
    cfg=config()
    verifier.validate_ablation_overrides(cfg)
    for arm,changes in ablation.EXPECTED_ARMS.items():
        effective=ablation.effective_config(cfg,arm)
        assert {k:v for k,v in effective.items() if v!=cfg[k]}==changes
    bad=copy.deepcopy(cfg)
    bad['arm_overrides'][cfg['arms'][1]]['similarity_weight']=0.
    with pytest.raises(ValueError,match='single-factor'):
        ablation.effective_config(bad,cfg['arms'][1])
    with pytest.raises(ValueError,match='Single-factor'):
        verifier.validate_ablation_overrides(bad)


@pytest.mark.parametrize('round_index',[0,1])
def test_reference_exactly_matches_frozen_update(round_index):
    module,model_cfg,utils=fixture()
    cfg=config()
    left=base.reference.build_models(module,model_cfg,123,'cpu')
    right=base.reference.build_models(module,model_cfg,123,'cpu')
    lo,ld=optim(left,cfg)
    ro,rd=optim(right,cfg)
    sx,tx=torch.randn(4,1,3000),torch.randn(4,1,3000)
    sy,py=torch.arange(4),torch.arange(4)
    torch.manual_seed(88)
    expected=ablation.original_update(left,lo,ld,utils,sx,sy,tx,py,round_index,cfg,'adast')
    torch.manual_seed(88)
    actual=ablation.measured_update(right,ro,rd,utils,sx,sy,tx,py,round_index,cfg,'adast')
    assert actual==expected
    assert base.reference.state_digest(left)==base.reference.state_digest(right)


def test_no_alignment_never_forwards_or_steps_discriminator_and_loss_reconstructs():
    module,model_cfg,utils=fixture()
    cfg=ablation.effective_config(config(),'adast_no_adversarial_full_source')
    models=base.reference.build_models(module,model_cfg,123,'cpu')
    original=copy.deepcopy(models['discriminator'].state_dict())
    def forbidden(*args):
        raise AssertionError('Removed alignment must not forward discriminator')
    models['discriminator'].register_forward_pre_hook(forbidden)
    optimizer,disc_optimizer=optim(models,cfg)
    sx,tx=torch.randn(4,1,3000),torch.randn(4,1,3000)
    sy,py=torch.arange(4),torch.arange(4)
    losses=ablation.measured_update(models,optimizer,disc_optimizer,utils,sx,sy,tx,py,1,cfg,'adast')
    assert not disc_optimizer.state
    assert all(torch.equal(v,original[k]) for k,v in models['discriminator'].state_dict().items())
    assert losses['adversarial']==losses['discriminator']==0.
    coefficients={'source_ce':.1,'similarity':.001,'adversarial':0.,'target_pseudo_ce':.01}
    verifier.verify_loss_components(losses,coefficients,True)
    assert np.isfinite(list(losses.values())).all()


def test_tiny_ablations_save_best_final_and_effective_coefficients(tmp_path,monkeypatch):
    module,model_cfg,utils=fixture()
    cfg=config()
    cfg.update(rounds=2,epochs_per_round=1,batch_size=4,limited_steps_per_epoch=1)
    arrays=[np.random.default_rng(1).normal(size=(8,3000)).astype(np.float32),np.arange(8,dtype=np.int64)%5]
    arrays+=arrays[:]+[arrays[0][:4]]
    monkeypatch.setattr(base,'measured_update',ablation.measured_update)
    guard=base.reference.Guard(tmp_path,60)
    for key in cfg['arms']:
        models=base.reference.build_models(module,model_cfg,123,'cpu')
        result=ablation.train_arm(key,models,utils,arrays,cfg,tmp_path/key,{'arm':key},guard,'cpu')
        assert result['epochs_completed']==2 and result['total_updates']==4
        assert (tmp_path/key/'best.pt').exists() and (tmp_path/key/'final.pt').exists()
        effective=ablation.effective_config(cfg,key)
        for index,row in enumerate(result['history']):
            assert row['loss_coefficients']['source_ce']==effective['source_loss_weights_by_round'][index]
            assert row['loss_coefficients']['target_pseudo_ce']==effective['target_loss_weights_by_round'][index]
            assert row['loss_coefficients']['adversarial']==effective['adversarial_weight']


def test_cpu_production_fallback_refused_before_output(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA required'):
        base.main(type('Args',(),{'output':tmp_path/'out','max_seconds':18000})())
    assert not (tmp_path/'out').exists()
