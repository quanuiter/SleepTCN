import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reuse_audit',ROOT/'scripts/audit_adast_fullsource_reuse.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def test_complete_disjoint_roles():
    audit.roles_valid({'train':np.array([0,2]),'validation':np.array([1]),
                       'test':np.array([3])},4)

@pytest.mark.parametrize('kind',['overlap','missing','duplicate','negative','float'])
def test_invalid_roles_rejected(kind):
    roles={'train':np.array([0,2]),'validation':np.array([1]),'test':np.array([3])}
    if kind=='overlap': roles['validation']=np.array([0])
    if kind=='missing': roles['test']=np.array([],dtype=np.int64)
    if kind=='duplicate': roles['train']=np.array([0,0])
    if kind=='negative': roles['test']=np.array([-1])
    if kind=='float': roles['train']=np.array([0.,2.])
    with pytest.raises(ValueError):
        audit.roles_valid(roles,4)

def test_subset_preserves_original_order_and_values():
    full=np.arange(24,dtype=np.float32).reshape(8,3)
    indices=np.array([6,0,3])
    audit.compare_subset(full,indices,full[indices].copy())
    wrong=full[indices].copy()
    wrong[0,0]+=1
    with pytest.raises(ValueError,match='differs'):
        audit.compare_subset(full,indices,wrong)
    with pytest.raises(ValueError,match='differs'):
        audit.compare_subset(full,indices,full[indices[::-1]].copy())

def test_training_setting_change_rejected():
    cfg={k:1 for k in audit.KEYS}
    audit.compatible_configs(cfg,dict(cfg))
    for key in audit.KEYS:
        changed=dict(cfg)
        changed[key]=2
        with pytest.raises(ValueError,match='Training setting'):
            audit.compatible_configs(cfg,changed)

def pair_fixture():
    cfg={'source_train_epochs':257,'batch_size':128}
    rows=[]
    for arm in ('source','adast'):
        history=[{'global_epoch':i+1,'updates':3,'source_presentations':257,
                  'source_unique_seen':257,'source_order_sha256':str(i),
                  'optimizer_lr_before_epoch':.001 if i<10 else .0001,
                  'loss_coefficients':{'source_ce':1. if i<15 else .1},
                  'target_training_presentations':0 if arm=='source' else 257,
                  'validation':{'source':{'macro_f1':float(i<2)}}}
                 for i in range(30)]
        rows.append({'initial_state_sha256':'same','epochs_completed':30,
                     'total_updates':90,'best_epoch':1,'history':history})
    return cfg,rows

def test_matched_pair_first_tie_selection():
    cfg,rows=pair_fixture()
    audit.validate_pair(rows,cfg)

@pytest.mark.parametrize('kind',['init','order','budget','coverage','lr','loss','target_leak','selection'])
def test_bad_reuse_pair_rejected(kind):
    cfg,rows=pair_fixture()
    if kind=='init': rows[1]['initial_state_sha256']='other'
    if kind=='order': rows[1]['history'][0]['source_order_sha256']='other'
    if kind=='budget': rows[1]['total_updates']=89
    if kind=='coverage': rows[1]['history'][0]['source_presentations']=256
    if kind=='lr': rows[1]['history'][10]['optimizer_lr_before_epoch']=.001
    if kind=='loss': rows[1]['history'][15]['loss_coefficients']['source_ce']=1.
    if kind=='target_leak': rows[0]['history'][0]['target_training_presentations']=257
    if kind=='selection': rows[1]['best_epoch']=2
    with pytest.raises(ValueError):
        audit.validate_pair(rows,cfg)
