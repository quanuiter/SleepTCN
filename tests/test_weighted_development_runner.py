import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import json
import numpy as np


def test_runner_refuses_cpu_fallback(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/run_weighted_development_cuda.py'
    spec = importlib.util.spec_from_file_location('weighted_dev_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='refusing CPU'):
        module.run(SimpleNamespace(output=tmp_path / 'results', max_seconds=18000))
    assert not (tmp_path / 'results').exists()


def test_gpu_preflight_reaches_trainer_and_retains_failure_export(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/run_weighted_development_cuda.py'
    spec = importlib.util.spec_from_file_location('weighted_dev_preflight', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'configs').mkdir()
    cfg = {'max_seconds':18000,'fold':0,'powers':[0.0],'training':{}}
    (tmp_path / 'configs/teacher_revision_development_v2.json').write_text(json.dumps({'weighted_development':cfg}))
    records = []
    for role in ('train','validation'):
        folder = tmp_path / 'data' / role
        folder.mkdir(parents=True)
        np.savez(folder / '000.npz',features=np.zeros((5,128),dtype=np.float32),labels=np.arange(5))
        records.append({'path':f'data/{role}/000.npz','role':role,'epochs':5,'valid_epochs':5})
    manifest = {'outer_test_included':False,'target_data_included':False,'participant_ids_included':False,
                'records':records,'files':{},'train_class_counts':[1]*5,'encoder_sha256':'fixture'}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(module.torch.cuda,'get_device_name',lambda _: 'Mock; no GPU operations')
    def stop_before_training(*args,**kwargs):
        raise RuntimeError('test sentinel: preflight reached trainer; no training')
    monkeypatch.setattr(module,'train_arm',stop_before_training)
    with pytest.raises(RuntimeError,match='test sentinel'):
        module.run(SimpleNamespace(output=tmp_path/'results',max_seconds=18000))
    assert (tmp_path/'results/execution_specification.json').exists()
    assert json.loads((tmp_path/'results/progress.json').read_bytes())['status']=='failed'
    assert (tmp_path/'SleepTCN_Weighted_Development_Results.zip').exists()
