import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

load('adast_reference_adapter', ROOT/'scripts/run_colab_adast_training.py')
load('development_metrics', ROOT/'src/sleeptcn/metrics.py')
base = load('run_adast_development_cuda', ROOT/'scripts/run_adast_development_cuda.py')
runner = load('confirmation_runner_test', ROOT/'scripts/run_adast_small_confirmation_cuda.py')
builder = load('confirmation_builder_test', ROOT/'scripts/build_adast_small_confirmation_bundle.py')

def config():
    return json.loads((ROOT/'configs/adast_small_confirmation_v1_20261004.json').read_bytes())

def test_frozen_scope_budget_and_single_intervention():
    cfg = config()
    assert cfg['fold'] == 1 and cfg['seed'] == 123
    assert cfg['full_source_steps_per_epoch'] == int(np.ceil(cfg['source_train_epochs']/cfg['batch_size']))
    assert cfg['updates_per_arm'] == cfg['full_source_steps_per_epoch']*30
    for arm, changes in runner.EXPECTED_ARMS.items():
        effective = runner.effective_config(cfg, arm)
        assert {k:v for k,v in effective.items() if v != cfg[k]} == changes
    wrong = copy.deepcopy(cfg)
    wrong['arm_overrides'][cfg['arms'][2]]['adversarial_weight'] = 0.
    with pytest.raises(ValueError, match='locked three'):
        runner.effective_config(wrong, cfg['arms'][2])

def test_all_three_sampling_orders_are_matched():
    cfg = config()
    rows = [{'arm':name,'initial_state_sha256':'same','history':[{'source_order_sha256':'x','target_order_sha256':None if i==0 else 'y'}]}
            for i,name in enumerate(cfg['arms'])]
    runner.check_matched(rows)
    bad = copy.deepcopy(rows)
    bad[2]['history'][0]['source_order_sha256'] = 'different'
    with pytest.raises(ValueError,match='source sampling'):
        runner.check_matched(bad)
    bad = copy.deepcopy(rows)
    bad[2]['history'][0]['target_order_sha256'] = 'different'
    with pytest.raises(ValueError,match='adaptation sampling'):
        runner.check_matched(bad)
    with pytest.raises(ValueError,match='three completed'):
        runner.check_matched(rows[:2])

@pytest.mark.parametrize('bad_kind',['overlap','duplicate','negative','outside'])
def test_forbidden_source_roles_rejected(bad_kind):
    roles = {'train':np.array([0,1]),'validation':np.array([2,3]),'test':np.array([4,5])}
    builder.validate_roles(roles,6)
    roles['validation'] = {'overlap':np.array([1,3]),'duplicate':np.array([2,2]),
                           'negative':np.array([-1,3]),'outside':np.array([2,6])}[bad_kind]
    with pytest.raises(ValueError):
        builder.validate_roles(roles,6)

def test_tiny_three_arms_save_selection_final_rng_and_losses(tmp_path):
    torch.set_num_threads(1)
    folder = ROOT/'runs/adast_development_20261004/payload/upstream'
    module = load('confirmation_upstream_test',folder/'models.py')
    model_cfg = load('confirmation_upstream_cfg_test',folder/'configs.py').Config()
    utils = load('confirmation_upstream_utils_test',folder/'utils.py')
    cfg = config()
    cfg.update(rounds=2,epochs_per_round=1,batch_size=4,limited_steps_per_epoch=1)
    x = np.random.default_rng(7).normal(size=(8,3000)).astype(np.float32)
    y = np.arange(8,dtype=np.int64)%5
    arrays = [x,y,x,y,x[:4]]
    guard = base.reference.Guard(tmp_path,60)
    rows = []
    for key in cfg['arms']:
        models = base.reference.build_models(module,model_cfg,123,'cpu')
        result = runner.train_arm(key,models,utils,arrays,cfg,tmp_path/key,{'arm':key},guard,'cpu')
        assert result['epochs_completed'] == 2 and result['total_updates'] == 4
        checkpoint = torch.load(tmp_path/key/'final.pt',map_location='cpu',weights_only=True)
        assert 'optimizer' in checkpoint and 'rng_state' in checkpoint and checkpoint['best_epoch'] == result['best_epoch']
        assert (tmp_path/key/'best.pt').exists()
        expected = runner.effective_config(cfg,key)
        for i,row in enumerate(result['history']):
            assert row['loss_coefficients']['source_ce'] == expected['source_loss_weights_by_round'][i]
            if key.startswith('source_only'):
                assert row['loss_coefficients']['target_pseudo_ce'] == row['loss_coefficients']['adversarial'] == 0.
        rows.append(result)
    runner.check_matched(rows)
    with np.load(tmp_path/cfg['arms'][1]/'validation_epoch_001.npz') as left, np.load(tmp_path/cfg['arms'][2]/'validation_epoch_001.npz') as right:
        for key in left.files:
            np.testing.assert_array_equal(left[key],right[key])

def test_no_production_cpu_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA required'):
        runner.main(type('Args',(),{'output':tmp_path/'results','max_seconds':18000})())
    assert not (tmp_path/'results').exists()


def test_launcher_replaces_overlapping_hash_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, 'RECORD', tmp_path)
    monkeypatch.setattr(builder, 'sha', lambda path: 'a'*64)
    binding = {'archive_path':str(tmp_path/'input.zip'), 'archive_sha256':'a'*64,
               'archive_bytes':123, 'manifest_sha256':'b'*64}
    builder.write_launcher(binding, [])
    text = (tmp_path/'colab_launch.py').read_text(encoding='utf8')
    compile(text, 'launcher_regression.py', 'exec')
    assert 'PLACEHOLDER' not in text
    assert repr('a'*64) in text and repr('b'*64) in text
    with pytest.raises(FileExistsError, match='Preserve existing'):
        builder.write_launcher(binding, [])


def test_independent_verifier_rejects_changed_scope():
    sys.path.insert(0, str(ROOT/'scripts'))
    import verify_adast_small_confirmation_results as verifier
    cfg = config()
    verifier.validate_protocol(cfg)
    for field, wrong_value in [('fold',0), ('seed',124), ('updates_per_arm',1140),
                               ('adaptation_subjects',6), ('source_train_epochs',157200)]:
        altered = copy.deepcopy(cfg)
        altered[field] = wrong_value
        with pytest.raises(ValueError, match='protocol differs'):
            verifier.validate_protocol(altered)


def test_prelocked_gate_reports_tradeoffs_without_changing_thresholds():
    sys.path.insert(0, str(ROOT/'scripts'))
    import analyze_adast_small_confirmation_results as analysis
    def metrics(macro):
        return {'macro_f1':macro,'per_class':{s:{'f1':.5} for s in analysis.CLASSES}}
    baseline = {'source':metrics(.7),'target':metrics(.5)}
    candidate = copy.deepcopy(baseline)
    candidate['source']['macro_f1'] = .71
    candidate['source']['per_class']['N1']['f1'] = .52
    candidate['source']['per_class']['REM']['f1'] = .49
    rule = config()['decision_rule']
    gate = analysis.development_gate(baseline,baseline,candidate,rule)
    assert gate['passed'] and gate['automatic_expansion'] is False
    candidate['source']['per_class']['REM']['f1'] = .489
    gate = analysis.development_gate(baseline,baseline,candidate,rule)
    assert not gate['passed']
    assert not gate['checks']['other_source_stage_f1_within_prelocked_decline']
    candidate['source']['per_class']['REM']['f1'] = .5
    candidate['target']['per_class']['N3']['f1'] = .48
    gate = analysis.development_gate(baseline,baseline,candidate,rule)
    assert not gate['checks']['target_path_N1_N3_f1_within_prelocked_decline']
    candidate['target']['per_class']['N3']['f1'] = .5
    candidate['source']['macro_f1'] = .7
    assert not analysis.development_gate(baseline,baseline,candidate,rule)['passed']
