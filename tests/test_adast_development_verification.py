import importlib.util
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('adast_dev_verification_tests',ROOT/'scripts/verify_adast_development_results.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_independent_sampler_matches_frozen_spec():
    cfg=json.loads((ROOT/'configs/adast_development_budget_v1_20261004.json').read_bytes())
    limited,lt=module.replay_sampling(cfg,2,False)
    full,ft=module.replay_sampling(cfg,2,True)
    assert len(limited)==4864 and len(full)==157200
    np.testing.assert_array_equal(full[:4864],limited)
    np.testing.assert_array_equal(ft[:4864],lt)
    np.testing.assert_array_equal(np.sort(full),np.arange(157200))


def test_validation_metrics_reconstruct_and_reject_wrong_metrics(tmp_path):
    labels=np.arange(5,dtype=np.int64)
    logits=np.eye(5,dtype=np.float32)
    stream=io.BytesIO()
    np.savez_compressed(stream,source=logits,target=logits)
    archive=tmp_path/'validation.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('validation.npz',stream.getvalue())
    metric=module.metrics_from_confusion(module.confusion_matrix_5(labels,logits.argmax(1)))
    with zipfile.ZipFile(archive) as z:
        module.verify_logits(z,'validation.npz',labels,{'source':metric,'target':metric})
        bad={**metric,'macro_f1':0.}
        with pytest.raises(ValueError,match='reconstruct'):
            module.verify_logits(z,'validation.npz',labels,{'source':bad,'target':metric})


def test_wrong_download_hash_fails_before_data_access(tmp_path):
    archive=tmp_path/'result.zip'
    archive.write_bytes(b'not the observed result')
    with pytest.raises(ValueError,match='observed Colab SHA'):
        module.verify(type('Args',(),{'archive':archive,'expected_sha256':'0'*64,'max_seconds':30})())


def test_loss_components_reconstruct_and_reject_target_loss_for_source_only():
    coefficients={'source_ce':.1,'similarity':.001,'adversarial':0.,'target_pseudo_ce':0.}
    values={'total':.202,'source_ce':2.,'similarity':2.,'adversarial':0.,'target_pseudo_ce':0.,'discriminator':0.}
    module.verify_loss_components(values,coefficients,False)
    with pytest.raises(ValueError,match='reconstruct'):
        module.verify_loss_components({**values,'total':.5},coefficients,False)
    with pytest.raises(ValueError,match='nonzero'):
        module.verify_loss_components({**values,'adversarial':.2},coefficients,False)
