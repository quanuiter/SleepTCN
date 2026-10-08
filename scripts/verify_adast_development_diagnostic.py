"""Second-process reconstruction from diagnostic logits; no training or test access."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sleeptcn.preprocessing import sha256_file
from sleeptcn.metrics import metrics_from_confusion
from sleeptcn.revision_campaign import write_once_json


def run():
    folder = ROOT/'runs/development_20261004/adast_diagnostic'
    result = json.loads((folder/'aggregate_results.json').read_bytes())
    verification = json.loads((folder/'verification.json').read_bytes())
    aggregate_sha = sha256_file(folder/'aggregate_results.json')
    assert verification['status']=='passed' and verification['aggregate_results_sha256']==aggregate_sha
    assert not result['target_test_access'] and not result['target_true_label_access']
    assert [f['fold'] for f in result['folds']]==list(range(10))
    plan = json.loads((folder/'execution_specification.json').read_bytes())
    views = plan['protocol']['adast_diagnostic']['views']
    pooled = {view:{rule:np.zeros((5,5),dtype=np.int64) for rule in ('head1','head2','maximum','mean')} for view in views}
    for row in result['folds']:
        path = folder/f"fold_{row['fold']:02d}_logits.npz"
        assert sha256_file(path)==row['logits_sha256']
        with np.load(path,allow_pickle=False) as data:
            for role in ('validation','adaptation'):
                for view in views:
                    a,b = data[f'{role}_{view}_head1'],data[f'{role}_{view}_head2']
                    assert a.shape==b.shape and a.shape[1]==5 and np.isfinite(a).all() and np.isfinite(b).all()
                    rules = {'head1':a,'head2':b,'maximum':np.maximum(a,b),'mean':(a+b)/2}
                    for rule,logits in rules.items():
                        pred = logits.argmax(1)
                        expected = row['validation' if role=='validation' else 'adaptation_unlabelled'][view][rule]
                        assert np.bincount(pred,minlength=5).tolist()==expected['prediction_counts']
                        if role=='validation':
                            labels = data['validation_labels']
                            cm = np.bincount(labels.astype(np.int64)*5+pred,minlength=25).reshape(5,5)
                            assert cm.tolist()==expected['metrics']['confusion_matrix']
                            assert metrics_from_confusion(cm)==expected['metrics']
                            pooled[view][rule] += cm
    for view,rules in pooled.items():
        for rule,cm in rules.items():
            assert metrics_from_confusion(cm)==result['pooled_validation'][view][rule]
    for name,digest in plan['frozen_sha256'].items():
        assert sha256_file(ROOT/name)==digest
    proof = {'status':'passed','aggregate_results_sha256':aggregate_sha,
             'all_ten_fold_saved_predictions_and_confusions_reconstructed':True,
             'pooled_validation_metrics_reconstructed':True,'frozen_input_checkpoint_hashes_unchanged':True,
             'training':False,'target_test_access':False,'target_true_label_access':False}
    write_once_json(folder/'independent_verification.json',proof)
    print(json.dumps(proof,indent=2))

if __name__=='__main__':
    run()
