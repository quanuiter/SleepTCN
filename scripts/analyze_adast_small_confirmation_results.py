"""Apply the prelocked fold-1 development gate only to independently verified results."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from analyze_adast_loss_ablation_results import CLASSES, difference, view
from verify_adast_small_confirmation_results import validate_protocol, STATUS

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'runs/adast_small_confirmation_20261004'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_gate(source_only, reference, candidate, rule):
    """Each row is a best-checkpoint validation dict with source/target paths."""
    s, r, c = source_only['source'], reference['source'], candidate['source']
    rt, ct = reference['target'], candidate['target']
    f1_delta = lambda a, b, name: a['per_class'][name]['f1']-b['per_class'][name]['f1']
    other = {name:f1_delta(c,r,name) for name in ('W','N2','N3','REM')}
    target = {name:f1_delta(ct,rt,name) for name in ('N1','N3')}
    checks = {
        'source_macro_f1_above_reference':c['macro_f1'] > r['macro_f1'],
        'source_macro_f1_above_source_only':c['macro_f1'] > s['macro_f1'],
        'source_N1_f1_above_reference':f1_delta(c,r,'N1') > 0,
        'other_source_stage_f1_within_prelocked_decline':all(
            v >= -rule['max_other_stage_f1_decline_vs_reference']-1e-12 for v in other.values()),
        'target_path_macro_f1_within_prelocked_decline':ct['macro_f1']-rt['macro_f1'] >=
            -rule['max_target_attention_macro_f1_decline_vs_reference']-1e-12,
        'target_path_N1_N3_f1_within_prelocked_decline':all(
            v >= -rule['max_target_attention_N1_N3_f1_decline_vs_reference']-1e-12 for v in target.values()),
    }
    return {'passed':all(checks.values()),'checks':checks,
        'source_macro_f1_delta_vs_reference':c['macro_f1']-r['macro_f1'],
        'source_macro_f1_delta_vs_source_only':c['macro_f1']-s['macro_f1'],
        'other_source_stage_f1_deltas_vs_reference':other,
        'target_path_macro_f1_delta_vs_reference':ct['macro_f1']-rt['macro_f1'],
        'target_path_N1_N3_f1_deltas_vs_reference':target,
        'scope':rule['scope'],'automatic_expansion':False}


def analyze(args):
    pointer = json.loads((RECORD/'verified_results_pointer.json').read_bytes())
    directory = Path(pointer['results_root']).resolve()
    proof_path = directory/'independent_verification.json'
    proof = json.loads(proof_path.read_bytes())
    internal = json.loads((directory/'verification.json').read_bytes())
    aggregate_sha = sha(directory/'aggregate_results.json')
    gates = ('all_93_validation_diagnostics_recomputed',
             'all_six_best_final_checkpoints_CPU_validation_replayed',
             'sampling_coverage_and_budget_replayed',
             'all_three_initializations_and_source_orders_match',
             'round_one_reference_keep_prefix_verified')
    if (pointer['status'] != 'passed' or sha(proof_path) != pointer['independent_verification_sha256']
        or proof['status'] != 'passed' or internal['status'] != 'passed'
        or not all(proof.get(k) for k in gates)
        or any(p['aggregate_results_sha256'] != aggregate_sha for p in (pointer,proof,internal))
        or proof['source_outer_test_access'] or proof['target_test_access']):
        raise ValueError('Require complete independent confirmation verification')
    result = json.loads((directory/'aggregate_results.json').read_bytes())
    cfg = json.loads((directory/'execution_specification.json').read_bytes())['protocol']
    validate_protocol(cfg)
    if cfg != json.loads((ROOT/'configs/adast_small_confirmation_v1_20261004.json').read_bytes()):
        raise ValueError('Prelocked local decision rule differs')
    if result['status'] != STATUS or [r['arm'] for r in result['arms']] != cfg['arms']:
        raise ValueError('Require three completed confirmation arms')
    arms = {a['arm']:a for a in result['arms']}
    selected = {name:a['best_source_validation'] for name,a in arms.items()}
    gate = development_gate(*(selected[name] for name in cfg['arms']),cfg['decision_rule'])
    report = {'scope':'pooled_epoch_source_validation_fold_1_seed_123',
        'fold':1,'seed':123,'validation_epochs':23881,'updates_per_arm':35820,
        'training_seconds':result['elapsed_seconds'],'verification_seconds':proof['elapsed_seconds'],
        'development_gate':gate,'selected':{},'final':{},'epoch_trajectories':{},
        'previous_fold_0_kept_separate':True,'target_attention_is_not_SHHS_performance':True,
        'training_variability_estimated':False,'source_outer_test_access':False,'target_test_access':False,
        'provenance':{'aggregate_results_sha256':aggregate_sha,
            'independent_verification_sha256':sha(proof_path),'archive_sha256':proof['archive_sha256'],
            'analysis_script_sha256':sha(Path(__file__).resolve())}}
    csv_buffer = io.StringIO(newline='')
    writer = csv.writer(csv_buffer,lineterminator='\n')
    writer.writerow(['arm','selection','epoch','attention_on_source_validation','macro_f1','accuracy',
                     'class','precision','recall','f1','support'])
    reference = arms[cfg['arms'][1]]
    for name,a in arms.items():
        if a['epochs_completed'] != 30 or a['total_updates'] != 35820:
            raise ValueError('Incomplete learning budget')
        for selection,key,epoch in [('selected','best_source_validation',a['best_epoch']),
                                    ('final','final_source_validation',30)]:
            report[selection][name] = {'epoch':epoch}
            for attention in ('source','target'):
                metrics = a[key][attention]
                report[selection][name][attention+'_attention_on_source_validation'] = view(metrics)
                report[selection][name]['delta_'+attention+'_vs_reference_same_selection'] = difference(
                    metrics,reference[key][attention])
                for stage in CLASSES:
                    row = metrics['per_class'][stage]
                    writer.writerow([name,selection,epoch,attention,metrics['macro_f1'],metrics['accuracy'],
                                     stage,*(row[k] for k in ('precision','recall','f1','support'))])
        report['epoch_trajectories'][name] = [
            {'epoch':h['global_epoch'],'loss_coefficients':h['loss_coefficients'],
             'source':view(h['validation']['source']),'target_on_source_validation':view(h['validation']['target'])}
            for h in a['history']]
    output = args.output.resolve()
    allowed = (ROOT/'Reports/analysis').resolve()
    if output == allowed or not output.is_relative_to(allowed) or output.exists():
        raise ValueError('Require a new Reports/analysis output subdirectory')
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (output/'all_class_metrics.csv').write_text(csv_buffer.getvalue(),encoding='utf8')
    manifest = {name:sha(output/name) for name in ('summary.json','all_class_metrics.csv')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'completed','development_gate':gate,'files_sha256':manifest},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args())
