"""Explicitly authorised continuation of best-checkpoint local evaluation; never trains.

Keep the original frozen code/specification, failed progress and 149 saved predictions
unchanged. Only absent target files are inferred. Reporting tolerates transient Windows
replace locks without suppressing data/checkpoint errors or the original resource limit.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import torch
import run_adast_fullsource_local_evaluation as runner

ORIGINAL_RUNNER_SHA = 'ec5d5cf82282c2eae86ff56e16d5424af1622bb6e6f38063e96c6a8229596aeb'
OLD_PROCESS = runner.RECORD / 'verification_path_repair_20261007/local_evaluation_best_process.json'
RESUME_NAME = 'progress_lock_resume_20261007'


def progress_json(path, data, sleep=time.sleep):
    """Progress only: bounded replace retry, then retain a uniquely named status snapshot.

    Never used for model, prediction, proof or scientific aggregate files. Serialization
    and all non-lock filesystem errors remain fatal; readers never see partial JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.stem+'-pending-', suffix='.json', delete=False) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    for attempt, delay in enumerate((.05, .1, .2, .4, .8, 0)):
        try:
            temporary.replace(path)
            return None
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33):
                raise
            if attempt == 5:
                warnings.warn('Progress replace remained locked; retained status snapshot: '+temporary.name,
                              RuntimeWarning)
                return temporary
            sleep(delay)


class ReportingBudget(runner.common.CampaignBudget):
    def publish(self, **values):
        self.context.update(values)
        progress_json(self.progress_path, {**self.context,
            'elapsed_attempt_seconds': self.clock()-self.started,
            'remaining_attempt_seconds': self.remaining()})


def check_original(output, budget=lambda: None):
    spec = runner.read(output/'execution_specification.json')
    frozen = runner.read(output/'all_checkpoints_frozen.json')
    runner.require(spec['selection'] == frozen['selection'] == 'best', 'Only the unfinished best selection')
    runner.require(not spec['target_score_based_selection'] and spec['cpu_threads'] == 4
                   and spec['batch_size'] == 128 and spec['cpu_flush_denormal'], 'Original inference settings differ')
    runner.require(spec['torch'] == str(torch.__version__), 'Original torch version differs')
    runner.require(runner.sha(runner.ROOT/'scripts/run_adast_fullsource_local_evaluation.py') == ORIGINAL_RUNNER_SHA,
                   'Original inference runner changed')
    for relative, digest in spec['code_sha256'].items():
        budget()
        runner.require(runner.sha(runner.ROOT/relative) == digest
                       and runner.sha(output/'code_snapshot'/relative) == digest, 'Original executed code changed')
    for path, digest in spec['bound_file_sha256'].items():
        budget()
        runner.require(runner.sha(path) == digest, 'Bound checkpoint, proof or input changed')
    actual, _, _ = runner.frozen_models('best', budget)
    runner.require(actual == frozen, 'Source-validation-selected models differ from original frozen inventory')
    return spec, frozen


def remaining_original_budget(spec, process):
    runner.require(process['status'] == 'failed' and process['exit_code'] == 1
                   and process['selection'] == 'best' and not process['training'], 'Require terminated failed inference')
    spent = (datetime.fromisoformat(process['finished_at'])-datetime.fromisoformat(process['start_time'])).total_seconds()
    runner.require(0 < spent < spec['max_seconds'] <= 18000, 'Original active resource budget exhausted or invalid')
    return spent, spec['max_seconds']-spent


def audit_saved(output, entries, frozen_hash, spec_hash, budget=lambda: None):
    folder = output/'target_predictions'
    expected = {entry['record_key']+'.npz' for entry in entries}
    present = {path.name for path in folder.iterdir() if path.is_file()}
    runner.require(present <= expected, 'Unknown/partial prediction file; retain it and stop')
    audited = []
    for entry in entries:
        budget()
        path = folder/(entry['record_key']+'.npz')
        if path.exists():
            runner.common.audit_prediction(path, entry, frozen_hash, spec_hash)
            audited.append({**entry, 'path': path.resolve().as_posix(), 'sha256': runner.sha(path)})
    runner.require(len(audited) == 149, 'Require the observed 149 complete saved predictions; no silent discard')
    return audited


def verify_resumption(args):
    output, record = args.output, args.output/RESUME_NAME
    provenance = runner.read(record/'resumption_provenance.json')
    runner.require(runner.sha(Path(__file__)) == provenance['resume_script_sha256']
                   and runner.sha(record/'resume_code_snapshot.py') == provenance['resume_script_sha256'],
                   'Resume code changed after launch')
    check_original(output)
    audit = runner.read(record/'private_reuse_audit.json')
    runner.require(runner.sha(record/'private_reuse_audit.json') == provenance['reuse_audit_sha256']
                   and len(audit['records']) == 149, 'Initial reuse audit changed')
    for entry in audit['records']:
        runner.require(runner.sha(Path(entry['path'])) == entry['sha256'], 'A reused prediction was overwritten')
    proof = runner.read(output/'independent_verification.json')
    runner.require(proof['status'] == 'passed'
                   and proof['aggregate_results_sha256'] == runner.sha(output/'aggregate_results.json'),
                   'Full independent evaluation verification absent')
    runner.write(record/'resumption_verification.json', {
        'status':'passed', 'original_specification_and_code_preserved':True,
        'all_149_reused_prediction_hashes_preserved':True, 'new_target_predictions':31,
        'aggregate_results_sha256':runner.sha(output/'aggregate_results.json'),
        'training':False, 'resource_budget_increased':False})


def run(args):
    output, record = args.output, args.output/RESUME_NAME
    runner.require(output == runner.EVALUATION/'best', 'Only the explicitly requested unfinished best evaluation')
    runner.require(not record.exists(), 'Preserve prior continuation; no automatic restart')
    for name in ('aggregate_results.json','verification.json','independent_verification.json','private_inference_manifest.json','private_source_manifest.json'):
        runner.require(not (output/name).exists(), 'Unexpected completed/staged result; stop without overwrite')
    spec = runner.read(output/'execution_specification.json')
    previous = runner.read(OLD_PROCESS)
    spent, remaining = remaining_original_budget(spec, previous)
    record.mkdir()
    runner.write(record/'process.json', {'pid':os.getpid(),'start_time':datetime.now().astimezone().isoformat(),
        'executable':sys.executable,'script':Path(__file__).resolve().as_posix(),
        'arguments':sys.argv[1:],'training':False,'prior_active_seconds':spent,
        'max_seconds':remaining,'original_max_seconds':spec['max_seconds']})
    budget = ReportingBudget(remaining, record/'progress.json')
    try:
        budget.publish(status='running', phase='check_original_inputs_and_code', training=False,
                       saved_target_predictions=149, remaining_target_predictions=31)
        spec, frozen = check_original(output, budget)
        entries = runner.read(runner.common.PILOT/'private_inference_manifest.json')['records']
        runner.require(len(entries) == 180 and sum(e['epochs'] for e in entries)==183528, 'Original target coverage differs')
        frozen_hash, spec_hash = runner.sha(output/'all_checkpoints_frozen.json'), runner.sha(output/'execution_specification.json')
        budget.publish(phase='audit_saved_predictions')
        targets = audit_saved(output, entries, frozen_hash, spec_hash, budget)
        runner.write(record/'private_reuse_audit.json', {'records':targets})
        source = Path(__file__).read_bytes()
        with (record/'resume_code_snapshot.py').open('xb') as stream:stream.write(source)
        runner.write(record/'resumption_provenance.json', {
            'resume_script_sha256':runner.sha(Path(__file__)), 'original_runner_sha256':ORIGINAL_RUNNER_SHA,
            'original_execution_specification_sha256':spec_hash, 'frozen_checkpoints_sha256':frozen_hash,
            'reuse_audit_sha256':runner.sha(record/'private_reuse_audit.json'), 'reused_target_predictions':149,
            'remaining_target_predictions':31, 'prior_active_seconds':spent,
            'continuation_max_seconds':remaining,'original_max_seconds':spec['max_seconds'],
            'user_authorised_continuation':True,'training':False})
        print('149 SAVED PREDICTIONS AUDITED; CONTINUING ONLY 31 MISSING RECORDS. NO TRAINING.',flush=True)
        networks = runner.common.load_networks(frozen)
        signal = runner.common.load_module('resumed_fullsource_signal',runner.ROOT/'scripts/run_recovered_e3_cpu_pilot.py')
        saved = {e['record_key']:e for e in targets}
        targets=[]
        new_count=0
        for ordinal,entry in enumerate(entries,1):
            budget()
            if entry['record_key'] in saved:
                targets.append(saved[entry['record_key']]);continue
            budget.publish(phase='remaining_target_inference',target_records_completed=len(targets),
                           new_target_predictions_completed=new_count)
            x,_ = signal.read_full_record(args.shhs_root/'shhs/polysomnography/edfs/shhs1'/(entry['record_key']+'.edf'),entry['source_edf_sha256'])
            runner.require(x.shape == (entry['epochs'],3000), 'Full recording support differs')
            parts={a:[] for a in runner.ARMS}
            for (fold,arm),models in networks.items():
                budget.publish(fold=fold,arm=arm)
                parts[arm].append(runner.common.softmax_logits(runner.common.predict(models,x,'source' if arm=='source_only' else 'target','cpu',budget)))
            parts={a:np.stack(v) for a,v in parts.items()}
            path=output/'target_predictions'/(entry['record_key']+'.npz')
            metadata={'subject_id':entry['subject_id'],'source_edf_sha256':entry['source_edf_sha256'],
                      'frozen_checkpoints_sha256':frozen_hash,'specification_sha256':spec_hash}
            with path.open('xb') as stream:
                np.savez_compressed(stream,**{a:p.astype(np.float64).mean(0).astype(np.float32) for a,p in parts.items()},
                    **{'fold_'+a:p for a,p in parts.items()},original_epoch_index=np.arange(len(x)),metadata_json=np.array(json.dumps(metadata)))
            runner.common.audit_prediction(path,entry,frozen_hash,spec_hash)
            targets.append({**entry,'path':path.resolve().as_posix(),'sha256':runner.sha(path)})
            new_count+=1
            print(f'Continued target {ordinal}/180; new {new_count}/31',flush=True)
        runner.require(new_count==31 and len(targets)==180, 'Require original full target coverage')
        runner.write(output/'private_inference_manifest.json',{'records':targets})
        budget.publish(phase='target_scoring',target_records_completed=180,fold=None,arm=None)
        target_cms,refs=runner.common.target_confusions(args,targets,frozen_hash,spec_hash,budget)
        runner.write(output/'private_reference_hashes.json',refs)
        budget.publish(phase='source_OOF_inference')
        source_cms,source_entries,source_refs=runner.common.source_confusions(output,networks,budget)
        runner.write(output/'private_source_manifest.json',{'records':source_entries,'reference_sha256':source_refs})
        summaries={}
        for name,matrices in [('target',target_cms),('source',source_cms)]:
            budget.publish(phase=name+'_bootstrap')
            summaries[name]=runner.common.summarize_pair(matrices['adast'],matrices['source_only'],'adast','source_only')
            with (output/('private_'+name+'_confusions.npz')).open('xb') as stream:np.savez_compressed(stream,**matrices)
        result={'status':'complete_fullsource_best','selection':'best',**summaries,
            'execution_specification_sha256':spec_hash,'frozen_checkpoints_sha256':frozen_hash,
            'inference_manifest_sha256':runner.sha(output/'private_inference_manifest.json'),
            'source_manifest_sha256':runner.sha(output/'private_source_manifest.json'),
            'elapsed_inference_and_scoring_seconds':spent+remaining-budget.remaining(),
            'training_backend':'CUDA_all_twenty_models_sixteen_new_four_reused','inference_backend':'local_CPU',
            'target_labels_used_for_selection':False,'previously_examined_cohort_post_hoc':True,
            'training_variance_not_in_bootstrap':True,'published_ADAST_exact_reproduction':False,
            'reused_target_predictions':149,'new_target_predictions':31,'training':False}
        runner.write(output/'aggregate_results.json',result)
        runner.write(output/'verification.json',{'status':'passed','checkpoint_count':20,
            'target_predictions_audited':180,'source_OOF_folds':10,'aggregate_results_sha256':runner.sha(output/'aggregate_results.json')})
        budget.publish(phase='independent_verification')
        subprocess.run([sys.executable,'-u',str(runner.ROOT/'scripts/run_adast_fullsource_local_evaluation.py'),
            '--verify-only','--selection','best','--output',output.as_posix(),'--shhs-root',args.shhs_root.as_posix()],
            timeout=budget.remaining(),check=True)
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--verify-resumption',
            '--output',output.as_posix(),'--shhs-root',args.shhs_root.as_posix()],timeout=budget.remaining(),check=True)
        budget.publish(status='complete_fullsource_best',phase='complete',target_records_completed=180,
                       new_target_predictions_completed=31)
        print('RESUMED BEST EVALUATION AND INDEPENDENT VERIFICATION COMPLETE.',flush=True)
        return 0
    except (runner.common.CampaignStop,subprocess.TimeoutExpired,KeyboardInterrupt) as error:
        budget.publish(status='stopped_resource_budget_or_interruption',reason=str(error),artifacts_retained=True)
        return 2
    except Exception as error:
        budget.publish(status='failed',reason=f'{type(error).__name__}: {error}',artifacts_retained=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=runner.EVALUATION/'best')
    parser.add_argument('--shhs-root',type=Path,default=Path('E:/research/Dataset/SHHS_v1'))
    parser.add_argument('--verify-resumption',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve();args.shhs_root=args.shhs_root.resolve()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    runner.require(torch.set_flush_denormal(True),'Require original CPU flush-denormal mode')
    if args.verify_resumption:verify_resumption(args)
    else:sys.exit(run(args))
