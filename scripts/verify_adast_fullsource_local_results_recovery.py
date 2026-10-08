"""Finish verification of retained predictions after a Path/str bug; never train or regenerate them."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
import run_adast_fullsource_local_evaluation as runner

ROOT = runner.ROOT
RELATIVE = 'scripts/run_adast_fullsource_local_evaluation.py'
OLD_ALIAS = 'sha = common.sha256_file'
NEW_ALIAS = 'sha = lambda path: common.sha256_file(Path(path))'


def check_code_provenance(output, specification):
    """Require original snapshots and all dependencies; permit exactly the one-line path repair."""
    repair = None
    for relative, digest in specification['code_sha256'].items():
        recorded, current = output / 'code_snapshot' / relative, ROOT / relative
        runner.require(runner.sha(recorded) == digest, 'Original executed code snapshot changed')
        actual = runner.sha(current)
        if actual != digest:
            runner.require(relative == RELATIVE, 'Unrelated executed dependency changed')
            original = recorded.read_text(encoding='utf-8')
            repaired = current.read_text(encoding='utf-8')
            runner.require(original.count(OLD_ALIAS) == 1 and
                           original.replace(OLD_ALIAS, NEW_ALIAS, 1) == repaired,
                           'Repair must be exactly the Path conversion; no inference changes')
            repair = {'file': relative, 'original_sha256': digest, 'repaired_sha256': actual,
                      'change': 'Coerce str/Path to Path at SHA256 call boundary only'}
    runner.require(repair is not None, 'Require the documented path-only repair')
    return repair


def verify(args):
    output = args.output.resolve()
    runner.require(output == runner.EVALUATION / 'final', 'Recover only retained final evaluation')
    runner.require(not (output / 'independent_verification.json').exists(), 'Do not repeat completed verification')
    budget = runner.common.CampaignBudget(args.max_seconds, output / 'verification_recovery_progress.json')
    budget.publish(status='running', phase='input_and_code_hashes', training=False)
    spec, frozen, result = [runner.read(output / name) for name in
                           ('execution_specification.json', 'all_checkpoints_frozen.json', 'aggregate_results.json')]
    aggregate_digest = runner.sha(output / 'aggregate_results.json')
    for path, digest in spec['bound_file_sha256'].items():
        budget()
        runner.require(runner.sha(path) == digest, 'Bound checkpoint/proof/input changed')
    repair = check_code_provenance(output, spec)
    runner.require(result['status'] == 'complete_fullsource_final' and result['selection'] == 'final' and
                   result['frozen_checkpoints_sha256'] == runner.sha(output / 'all_checkpoints_frozen.json') and
                   result['execution_specification_sha256'] == runner.sha(output / 'execution_specification.json') and
                   result['inference_manifest_sha256'] == runner.sha(output / 'private_inference_manifest.json') and
                   result['source_manifest_sha256'] == runner.sha(output / 'private_source_manifest.json'), 'Result/provenance differs')
    runner.require([row['fold'] for row in frozen['records']] == list(range(10)), 'Fold coverage differs')
    entries = runner.read(output / 'private_inference_manifest.json')['records']
    original = runner.read(runner.common.PILOT / 'private_inference_manifest.json')['records']
    keys = ('subject_id', 'record_key', 'epochs', 'source_edf_sha256')
    runner.require([{k: e[k] for k in keys} for e in entries] == [{k: e[k] for k in keys} for e in original], 'Target identity/order differs')
    budget.publish(phase='recompute_scoring_and_bootstrap', target_prediction_files=180)
    target_cms, refs = runner.common.target_confusions(args, entries, runner.sha(output / 'all_checkpoints_frozen.json'),
                                                     runner.sha(output / 'execution_specification.json'), budget)
    source_cms, source_entries, source_refs = runner.common.source_confusions(output, budget=budget)
    runner.require(runner.read(output / 'private_source_manifest.json') == {'records': source_entries, 'reference_sha256': source_refs} and
                   runner.read(output / 'private_reference_hashes.json') == refs, 'Prediction/reference manifest differs')
    for name, matrices in [('target', target_cms), ('source', source_cms)]:
        budget()
        runner.require(runner.common.summarize_pair(matrices['adast'], matrices['source_only'], 'adast', 'source_only') == result[name],
                       'Metrics/bootstrap do not recompute')
        with np.load(output / ('private_' + name + '_confusions.npz'), allow_pickle=False) as values:
            for arm, cm in matrices.items():
                np.testing.assert_array_equal(values[arm], cm)
    networks = runner.common.load_networks(frozen)
    signal = runner.common.load_module('recovery_verification_signal', ROOT / 'scripts/run_recovered_e3_cpu_pilot.py')
    replays = 0
    for entry in (entries[0], entries[-1]):
        budget.publish(phase='target_model_record_replay', target_replays_completed=replays)
        x, _ = signal.read_full_record(args.shhs_root / 'shhs/polysomnography/edfs/shhs1' / (entry['record_key'] + '.edf'), entry['source_edf_sha256'])
        with np.load(entry['path'], allow_pickle=False) as values:
            for (fold, arm), models in networks.items():
                actual = runner.common.softmax_logits(runner.common.predict(models, x, 'source' if arm == 'source_only' else 'target', 'cpu', budget))
                np.testing.assert_array_equal(actual, values['fold_' + arm][fold])
                replays += 1
                budget.publish(target_replays_completed=replays)
    x = np.load(runner.common.WORK / 'data/source_x.npy', mmap_mode='r', allow_pickle=False)
    source_replays = 0
    for fold in range(10):
        budget.publish(phase='source_model_batch_replay', source_replays_completed=source_replays)
        with np.load(output / 'source_predictions' / f'fold_{fold:02d}.npz', allow_pickle=False) as values:
            for arm in runner.ARMS:
                actual = runner.common.softmax_logits(runner.common.predict(networks[fold, arm], x[values['indices'][:128]], 'source', 'cpu', budget))
                np.testing.assert_array_equal(actual, values[arm][:128])
                source_replays += 1
    runner.require(runner.sha(output / 'aggregate_results.json') == aggregate_digest, 'Aggregate changed during verification')
    check_code_provenance(output, spec)
    for path, digest in spec['bound_file_sha256'].items():
        budget()
        runner.require(runner.sha(path) == digest, 'Bound input changed during verification')
    proof = {'status': 'passed', 'selection': 'final', 'checkpoint_hashes_verified': 20,
             'target_prediction_files_verified': 180, 'source_OOF_folds_verified': 10,
             'target_model_record_replays': replays, 'source_model_batch_replays': source_replays,
             'all_confusions_and_bootstrap_recomputed': True, 'aggregate_results_sha256': aggregate_digest,
             'path_only_code_repair': repair, 'original_execution_snapshot_preserved': True,
             'predictions_regenerated': False, 'training': False,
             'recovery_verifier_sha256': runner.sha(Path(__file__)),
             'verification_finished_local': datetime.now().astimezone().isoformat()}
    runner.write(output / 'independent_verification.json', proof)
    prior_progress = runner.read(output / 'progress.json')
    runner.write(output / 'progress_before_verification_repair.json', prior_progress)
    # Retain the failed attempt verbatim above; publish recovery completion only after the full proof.
    completed = {**prior_progress, 'status': 'complete_fullsource_final', 'phase': 'complete',
                 'verification_recovered': True, 'recovery_verification_seconds': args.max_seconds - budget.remaining()}
    completed.pop('reason', None)
    with (output / 'progress.json').open('w', encoding='utf-8') as stream:
        json.dump(completed, stream, indent=2, allow_nan=False)
        stream.write('\n')
    budget.publish(status='passed', phase='complete', target_replays_completed=replays, source_replays_completed=source_replays)
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=runner.EVALUATION / 'final')
    parser.add_argument('--shhs-root', type=Path, default=Path('E:/research/Dataset/SHHS_v1'))
    parser.add_argument('--max-seconds', type=float, default=1800)
    args = parser.parse_args()
    runner.require(0 < args.max_seconds <= 1800, 'At most 30 minutes for this verification-only repair')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    runner.require(torch.set_flush_denormal(True), 'Require original CPU inference mode')
    verify(args)
