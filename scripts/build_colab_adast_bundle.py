"""Build an allowlisted EEG bundle for ten-fold ADAST; private mapping stays local."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/colab_adast_20261004'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / 'SleepTCN_ADAST_10Fold_20261004.zip'
    if archive.exists():
        verified = json.loads((OUT / 'bundle_verification.json').read_bytes())
        if digest(archive) != verified['archive_sha256']:
            raise ValueError('Existing verified archive changed')
        print(json.dumps(verified), flush=True)
        return
    work = OUT / 'payload'
    if work.exists():
        raise FileExistsError('Partial bundle exists; inspect, do not overwrite')
    if shutil.disk_usage(OUT).free < 6_000_000_000:
        raise RuntimeError('Need 6GB available to preserve lossless input bundle')
    work.mkdir()
    (work / 'data').mkdir()
    upstream = ROOT / 'runs/teacher_revision_cpu_20261001/adast_upstream'
    old_cfg = json.loads((ROOT / 'configs/teacher_revision_adast_fold0_pilot_v1.json').read_bytes())
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != old_cfg['upstream_commit']:
        raise ValueError('Pinned upstream revision differs')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Upstream modified')
    (work / 'upstream').mkdir()
    upstream_hashes = {}
    for original, name in [('models/models.py', 'models.py'), ('config_files/configs.py', 'configs.py'),
                           ('utils.py', 'utils.py'), ('LICENSE', 'LICENSE')]:
        path = upstream / original
        shutil.copyfile(path, work / 'upstream' / name)
        upstream_hashes[original] = digest(path)
    shutil.copyfile(ROOT / 'scripts/run_colab_adast_training.py', work / 'run_colab_adast_training.py')
    cfg = dict(old_cfg)
    cfg.pop('outer_fold')
    cfg.update(status='post_hoc_ten_fold_protocol_fixed_before_new_CUDA_baseline_results',
               outer_folds=list(range(10)),
               source_sampling='same_seeded_permutation_each_epoch_take_first_4864_valid_train_epochs_of_each_fold',
               scheduler='all_joint_optimizer_groups_step10_gamma0.1_round0_matches_existing_adapter_actual_behavior',
               evaluation='local_only_180_locked_SHHS_test_subjects_after_all_twenty_final_checkpoints_frozen',
               ensemble='per_fold_elementwise_maximum_dual_head_logits_then_softmax_then_mean_of_ten_probabilities',
               backend='all_ten_matched_pairs_fresh_CUDA_float32; historical_single_fold_CPU_pilot_not_imported',
               limitations=['Harmonized-protocol implementation, not exact published-dataset reproduction.',
                            'One seed; SHHS previously examined; post-hoc supplementary analysis, not untouched holdout.',
                            'Five adaptation subjects; fixed 38 updates/epoch, not a full pass through source data.',
                            'Matched source-only retains dual-head similarity regularization.',
                            'No target validation logging, true adaptation labels or target-label model selection.'])
    (work / 'protocol.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (OUT / 'protocol.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    split_path = ROOT / 'data/splits/sleepedf_sc_10fold_seed42_v2.json'
    splits = json.loads(split_path.read_bytes())['outer_runs']
    keys = sorted({k for s in splits for r in ['train', 'validation', 'test'] for k in s[r]['record_keys']})
    if len(keys) != 153 or len(splits) != 10:
        raise ValueError('Source split support differs')
    private_sources, offset = [], 0
    source_x = np.lib.format.open_memmap(work / 'data/source_x.npy', mode='w+', dtype=np.float32, shape=(195469, 3000))
    source_y = np.lib.format.open_memmap(work / 'data/source_y.npy', mode='w+', dtype=np.int64, shape=(195469,))
    ranges = {}
    for ordinal, key in enumerate(keys):
        path = ROOT / 'data/processed/filtered_v2' / (key + '.npz')
        with np.load(path, allow_pickle=False) as z:
            valid = z['y'] >= 0
            x, y = z['x'][valid], z['y'][valid]
            if x.dtype != np.float32 or x.shape != (len(y), 3000) or not np.isfinite(x).all() or not np.isin(y, range(5)).all():
                raise ValueError('Source schema differs')
            source_x[offset:offset + len(y)] = x
            source_y[offset:offset + len(y)] = y
            ranges[key] = (offset, offset + len(y))
            private_sources.append({'ordinal': ordinal, 'record_key': key, 'sha256': digest(path), 'epochs': len(y)})
            offset += len(y)
        if ordinal % 25 == 0:
            print(f'Prepared source records {ordinal + 1}/153', flush=True)
    if offset != 195469:
        raise ValueError('Source valid-epoch support differs')
    source_x.flush()
    source_y.flush()
    del source_x, source_y
    y_all = np.load(work / 'data/source_y.npy', mmap_mode='r', allow_pickle=False)
    role_summary = []
    for fold, split in enumerate(splits):
        subject_sets = [set(split[role]['subject_ids']) for role in ['train', 'validation', 'test']]
        if any(subject_sets[i] & subject_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('Source subject overlap')
        indices = {}
        for role in ['train', 'validation', 'test']:
            indices[role] = np.concatenate([np.arange(*ranges[k]) for k in sorted(split[role]['record_keys'])])
            if len(indices[role]) != split[role]['valid_epochs']:
                raise ValueError('Source role support differs')
            expected = [split[role]['label_counts'][str(i)] for i in range(5)]
            if np.bincount(y_all[indices[role]], minlength=5).tolist() != expected:
                raise ValueError('Source role class counts differ')
        if not np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(195469)):
            raise ValueError('Source roles must cover all source epochs exactly once')
        np.savez_compressed(work / f'data/fold_{fold:02d}_roles.npz', **indices)
        role_summary.append({'fold': fold, 'valid_epochs': {role: len(v) for role, v in indices.items()}})
    del y_all
    adaptation_manifest = ROOT / 'data/cache/shhs_label_free_adaptation_v1_20261001/private_manifest.json'
    adaptation = json.loads(adaptation_manifest.read_bytes())
    old_inputs = json.loads((ROOT / 'runs/teacher_revision_cpu_20261001/adast_fold0/private_inputs.json').read_bytes())
    if digest(adaptation_manifest) != old_inputs['adaptation_manifest_sha256']:
        raise ValueError('Locked adaptation manifest differs from historical verified run')
    pilot = ROOT / 'runs/teacher_revision_cpu_20261001/recovered_e3_fold0'
    pilot_result = json.loads((pilot / 'aggregate_results.json').read_bytes())
    target_manifest = pilot / 'private_inference_manifest.json'
    if digest(target_manifest) != pilot_result['inference_manifest_sha256']:
        raise ValueError('Locked target-test manifest changed')
    test_ids = {e['subject_id'] for e in json.loads(target_manifest.read_bytes())['records']}
    adaptation_ids = {e['subject_id'] for e in adaptation['records']}
    if len(adaptation['records']) != 5 or len(adaptation_ids) != 5 or adaptation_ids & test_ids:
        raise ValueError('Adaptation/test subject overlap or support differs')
    pieces = []
    for entry in sorted(adaptation['records'], key=lambda e: e['subject_id']):
        signal = entry['variants']['filtered_v2']
        if digest(signal['path']) != signal['sha256']:
            raise ValueError('Adaptation cache changed')
        with np.load(signal['path'], allow_pickle=False) as z:
            if set(z.files) != {'x', 'original_epoch_index', 'metadata_json'}:
                raise ValueError('Target true labels or unexpected fields')
            meta = json.loads(str(z['metadata_json']))
            if meta['role'] != 'adaptation' or meta['uses_sleep_annotations'] or not np.array_equal(z['original_epoch_index'], np.arange(len(z['x']))):
                raise ValueError('Adaptation window not label-independent')
            pieces.append(z['x'])
    target = np.concatenate(pieces)
    if target.shape != (4989, 3000) or target.dtype != np.float32 or not np.isfinite(target).all():
        raise ValueError('Adaptation schema/support differs')
    np.save(work / 'data/adaptation_x.npy', target, allow_pickle=False)
    if digest(work / 'data/adaptation_x.npy') != old_inputs['cached_arrays']['target_x']['sha256']:
        raise ValueError('Adaptation values differ from original label-free cache')
    private = {'source_mapping': private_sources, 'source_split_sha256': digest(split_path),
               'adaptation_manifest_sha256': digest(adaptation_manifest), 'target_test_manifest_sha256': digest(target_manifest)}
    (OUT / 'private_local_provenance.json').write_text(json.dumps(private, indent=2), encoding='utf-8')
    paths = sorted(p for p in work.rglob('*') if p.is_file())
    manifest = {'status': 'lossless_allowlisted_ADAST_training_inputs', 'source_split_sha256': digest(split_path),
                'source_valid_epochs': 195469, 'source_records': 153, 'folds': role_summary,
                'adaptation_subjects': 5, 'adaptation_epochs': 4989, 'full_record_label_independent': True,
                'target_test_data_included': False, 'target_labels_included': False,
                'participant_ids_included': False, 'raw_edf_included': False,
                'upstream_commit': cfg['upstream_commit'], 'upstream_files_sha256': upstream_hashes,
                'files': {p.relative_to(work).as_posix(): digest(p) for p in paths}}
    (work / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Compressing lossless source signals and label-free adaptation; no test data', flush=True)
    with zipfile.ZipFile(archive.with_suffix('.tmp.zip'), 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths + [work / 'manifest.json']:
            z.write(p, arcname=p.relative_to(work).as_posix())
    archive.with_suffix('.tmp.zip').replace(archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist()) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Archive CRC or allowlist differs')
    result = {'status': 'passed', 'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
              'target_labels_included': False, 'target_test_data_included': False, 'participant_ids_included': False,
              'adaptation_subjects': 5, 'adaptation_epochs': 4989, 'source_folds': 10,
              'manifest_sha256': digest(work / 'manifest.json')}
    (OUT / 'bundle_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
