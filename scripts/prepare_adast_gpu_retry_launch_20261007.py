"""Prepare a fresh Colab launcher, preserving the frozen scientific payload."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1]
    prior = root / 'runs/adast_fullsource_completion_20261006'
    record = root / 'runs/adast_fullsource_retry_20261007'
    record.mkdir(exist_ok=True)
    source = (prior / 'colab_launch.py').read_text(encoding='utf8')
    source = source.replace("'sleeptcn_adast_fullsource_completion_20261006'",
                            "'sleeptcn_adast_fullsource_retry_20261007'")
    source = source.replace("'adast_fullsource_completion_20261006' / 'results'",
                            "'adast_fullsource_retry_20261007' / 'results'")
    marker = "with (work / 'launcher.log').open('x', encoding='utf8') as log:"
    backup = '''# Orchestration only: export immutable completed fold pairs during training.
# No changes to models, sampling, loss, checkpoint selection or training budget.
backed_up = set()
def backup_completed_pairs():
    for fold in range(2, 10):
        folder = output / f'fold_{fold:02d}'
        if fold in backed_up or not (folder / 'pair_verification.json').exists():
            continue
        paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.pt', '.json', '.npz', '.npy', '.log'})
        paths.append(output / 'execution_specification.json')
        destination = output.parent / f'SleepTCN_ADAST_Fullsource_Retry_20261007_Fold_{fold:02d}.zip'
        with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in paths:
                z.write(p, p.relative_to(output).as_posix())
            z.writestr('export_manifest.json', json.dumps({p.relative_to(output).as_posix(): sha(p) for p in paths}, indent=2))
        backed_up.add(fold)
        print('FOLD_BACKUP_SHA256', fold, sha(destination), 'BYTES', destination.stat().st_size, flush=True)
        try:
            files.download(str(destination))
        except Exception as error:
            print('FOLD_BACKUP_DOWNLOAD_FAILED', fold, type(error).__name__, flush=True)
        print('FOLD_BACKUP_REQUESTED', fold, flush=True)

'''
    assert source.count(marker) == 1
    source = source.replace(marker, backup + marker)
    source = source.replace('        log.flush()\n    returncode = process.wait()',
                            '        log.flush()\n        backup_completed_pairs()\n    returncode = process.wait()\n    backup_completed_pairs()')
    assert 'backup_completed_pairs()\n    returncode' in source
    target = record / 'colab_launch.py'
    assert not target.exists(), 'Preserve existing retry preparation.'
    target.write_text(source, encoding='utf8')
    info = json.loads((prior / 'upload_parts.json').read_bytes())
    storage_root = Path(info['payload_root']).parent / 'retry_results_20261007'
    storage_root.mkdir(exist_ok=True)
    (record / 'preparation.json').write_text(json.dumps({
        'status': 'prepared_not_uploaded_not_training',
        'user_authorization': 'Continue GPU training, 2026-10-07',
        'previous_runtime_recovery': 'GPU T4 available; both previous work/results paths absent',
        'models_to_rerun': 16, 'folds': list(range(2, 10)),
        'reuse_local_folds': [0, 1], 'scientific_payload_changed': False,
        'input_archive_sha256': info['archive_sha256'],
        'manifest_sha256': info['manifest_sha256'],
        'launcher_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'local_results_root': str(storage_root),
        'max_training_seconds': 18000, 'cpu_training_fallback': False,
        'completed_pair_backup': True, 'backup_download_is_not_confirmed_local_retention': True,
        'previous_results_not_reused_from_log': True,
    }, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'launcher': str(target), 'results_root': str(storage_root)}))


if __name__ == '__main__':
    main()
