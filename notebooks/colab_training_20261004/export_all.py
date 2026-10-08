# Export all eight newly trained checkpoints and their exact per-fold archives.
import hashlib, json, pathlib, zipfile
from google.colab import files
root = pathlib.Path('/content/sleeptcn_training_20261004/results')
rows, archives = [], {}
for fold in range(6, 10):
    folder = root / f'fold_{fold:02d}'
    result = json.loads((folder / 'aggregate_results.json').read_bytes())
    verification = json.loads((folder / 'verification.json').read_bytes())
    assert result['status'] == 'complete_source_only_cuda_pair'
    assert verification['status'] == 'passed'
    assert hashlib.sha256((folder / 'aggregate_results.json').read_bytes()).hexdigest() == verification['aggregate_results_sha256']
    for arm, selection in result['selections'].items():
        assert hashlib.sha256((folder / arm / 'best.pt').read_bytes()).hexdigest() == selection['checkpoint_sha256']
        rows.append({'fold': fold, 'arm': arm, 'epochs_completed': selection['epochs_completed'],
                     'selected_epoch': selection['selected_epoch'],
                     'train_validation_seconds': selection['training_and_validation_seconds']})
    path = root / f'SleepTCN_Fold{fold:02d}_CUDA_Checkpoints.zip'
    archives[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {'status': 'complete_source_only_four_cuda_pairs', 'new_selected_checkpoints': 8,
            'target_access': False, 'rows': rows, 'archives_sha256': archives}
(root / 'campaign_export_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
combined = pathlib.Path('/content/SleepTCN_CUDA_Folds06_09_20261004.zip')
with zipfile.ZipFile(combined, 'w', compression=zipfile.ZIP_STORED) as z:
    for name in archives:
        z.write(root / name, arcname=name)
    z.write(root / 'campaign_export_manifest.json', arcname='campaign_export_manifest.json')
print('ALL 8 REMAINING SOURCE MODELS TRAINED; SHHS NOT EVALUATED')
print('fold  arm          epochs  selected  train+validation_seconds')
for row in rows:
    print(f"{row['fold']:4}  {row['arm']:12} {row['epochs_completed']:6} {row['selected_epoch']:9} {row['train_validation_seconds']:25.2f}")
print('TOTAL_TRAIN_VALIDATION_SECONDS', round(sum(row['train_validation_seconds'] for row in rows), 2))
print('COMBINED_ARCHIVE_SHA256', hashlib.sha256(combined.read_bytes()).hexdigest())
files.download(str(combined))
