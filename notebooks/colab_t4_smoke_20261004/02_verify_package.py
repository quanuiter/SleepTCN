# 2. Verify the uploaded allowlisted package before extracting or running code.
import hashlib, json, pathlib, zipfile

archive_path = pathlib.Path('/content/SleepTCN_T4_Smoke_20261004.zip')
expected_sha256 = '63a2dc05d1888c412d99da39b24229a8fc7eae19afc63b16c4b838c5ef137f44'
assert archive_path.exists(), 'Upload SleepTCN_T4_Smoke_20261004.zip into the Files pane first.'
actual_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
assert actual_sha256 == expected_sha256, 'Package hash differs; stop and recheck the upload.'
work = pathlib.Path('/content/sleeptcn_t4_smoke_20261004')
allowed = {'src/sleeptcn/__init__.py', 'src/sleeptcn/models.py',
           'src/sleeptcn/training.py', 'scripts/benchmark_colab_tcn.py',
           'sample_records.npz', 'manifest.json'}
with zipfile.ZipFile(archive_path) as archive:
    assert set(archive.namelist()) == allowed
    assert archive.testzip() is None
    manifest = json.loads(archive.read('manifest.json'))
    assert manifest['target_data_included'] is False
    assert manifest['checkpoint_included'] is False
    assert manifest['participant_ids_included'] is False
    assert set(manifest['files']) == allowed - {'manifest.json'}
    for name, expected in manifest['files'].items():
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == expected
        destination = work / name
        if destination.exists():
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
print('PACKAGE_VERIFIED', actual_sha256)
print('Source samples only; no participant IDs, raw EEG, SHHS or campaign checkpoint.')

