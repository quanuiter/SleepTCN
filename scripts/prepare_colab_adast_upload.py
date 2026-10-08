"""Split verified lossless bundle into browser-sized parts; generate reviewed cell."""
import json
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/colab_adast_20261004'
SIZE = 256 * 1024 * 1024


def main():
    verification = json.loads((OUT / 'bundle_verification.json').read_bytes())
    archive = OUT / 'SleepTCN_ADAST_10Fold_20261004.zip'
    parts = []
    with archive.open('rb') as stream:
        for index in range((verification['archive_bytes'] + SIZE - 1) // SIZE):
            data = stream.read(SIZE)
            name = f'SleepTCN_ADAST_20261004.part{index:02d}'
            path = OUT / name
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('Existing part changed')
            else:
                path.write_bytes(data)
            parts.append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        if stream.read(1):
            raise ValueError('Archive support changed')
    manifest = {'archive_sha256': verification['archive_sha256'], 'archive_bytes': verification['archive_bytes'], 'parts': parts}
    (OUT / 'upload_parts.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cell = '''# ADAST + matched source-only, all ten folds, CUDA only. No SHHS test data/true labels.
import hashlib, json, pathlib, shutil, subprocess, sys, time, zipfile
from google.colab import files
parts = PARTS_JSON
expected_archive_sha256 = ARCHIVE_SHA
expected_archive_bytes = ARCHIVE_BYTES
root = pathlib.Path('/content')
print('Waiting for lossless EEG upload parts. No training until all hashes match.', flush=True)
started = time.monotonic()
while any(not (root / p['name']).exists() or (root / p['name']).stat().st_size != p['bytes'] for p in parts):
    assert time.monotonic() - started < 3600, 'Upload timeout; no training started, uploaded files retained.'
    time.sleep(2)
def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
archive_path = root / 'SleepTCN_ADAST_10Fold_20261004.zip'
if not archive_path.exists():
    temporary = archive_path.with_suffix('.tmp.zip')
    with temporary.open('wb') as destination:
        for part in parts:
            path = root / part['name']
            assert file_sha(path) == part['sha256'], 'Upload part SHA256 differs; stop.'
            with path.open('rb') as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(archive_path)
assert archive_path.stat().st_size == expected_archive_bytes
assert file_sha(archive_path) == expected_archive_sha256
work = root / 'sleeptcn_adast_20261004'
with zipfile.ZipFile(archive_path) as z:
    manifest = json.loads(z.read('manifest.json'))
    assert not manifest['participant_ids_included'] and not manifest['target_labels_included']
    assert not manifest['target_test_data_included'] and manifest['adaptation_subjects'] == 5
    assert len(z.namelist()) == len(manifest['files']) + 1
    assert set(z.namelist()) == set(manifest['files']) | {'manifest.json'}
    for name in z.namelist():
        relative = pathlib.PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        destination = work / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if name != 'manifest.json':
                assert file_sha(destination) == manifest['files'][name]
            else:
                assert destination.read_bytes() == z.read(name)
        else:
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            with z.open(name) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if name != 'manifest.json':
                assert file_sha(temporary) == manifest['files'][name]
            temporary.replace(destination)
print('ALLOWLIST/HASHES VERIFIED: 10 source folds; 5 label-free adaptation people; NO target-test data.', flush=True)
output = root / 'sleeptcn_adast_results_20261004'
output.mkdir(exist_ok=True)
command = [sys.executable, '-u', str(work / 'run_colab_adast_training.py'), '--output', str(output), '--max-seconds', '18000']
with (output / 'run.log').open('a', encoding='utf-8') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / 'process.json').write_text(json.dumps({'pid': process.pid, 'command': command}), encoding='utf-8')
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    returncode = process.wait()
result_archive = root / 'SleepTCN_ADAST_CUDA_10Fold_Results_20261004.zip'
if result_archive.exists():
    files.download(str(result_archive))
assert returncode == 0, 'Stopped/failed; epoch checkpoints retained. Do not automatically restart.'
print('ALL 20 ADAST/SOURCE-ONLY CHECKPOINTS COMPLETE. SHHS TEST EVALUATION MUST BE LOCAL.', flush=True)
'''
    cell = cell.replace('PARTS_JSON', repr(parts)).replace('ARCHIVE_SHA', repr(verification['archive_sha256'])).replace('ARCHIVE_BYTES', str(verification['archive_bytes']))
    (OUT / 'colab_train_adast.py').write_text(cell, encoding='utf-8')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': cell.splitlines(keepends=True)}]}
    (OUT / 'SleepTCN_ADAST_Training_20261004.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
