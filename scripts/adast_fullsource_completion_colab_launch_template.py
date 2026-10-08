# Eight remaining full-source source-only/ADAST pairs; CUDA only.
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile
import torch
from google.colab import files

assert torch.cuda.is_available(), 'CUDA unavailable; no CPU training fallback.'
assert str(torch.__version__) == '2.11.0+cu130', 'PyTorch differs from the four reused models.'
root = pathlib.Path('/content')
work = root / 'sleeptcn_adast_fullsource_completion_20261006'
output = root / 'adast_fullsource_completion_20261006' / 'results'
parts = PARTS_PLACEHOLDER
expected_sha = SHA_PLACEHOLDER
expected_bytes = BYTES_PLACEHOLDER
expected_manifest_sha = MANIFEST_SHA_PLACEHOLDER
assert not work.exists() and not output.exists(), 'Preserve existing attempt; no automatic restart.'
assert shutil.disk_usage(root).free > 3 * expected_bytes + 5_000_000_000, 'Insufficient runtime storage.'
print('GPU', torch.cuda.get_device_name(0), 'TORCH', str(torch.__version__), flush=True)
print('WAITING FOR INPUT PARTS. NO TRAINING STARTED.', flush=True)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


started = time.monotonic()
while any(not (root / p['name']).exists() or (root / p['name']).stat().st_size != p['bytes'] for p in parts):
    assert time.monotonic() - started < 3600, 'Upload timeout; no training started.'
    time.sleep(2)
archive = root / 'SleepTCN_ADAST_Fullsource_Completion_Input_20261006.zip'
assert not archive.exists(), 'Preserve existing input archive.'
with archive.open('xb') as sink:
    for part in parts:
        path = root / part['name']
        assert sha(path) == part['sha256'], 'Uploaded part differs.'
        with path.open('rb') as stream:
            shutil.copyfileobj(stream, sink, 8 * 1024 * 1024)
assert archive.stat().st_size == expected_bytes and sha(archive) == expected_sha, 'Input archive differs.'
with zipfile.ZipFile(archive) as zipped:
    manifest_data = zipped.read('manifest.json')
    assert hashlib.sha256(manifest_data).hexdigest() == expected_manifest_sha
    manifest = json.loads(manifest_data)
    assert manifest['folds_to_train'] == list(range(2, 10)) and manifest['adaptation_subjects'] == 5
    assert manifest['source_all_records_included'] and manifest['source_role_training_restricted']
    assert not any(manifest[k] for k in ('target_test_data_included', 'target_labels_included', 'participant_ids_included', 'raw_edf_included'))
    assert len(zipped.namelist()) == len(set(zipped.namelist()))
    assert set(zipped.namelist()) == set(manifest['files']) | {'manifest.json'}
    assert zipped.testzip() is None
    for member in zipped.infolist():
        name = member.orig_filename
        relative = pathlib.PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and ':' not in name
        path = work / name
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipped.open(member) as stream, path.open('xb') as sink:
            shutil.copyfileobj(stream, sink, 8 * 1024 * 1024)
        if name != 'manifest.json':
            assert sha(path) == manifest['files'][name], 'Payload file differs.'
print('ALL INPUT HASHES VERIFIED. START 16 FULL-SOURCE CUDA MODELS: FOLDS 2-9.', flush=True)
command = [sys.executable, '-u', str(work / 'run_adast_fullsource_completion_cuda.py'),
           '--output', str(output), '--max-seconds', '18000']
with (work / 'launcher.log').open('x', encoding='utf8') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (work / 'process.json').write_text(json.dumps({'pid': process.pid, 'command': command}))
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    returncode = process.wait()
result_archive = output.parent / 'SleepTCN_ADAST_Fullsource_Completion_Results_20261006.zip'
if result_archive.exists():
    print('DOWNLOAD_SHA256', sha(result_archive), flush=True)
    files.download(str(result_archive))
assert returncode == 0, 'Stopped/failed; saved checkpoints retained; no automatic restart.'
print('EIGHT MATCHED FULL-SOURCE PAIRS COMPLETE. DOWNLOAD RESULTS FOR LOCAL VERIFICATION.', flush=True)
