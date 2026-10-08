# Three fresh models, fold 1: source-only, ADAST reference, retained source learning.
import hashlib, json, pathlib, shutil, subprocess, sys, time, zipfile
import torch
from google.colab import files
assert torch.cuda.is_available(), 'CUDA unavailable; no CPU training fallback.'
root = pathlib.Path('/content')
work = root/'sleeptcn_adast_small_confirmation_20261004'
output = root/'adast_small_confirmation_20261004'/'results'
parts = PARTS_PLACEHOLDER
expected_sha = SHA_PLACEHOLDER
expected_bytes = BYTES_PLACEHOLDER
expected_manifest_sha = MANIFEST_SHA_PLACEHOLDER
assert not work.exists() and not output.exists(), 'Preserve existing attempt; no automatic restart.'
print('GPU', torch.cuda.get_device_name(0), 'TORCH', str(torch.__version__), flush=True)
print('WAITING FOR AUTHORIZED FOLD-1 TRAIN/VALIDATION AND FIVE UNLABELLED ADAPTATION PEOPLE. NO TRAINING YET.', flush=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

started = time.monotonic()
while any(not (root/p['name']).exists() or (root/p['name']).stat().st_size != p['bytes'] for p in parts):
    assert time.monotonic()-started < 3600, 'Upload timeout; no training started.'
    time.sleep(2)
archive = root/'SleepTCN_ADAST_Small_Confirmation_Input_20261004.zip'
assert not archive.exists(), 'Preserve existing input archive.'
with archive.open('xb') as sink:
    for part in parts:
        path = root/part['name']
        assert sha(path) == part['sha256'], 'Uploaded part differs.'
        with path.open('rb') as stream:
            shutil.copyfileobj(stream, sink, 1024*1024)
assert archive.stat().st_size == expected_bytes and sha(archive) == expected_sha
with zipfile.ZipFile(archive) as z:
    manifest_data = z.read('manifest.json')
    assert hashlib.sha256(manifest_data).hexdigest() == expected_manifest_sha
    manifest = json.loads(manifest_data)
    assert manifest['fold'] == 1 and manifest['adaptation_subjects'] == 5
    assert not any(manifest[k] for k in ('source_outer_test_included', 'target_test_data_included', 'target_labels_included', 'participant_ids_included'))
    assert len(z.namelist()) == len(set(z.namelist()))
    assert set(z.namelist()) == set(manifest['files']) | {'manifest.json'}
    assert z.testzip() is None
    for member in z.infolist():
        name = member.orig_filename
        relative = pathlib.PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and ':' not in name
        path = work/name
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with z.open(member) as stream, path.open('xb') as sink:
            shutil.copyfileobj(stream, sink, 1024*1024)
        if name != 'manifest.json':
            assert sha(path) == manifest['files'][name], 'Frozen payload differs.'
print('ALLOWLIST AND ALL HASHES VERIFIED. START THREE FRESH FULL-SOURCE CUDA MODELS ON FOLD 1.', flush=True)
command = [sys.executable, '-u', str(work/'run_adast_small_confirmation_cuda.py'), '--output', str(output), '--max-seconds', '18000']
with (work/'launcher.log').open('x', encoding='utf8') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (work/'process.json').write_text(json.dumps({'pid':process.pid,'command':command}))
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    returncode = process.wait()
result_archive = output.parent/'SleepTCN_ADAST_Small_Confirmation_Results_20261004.zip'
if result_archive.exists():
    print('DOWNLOAD_SHA256', sha(result_archive), flush=True)
    files.download(str(result_archive))
assert returncode == 0, 'Stopped/failed; checkpoints retained; no automatic restart.'
print('THREE FOLD-1 CONFIRMATION MODELS COMPLETE. SOURCE VALIDATION ONLY.', flush=True)
