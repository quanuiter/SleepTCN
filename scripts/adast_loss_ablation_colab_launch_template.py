# ADAST full-source loss ablations: one fresh reference + three single-factor changes.
import base64, hashlib, json, pathlib, shutil, subprocess, sys, time, zipfile
import torch
from google.colab import files
assert torch.cuda.is_available(), 'GPU unavailable; no CPU training fallback.'
root = pathlib.Path('/content')
old = root/'sleeptcn_adast_development_20261004'
work = root/'sleeptcn_adast_loss_ablation_20261004'
output = root/'adast_loss_ablation_20261004'/'results'
patch = root/PATCH_NAME_PLACEHOLDER
patch_sha = PATCH_SHA_PLACEHOLDER
patch_size = PATCH_SIZE_PLACEHOLDER
patch_data = PATCH_BASE64_PLACEHOLDER
base_parts = BASE_PARTS_PLACEHOLDER
base_archive_sha = BASE_ARCHIVE_SHA_PLACEHOLDER
base_manifest_sha = BASE_MANIFEST_SHA_PLACEHOLDER
assert not work.exists() and not output.exists(), 'Preserve existing attempt; no automatic restart.'
print('GPU', torch.cuda.get_device_name(0), 'TORCH', str(torch.__version__), flush=True)
print('VERIFYING CODE PATCH; REUSING THE SAME AUTHORIZED DATA.', flush=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def unpack(archive, destination, expected_names):
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == expected_names
        assert z.testzip() is None
        for member in z.infolist():
            name = member.orig_filename
            relative = pathlib.PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and ':' not in name
            target = destination/name
            assert not target.exists(), 'Preserve existing payload.'
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as stream, target.open('xb') as sink:
                shutil.copyfileobj(stream, sink, 1024*1024)

started = time.monotonic()
decoded = base64.b64decode(patch_data,validate=True)
assert len(decoded)==patch_size and hashlib.sha256(decoded).hexdigest()==patch_sha
if not patch.exists():
    with patch.open('xb') as sink:
        sink.write(decoded)
assert patch.stat().st_size==patch_size and sha(patch) == patch_sha
# An idle-disconnected VM may retain the data. If the VM was replaced, restore only the same frozen input.
if not old.exists():
    archive = root/'SleepTCN_ADAST_Development_20261004.zip'
    if not archive.exists():
        while any(not (root/p['name']).exists() or (root/p['name']).stat().st_size != p['bytes'] for p in base_parts):
            assert time.monotonic()-started < 3600, 'Original data upload missing; no training started.'
            time.sleep(2)
        with archive.open('xb') as destination:
            for part in base_parts:
                path = root/part['name']
                assert sha(path) == part['sha256']
                with path.open('rb') as stream:
                    shutil.copyfileobj(stream, destination, 1024*1024)
    assert sha(archive) == base_archive_sha
    with zipfile.ZipFile(archive) as z:
        original = json.loads(z.read('manifest.json'))
    unpack(archive, old, set(original['files']) | {'manifest.json'})
assert sha(old/'manifest.json') == base_manifest_sha
original = json.loads((old/'manifest.json').read_bytes())
assert not any(original[k] for k in ('source_outer_test_included','target_test_data_included','target_labels_included','participant_ids_included'))
with zipfile.ZipFile(patch) as z:
    manifest = json.loads(z.read('manifest.json'))
assert manifest['base_manifest_sha256'] == base_manifest_sha and manifest['adaptation_subjects'] == 5
assert not any(manifest[k] for k in ('source_outer_test_included','target_test_data_included','target_labels_included','participant_ids_included'))
assert manifest['reused_data_files'] == {k:v for k,v in original['files'].items() if k.startswith('data/')}
unpack(patch, work, (set(manifest['files'])-set(manifest['reused_data_files'])) | {'manifest.json'})
for name, expected in manifest['reused_data_files'].items():
    assert sha(old/name) == expected
    destination = work/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(old/name)
for name, expected in manifest['files'].items():
    assert sha(work/name) == expected, 'Payload hash differs.'
print('ALL INPUT AND CODE HASHES VERIFIED. START FOUR FRESH FULL-SOURCE CUDA ARMS.', flush=True)
command = [sys.executable, '-u', str(work/'run_adast_loss_ablation_cuda.py'), '--output', str(output), '--max-seconds', '18000']
with (work/'launcher.log').open('x', encoding='utf8') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    returncode = process.wait()
result_archive = output.parent/'SleepTCN_ADAST_Loss_Ablation_Results_20261004.zip'
if result_archive.exists():
    print('DOWNLOAD_SHA256', sha(result_archive), flush=True)
    files.download(str(result_archive))
assert returncode == 0, 'Stopped/failed; saved checkpoints retained; no automatic restart.'
print('FOUR ADAST LOSS ABLATION ARMS COMPLETE.', flush=True)
