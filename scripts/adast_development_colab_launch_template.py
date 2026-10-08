# Four fresh matched ADAST development arms, CUDA only, no source outer test or SHHS test.
import hashlib, json, pathlib, shutil, subprocess, sys, time, zipfile
from google.colab import files
import torch
assert torch.cuda.is_available(), 'CUDA required; no CPU fallback.'
parts = PARTS_PLACEHOLDER
expected_sha = SHA_PLACEHOLDER
expected_bytes = BYTES_PLACEHOLDER
root = pathlib.Path('/content')
work = root / 'sleeptcn_adast_development_20261004'
output = root / 'adast_development_20261004' / 'results'
assert not work.exists() and not output.exists(), 'Existing attempt retained; no automatic restart.'
print('GPU', torch.cuda.get_device_name(0), 'TORCH', str(torch.__version__), flush=True)
print('WAITING FOR AUTHORIZED SOURCE TRAIN/VALIDATION + FIVE UNLABELLED ADAPTATION PEOPLE. NO TRAINING YET.', flush=True)
started = time.monotonic()
while any(not (root/p['name']).exists() or (root/p['name']).stat().st_size != p['bytes'] for p in parts):
    assert time.monotonic()-started < 3600, 'Upload timeout; no training started; preserve uploaded files.'
    time.sleep(2)
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()
archive = root / 'SleepTCN_ADAST_Development_20261004.zip'
assert not archive.exists(), 'Preserve existing archive.'
with archive.open('xb') as destination:
    for part in parts:
        path=root/part['name']
        assert file_sha(path)==part['sha256'], 'Upload part hash differs.'
        with path.open('rb') as stream:
            shutil.copyfileobj(stream,destination,1024*1024)
assert archive.stat().st_size==expected_bytes and file_sha(archive)==expected_sha
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('manifest.json'))
    assert not any(manifest[k] for k in ('source_outer_test_included','target_test_data_included','target_labels_included','participant_ids_included'))
    assert manifest['adaptation_subjects']==5
    assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(manifest['files'])|{'manifest.json'}
    for member in z.infolist():
        name=member.orig_filename
        relative=pathlib.PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and ':' not in name
        destination=work/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        with z.open(member) as stream,destination.open('xb') as target:
            shutil.copyfileobj(stream,target,1024*1024)
        if name!='manifest.json':
            assert file_sha(destination)==manifest['files'][name], 'Payload hash differs.'
print('ALLOWLIST/HASHES VERIFIED. TRAIN/VALIDATION ONLY; NO OUTER TEST; NO SHHS TEST OR TARGET LABELS.',flush=True)
print('START FOUR FRESH CUDA ARMS: LIMITED/FULL SOURCE × SOURCE-ONLY/ADAST. SHARED FIVE-HOUR LIMIT.',flush=True)
command=[sys.executable,'-u',str(work/'run_adast_development_cuda.py'),'--output',str(output),'--max-seconds','18000']
with (work/'launcher.log').open('x',encoding='utf8') as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True)
        log.write(line)
        log.flush()
    returncode=process.wait()
result_archive=output.parent/'SleepTCN_ADAST_Development_Results_20261004.zip'
if result_archive.exists():
    print('DOWNLOAD_SHA256',file_sha(result_archive),flush=True)
    files.download(str(result_archive))
assert returncode==0,'Stopped/failed; saved epoch checkpoints retained; do not restart automatically.'
print('FOUR ADAST DEVELOPMENT ARMS COMPLETE. SOURCE VALIDATION ONLY; NOT SHHS PERFORMANCE.',flush=True)
