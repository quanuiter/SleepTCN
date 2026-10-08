"""Sanitize fold-1 roles into a new temporary C-drive payload, never overwrite runs."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'runs/adast_small_confirmation_20261004'
OLD = ROOT/'runs/colab_adast_20261004/payload'
BASE = ROOT/'runs/adast_development_20261004/payload'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def validate_roles(roles, source_size):
    required = {'train', 'validation', 'test'}
    if not required <= set(roles):
        raise ValueError('Missing source role')
    for key in required:
        value = roles[key]
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer) or len(np.unique(value)) != len(value):
            raise ValueError('Invalid/duplicate role index')
        if len(value) == 0 or value.min() < 0 or value.max() >= source_size:
            raise ValueError('Invalid source role range')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if np.intersect1d(roles[a], roles[b]).size:
            raise ValueError('Source roles overlap')

def build():
    if RECORD.exists():
        raise FileExistsError('Preserve existing confirmation attempt')
    cfg = json.loads((ROOT/'configs/adast_small_confirmation_v1_20261004.json').read_bytes())
    previous = ROOT/'runs/adast_loss_ablation_20261004/verified_results'
    proof = json.loads((previous/'independent_verification.json').read_bytes())
    if proof['status'] != 'passed' or proof['aggregate_results_sha256'] != cfg['previous_loss_aggregate_sha256'] or sha(previous/'aggregate_results.json') != cfg['previous_loss_aggregate_sha256']:
        raise ValueError('Previous loss experiment verification gate failed')
    old_verification = json.loads((OLD.parent/'bundle_verification.json').read_bytes())
    if sha(OLD/'manifest.json') != old_verification['manifest_sha256']:
        raise ValueError('Historical source manifest changed')
    manifest = json.loads((OLD/'manifest.json').read_bytes())
    if any(manifest[k] for k in ('target_test_data_included','target_labels_included','participant_ids_included')):
        raise ValueError('Forbidden historical target role')
    role_name = 'data/fold_01_roles.npz'
    for name in ('data/source_x.npy','data/source_y.npy',role_name,'data/adaptation_x.npy'):
        if sha(OLD/name) != manifest['files'][name]:
            raise ValueError('Historical frozen source input changed: '+name)
    frozen = json.loads((BASE/'manifest.json').read_bytes())
    copies = ['adast_reference_adapter.py','development_metrics.py','run_adast_development_cuda.py',
              'upstream/models.py','upstream/configs.py','upstream/utils.py','upstream/LICENSE']
    for name in copies:
        if sha(BASE/name) != frozen['files'][name]:
            raise ValueError('Verified base code changed: '+name)
    if sha(BASE/'run_adast_development_cuda.py') != cfg['previous_development_runner_sha256']:
        raise ValueError('Frozen measured training runner differs')
    source = np.load(OLD/'data/source_x.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(OLD/'data/source_y.npy', mmap_mode='r', allow_pickle=False)
    with np.load(OLD/role_name, allow_pickle=False) as z:
        roles = {k:z[k].copy() for k in ('train','validation','test')}
    validate_roles(roles, len(source))
    if len(roles['train']) != cfg['source_train_epochs'] or len(roles['validation']) != cfg['source_validation_epochs']:
        raise ValueError('Locked fold-1 sizes differ')
    if source.shape != (len(labels),3000) or source.dtype != np.float32 or labels.dtype != np.int64:
        raise ValueError('Historical source schema differs')
    temp_parent = Path(tempfile.gettempdir()).resolve()
    if shutil.disk_usage(temp_parent).free < 8_000_000_000:
        raise RuntimeError('Need 8 GB free in the temporary drive; do not delete historical artifacts')
    storage = Path(tempfile.mkdtemp(prefix='SleepTCN_ADAST_Confirm_20261004_', dir=temp_parent))
    work = storage/'payload'
    (work/'data').mkdir(parents=True)
    RECORD.mkdir(parents=True)
    pointer = {'status':'building','storage_root':storage.as_posix(),'payload_root':work.as_posix(),
               'data_upload_consent':'Direct user confirmation: fold-1 sanitized train/validation plus the same five unlabelled SHHS adaptation people to notebook 19wRkGwBOlqm4voA-z-6Al52gzIaRDW68',
               'fold':1,'seed':123,'source_outer_test_included':False,'target_test_data_included':False}
    (RECORD/'storage.json').write_text(json.dumps(pointer,indent=2)+'\n',encoding='utf8')
    for role in ('train','validation'):
        indices = roles[role]
        destination = np.lib.format.open_memmap(work/f'data/{role}_x.npy', mode='w+', dtype=np.float32, shape=(len(indices),3000))
        for start in range(0,len(indices),2048):
            destination[start:start+2048] = source[indices[start:start+2048]]
        destination.flush()
        del destination
        np.save(work/f'data/{role}_y.npy', labels[indices].astype(np.int64), allow_pickle=False)
        print('PREPARED',role,len(indices),'source epochs; no IDs or outer test',flush=True)
    shutil.copyfile(OLD/'data/adaptation_x.npy', work/'data/adaptation_x.npy')
    for name in copies:
        target = work/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(BASE/name,target)
    shutil.copyfile(ROOT/'scripts/run_adast_small_confirmation_cuda.py',work/'run_adast_small_confirmation_cuda.py')
    shutil.copyfile(ROOT/'configs/adast_small_confirmation_v1_20261004.json',work/'protocol.json')
    paths = sorted(p for p in work.rglob('*') if p.is_file())
    fresh = {'status':'allowlisted_fold1_confirmation_no_outer_test','fold':1,'seed':123,
        'source_outer_test_included':False,'target_test_data_included':False,'target_labels_included':False,
        'participant_ids_included':False,'raw_edf_included':False,'adaptation_subjects':5,'adaptation_epochs':4989,
        'source_train_epochs':len(roles['train']),'source_validation_epochs':len(roles['validation']),
        'train_class_counts':np.bincount(labels[roles['train']],minlength=5).tolist(),
        'validation_class_counts':np.bincount(labels[roles['validation']],minlength=5).tolist(),
        'upstream_commit':frozen['upstream_commit'],'upstream_files_sha256':frozen['upstream_files_sha256'],
        'adaptation_input_matches_previous':sha(work/'data/adaptation_x.npy') == frozen['files']['data/adaptation_x.npy'],
        'files':{p.relative_to(work).as_posix():sha(p) for p in paths}}
    if not fresh['adaptation_input_matches_previous']:
        raise ValueError('Adaptation participants/data changed')
    (work/'manifest.json').write_text(json.dumps(fresh,indent=2)+'\n',encoding='utf8')
    archive = storage/'SleepTCN_ADAST_Small_Confirmation_Input_20261004.zip'
    print('COMPRESSING AUTHORIZED FOLD-1 PAYLOAD',flush=True)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in paths+[work/'manifest.json']:
            z.write(path,path.relative_to(work).as_posix())
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != set(fresh['files'])|{'manifest.json'} or len(z.namelist()) != len(fresh['files'])+1 or z.testzip() is not None:
            raise ValueError('Input ZIP allowlist/CRC failed')
    binding = {'status':'passed','archive_filename':archive.name,'archive_path':archive.as_posix(),
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'manifest_sha256':sha(work/'manifest.json'),
        'payload_root':work.as_posix(),'source_outer_test_included':False,'target_test_data_included':False,
        'target_labels_included':False,'participant_ids_included':False,'fold_roles_disjoint':True}
    (RECORD/'bundle_verification.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf8')
    (RECORD/'private_local_provenance.json').write_text(json.dumps({
        'old_manifest_sha256':old_verification['manifest_sha256'],'fold_roles_sha256':manifest['files'][role_name],
        'train_indices_sha256':hashlib.sha256(roles['train'].astype(np.int64).tobytes()).hexdigest(),
        'validation_indices_sha256':hashlib.sha256(roles['validation'].astype(np.int64).tobytes()).hexdigest(),
        'test_indices_sha256':hashlib.sha256(roles['test'].astype(np.int64).tobytes()).hexdigest()},indent=2)+'\n',encoding='utf8')
    parts = []
    with archive.open('rb') as stream:
        index = 0
        while block := stream.read(256*1024*1024):
            part = storage/f'SleepTCN_ADAST_Small_Confirmation_20261004.part{index:02d}'
            with part.open('xb') as sink:
                sink.write(block)
            parts.append({'name':part.name,'path':part.as_posix(),'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
            index += 1
    (RECORD/'upload_parts.json').write_text(json.dumps({**binding,'parts':parts},indent=2)+'\n',encoding='utf8')
    write_launcher(binding, parts)
    pointer.update(status='bundle_verified_ready_for_upload',archive_sha256=binding['archive_sha256'])
    (RECORD/'storage.json').write_text(json.dumps(pointer,indent=2)+'\n',encoding='utf8')
    print(json.dumps(binding,indent=2),flush=True)

def write_launcher(binding, parts):
    destination = RECORD/'colab_launch.py'
    if destination.exists():
        raise FileExistsError('Preserve existing launcher')
    if sha(Path(binding['archive_path'])) != binding['archive_sha256']:
        raise ValueError('Frozen bundle changed')
    for part in parts:
        path = Path(part['path'])
        if path.stat().st_size != part['bytes'] or sha(path) != part['sha256']:
            raise ValueError('Frozen upload part changed')
    template = (ROOT/'scripts/adast_small_confirmation_colab_launch_template.py').read_text(encoding='utf8')
    replacements = {'PARTS_PLACEHOLDER':repr([{k:v for k,v in p.items() if k!='path'} for p in parts]),
                    'SHA_PLACEHOLDER':repr(binding['archive_sha256']), 'BYTES_PLACEHOLDER':str(binding['archive_bytes']),
                    'MANIFEST_SHA_PLACEHOLDER':repr(binding['manifest_sha256'])}
    for token in sorted(replacements, key=len, reverse=True):
        template = template.replace(token,replacements[token])
    compile(template,'confirmation_colab_launch.py','exec')
    destination.write_text(template,encoding='utf8')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher-only',action='store_true',help='Prepare missing launcher from a verified unchanged input, no training or rebuild')
    args = parser.parse_args()
    if args.launcher_only:
        binding = json.loads((RECORD/'bundle_verification.json').read_bytes())
        parts = json.loads((RECORD/'upload_parts.json').read_bytes())['parts']
        write_launcher(binding,parts)
        pointer = json.loads((RECORD/'storage.json').read_bytes())
        pointer.update(status='bundle_verified_ready_for_upload',archive_sha256=binding['archive_sha256'])
        (RECORD/'storage.json').write_text(json.dumps(pointer,indent=2)+'\n',encoding='utf8')
        print(json.dumps(binding,indent=2),flush=True)
    else:
        build()
