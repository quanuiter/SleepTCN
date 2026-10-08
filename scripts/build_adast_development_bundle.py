"""De-identify fold-0 train/validation only from the already verified source payload."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/adast_development_20261004'
OLD=ROOT/'runs/colab_adast_20261004/payload'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def build():
    work=OUT/'payload'
    if work.exists():
        raise FileExistsError('Preserve existing development bundle; inspect instead of overwrite')
    OUT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(OUT).free<7_000_000_000:
        raise RuntimeError('Require 7 GB free for private lossless development bundle')
    old_ver=json.loads((OLD.parent/'bundle_verification.json').read_bytes())
    if sha(OLD/'manifest.json')!=old_ver['manifest_sha256']:
        raise ValueError('Historical verified manifest changed')
    old=json.loads((OLD/'manifest.json').read_bytes())
    if any(old[k] for k in ('target_test_data_included','target_labels_included','participant_ids_included')):
        raise ValueError('Historical payload contains forbidden target roles')
    needed=['data/source_x.npy','data/source_y.npy','data/fold_00_roles.npz','data/adaptation_x.npy',
            'upstream/models.py','upstream/configs.py','upstream/utils.py','upstream/LICENSE','run_colab_adast_training.py']
    for name in needed:
        if sha(OLD/name)!=old['files'][name]:
            raise ValueError('Historical locked file changed: '+name)
    cfg=json.loads((ROOT/'configs/adast_development_budget_v1_20261004.json').read_bytes())
    source_x=np.load(OLD/'data/source_x.npy',mmap_mode='r',allow_pickle=False)
    source_y=np.load(OLD/'data/source_y.npy',mmap_mode='r',allow_pickle=False)
    with np.load(OLD/'data/fold_00_roles.npz',allow_pickle=False) as roles:
        train=roles['train'].copy().astype(np.int64)
        validation=roles['validation'].copy().astype(np.int64)
        if np.intersect1d(train,validation).size or any(np.intersect1d(roles[a],roles['test']).size for a in ['train','validation']):
            raise ValueError('Source roles overlap')
    if len(train)!=cfg['source_train_epochs'] or len(validation)!=cfg['source_validation_epochs']:
        raise ValueError('Locked development role sizes differ')
    work.mkdir()
    (work/'data').mkdir()
    for role,indices in [('train',train),('validation',validation)]:
        target=np.lib.format.open_memmap(work/f'data/{role}_x.npy',mode='w+',dtype=np.float32,shape=(len(indices),3000))
        for start in range(0,len(indices),2048):
            target[start:start+2048]=source_x[indices[start:start+2048]]
        target.flush()
        del target
        np.save(work/f'data/{role}_y.npy',source_y[indices].astype(np.int64),allow_pickle=False)
        print('Prepared',role,len(indices),'source epochs; no identifiers',flush=True)
    shutil.copyfile(OLD/'data/adaptation_x.npy',work/'data/adaptation_x.npy')
    (work/'upstream').mkdir()
    for name in ('models.py','configs.py','utils.py','LICENSE'):
        shutil.copyfile(OLD/'upstream'/name,work/'upstream'/name)
    shutil.copyfile(OLD/'run_colab_adast_training.py',work/'adast_reference_adapter.py')
    shutil.copyfile(ROOT/'src/sleeptcn/metrics.py',work/'development_metrics.py')
    shutil.copyfile(ROOT/'scripts/run_adast_development_cuda.py',work/'run_adast_development_cuda.py')
    shutil.copyfile(ROOT/'configs/adast_development_budget_v1_20261004.json',work/'protocol.json')
    paths=sorted(p for p in work.rglob('*') if p.is_file())
    manifest={'status':'allowlisted_fold0_development_inputs_no_outer_test',
        'source_outer_test_included':False,'target_test_data_included':False,'target_labels_included':False,
        'participant_ids_included':False,'raw_edf_included':False,'adaptation_subjects':5,'adaptation_epochs':4989,
        'source_train_epochs':len(train),'source_validation_epochs':len(validation),
        'train_class_counts':np.bincount(source_y[train],minlength=5).tolist(),
        'validation_class_counts':np.bincount(source_y[validation],minlength=5).tolist(),
        'upstream_commit':old['upstream_commit'],'upstream_files_sha256':old['upstream_files_sha256'],
        'files':{p.relative_to(work).as_posix():sha(p) for p in paths}}
    (work/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    archive=OUT/'SleepTCN_ADAST_Development_20261004.zip'
    print('Compressing source train/validation and previously authorized unlabelled adaptation',flush=True)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in paths+[work/'manifest.json']:
            z.write(p,p.relative_to(work).as_posix())
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(manifest['files'])|{'manifest.json'} or z.testzip() is not None:
            raise ValueError('Bundle CRC/allowlist failed')
    verification={'status':'passed','archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'manifest_sha256':sha(work/'manifest.json'),'source_outer_test_included':False,
        'target_test_data_included':False,'target_labels_included':False,'participant_ids_included':False}
    (OUT/'bundle_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf8')
    # Private positional provenance remains local; it is not in the upload archive.
    (OUT/'private_local_provenance.json').write_text(json.dumps({'old_manifest_sha256':old_ver['manifest_sha256'],
        'source_train_indices_sha256':hashlib.sha256(train.tobytes()).hexdigest(),
        'source_validation_indices_sha256':hashlib.sha256(validation.tobytes()).hexdigest()},indent=2),encoding='utf8')
    parts=[]
    part_size=256*1024*1024
    with archive.open('rb') as stream:
        index=0
        while data:=stream.read(part_size):
            path=OUT/f'SleepTCN_ADAST_Development_20261004.part{index:02d}'
            with path.open('xb') as destination:
                destination.write(data)
            parts.append({'name':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            index+=1
    (OUT/'upload_parts.json').write_text(json.dumps({**verification,'parts':parts},indent=2),encoding='utf8')
    template=(ROOT/'scripts/adast_development_colab_launch_template.py').read_text(encoding='utf8')
    cell=template.replace('PARTS_PLACEHOLDER',repr(parts)).replace('SHA_PLACEHOLDER',repr(verification['archive_sha256'])).replace('BYTES_PLACEHOLDER',str(verification['archive_bytes']))
    (OUT/'colab_launch.py').write_text(cell,encoding='utf8')
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    build()
