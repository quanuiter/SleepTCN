"""Allowlisted source train/validation feature bundle, no outer test or SHHS."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sleeptcn.preprocessing import sha256_file
from sleeptcn.development import softened_class_weights, require_development_roles
from sleeptcn.revision_campaign import write_once_json


def build(output=None, archive_name='SleepTCN_Weighted_Development_20261004.zip'):
    output = (output or ROOT / 'runs/development_20261004/weighted_bundle').resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or Path(archive_name).name != archive_name:
        raise ValueError('Require a new runs subdirectory and simple archive name')
    if output.exists():
        raise FileExistsError('Existing bundle: verify, do not overwrite')
    protocol = ROOT / 'configs/teacher_revision_development_v2.json'
    cfg = json.loads(protocol.read_bytes())['weighted_development']
    cache = ROOT / 'data/cache/recovered_e3_fold0_features_20261001'
    original = json.loads((cache / 'private_manifest.json').read_bytes())
    split_path = ROOT / 'data/splits/sleepedf_sc_10fold_seed42_v2.json'
    if sha256_file(split_path) != original['summary']['source_split_sha256']:
        raise ValueError('Source split hash differs')
    split = json.loads(split_path.read_bytes())['outer_runs'][cfg['fold']]
    if set(split['train']['subject_ids']) & set(split['validation']['subject_ids']):
        raise ValueError('Subject overlap')
    encoder_hash = original['summary']['encoder_sha256']
    expected_encoder = json.loads((ROOT / 'runs/teacher_revision_cpu_20261001/recovered_e3_fold0/execution_specification.json').read_bytes())['checkpoint_sha256']['extractor']
    if encoder_hash != expected_encoder:
        raise ValueError('Verified frozen encoder differs')
    files = {p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted((ROOT / 'src/sleeptcn').rglob('*.py'))}
    for path in [protocol, ROOT / 'scripts/run_weighted_development_cuda.py']:
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    records, counts = [], np.zeros(5,dtype=np.int64)
    for role in ('train','validation'):
        entries = sorted([e for e in original['records'] if e['role'] == role],key=lambda e:e['record_key'])
        if {e['record_key'] for e in entries} != set(split[role]['record_keys']):
            raise ValueError('Source role membership differs')
        support = 0
        for ordinal, entry in enumerate(entries):
            path = Path(entry['path'])
            if sha256_file(path) != entry['sha256'] or entry['encoder_sha256'] != encoder_hash:
                raise ValueError('Cached features changed')
            with np.load(path,allow_pickle=False) as z:
                x,y = z['features'].copy(),z['labels'].astype(np.int64)
            if x.shape != (len(y),128) or not np.isfinite(x).all() or not np.isin(y,[-1,0,1,2,3,4]).all():
                raise ValueError('Feature cache invalid')
            n = int((y >= 0).sum())
            if n != entry['valid_epochs']:
                raise ValueError('Valid support differs')
            support += n
            if role == 'train':
                counts += np.bincount(y[y>=0],minlength=5)
            name = f'data/{role}/{ordinal:03d}.npz'
            stream = io.BytesIO()
            np.savez_compressed(stream,features=x,labels=y)
            files[name] = stream.getvalue()
            records.append({'path':name,'role':role,'epochs':len(y),'valid_epochs':n})
        if support != split[role]['valid_epochs']:
            raise ValueError('Source role support differs')
    np.testing.assert_array_equal(counts,original['summary']['train_class_counts'])
    require_development_roles(records)
    manifest = {'status':'source_only_development_bundle','fold':cfg['fold'],'encoder_sha256':encoder_hash,
                'source_split_sha256':sha256_file(split_path),'source_cache_manifest_sha256':sha256_file(cache/'private_manifest.json'),
                'train_class_counts':counts.tolist(),'records':records,'outer_test_included':False,
                'target_data_included':False,'participant_ids_included':False,'raw_eeg_included':False,
                'files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
    output.mkdir(parents=True)
    archive = output / archive_name
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():
            z.writestr(name,data)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist()) != set(files)|{'manifest.json'}:
            raise ValueError('Archive CRC/allowlist differs')
        for name,digest in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('Archive content differs')
    verification = {'status':'passed','archive_sha256':sha256_file(archive),'archive_bytes':archive.stat().st_size,
                    'outer_test_included':False,'target_data_included':False,'participant_ids_included':False,
                    'train_class_counts':counts.tolist(),
                    'weights_by_power':{str(p):softened_class_weights(counts,p).tolist() for p in cfg['powers']},
                    'source_role_valid_epochs':{r:sum(e['valid_epochs'] for e in records if e['role']==r) for r in ('train','validation')}}
    write_once_json(output / 'bundle_verification.json',verification)
    print(json.dumps(verification,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--archive-name', default='SleepTCN_Weighted_Development_20261004.zip')
    args = parser.parse_args()
    build(args.output, args.archive_name)
