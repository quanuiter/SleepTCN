"""Small immutable code patch; reuse byte-identical authorized development data."""
import hashlib
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/adast_development_20261004'
OUT = ROOT / 'runs/adast_loss_ablation_20261004'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build():
    if OUT.exists():
        raise FileExistsError('Preserve existing ablation bundle; no overwrite')
    proof = json.loads((BASE / 'verified_results/independent_verification.json').read_bytes())
    plan = json.loads((ROOT / 'configs/adast_loss_ablation_v1_20261004.json').read_bytes())
    old = BASE / 'payload'
    manifest = json.loads((old / 'manifest.json').read_bytes())
    verification = json.loads((BASE / 'bundle_verification.json').read_bytes())
    if proof['status'] != 'passed' or proof['aggregate_results_sha256'] != plan['verified_reference_aggregate_sha256']:
        raise ValueError('Development verification gate not satisfied')
    for path, expected in [(old/'protocol.json', plan['base_protocol_sha256']),
                           (old/'run_adast_development_cuda.py', plan['base_runner_sha256']),
                           (old/'manifest.json', plan['base_manifest_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Frozen development identity changed')
    for name, expected in manifest['files'].items():
        if sha(old/name) != expected:
            raise ValueError('Frozen input changed')
    if any(manifest[k] for k in ('source_outer_test_included', 'target_test_data_included', 'target_labels_included', 'participant_ids_included')):
        raise ValueError('Forbidden data role')
    work = OUT / 'payload'
    work.mkdir(parents=True)
    # Hardlinks avoid duplicating multi-GB EEG on a nearly full local drive; never edit these files.
    for name in manifest['files']:
        if name == 'protocol.json':
            continue
        destination = work/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith('data/'):
            os.link(old/name, destination)
        else:
            shutil.copyfile(old/name, destination)
    cfg = json.loads((old/'protocol.json').read_bytes())
    cfg.update({k: plan[k] for k in ('status', 'campaign_kind', 'arms', 'arm_overrides', 'base_protocol_sha256',
                                   'base_runner_sha256', 'base_manifest_sha256', 'verified_reference_aggregate_sha256')})
    cfg['automatic_ablation_or_ten_fold_extension'] = False
    (work/'protocol.json').write_text(json.dumps(cfg, indent=2), encoding='utf8')
    shutil.copyfile(ROOT/'scripts/run_adast_loss_ablation_cuda.py', work/'run_adast_loss_ablation_cuda.py')
    paths = sorted(p for p in work.rglob('*') if p.is_file())
    patched = {p.relative_to(work).as_posix(): sha(p) for p in paths}
    manifest.update(status='allowlisted_loss_ablation_reusing_verified_development_data', files=patched,
                    reused_data_files={k: v for k, v in patched.items() if k.startswith('data/')},
                    base_manifest_sha256=plan['base_manifest_sha256'])
    (work/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    archive = OUT/'SleepTCN_ADAST_Loss_Ablation_Patch_20261004.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in paths + [work/'manifest.json']:
            if not p.relative_to(work).as_posix().startswith('data/'):
                z.write(p, p.relative_to(work).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == (set(patched)-set(manifest['reused_data_files'])) | {'manifest.json'}
        assert z.testzip() is None
    binding = {'status': 'passed', 'archive_filename': archive.name, 'archive_sha256': sha(archive),
               'archive_bytes': archive.stat().st_size, 'manifest_sha256': sha(work/'manifest.json'),
               'base_archive_sha256': verification['archive_sha256'], 'base_manifest_sha256': plan['base_manifest_sha256'],
               'no_new_EEG_data': True, 'read_only_local_data_hardlinks': True}
    (OUT/'bundle_verification.json').write_text(json.dumps(binding, indent=2), encoding='utf8')
    write_launcher(binding, OUT/'colab_launch.py')
    print(json.dumps(binding, indent=2), flush=True)


def write_launcher(binding, destination):
    if destination.exists():
        raise FileExistsError('Preserve existing launcher')
    archive=OUT/binding['archive_filename']
    if sha(archive)!=binding['archive_sha256']:
        raise ValueError('Patch identity differs')
    template = (ROOT/'scripts/adast_loss_ablation_colab_launch_template.py').read_text(encoding='utf8')
    parts = json.loads((BASE/'upload_parts.json').read_bytes())['parts']
    substitutions = {'PATCH_NAME_PLACEHOLDER': repr(archive.name), 'PATCH_SHA_PLACEHOLDER': repr(binding['archive_sha256']),
                     'PATCH_BASE64_PLACEHOLDER': repr(base64.b64encode(archive.read_bytes()).decode('ascii')),
                     'PATCH_SIZE_PLACEHOLDER': str(binding['archive_bytes']), 'BASE_PARTS_PLACEHOLDER': repr(parts),
                     'BASE_ARCHIVE_SHA_PLACEHOLDER': repr(binding['base_archive_sha256']),
                     'BASE_MANIFEST_SHA_PLACEHOLDER': repr(binding['base_manifest_sha256'])}
    for key, value in substitutions.items():
        template = template.replace(key, value)
    compile(template, 'ablation_colab_launch.py', 'exec')
    destination.write_text(template, encoding='utf8')
    print('Launcher ready:',destination.name,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedded-launcher-only',action='store_true')
    args=parser.parse_args()
    if args.embedded_launcher_only:
        write_launcher(json.loads((OUT/'bundle_verification.json').read_bytes()),OUT/'colab_launch_embedded.py')
    else:
        build()
