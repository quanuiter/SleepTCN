"""Build a local source-only checkout candidate and test its Git byte round-trip.

Never touches the real repository index or contacts a remote. Copies public source,
tracked light historical test evidence, and current public delivery inputs only.
The local report is not a privacy approval or permission to push these candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

import build_public_delivery as delivery

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=ROOT, data=None, check=True):
    result = subprocess.run(['git', *args], cwd=cwd, input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result


def candidate_paths():
    names = git('ls-files', '-c', '-o', '--exclude-standard', '-z').stdout.split(b'\0')
    names = sorted(set(n.decode('utf8') for n in names if n))
    excluded = git('check-ignore', '--no-index', '-z', '--stdin',
                   data=('\0'.join(names) + '\0').encode('utf8'), check=False)
    if excluded.returncode not in (0, 1):
        raise RuntimeError(excluded.stderr.decode())
    ignored = set(excluded.stdout.decode('utf8').split('\0'))
    public = set(delivery.INPUTS) | set(delivery.PDFS.values()) | {
        delivery.ARCHIVE, delivery.MANIFEST, delivery.SUBMISSION, delivery.EVIDENCE,
        'Reports/paper_en/BUILD.md',
    }
    public |= {p.relative_to(ROOT).as_posix() for p, _ in
               delivery.verify_manifest(ROOT / delivery.EVIDENCE, ROOT)}
    public |= {p.relative_to(ROOT).as_posix() for p in (ROOT / 'Reports/analysis').rglob('*')
               if p.suffix in ('.json', '.csv')}
    directories = ('src/', 'scripts/', 'tests/', 'demo/', 'configs/', 'requirements/', 'environment/',
                   'docs/', 'notebooks/', '.github/', 'data/manifests/', 'data/splits/')
    top = {'.gitattributes', '.gitignore', 'pyproject.toml', 'README.md'}
    # Only historical evidence already tracked, never a private modern runs/ folder.
    tracked = set(git('ls-files', '-z').stdout.decode('utf8').split('\0'))
    for name in names:
        if name in ignored:
            continue
        if name in public or name in top or name.startswith(directories) or (
            name in tracked and name.startswith(('runs/v2/analysis/', 'runs/v2/gate8/',
                                                'runs/v2/publication/'))):
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError('Candidate is not a regular in-repo file: ' + name)
            yield name


def main(args):
    scratch = ROOT / 'tmp/github_source_check'
    scratch.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='source_', dir=scratch))
    names = list(candidate_paths())
    for name in names:
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    git('init', '-q', cwd=stage)
    git('config', 'core.autocrlf', 'true', cwd=stage)
    git('add', '--all', cwd=stage)  # Isolated disposable repo, never the real index.
    differing = []
    for name in names:
        raw = git('hash-object', '--no-filters', '--', name, cwd=stage).stdout.strip()
        index = git('rev-parse', ':' + name, cwd=stage).stdout.strip()
        if raw != index:
            differing.append(name)
    # Exact-byte checks apply to signed research/publication files, not ordinary docs.
    signed = ('src/', 'scripts/', 'configs/', 'requirements/', 'data/', 'Reports/', 'runs/')
    bad = [p for p in differing if p.startswith(signed)]
    if bad:
        raise ValueError('Git filters changed signed candidate bytes: ' + ', '.join(bad))
    checkout = stage.parent / (stage.name + '_roundtrip')
    checkout.mkdir()
    git('checkout-index', '--all', '--prefix=' + checkout.as_posix() + '/', cwd=stage)
    # Also test a Linux-style checkout (autocrlf=false), not only Windows defaults.
    posix_checkout = stage.parent / (stage.name + '_lf_roundtrip')
    posix_checkout.mkdir()
    git('-c', 'core.autocrlf=false', 'checkout-index', '--all',
        '--prefix=' + posix_checkout.as_posix() + '/', cwd=stage)
    for name in names:
        if name.startswith(signed) or name == '.gitattributes':
            expected = (ROOT / name).read_bytes()
            if any((folder / name).read_bytes() != expected for folder in (checkout, posix_checkout)):
                raise ValueError('Checkout bytes differ: ' + name)
    # Every public manifest must also hold on the actual checkout bytes.
    test_temp = stage / 'test_temp'
    test_temp.mkdir()
    # Use a writable, isolated scratch for unittest tempfile and pytest alike.
    # Some sandboxed Windows user TEMP paths deny rename even after successful write.
    env = dict(os.environ, PYTHONPATH=str(checkout / 'src'), TMP=str(test_temp),
               TEMP=str(test_temp), TMPDIR=str(test_temp), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    commands = [
        [sys.executable, 'scripts/build_public_delivery.py', '--verify-only'],
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider'],
    ]
    if args.test:
        commands.append([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                         '--basetemp', str(test_temp / 'pytest')])
    records = []
    posix_proof = subprocess.run([sys.executable, 'scripts/build_public_delivery.py', '--verify-only'],
                                cwd=posix_checkout, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stage / 'lf_checkout.log').write_bytes(posix_proof.stdout)
    if posix_proof.returncode:
        raise RuntimeError('Linux-style checkout publication proof failed; see ' + str(stage))
    for i, command in enumerate(commands):
        run = subprocess.run(command, cwd=checkout, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (stage / f'check_{i}.log').write_bytes(run.stdout)
        records.append({'command': command[1:], 'exit_code': run.returncode})
        if run.returncode:
            print('\n'.join(run.stdout.decode(errors='replace').splitlines()[-35:]))
            raise RuntimeError('Source checkout check failed; log directory: ' + str(stage))
    proof = {'status': 'passed', 'files': len(names), 'real_repository_index_changed': False,
             'signed_bytes_preserved_through_git': True, 'autocrlf_true_and_false_checked': True,
             'ordinary_text_normalized': differing,
             'checks': records, 'private_payloads_copied': False, 'training': False,
             'source_checkout': str(checkout)}
    (stage / 'verification.json').write_text(json.dumps(proof, indent=2), encoding='utf8')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', help='Run default no-training suite after collection')
    main(parser.parse_args())
