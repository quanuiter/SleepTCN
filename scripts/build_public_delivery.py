"""Build current EN/VI/supplement from public files, without training.

Evidence hashes are checked, never refreshed. Existing deliveries are preserved
until every build succeeds. --verify-only is read-only. Requires TeX on PATH.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EN = 'Reports/paper_en/'
VI = 'Reports/paper_vi_translation/'
EN_ENTRIES = (
    'main.tex', 'supplement.tex', 'references.bib', 'ieeetr-doi.bst', 'highlights.txt',
    'figures/study_design_en.pdf', 'figures/study_design_en.svg',
    'figures/gate8_primary_speedup_f1_en.pdf', 'figures/gate8_feature_silhouette_en.pdf',
    'figures/gate8_context_ablation_effects_en.pdf',
    'figure_sources/gate8_primary_speedup_f1_en.tex', 'figure_sources/study_design.py',
    'SUBMISSION_BUILD.md',
)
VI_ENTRIES = ('main.tex', 'figures/study_design_vi.pdf', 'figures/study_design_vi.svg',
              'figures/speedup_vi.pdf', 'figures/speedup_vi.tex')
INPUTS = tuple(EN + p for p in EN_ENTRIES) + tuple(VI + p for p in VI_ENTRIES)
PDFS = {
    EN + 'main.pdf': 'Reports/output/pdf/SleepTCN_Scientific_Article_EN.pdf',
    VI + 'main.pdf': 'Reports/output/pdf/SleepTCN_BSPC_Ban_dich_Tieng_Viet.pdf',
    EN + 'supplement.pdf': 'Reports/output/pdf/SleepTCN_Supplement_EN.pdf',
}
ARCHIVE = 'Reports/output/source/SleepTCN_BSPC_Manuscript_Source.zip'
EVIDENCE = 'Reports/publication/evidence.sha256'
MANIFEST = 'Reports/REPORT_MANIFEST.sha256'
SUBMISSION = EN + 'SUBMISSION_MANIFEST.sha256'
TOOLS = ('scripts/build_public_delivery.py', 'scripts/rebuild_bspc_delivery.ps1',
         'docs/PUBLIC_REPRODUCIBILITY.md', EN + 'BUILD.md', '.gitattributes')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest_entries(path, base):
    seen, entries = set(), []
    for line in path.read_text(encoding='utf8').splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
        if not match:
            raise ValueError('Invalid manifest line in ' + path.name)
        digest, relative = match.groups()
        target = (base / relative).resolve()
        if not target.is_relative_to(ROOT.resolve()) or target in seen:
            raise ValueError('Outside root or duplicate manifest path')
        seen.add(target)
        entries.append((target, digest))
    if not entries:
        raise ValueError('Empty evidence/manifest')
    return entries


def verify_manifest(path, base):
    rows = manifest_entries(path, base)
    for target, expected in rows:
        if sha(target) != expected:
            raise ValueError('Evidence or release hash changed: ' + target.relative_to(ROOT).as_posix())
    return rows


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(path)  # Release writes fail closed, unlike progress reporting.


def write_manifest(path, targets):
    lines = [sha(p) + '  ' + Path(os.path.relpath(p, path.parent)).as_posix()
             for p in sorted(set(targets))]
    atomic_bytes(path, ('\n'.join(lines) + '\n').encode('utf8'))


def command(args, cwd, log):
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError('Build command failed; see ' + str(log))


def tex(stage, folder, name, bibliography=False, engine='pdflatex'):
    cwd = stage / folder
    for number in range(3):
        command([engine, '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
                cwd, cwd / f'{name}.pass{number}.txt')
        if number == 0 and bibliography:
            command(['bibtex', name], cwd, cwd / f'{name}.bibtex.txt')
    log = (cwd / (name + '.log')).read_text(encoding='utf8', errors='replace')
    if re.search(r'There were undefined references|(?:Citation|Reference) .+ undefined|Missing character:', log):
        raise RuntimeError('Unresolved reference or missing glyph in ' + folder + name)


def make_archive(destination, source_root):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in EN_ENTRIES:
            name = 'README.md' if relative == 'SUBMISSION_BUILD.md' else relative
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (source_root / EN / relative).read_bytes())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(EN_ENTRIES):
            raise ValueError('Source archive verification failed')
    temporary.replace(destination)


def verify_archive(path):
    expected = {'README.md' if p == 'SUBMISSION_BUILD.md' else p: ROOT / EN / p for p in EN_ENTRIES}
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError('Source archive membership differs')
        for name, source in expected.items():
            if archive.read(name) != source.read_bytes():
                raise ValueError('Source archive entry differs: ' + name)


def refresh_manifests():
    evidence = verify_manifest(ROOT / EVIDENCE, ROOT)
    paths = [ROOT / p for p in (*INPUTS, *PDFS.values(), ARCHIVE, EVIDENCE, *TOOLS)]
    paths += [p for p, _ in evidence]
    write_manifest(ROOT / SUBMISSION, [ROOT / EN / p for p in EN_ENTRIES] + [ROOT / ARCHIVE])
    write_manifest(ROOT / MANIFEST, paths + [ROOT / SUBMISSION])


def main(args):
    verify_manifest(ROOT / EVIDENCE, ROOT)
    if args.verify_only or args.package_only:
        for name in (MANIFEST, SUBMISSION):
            verify_manifest(ROOT / name, (ROOT / name).parent)
        verify_archive(ROOT / ARCHIVE)
        print('PUBLIC DELIVERY HASHES AND SOURCE ARCHIVE VERIFIED; no source/PDF changes')
        return
    for executable in ('pdflatex', 'xelatex', 'bibtex'):
        if not shutil.which(executable):
            raise RuntimeError('Missing build prerequisite: ' + executable)
    scratch = ROOT / 'tmp/public_delivery'
    scratch.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='build_', dir=scratch))
    for relative in INPUTS:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    tex(stage, EN, 'supplement')
    tex(stage, EN, 'main', bibliography=True)
    tex(stage, VI, 'main', engine='xelatex')
    make_archive(stage / 'source.zip', stage)
    portable = stage / 'portable'
    with zipfile.ZipFile(stage / 'source.zip') as archive:
        archive.extractall(portable)  # Members are the fixed allowlist above.
    tex(stage, 'portable', 'supplement')
    tex(stage, 'portable', 'main', bibliography=True)
    verify_manifest(ROOT / EVIDENCE, ROOT)
    for relative in INPUTS:
        if (stage / relative).read_bytes() != (ROOT / relative).read_bytes():
            raise ValueError('Source changed during build: ' + relative)
    if args.stage_only:
        print('BUILD VERIFIED IN STAGING ONLY: ' + str(stage))
        return
    # Per-file atomic publish; old manifest stays unchanged if a publish fails.
    # The backup retains the complete previous release for recovery.
    backup = stage / 'previous_delivery'
    backup.mkdir()
    for relative in (*PDFS.values(), ARCHIVE, MANIFEST, SUBMISSION):
        path = ROOT / relative
        if path.exists():
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    for source, relative in PDFS.items():
        atomic_bytes(ROOT / relative, (stage / source).read_bytes())
    atomic_bytes(ROOT / ARCHIVE, (stage / 'source.zip').read_bytes())
    refresh_manifests()
    print('THREE PDFS, PORTABLE EN SOURCE ZIP AND PUBLIC MANIFESTS BUILT: ' + str(stage))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verify-only', action='store_true')
    group.add_argument('--stage-only', action='store_true', help='Build without replacing deliveries')
    group.add_argument('--package-only', action='store_true', help='Verify existing package; refuses stale PDFs')
    main(parser.parse_args())
