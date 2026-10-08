"""Packaging contracts with tiny text fixtures; no TeX/model training required."""
import importlib.util
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_delivery_test', ROOT / 'scripts/build_public_delivery.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_historical_evidence_not_silently_rebased(tmp_path, monkeypatch):
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    source, manifest = tmp_path / 'evidence.json', tmp_path / 'evidence.sha256'
    source.write_text('{}')
    manifest.write_text(build.sha(source) + '  evidence.json\n')
    before = manifest.read_bytes()
    assert len(build.verify_manifest(manifest, tmp_path)) == 1
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='hash changed'):
        build.verify_manifest(manifest, tmp_path)
    assert manifest.read_bytes() == before


def test_duplicate_and_outside_manifest_paths_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    manifest = tmp_path / 'manifest'
    for text in ['0' * 64 + '  ../outside\n', ('0' * 64 + '  duplicate\n') * 2]:
        manifest.write_text(text)
        with pytest.raises(ValueError, match='Outside root or duplicate'):
            build.manifest_entries(manifest, tmp_path)


def test_archive_failure_preserves_existing_and_success_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(build, 'EN', '')
    monkeypatch.setattr(build, 'EN_ENTRIES', ('main.tex', 'SUBMISSION_BUILD.md'))
    destination = tmp_path / 'source.zip'
    destination.write_bytes(b'previous delivery')
    with pytest.raises(FileNotFoundError):
        build.make_archive(destination, tmp_path)
    assert destination.read_bytes() == b'previous delivery'
    (tmp_path / 'main.tex').write_text('tiny source')
    (tmp_path / 'SUBMISSION_BUILD.md').write_text('tiny instructions')
    build.make_archive(destination, tmp_path)
    first = destination.read_bytes()
    build.make_archive(destination, tmp_path)
    assert first == destination.read_bytes()
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {'main.tex', 'README.md'}


def test_current_translation_is_built_and_no_private_packaging_dependencies():
    assert build.VI + 'main.tex' in build.INPUTS
    assert build.PDFS[build.VI + 'main.pdf'].endswith('SleepTCN_BSPC_Ban_dich_Tieng_Viet.pdf')
    assert not any('/tmp/' in p or '.private.' in p for p in build.INPUTS)


def test_archive_rejects_extra_member(tmp_path, monkeypatch):
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(build, 'EN', '')
    monkeypatch.setattr(build, 'EN_ENTRIES', ('main.tex',))
    (tmp_path / 'main.tex').write_text('source')
    destination = tmp_path / 'source.zip'
    build.make_archive(destination, tmp_path)
    build.verify_archive(destination)
    with zipfile.ZipFile(destination, 'a') as archive:
        archive.writestr('unapproved', 'no')
    with pytest.raises(ValueError, match='membership'):
        build.verify_archive(destination)
