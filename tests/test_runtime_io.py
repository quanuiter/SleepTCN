import json
from pathlib import Path

import pytest

from sleeptcn.runtime_io import ImmutableHashCache, ReportingBudget, file_sha256, progress_json
from sleeptcn.revision_weighted_campaign import CampaignStop


def lock():
    error = PermissionError('synthetic Windows sharing lock')
    error.winerror = 32
    return error


def test_transient_lock_and_unique_pending_files(tmp_path, monkeypatch):
    path = tmp_path / 'progress.json'
    original, calls, sleeps = Path.replace, [], []
    def replace(self, target):
        calls.append(self)
        if len(calls) < 3:
            raise lock()
        return original(self, target)
    monkeypatch.setattr(Path, 'replace', replace)
    assert progress_json(path, {'ok': 1}, sleep=sleeps.append) is None
    assert sleeps == [.05, .1]
    assert json.loads(path.read_bytes()) == {'ok': 1}
    assert not list(tmp_path.glob('*.pending'))


def test_persistent_lock_preserves_old_and_new_snapshots(tmp_path, monkeypatch):
    path = tmp_path / 'progress.json'
    path.write_text('{}')
    def replace(*args):
        raise lock()
    monkeypatch.setattr(Path, 'replace', replace)
    with pytest.warns(RuntimeWarning, match='retained complete'):
        a = progress_json(path, {'n': 1}, sleep=lambda _: None)
    with pytest.warns(RuntimeWarning):
        b = progress_json(path, {'n': 2}, sleep=lambda _: None)
    assert a != b and json.loads(a.read_bytes()) == {'n': 1}
    assert json.loads(b.read_bytes()) == {'n': 2} and path.read_text() == '{}'


def test_other_io_errors_and_invalid_json_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / 'progress.json'
    def replace(*args):
        raise OSError('disk fault')
    monkeypatch.setattr(Path, 'replace', replace)
    with pytest.raises(OSError, match='disk fault'):
        progress_json(path, {})
    with pytest.raises(ValueError):
        progress_json(path, {'x': float('nan')})
    assert not path.exists()


def test_budget_deadline_unchanged(tmp_path):
    tick = [0.]
    budget = ReportingBudget(2, tmp_path / 'progress.json', clock=lambda: tick[0])
    budget.publish(status='running')
    tick[0] = 2
    with pytest.raises(CampaignStop):
        budget()


def test_cache_hashes_once_checks_every_expected_and_invalidates(tmp_path):
    path = tmp_path / 'input'
    path.write_bytes(b'abc')
    calls = []
    def digest(p):
        calls.append(p)
        return file_sha256(p)
    cache = ImmutableHashCache(digest)
    expected = file_sha256(path)
    assert cache.check(path, expected) == cache.check(path, expected) == expected
    assert len(calls) == 1
    with pytest.raises(ValueError, match='hash differs'):
        cache.check(path, '0' * 64)
    path.write_bytes(b'changed-longer')
    with pytest.raises(ValueError):
        cache.check(path, expected)
    assert len(calls) == 2


def test_file_changed_during_hash_is_rejected(tmp_path):
    path = tmp_path / 'input'
    path.write_bytes(b'abc')
    def digest(p):
        value = file_sha256(p)
        p.write_bytes(b'changed')
        return value
    with pytest.raises(ValueError, match='while hashing'):
        ImmutableHashCache(digest).check(path)
