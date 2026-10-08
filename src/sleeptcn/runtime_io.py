"""Versioned operational I/O; historical campaign helpers remain byte-for-byte intact.

Only progress reporting may retain a pending snapshot after Windows sharing locks.
Scientific results/checkpoints must continue to use fail-closed writers.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import warnings

from .revision_weighted_campaign import CampaignBudget as HistoricalBudget


def progress_json(path, data, *, sleep=time.sleep):
    path = Path(path)
    encoded = (json.dumps(data, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                                     prefix=path.name + '.', suffix='.pending', delete=False) as stream:
        stream.write(encoded)
        pending = Path(stream.name)
    for delay in (.05, .1, .2, .4, .8, None):
        try:
            pending.replace(path)
            return None
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33):
                raise
            if delay is None:
                warnings.warn('Progress locked; retained complete status snapshot: ' + pending.name,
                              RuntimeWarning, stacklevel=2)
                return pending
            sleep(delay)


class ReportingBudget(HistoricalBudget):
    """Same deadline contract, with bounded retries for status publication only."""

    def publish(self, **values):
        self.context.update(values)
        self.pending_snapshot = progress_json(self.progress_path, {
            **self.context,
            'elapsed_attempt_seconds': self.clock() - self.started,
            'remaining_attempt_seconds': self.remaining(),
        })


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ImmutableHashCache:
    """Per-operation cache for trusted immutable inputs, not an adversarial file lock.

    Stat changes invalidate a cached value. Final independent verification must still
    hash all bound files afresh. A concurrent change during hashing is an error.
    """

    def __init__(self, hasher=file_sha256):
        self.hasher, self._cache, self.ledger = hasher, {}, {}

    @staticmethod
    def signature(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def check(self, path, expected=None):
        path = Path(path).resolve()
        key, before = path.as_posix(), self.signature(path)
        cached = self._cache.get(key)
        if cached is None or cached[0] != before:
            actual = self.hasher(path)
            if self.signature(path) != before:
                raise ValueError('Input changed while hashing: ' + path.name)
            self._cache[key] = (before, actual)
        else:
            actual = cached[1]
        if expected is not None and actual != expected:
            raise ValueError('Bound input/checkpoint/proof hash differs: ' + path.name)
        self.ledger[key] = actual
        return actual
