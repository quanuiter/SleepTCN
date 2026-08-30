"""Deterministic hashing helpers used by manifests and artifact provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(named_hashes: Mapping[str, str]) -> str:
    """Hash a canonical JSON mapping of named SHA-256 values."""

    for name, value in named_hashes.items():
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid SHA-256 for {name}")
    canonical = json.dumps(dict(named_hashes), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
