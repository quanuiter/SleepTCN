"""Canonical NPZ checks and rewrites."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import sha256_file
from .serialization import atomic_savez


DEFAULT_VARIANTS = (
    "paper_raw_v1",
    "filtered_v2",
    "bandpass_v2",
    "bandpass_clip_v2",
    "filtered_zscore_v2",
)


def canonical_bytes(path: Path) -> bytes:
    """Return canonical bytes without modifying ``path``."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    temporary = path.with_name(f".{path.name}.canonical-check")
    try:
        atomic_savez(temporary, arrays)
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonicalize_processed(
    processed_root: Path,
    variants: Iterable[str],
    *,
    rewrite: bool,
    progress: Callable[[int, int], Any] | None = None,
) -> dict[str, Any]:
    """Check or rewrite all NPZ files under the selected variants."""

    files: list[Path] = []
    for variant in variants:
        root = processed_root / variant
        if not root.is_dir():
            return {
                "files": [],
                "drifted": [],
                "remaining": [],
                "missing_variants": [(variant, root)],
            }
        files.extend(sorted(root.glob("*.npz")))

    drifted: list[Path] = []
    for index, path in enumerate(files, start=1):
        old_hash = sha256_file(path)
        new_bytes = canonical_bytes(path)
        new_hash = _sha256_bytes(new_bytes)
        if old_hash != new_hash:
            drifted.append(path)
            if rewrite:
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                atomic_savez(path, arrays)
        if progress is not None:
            progress(index, len(files))

    remaining = [
        path
        for path in drifted
        if sha256_file(path) != _sha256_bytes(canonical_bytes(path))
    ]
    return {
        "files": files,
        "drifted": drifted,
        "remaining": remaining,
        "missing_variants": [],
    }
