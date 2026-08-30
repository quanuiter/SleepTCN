"""Portable path helpers for manifests and provenance records."""

from __future__ import annotations

from pathlib import Path


def portable_path(path: Path, root: Path) -> str:
    """Return a stable path relative to *root*, never an absolute mount path."""

    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        # External inputs are identified by their content hash; retaining only
        # the basename avoids leaking a machine-specific mount point.
        return resolved_path.name
