"""Portable audits for canonical processed-artifact manifests."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .hashing import sha256_file
from .serialization import read_json


def audit_artifact_manifest(
    manifest_path: Path,
    workspace: Path,
    *,
    variants: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Verify files, canonical ZIP metadata, NPZ readability and lock hashes."""

    manifest = read_json(manifest_path)
    root = workspace.resolve()
    errors: list[str] = []
    checked = 0
    canonical_zip_files = 0
    selected_variants = set(variants) if variants is not None else None
    selected_records = [
        record
        for record in manifest.get("records", [])
        if selected_variants is None or record["variant"] in selected_variants
    ]
    for record in selected_records:
        relative = Path(record["output_path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"nonportable_path:{record['output_path']}")
            continue
        path = root / relative
        checked += 1
        if not path.is_file():
            errors.append(f"missing:{record['output_path']}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record["output_sha256"]:
            errors.append(f"sha256:{record['output_path']}")
        if path.stat().st_size != record.get("size_bytes", path.stat().st_size):
            errors.append(f"size:{record['output_path']}")
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if not infos or any(
                    info.create_system != 3
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != zipfile.ZIP_STORED
                    for info in infos
                ):
                    errors.append(f"noncanonical_zip:{record['output_path']}")
                else:
                    canonical_zip_files += 1
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    errors.append(f"empty_npz:{record['output_path']}")
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            errors.append(f"unreadable:{record['output_path']}:{type(error).__name__}")

    summary = manifest.get("summary", {})
    expected_files = (
        len(selected_records)
        if selected_variants is not None
        else int(summary.get("files", len(manifest.get("records", []))))
    )
    if checked != expected_files:
        errors.append(f"file_count:{checked}!={expected_files}")
    lock_info = manifest.get("environment_lock")
    if lock_info:
        lock_path = root / lock_info["path"]
        if not lock_path.is_file():
            errors.append(f"missing_environment_lock:{lock_info['path']}")
        elif sha256_file(lock_path) != lock_info["sha256"]:
            errors.append("environment_lock_sha256")
        else:
            lock_text = lock_path.read_text(encoding="utf-8")
            match = re.search(r"Included freeze SHA-256:\s*([0-9a-f]{64})", lock_text)
            freeze_path = root / "environment/pip-freeze.txt"
            if match and (
                not freeze_path.is_file() or sha256_file(freeze_path) != match.group(1)
            ):
                errors.append("environment_freeze_sha256")

    return {
        "schema_version": 1,
        "manifest": manifest_path.as_posix(),
        "summary": {
            "files_checked": checked,
            "canonical_zip_files": canonical_zip_files,
            "errors": errors,
            "passed": not errors,
        },
    }
