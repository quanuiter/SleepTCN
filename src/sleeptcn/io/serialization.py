"""Platform-independent serialization for canonical NPZ artifacts."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


NPZ_SERIALIZATION_FORMAT = "npz_zip_stored_v1"


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document from *path*."""

    return json.loads(path.read_text(encoding="utf-8"))


def _npy_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer, np.asarray(value), allow_pickle=False, version=(1, 0)
    )
    return buffer.getvalue()


def atomic_savez(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write the locked ``npz_zip_stored_v1`` format atomically.

    The format deliberately uses ZIP_STORED, fixed timestamps and fixed file
    metadata.  All new processed and model-side artifacts use this same
    contract; changing it requires a format/version decision rather than an
    incidental serializer refactor.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            for name, value in arrays.items():
                info = zipfile.ZipInfo(f"{name}.npy")
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.create_version = 20
                info.extract_version = 20
                info.flag_bits = 0
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(value))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = True,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    """Write a newline-terminated JSON document through an atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=ensure_ascii,
                indent=indent,
                sort_keys=sort_keys,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
