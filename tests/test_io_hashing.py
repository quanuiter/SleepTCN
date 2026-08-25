from __future__ import annotations

from pathlib import Path

import pytest

from sleeptcn.io.hashing import combined_sha256, sha256_file
from sleeptcn.io.paths import portable_path
from sleeptcn.io.serialization import atomic_write_json, read_json


def test_sha256_file_matches_known_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"SleepTCN reproducibility\n")
    assert sha256_file(path) == (
        "0dcb88e9d919a2acd376a233b457f59d84848706446b0ac859ecc264272abca0"
    )


def test_combined_sha256_is_order_independent() -> None:
    values = {"split": "a" * 64, "checkpoint": "b" * 64}
    assert combined_sha256(values) == combined_sha256(
        {"checkpoint": "b" * 64, "split": "a" * 64}
    )


def test_combined_sha256_rejects_invalid_digest() -> None:
    with pytest.raises(ValueError, match="invalid SHA-256"):
        combined_sha256({"split": "not-a-digest"})


def test_atomic_write_json_is_newline_terminated_and_deterministic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "nested" / "first.json"
    second = tmp_path / "nested" / "second.json"
    atomic_write_json(first, {"b": 2, "a": 1})
    atomic_write_json(second, {"a": 1, "b": 2})
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")


def test_read_json_uses_the_shared_utf8_contract(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"message": "tái lập"}\n', encoding="utf-8")
    assert read_json(path) == {"message": "tái lập"}


def test_portable_path_never_leaks_external_mount_root(tmp_path: Path) -> None:
    inside = tmp_path / "data" / "artifact.npz"
    inside.parent.mkdir()
    inside.write_bytes(b"x")
    assert portable_path(inside, tmp_path) == "data/artifact.npz"

    external = Path("/tmp") / "external-artifact.npz"
    assert portable_path(external, tmp_path) == external.name
