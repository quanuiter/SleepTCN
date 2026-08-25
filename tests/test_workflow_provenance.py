from __future__ import annotations

from pathlib import Path

import pytest

import sleeptcn.workflows.provenance as provenance
from sleeptcn.workflows.provenance import clean_git_commit, runner_code_sha256


def test_runner_provenance_hashes_are_stable_shape_and_gate8_is_distinct() -> None:
    workspace = Path(__file__).resolve().parents[1]
    base = runner_code_sha256(workspace)
    gate8 = runner_code_sha256(workspace, include_gate8=True)

    assert len(base) == len(gate8) == 64
    assert base != gate8


def test_runner_hash_covers_extracted_workflow_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Path(__file__).resolve().parents[1]
    original = runner_code_sha256(workspace)
    target = workspace / "src/sleeptcn/workflows/context_ablation.py"
    real_sha256_file = provenance.sha256_file

    def changed_hash(path: Path) -> str:
        if path == target:
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(provenance, "sha256_file", changed_hash)
    assert runner_code_sha256(workspace) != original


def test_clean_git_commit_rejects_non_repository(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="readable Git repository"):
        clean_git_commit(tmp_path)
