"""Shared source-provenance hashing for experiment runners."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..io.hashing import combined_sha256, sha256_file


_BASE_RUNNER_PATHS = (
    "src/sleeptcn/artifacts.py",
    "src/sleeptcn/dataset.py",
    "src/sleeptcn/engine.py",
    "src/sleeptcn/evaluation/persistence.py",
    "src/sleeptcn/evaluation/tables.py",
    "src/sleeptcn/evaluation/__init__.py",
    "src/sleeptcn/experiment.py",
    "src/sleeptcn/features.py",
    "src/sleeptcn/io/hashing.py",
    "src/sleeptcn/io/serialization.py",
    "src/sleeptcn/metrics.py",
    "src/sleeptcn/workflows/checkpoints.py",
    "src/sleeptcn/workflows/model_factory.py",
    "src/sleeptcn/models.py",
    "src/sleeptcn/training.py",
    "src/sleeptcn/training_data.py",
    "src/sleeptcn/workflows/layout.py",
    "src/sleeptcn/workflows/context_ablation.py",
    "src/sleeptcn/workflows/gate8_protocol.py",
    "src/sleeptcn/workflows/provenance.py",
    "src/sleeptcn/workflows/stages.py",
    "src/sleeptcn/run_validation.py",
)


def runner_code_sha256(workspace: Path, *, include_gate8: bool = False) -> str:
    """Hash the exact source set participating in a runner's provenance.

    Gate 8 has one additional orchestrator file; the shared base list keeps
    the two runner contracts aligned.  Extracted workflow modules and the
    validator are included explicitly so refactoring a dependency changes the
    recorded provenance hash.
    """

    paths = _BASE_RUNNER_PATHS
    if include_gate8:
        insertion = paths.index("src/sleeptcn/io/hashing.py")
        paths = paths[:insertion] + ("src/sleeptcn/gate8.py",) + paths[insertion:]
    return combined_sha256({path: sha256_file(workspace / path) for path in paths})


def clean_git_commit(
    workspace: Path,
    *,
    unreadable_message: str = "workspace must be a readable Git repository",
    dirty_message: str = "workspace must have a clean Git worktree",
) -> str:
    """Return HEAD only when *workspace* is a readable, clean Git tree."""

    commit = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode or status.returncode:
        raise RuntimeError(unreadable_message)
    if status.stdout.strip():
        raise RuntimeError(dirty_message)
    return commit.stdout.strip()


__all__ = ["clean_git_commit", "runner_code_sha256"]
