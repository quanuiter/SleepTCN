"""Pure filesystem layout for SleepTCN experiment workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_IDS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")


@dataclass(frozen=True)
class ExperimentLayout:
    """Stable run/cache locations independent of torch or dataset loading."""

    workspace: Path
    experiment_id: str
    outer_fold: int
    seed: int
    smoke: bool
    run_root: Path
    cache_root: Path


def build_experiment_layout(
    workspace: Path,
    experiment_id: str,
    outer_fold: int,
    seed: int,
    *,
    smoke: bool,
    artifact_root: Path | None = None,
) -> ExperimentLayout:
    """Validate identifiers and return the canonical run/cache layout."""

    workspace = workspace.resolve()
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(f"experiment_id must be one of {EXPERIMENT_IDS}")
    if outer_fold not in range(10) or seed < 0:
        raise ValueError("invalid fold or seed")
    mode = "smoke" if smoke else "full"
    if artifact_root is None:
        run_base = workspace / "runs" / "v2"
        cache_base = workspace / "data" / "cache" / "features" / "v2"
    else:
        artifact_root = artifact_root.resolve()
        run_base = artifact_root
        cache_base = artifact_root / "cache" / "features"
    run_root = run_base / mode / experiment_id / f"fold_{outer_fold:02d}" / f"seed_{seed}"
    cache_root = cache_base / mode / f"fold_{outer_fold:02d}" / f"seed_{seed}"
    return ExperimentLayout(
        workspace=workspace,
        experiment_id=experiment_id,
        outer_fold=outer_fold,
        seed=seed,
        smoke=smoke,
        run_root=run_root,
        cache_root=cache_root,
    )
