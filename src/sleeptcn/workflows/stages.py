"""Checkpoint-stage provenance and completion markers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..io.hashing import sha256_file
from ..io.serialization import atomic_write_json, read_json


class StageContext(Protocol):
    """Minimal context contract needed by stage provenance."""

    experiment_id: str
    outer_fold: int
    config_sha256: str
    split_sha256: str
    data_variant: str
    smoke: bool


def checkpoint_metadata(
    context: StageContext, model: Any, stage: str, component_seed: int
) -> dict[str, Any]:
    """Build the stable metadata embedded in a model checkpoint."""

    return {
        "experiment_id": context.experiment_id,
        "stage": stage,
        "outer_fold": context.outer_fold,
        "seed": component_seed,
        "config_sha256": context.config_sha256,
        "split_sha256": context.split_sha256,
        "data_variant": context.data_variant,
        "model_class": type(model).__name__,
    }


def stage_marker_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "complete.json"


def stage_is_complete(
    context: StageContext,
    checkpoint_dir: Path,
    stage: str,
    component_seed: int,
) -> bool:
    """Validate a stage marker and its referenced best checkpoint."""

    marker_path = stage_marker_path(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    if not marker_path.is_file():
        return False
    marker = read_json(marker_path)
    expected = {
        "stage": stage,
        "outer_fold": context.outer_fold,
        "component_seed": component_seed,
        "config_sha256": context.config_sha256,
        "split_sha256": context.split_sha256,
        "data_variant": context.data_variant,
        "smoke": context.smoke,
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(f"stage completion marker mismatch: {mismatches}")
    if not best_path.is_file() or sha256_file(best_path) != marker.get(
        "best_checkpoint_sha256"
    ):
        raise ValueError(f"completed stage checkpoint mismatch: {stage}")
    return True


def mark_stage_complete(
    context: StageContext,
    checkpoint_dir: Path,
    stage: str,
    component_seed: int,
) -> None:
    """Write a deterministic completion marker for a finished stage."""

    best_path = checkpoint_dir / "best.pt"
    atomic_write_json(
        stage_marker_path(checkpoint_dir),
        {
            "schema_version": 1,
            "stage": stage,
            "outer_fold": context.outer_fold,
            "component_seed": component_seed,
            "config_sha256": context.config_sha256,
            "split_sha256": context.split_sha256,
            "data_variant": context.data_variant,
            "smoke": context.smoke,
            "best_checkpoint_sha256": sha256_file(best_path),
        },
    )
