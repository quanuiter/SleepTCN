"""Shared verified-checkpoint loading for workflow runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from ..engine import load_model_checkpoint
from .stages import checkpoint_metadata, stage_is_complete


def load_verified_checkpoint(
    context: Any,
    model: nn.Module,
    checkpoint_dir: Path,
    stage: str,
    component_seed: int,
    *,
    selection_metric: str,
    device: Any,
    incomplete_message: str | None = None,
) -> Path:
    """Validate a stage marker and load its ``best.pt`` checkpoint."""

    if not stage_is_complete(context, checkpoint_dir, stage, component_seed):
        raise ValueError(
            incomplete_message or f"incomplete checkpoint stage: {stage}"
        )
    path = checkpoint_dir / "best.pt"
    load_model_checkpoint(
        path,
        model,
        expected_metadata={
            **checkpoint_metadata(context, model, stage, component_seed),
            "selection_metric": selection_metric,
        },
        device=device,
    )
    return path
