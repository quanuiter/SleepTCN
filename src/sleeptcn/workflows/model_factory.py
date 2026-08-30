"""Canonical constructors for sequence models used by all runners."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from torch import nn

from ..models import BiLSTMSleepNet, SleepTCN
from .checkpoints import load_verified_checkpoint


SequenceKind = Literal["bilstm", "tcn"]


def sequence_kind(model_name: str) -> SequenceKind:
    """Normalize the protocol name to the checkpoint-stage name."""

    if model_name == "bilstm":
        return "bilstm"
    if model_name in {"tcn", "common_tcn"}:
        return "tcn"
    raise ValueError(f"unsupported sequence model: {model_name}")


def sequence_component_config(
    config: Mapping[str, Any], model_name: str
) -> Mapping[str, Any]:
    """Return the locked component config for a sequence model."""

    kind = sequence_kind(model_name)
    component_name = "bilstm" if kind == "bilstm" else "common_tcn"
    try:
        return config["components"][component_name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing sequence component config: {component_name}") from error


def build_sequence_model(
    config: Mapping[str, Any], model_name: str, *, input_dim: int | None = None
) -> nn.Module:
    """Construct a sequence model without loading weights or touching data."""

    kind = sequence_kind(model_name)
    component = sequence_component_config(config, kind)
    if kind == "bilstm":
        configured_dim = int(component["input_dim"])
        if input_dim is not None and input_dim != configured_dim:
            raise ValueError(
                f"BiLSTM input dimension mismatch: {input_dim} != {configured_dim}"
            )
        return BiLSTMSleepNet(
            input_dim=configured_dim,
            hidden_dim=int(component["hidden_dim"]),
        )
    if input_dim is None or input_dim <= 0:
        raise ValueError("TCN input_dim must be a positive integer")
    return SleepTCN(
        input_dim=input_dim,
        hidden_dim=int(component["hidden_dim"]),
        kernel_size=int(component["kernel_size"]),
        n_blocks=int(component["residual_blocks"]),
        dropout=float(component["dropout"]),
    )


def load_sequence_checkpoint(
    context: Any,
    model: nn.Module,
    kind: str,
    checkpoint_dir: Path,
    *,
    device: Any,
) -> Path:
    """Validate and load a sequence checkpoint using the shared stage contract."""

    normalized_kind = sequence_kind(kind)
    stage = f"sequence/{normalized_kind}"
    component_seed = context.seed
    return load_verified_checkpoint(
        context,
        model,
        checkpoint_dir,
        stage,
        component_seed,
        selection_metric="validation_macro_f1",
        device=device,
        incomplete_message=f"incomplete sequence stage: {stage}",
    )
