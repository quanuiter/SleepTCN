"""Shared persistence contract for role-level predictions and metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .tables import PredictionTable, save_prediction_table
from ..io.serialization import atomic_write_json


def save_role_artifacts(
    run_root: Path,
    table: PredictionTable,
    role: str,
    *,
    prediction_metadata: Mapping[str, Any],
    metrics_metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Write one role's prediction table and derived metrics atomically.

    The caller owns protocol-specific metadata (for example Gate 8's condition
    and protocol hash). This function owns the shared paths and the invariant
    that metrics are computed from the exact table written to disk.
    """

    if role not in {"validation", "test"}:
        raise ValueError("evaluation role must be validation or test")
    prediction_payload = dict(prediction_metadata)
    if prediction_payload.get("role") != role:
        raise ValueError("prediction metadata role does not match requested role")
    save_prediction_table(
        run_root / "predictions" / f"{role}.npz",
        table,
        prediction_payload,
    )
    metrics = table.metrics()
    atomic_write_json(
        run_root / "metrics" / f"{role}.json",
        {"metadata": dict(metrics_metadata), "metrics": metrics},
    )
    return metrics
