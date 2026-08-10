"""Quy tac tao dac trung 15CNN, giu dung bien cua tung ban ghi."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .dataset import SleepRecord


STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")
MANIPULATIONS = ("current", "previous", "next")
MANIPULATION_PREFIX = {"current": "C", "previous": "P", "next": "N"}


def shift_within_record(signals: np.ndarray, manipulation: str) -> np.ndarray:
    """Lay epoch hien tai/lien truoc/lien sau ma khong vuot bien ban ghi."""
    if signals.ndim != 2 or signals.shape[1] != 3000 or len(signals) == 0:
        raise ValueError("signals must have shape (T, 3000), T > 0")
    if manipulation == "current":
        return signals
    if manipulation == "previous":
        return np.concatenate((signals[:1], signals[:-1]), axis=0)
    if manipulation == "next":
        return np.concatenate((signals[1:], signals[-1:]), axis=0)
    raise ValueError(f"unknown manipulation: {manipulation}")


def concatenate_valid_epochs(
    records: Sequence[SleepRecord], manipulation: str
) -> tuple[np.ndarray, np.ndarray]:
    """Shift tung ban ghi truoc, sau do moi ghep cac epoch co nhan 0..4."""
    if not records:
        raise ValueError("records must not be empty")
    signal_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for record in records:
        shifted = shift_within_record(record.x, manipulation)
        valid = record.valid_mask
        signal_parts.append(shifted[valid])
        label_parts.append(record.y[valid].astype(np.int64, copy=False))
    return np.concatenate(signal_parts), np.concatenate(label_parts)


def class_specific_weights(
    target_class: int, class_counts: np.ndarray
) -> np.ndarray:
    """Trong so nhi phan cua tung Net_c theo cong thuc dung trong paper."""
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.shape != (5,) or np.any(counts <= 0):
        raise ValueError("class_counts must contain five positive counts")
    if target_class not in range(5):
        raise ValueError("target_class must be in 0..4")
    total = float(counts.sum())
    positive = float(counts[target_class])
    weights = np.full(5, 0.5 / ((total - positive) / total), dtype=np.float32)
    weights[target_class] = 0.5 / (positive / total)
    return weights


def expected_15cnn_keys() -> tuple[str, ...]:
    return tuple(
        f"{MANIPULATION_PREFIX[manipulation]}_{stage}"
        for manipulation in MANIPULATIONS
        for stage in STAGE_NAMES
    )


@torch.no_grad()
def extract_15cnn_features(
    signals: np.ndarray,
    models: Mapping[str, nn.Module],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Tra ve 75 xac suat theo thu tu C_*, P_*, N_* cho moi epoch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    keys = expected_15cnn_keys()
    missing = [key for key in keys if key not in models]
    extra = sorted(set(models) - set(keys))
    if missing or extra:
        raise ValueError(f"invalid 15CNN model keys; missing={missing}, extra={extra}")
    device = torch.device(device)
    columns: list[np.ndarray] = []
    for manipulation in MANIPULATIONS:
        shifted = shift_within_record(signals, manipulation)
        input_tensor = torch.from_numpy(shifted).unsqueeze(1)
        for stage in STAGE_NAMES:
            key = f"{MANIPULATION_PREFIX[manipulation]}_{stage}"
            model = models[key].to(device).eval()
            batches: list[torch.Tensor] = []
            for start in range(0, len(input_tensor), batch_size):
                logits = model(input_tensor[start : start + batch_size].to(device))
                if logits.shape != (min(batch_size, len(input_tensor) - start), 5):
                    raise ValueError(f"{key}: model output must have shape (B,5)")
                batches.append(torch.softmax(logits, dim=-1).cpu())
            columns.append(torch.cat(batches).numpy())
    result = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
    if result.shape != (len(signals), 75) or not np.isfinite(result).all():
        raise AssertionError("invalid 15CNN feature matrix")
    return result
