"""Pure C/P/N feature-group masking for the Gate-8 workflow.

This module intentionally has no torch dependency.  It owns the frozen feature
contract and the train-only replacement calculation; model training and artifact
orchestration remain in :mod:`sleeptcn.gate8`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from ..training_data import FeatureSequence


CONDITIONS = ("CP", "CN", "C")
GROUP_SLICES = {"C": (0, 25), "P": (25, 50), "N": (50, 75)}


def train_replacement_mean(
    sequences: Sequence[FeatureSequence],
) -> tuple[np.ndarray, int]:
    """Calculate a 75-dim replacement vector from valid train epochs only."""

    if not sequences:
        raise ValueError("training sequences must not be empty")
    total = np.zeros(75, dtype=np.float64)
    count = 0
    for sequence in sequences:
        if sequence.features.shape[1] != 75:
            raise ValueError("Gate 8 requires 75-dimensional 15CNN features")
        valid = (sequence.labels >= 0) & (sequence.labels < 5)
        total += sequence.features[valid].sum(axis=0, dtype=np.float64)
        count += int(valid.sum())
    if count <= 0:
        raise ValueError("training role has no valid epochs")
    mean = (total / count).astype(np.float32)
    if mean.shape != (75,) or not np.isfinite(mean).all():
        raise AssertionError("invalid Gate 8 replacement mean")
    return mean, count


def context_groups(condition: str) -> tuple[str, ...]:
    """Return the feature groups retained by a locked Gate-8 condition."""

    if condition == "CP":
        return ("C", "P")
    if condition == "CN":
        return ("C", "N")
    if condition == "C":
        return ("C",)
    raise ValueError(f"unknown Gate 8 condition: {condition}")


def mask_feature_sequences(
    sequences: Sequence[FeatureSequence],
    condition: str,
    replacement_mean: np.ndarray,
) -> tuple[FeatureSequence, ...]:
    """Replace masked groups while preserving the 75-dimensional contract."""

    if condition not in CONDITIONS or replacement_mean.shape != (75,):
        raise ValueError("invalid Gate 8 masking arguments")
    masked_groups = tuple(
        group for group in GROUP_SLICES if group not in context_groups(condition)
    )
    output: list[FeatureSequence] = []
    for sequence in sequences:
        if sequence.features.shape[1] != 75:
            raise ValueError("Gate 8 masking requires 75-dimensional features")
        features = sequence.features.copy()
        for group in masked_groups:
            start, stop = GROUP_SLICES[group]
            features[:, start:stop] = replacement_mean[start:stop]
            if not np.array_equal(
                features[:, start:stop],
                np.broadcast_to(
                    replacement_mean[start:stop], features[:, start:stop].shape
                ),
            ):
                raise AssertionError(f"failed to mask feature group {group}")
        output.append(
            replace(
                sequence,
                extractor_id=f"{sequence.extractor_id}_gate8_{condition}",
                features=features,
            )
        )
    return tuple(output)


__all__ = [
    "CONDITIONS",
    "GROUP_SLICES",
    "context_groups",
    "mask_feature_sequences",
    "train_replacement_mean",
]
