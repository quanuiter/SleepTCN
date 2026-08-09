"""Tiện ích huấn luyện chung, độc lập kiến trúc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


IGNORED_LABEL = -1
PAD_LABEL = -100
N_CLASSES = 5


@dataclass(frozen=True)
class PaddedBatch:
    features: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor
    valid_target_mask: torch.Tensor
    ignored_epoch_mask: torch.Tensor
    padding_mask: torch.Tensor


def collate_feature_sequences(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> PaddedBatch:
    """Pad chuỗi đặc trưng (T,F), không xóa epoch nhãn -1."""
    if not batch:
        raise ValueError("batch must not be empty")
    feature_dim = batch[0][0].shape[1]
    lengths = torch.tensor([features.shape[0] for features, _ in batch], dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError("all sequences must contain at least one epoch")
    max_length = int(lengths.max())
    features_pad = torch.zeros(
        (len(batch), max_length, feature_dim), dtype=torch.float32
    )
    targets_pad = torch.full(
        (len(batch), max_length), PAD_LABEL, dtype=torch.long
    )
    for index, (features, targets) in enumerate(batch):
        if features.ndim != 2 or features.shape[1] != feature_dim:
            raise ValueError("all feature tensors must have shape (T, same_feature_dim)")
        if features.dtype != torch.float32:
            raise ValueError("features must be float32")
        if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
            raise ValueError("targets must have shape (T,)")
        if targets.dtype != torch.long:
            raise ValueError("targets must be torch.long")
        if not torch.all(
            (targets == IGNORED_LABEL) | ((targets >= 0) & (targets < N_CLASSES))
        ):
            raise ValueError("target contains unsupported label")
        length = features.shape[0]
        features_pad[index, :length] = features
        targets_pad[index, :length] = targets

    padding_mask = targets_pad == PAD_LABEL
    ignored_mask = targets_pad == IGNORED_LABEL
    valid_mask = (targets_pad >= 0) & (targets_pad < N_CLASSES)
    if torch.any(valid_mask & ignored_mask) or torch.any(valid_mask & padding_mask):
        raise AssertionError("target masks overlap")
    return PaddedBatch(
        features=features_pad,
        targets=targets_pad,
        lengths=lengths,
        valid_target_mask=valid_mask,
        ignored_epoch_mask=ignored_mask,
        padding_mask=padding_mask,
    )


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy trên nhãn 0..4; che đồng thời -1 và -100."""
    if logits.ndim < 2 or logits.shape[-1] != N_CLASSES:
        raise ValueError("logits must end with five class scores")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets shape must equal logits shape without class axis")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    valid = (targets >= 0) & (targets < N_CLASSES)
    unsupported = ~(valid | (targets == IGNORED_LABEL) | (targets == PAD_LABEL))
    if torch.any(unsupported):
        raise ValueError("targets contain unsupported labels")
    if not torch.any(valid):
        raise ValueError("batch contains no valid target")
    if class_weights is not None:
        if class_weights.shape != (N_CLASSES,):
            raise ValueError("class_weights must have shape (5,)")
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits[valid], targets[valid], weight=class_weights)
