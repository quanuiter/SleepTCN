"""Luu/nap dac trung va du doan kem ma bam truy nguyen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dataset import SleepRecord
from .features import extract_15cnn_features
from .evaluation.tables import PredictionTable, save_prediction_table
from .io.hashing import combined_sha256, sha256_file
from .io.serialization import atomic_savez
from .training_data import FeatureSequence


ARTIFACT_SCHEMA_VERSION = 1


@torch.no_grad()
def extract_resnet_sequence(
    record: SleepRecord,
    model: nn.Module,
    *,
    extractor_id: str,
    device: torch.device | str,
    batch_size: int = 256,
) -> FeatureSequence:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    model = model.to(device).eval()
    features: list[np.ndarray] = []
    for start in range(0, len(record.x), batch_size):
        signals = torch.from_numpy(record.x[start : start + batch_size]).unsqueeze(1)
        batch_features = model.extract_features(signals.to(device))
        if batch_features.ndim != 2 or batch_features.shape[0] != len(signals):
            raise ValueError("ResNet extractor returned invalid shape")
        features.append(batch_features.cpu().numpy())
    matrix = np.concatenate(features).astype(np.float32, copy=False)
    return FeatureSequence(
        record_key=record.info.record_key,
        subject_id=record.info.subject_id,
        preprocess_version=record.info.preprocess_version,
        extractor_id=extractor_id,
        features=matrix,
        labels=record.y.copy(),
        original_epoch_index=record.original_epoch_index.copy(),
    )


def extract_15cnn_sequence(
    record: SleepRecord,
    models: dict[str, nn.Module],
    *,
    extractor_id: str,
    device: torch.device | str,
    batch_size: int = 256,
) -> FeatureSequence:
    features = extract_15cnn_features(
        record.x, models, device=device, batch_size=batch_size
    )
    return FeatureSequence(
        record_key=record.info.record_key,
        subject_id=record.info.subject_id,
        preprocess_version=record.info.preprocess_version,
        extractor_id=extractor_id,
        features=features,
        labels=record.y.copy(),
        original_epoch_index=record.original_epoch_index.copy(),
    )


def save_feature_sequence(
    path: Path,
    sequence: FeatureSequence,
    *,
    extractor_sha256: str,
    split_sha256: str,
    outer_fold: int,
    seed: int,
) -> None:
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "feature_sequence",
        "record_key": sequence.record_key,
        "subject_id": sequence.subject_id,
        "preprocess_version": sequence.preprocess_version,
        "extractor_id": sequence.extractor_id,
        "extractor_sha256": extractor_sha256,
        "split_sha256": split_sha256,
        "outer_fold": outer_fold,
        "seed": seed,
    }
    combined_sha256(
        {"extractor_sha256": extractor_sha256, "split_sha256": split_sha256}
    )
    atomic_savez(
        path,
        {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            "features": sequence.features,
            "labels": sequence.labels.astype(np.int8, copy=False),
            "original_epoch_index": sequence.original_epoch_index,
        },
    )


def load_feature_sequence(
    path: Path,
    *,
    expected_extractor_sha256: str,
    expected_split_sha256: str,
    expected_outer_fold: int,
    expected_seed: int,
) -> FeatureSequence:
    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"].item()))
        expected = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "feature_sequence",
            "extractor_sha256": expected_extractor_sha256,
            "split_sha256": expected_split_sha256,
            "outer_fold": expected_outer_fold,
            "seed": expected_seed,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"feature artifact metadata mismatch: {mismatches}")
        return FeatureSequence(
            record_key=metadata["record_key"],
            subject_id=metadata["subject_id"],
            preprocess_version=metadata["preprocess_version"],
            extractor_id=metadata["extractor_id"],
            features=npz["features"].copy(),
            labels=npz["labels"].copy(),
            original_epoch_index=npz["original_epoch_index"].copy(),
        )


def prediction_table_from_parts(
    parts: list[tuple[FeatureSequence, np.ndarray]],
) -> PredictionTable:
    if not parts:
        raise ValueError("prediction parts must not be empty")
    subjects: list[np.ndarray] = []
    records: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    true: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    seen_records: set[str] = set()
    for sequence, logits in parts:
        if sequence.record_key in seen_records:
            raise ValueError(f"duplicate prediction record: {sequence.record_key}")
        seen_records.add(sequence.record_key)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (len(sequence.labels), 5):
            raise ValueError("record logits do not align with full sequence")
        valid = (sequence.labels >= 0) & (sequence.labels < 5)
        count = int(valid.sum())
        subjects.append(np.full(count, sequence.subject_id, dtype="U5"))
        records.append(np.full(count, sequence.record_key, dtype="U7"))
        indices.append(sequence.original_epoch_index[valid])
        true.append(sequence.labels[valid].astype(np.int8, copy=False))
        selected_logits = logits[valid]
        logits_parts.append(selected_logits)
        predicted.append(selected_logits.argmax(axis=1).astype(np.int8))
    return PredictionTable(
        subject_id=np.concatenate(subjects),
        record_key=np.concatenate(records),
        original_epoch_index=np.concatenate(indices),
        true_label=np.concatenate(true),
        predicted_label=np.concatenate(predicted),
        logits=np.concatenate(logits_parts),
    )
