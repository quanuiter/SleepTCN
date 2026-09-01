"""Verified inference and error-analysis primitives for the thesis demo."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .dataset import load_record
from .features import expected_15cnn_keys, extract_15cnn_features
from .models import BiLSTMSleepNet, EEGResNet1D, SleepCNN, SleepTCN
from .preprocessing import PreprocessConfig, preprocess_signal_variant


STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")
DEMO_EXPERIMENTS = ("E0", "E3", "E6")
EXPERIMENT_VARIANTS = {
    "E0": "paper_raw_v1",
    "E3": "filtered_v2",
    "E6": "filtered_zscore_v2",
}
ASSET_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DemoRecord:
    record_key: str
    x: np.ndarray
    labels: np.ndarray | None
    original_epoch_index: np.ndarray
    source: str
    note: str
    data_variant: str

    def __post_init__(self) -> None:
        if self.x.dtype != np.float32 or self.x.ndim != 2 or self.x.shape[1] != 3000:
            raise ValueError("demo signal must be float32 with shape (epochs, 3000)")
        if len(self.x) == 0 or not np.isfinite(self.x).all():
            raise ValueError("demo signal must contain finite epochs")
        if self.original_epoch_index.shape != (len(self.x),):
            raise ValueError("original_epoch_index must align with signal epochs")
        if self.labels is not None:
            if self.labels.shape != (len(self.x),) or not np.isin(
                self.labels, [-1, 0, 1, 2, 3, 4]
            ).all():
                raise ValueError("labels must align with signal epochs")
        if self.data_variant not in set(EXPERIMENT_VARIANTS.values()):
            raise ValueError(f"unsupported demo data variant: {self.data_variant}")


@dataclass(frozen=True)
class DemoModels:
    experiment_id: str
    extractor_kind: str
    extractor: EEGResNet1D | Mapping[str, SleepCNN]
    sequence_kind: str
    sequence: SleepTCN | BiLSTMSleepNet
    device: torch.device
    outer_fold: int
    seed: int
    source_ref: str


@dataclass(frozen=True)
class DemoPrediction:
    experiment_id: str
    predicted: np.ndarray
    probabilities: np.ndarray
    elapsed_seconds: float
    epochs_per_second: float
    device: str

    def __post_init__(self) -> None:
        if self.experiment_id not in DEMO_EXPERIMENTS:
            raise ValueError(f"unsupported experiment: {self.experiment_id}")
        if self.predicted.ndim != 1 or self.probabilities.shape != (
            len(self.predicted),
            5,
        ):
            raise ValueError("prediction arrays are not aligned")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("probabilities contain non-finite values")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("prediction probabilities must sum to one")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_path(asset_root: Path, relative: str) -> Path:
    root = asset_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"asset path escapes asset root: {relative}") from error
    return path


def _validate_file_entry(asset_root: Path, label: str, entry: Any) -> None:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ValueError(f"invalid {label} asset entry")
    expected = str(entry["sha256"])
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"invalid {label} SHA-256")
    path = _asset_path(asset_root, str(entry["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")


def validate_asset_manifest(asset_root: Path) -> dict[str, Any]:
    """Validate every checkpoint and locked prediction before use."""

    asset_root = asset_root.resolve()
    manifest_path = asset_root / "demo_assets.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ASSET_SCHEMA_VERSION:
        raise ValueError("unsupported demo asset schema")
    if not isinstance(manifest.get("outer_fold"), int) or not isinstance(
        manifest.get("seed"), int
    ):
        raise ValueError("demo asset fold and seed must be integers")
    test_records = manifest.get("test_records")
    if not isinstance(test_records, list) or not test_records or any(
        not isinstance(value, str) or not value for value in test_records
    ):
        raise ValueError("demo manifest must list fold test records")
    experiments = manifest.get("experiments")
    if not isinstance(experiments, dict) or tuple(experiments) != DEMO_EXPERIMENTS:
        raise ValueError("demo manifest must contain E0, E3 and E6 in canonical order")
    for experiment_id, expected_variant in EXPERIMENT_VARIANTS.items():
        experiment = experiments[experiment_id]
        if experiment.get("data_variant") != expected_variant:
            raise ValueError(f"{experiment_id} data variant mismatch")
        _validate_file_entry(
            asset_root, f"{experiment_id} prediction", experiment.get("prediction")
        )
        _validate_file_entry(
            asset_root, f"{experiment_id} sequence", experiment.get("sequence")
        )
        if experiment_id == "E0":
            extractors = experiment.get("extractors")
            if not isinstance(extractors, dict) or tuple(extractors) != expected_15cnn_keys():
                raise ValueError("E0 manifest must contain the canonical 15CNN set")
            for key, entry in extractors.items():
                _validate_file_entry(asset_root, f"E0 extractor {key}", entry)
        else:
            _validate_file_entry(
                asset_root, f"{experiment_id} extractor", experiment.get("extractor")
            )
    return manifest


def _validate_checkpoint_metadata(
    payload: dict[str, Any], manifest: dict[str, Any], experiment_id: str, stage: str
) -> None:
    if payload.get("schema_version") != 1 or "model_state" not in payload:
        raise ValueError("unsupported checkpoint payload")
    metadata = payload.get("metadata", {})
    expected = {
        "experiment_id": experiment_id,
        "stage": stage,
        "outer_fold": manifest["outer_fold"],
        "data_variant": EXPERIMENT_VARIANTS[experiment_id],
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint metadata mismatch: {mismatches}")


def _load_payload(
    asset_root: Path,
    entry: Mapping[str, str],
    manifest: dict[str, Any],
    experiment_id: str,
    stage: str,
) -> dict[str, Any]:
    payload = torch.load(
        _asset_path(asset_root, entry["path"]),
        map_location="cpu",
        weights_only=False,
    )
    _validate_checkpoint_metadata(payload, manifest, experiment_id, stage)
    return payload


def load_demo_models(
    asset_root: Path,
    experiment_id: str,
    device: str | torch.device = "cpu",
    *,
    validated_manifest: dict[str, Any] | None = None,
) -> DemoModels:
    """Load one verified fold checkpoint set for live EDF inference."""

    if experiment_id not in DEMO_EXPERIMENTS:
        raise ValueError(f"unsupported experiment: {experiment_id}")
    asset_root = asset_root.resolve()
    manifest = validated_manifest or validate_asset_manifest(asset_root)
    entry = manifest["experiments"][experiment_id]
    device = torch.device(device)
    if experiment_id == "E0":
        extractor_kind = "cnn15"
        extractor: EEGResNet1D | dict[str, SleepCNN] = {}
        for index, key in enumerate(expected_15cnn_keys()):
            model = SleepCNN()
            payload = _load_payload(
                asset_root,
                entry["extractors"][key],
                manifest,
                experiment_id,
                f"cnn15/{key}",
            )
            expected_seed = manifest["seed"] + index
            if payload.get("metadata", {}).get("seed") != expected_seed:
                raise ValueError(f"E0 {key} component seed mismatch")
            model.load_state_dict(payload["model_state"], strict=True)
            extractor[key] = model.to(device).eval()
        sequence_kind = "bilstm"
        sequence: SleepTCN | BiLSTMSleepNet = BiLSTMSleepNet()
        sequence_stage = "sequence/bilstm"
    else:
        extractor_kind = "resnet1d"
        extractor = EEGResNet1D()
        payload = _load_payload(
            asset_root,
            entry["extractor"],
            manifest,
            experiment_id,
            "resnet1d",
        )
        extractor.load_state_dict(payload["model_state"], strict=True)
        extractor.to(device).eval()
        sequence_kind = "tcn"
        sequence = SleepTCN(input_dim=128)
        sequence_stage = "sequence/tcn"
    sequence_payload = _load_payload(
        asset_root,
        entry["sequence"],
        manifest,
        experiment_id,
        sequence_stage,
    )
    sequence.load_state_dict(sequence_payload["model_state"], strict=True)
    sequence.to(device).eval()
    return DemoModels(
        experiment_id=experiment_id,
        extractor_kind=extractor_kind,
        extractor=extractor,
        sequence_kind=sequence_kind,
        sequence=sequence,
        device=device,
        outer_fold=int(manifest["outer_fold"]),
        seed=int(manifest["seed"]),
        source_ref=str(manifest["source_ref"]),
    )


def load_processed_demo_record(path: Path, variant: str | None = None) -> DemoRecord:
    variant = variant or path.parent.name
    if variant not in set(EXPERIMENT_VARIANTS.values()):
        raise ValueError(f"unsupported demo data variant: {variant}")
    record = load_record(path, variant)
    return DemoRecord(
        record_key=record.info.record_key,
        x=record.x,
        labels=record.y,
        original_epoch_index=record.original_epoch_index,
        source=f"Sleep-EDF Expanded / {variant}",
        note="Nhãn chỉ được dùng để đánh giá sau suy luận, không được đưa vào mô hình.",
        data_variant=variant,
    )


def _read_edf_signal(path: Path, channel: str) -> np.ndarray:
    import pyedflib

    reader = pyedflib.EdfReader(str(path.resolve()))
    try:
        channels = [str(value).strip() for value in reader.getSignalLabels()]
        if channel not in channels:
            raise ValueError(f"EDF is missing channel {channel!r}; found {channels}")
        index = channels.index(channel)
        sampling_rate = float(reader.getSampleFrequency(index))
        if not math.isclose(sampling_rate, 100.0, abs_tol=1e-9):
            raise ValueError(f"Expected 100 Hz, found {sampling_rate:g} Hz")
        dimension = str(reader.getPhysicalDimension(index)).strip()
        normalized = dimension.lower().replace("μ", "u").replace("µ", "u")
        if normalized != "uv":
            raise ValueError(f"Expected physical unit uV, found {dimension!r}")
        signal = reader.readSignal(index).astype(np.float64, copy=False)
    finally:
        reader.close()
    complete_samples = signal.size - signal.size % 3000
    if complete_samples < 3000:
        raise ValueError("EDF does not contain a complete 30-second epoch")
    return signal[:complete_samples]


def inspect_edf_demo(path: Path, channel: str = "EEG Fpz-Cz") -> dict[str, Any]:
    """Inspect the EDF input contract without running preprocessing or inference."""

    import pyedflib

    reader = pyedflib.EdfReader(str(path.resolve()))
    try:
        channels = [str(value).strip() for value in reader.getSignalLabels()]
        result: dict[str, Any] = {
            "channels": channels,
            "required_channel": channel,
            "has_required_channel": channel in channels,
            "sampling_rate_hz": None,
            "physical_dimension": None,
            "unit_is_uv": False,
            "samples": 0,
            "complete_epochs": 0,
            "trailing_samples": 0,
            "file_duration_seconds": float(reader.getFileDuration()),
        }
        if channel in channels:
            index = channels.index(channel)
            sampling_rate = float(reader.getSampleFrequency(index))
            dimension = str(reader.getPhysicalDimension(index)).strip()
            normalized = dimension.lower().replace("μ", "u").replace("µ", "u")
            samples = int(reader.getNSamples()[index])
            samples_per_epoch = int(round(sampling_rate * 30.0)) if sampling_rate > 0 else 0
            result.update(
                {
                    "sampling_rate_hz": sampling_rate,
                    "physical_dimension": dimension,
                    "unit_is_uv": normalized == "uv",
                    "samples": samples,
                    "complete_epochs": samples // samples_per_epoch
                    if samples_per_epoch
                    else 0,
                    "trailing_samples": samples % samples_per_epoch
                    if samples_per_epoch
                    else samples,
                }
            )
    finally:
        reader.close()
    result["sampling_rate_is_100hz"] = bool(
        result["sampling_rate_hz"] is not None
        and math.isclose(float(result["sampling_rate_hz"]), 100.0, abs_tol=1e-9)
    )
    result["has_complete_epoch"] = result["complete_epochs"] > 0
    result["ready"] = bool(
        result["has_required_channel"]
        and result["sampling_rate_is_100hz"]
        and result["unit_is_uv"]
        and result["has_complete_epoch"]
    )
    return result


def load_edf_demo_records(
    path: Path, channel: str = "EEG Fpz-Cz"
) -> dict[str, DemoRecord]:
    """Create aligned E0/E3/E6 inputs from one unlabeled EDF."""

    signal = _read_edf_signal(path, channel)
    config = PreprocessConfig()
    arrays = {"paper_raw_v1": signal.astype(np.float32).reshape(-1, 3000)}
    for variant in ("filtered_v2", "filtered_zscore_v2"):
        processed, _, _ = preprocess_signal_variant(signal, variant, config)
        arrays[variant] = processed.reshape(-1, 3000)
    result: dict[str, DemoRecord] = {}
    for experiment_id, variant in EXPERIMENT_VARIANTS.items():
        epochs = arrays[variant]
        result[experiment_id] = DemoRecord(
            record_key=path.stem,
            x=epochs,
            labels=None,
            original_epoch_index=np.arange(len(epochs), dtype=np.int32),
            source=f"Uploaded EDF / {channel}",
            note=(
                "Suy luận khám phá: không có hypnogram nên không áp dụng được bước "
                "trim theo nhãn của giao thức đánh giá khóa luận."
            ),
            data_variant=variant,
        )
    return result


def load_edf_demo_record(
    path: Path, channel: str = "EEG Fpz-Cz", variant: str = "filtered_v2"
) -> DemoRecord:
    """Backward-compatible single-variant EDF loader."""

    experiment_id = next(
        (key for key, value in EXPERIMENT_VARIANTS.items() if value == variant), None
    )
    if experiment_id is None:
        raise ValueError(f"unsupported demo data variant: {variant}")
    return load_edf_demo_records(path, channel)[experiment_id]


def slice_demo_record(record: DemoRecord, start: int, count: int) -> DemoRecord:
    if start < 0 or count <= 0 or start + count > len(record.x):
        raise ValueError("invalid demo record slice")
    labels = None if record.labels is None else record.labels[start : start + count].copy()
    return DemoRecord(
        record_key=record.record_key,
        x=record.x[start : start + count].copy(),
        labels=labels,
        original_epoch_index=record.original_epoch_index[start : start + count].copy(),
        source=record.source,
        note=record.note,
        data_variant=record.data_variant,
    )


def slice_prediction(prediction: DemoPrediction, start: int, count: int) -> DemoPrediction:
    if start < 0 or count <= 0 or start + count > len(prediction.predicted):
        raise ValueError("invalid demo prediction slice")
    return DemoPrediction(
        experiment_id=prediction.experiment_id,
        predicted=prediction.predicted[start : start + count].copy(),
        probabilities=prediction.probabilities[start : start + count].copy(),
        elapsed_seconds=prediction.elapsed_seconds,
        epochs_per_second=prediction.epochs_per_second,
        device=prediction.device,
    )


@torch.inference_mode()
def predict_record(
    record: DemoRecord, models: DemoModels, *, batch_size: int = 128
) -> DemoPrediction:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    expected_variant = EXPERIMENT_VARIANTS[models.experiment_id]
    if record.data_variant != expected_variant:
        raise ValueError(
            f"{models.experiment_id} requires {expected_variant}, found {record.data_variant}"
        )
    start_time = time.perf_counter()
    if models.extractor_kind == "cnn15":
        features_array = extract_15cnn_features(
            record.x,
            models.extractor,
            device=models.device,
            batch_size=batch_size,
        )
        features = torch.from_numpy(features_array).unsqueeze(0).to(models.device)
    else:
        assert isinstance(models.extractor, EEGResNet1D)
        feature_parts: list[torch.Tensor] = []
        for start in range(0, len(record.x), batch_size):
            batch = torch.from_numpy(record.x[start : start + batch_size]).unsqueeze(1)
            feature_parts.append(
                models.extractor.extract_features(batch.to(models.device)).cpu()
            )
        features = torch.cat(feature_parts, dim=0).unsqueeze(0).to(models.device)
    if models.sequence_kind == "bilstm":
        assert isinstance(models.sequence, BiLSTMSleepNet)
        logits = models.sequence(
            features, torch.tensor([len(record.x)], dtype=torch.long)
        ).squeeze(0)
    else:
        assert isinstance(models.sequence, SleepTCN)
        logits = models.sequence(features, padding_mask=None).squeeze(0)
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
    predicted = probabilities.argmax(axis=1).astype(np.int8)
    elapsed = max(time.perf_counter() - start_time, np.finfo(float).eps)
    return DemoPrediction(
        experiment_id=models.experiment_id,
        predicted=predicted,
        probabilities=probabilities,
        elapsed_seconds=float(elapsed),
        epochs_per_second=float(len(record.x) / elapsed),
        device=str(models.device),
    )


def predict_e3(
    record: DemoRecord, models: DemoModels, *, batch_size: int = 128
) -> DemoPrediction:
    if models.experiment_id != "E3":
        raise ValueError("predict_e3 requires E3 models")
    return predict_record(record, models, batch_size=batch_size)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return (exponential / exponential.sum(axis=1, keepdims=True)).astype(np.float32)


def load_locked_prediction(
    asset_root: Path,
    experiment_id: str,
    record: DemoRecord,
    *,
    validated_manifest: dict[str, Any] | None = None,
) -> DemoPrediction:
    """Load one record from the immutable fold-level test prediction artifact."""

    if experiment_id not in DEMO_EXPERIMENTS:
        raise ValueError(f"unsupported experiment: {experiment_id}")
    manifest = validated_manifest or validate_asset_manifest(asset_root)
    entry = manifest["experiments"][experiment_id]["prediction"]
    with np.load(_asset_path(asset_root, entry["path"]), allow_pickle=False) as artifact:
        mask = artifact["record_key"] == record.record_key
        if not np.any(mask):
            raise ValueError(f"{record.record_key} is absent from {experiment_id} test predictions")
        artifact_labels = artifact["true_label"][mask].astype(np.int8, copy=False)
        artifact_indices = artifact["original_epoch_index"][mask].astype(np.int32, copy=False)
        artifact_predicted = artifact["predicted_label"][mask].astype(np.int8, copy=False)
        artifact_probabilities = _softmax(artifact["logits"][mask])
    if record.labels is None:
        raise ValueError(f"{experiment_id} locked prediction requires record labels")
    if not np.array_equal(
        artifact_predicted, artifact_probabilities.argmax(axis=1).astype(np.int8)
    ):
        raise ValueError(f"{experiment_id} stored labels disagree with logits")
    positions = np.searchsorted(record.original_epoch_index, artifact_indices)
    if (
        np.any(positions >= len(record.original_epoch_index))
        or not np.array_equal(record.original_epoch_index[positions], artifact_indices)
    ):
        raise ValueError(f"{experiment_id} prediction epoch alignment mismatch")
    if not np.array_equal(record.labels[positions], artifact_labels):
        raise ValueError(f"{experiment_id} prediction label alignment mismatch")
    predicted = np.full(len(record.x), -1, dtype=np.int8)
    probabilities = np.full((len(record.x), 5), .2, dtype=np.float32)
    predicted[positions] = artifact_predicted
    probabilities[positions] = artifact_probabilities
    return DemoPrediction(
        experiment_id=experiment_id,
        predicted=predicted,
        probabilities=probabilities,
        elapsed_seconds=0.0,
        epochs_per_second=0.0,
        device="locked test artifact",
    )


def stage_transition_mask(labels: np.ndarray, radius: int = 2) -> np.ndarray:
    """Mark epochs within ``radius`` of a valid true-stage change."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or radius < 0:
        raise ValueError("labels must be one-dimensional and radius non-negative")
    result = np.zeros(len(labels), dtype=bool)
    changes = np.flatnonzero(
        (labels[1:] >= 0) & (labels[:-1] >= 0) & (labels[1:] != labels[:-1])
    ) + 1
    for index in changes:
        result[max(0, index - radius) : min(len(labels), index + radius + 1)] = True
    return result


def n3_to_n2_mask(labels: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    if labels.shape != predicted.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be aligned one-dimensional arrays")
    return (labels == 3) & (predicted == 2)


def available_fold_records(
    asset_root: Path,
    processed_root: Path,
    *,
    validated_manifest: dict[str, Any] | None = None,
) -> list[Path]:
    manifest = validated_manifest or validate_asset_manifest(asset_root)
    processed_root = processed_root.resolve()
    return [
        processed_root / f"{record_key}.npz"
        for record_key in manifest["test_records"]
        if (processed_root / f"{record_key}.npz").is_file()
    ]
