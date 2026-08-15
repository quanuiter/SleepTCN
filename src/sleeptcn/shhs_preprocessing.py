"""Locked, auditable SHHS1 preprocessing for external checkpoint evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyedflib
from scipy.signal import resample_poly

from .preprocessing import (
    PreprocessConfig,
    atomic_savez,
    preprocess_signal_variant,
    sha256_file,
    trim_sleep_window,
)


SHHS_VARIANTS = ("paper_raw_v1", "filtered_v2", "filtered_zscore_v2")
RAW_STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: -1, 9: -1}
PRIMARY_ROLE_COUNTS = {"adaptation": 5, "validation": 15, "test": 180}


@dataclass(frozen=True)
class SHHSPreprocessConfig:
    source_channel: str = "EEG"
    source_montage: str = "C4-A1"
    source_sampling_rate_hz: float = 125.0
    target_sampling_rate_hz: float = 100.0
    epoch_seconds: int = 30
    resample_up: int = 4
    resample_down: int = 5
    resample_window_beta: float = 5.0
    resample_padtype: str = "constant"
    wake_edge_minutes: int = 30
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 30.0
    bandpass_order: int = 4
    clip_uv: float = 800.0
    scale_factor: float = 100.0
    trim_anchor_policy: str = "true_sleep_n1_to_rem"

    @property
    def source_samples_per_epoch(self) -> int:
        return int(self.source_sampling_rate_hz * self.epoch_seconds)

    @property
    def target_samples_per_epoch(self) -> int:
        return int(self.target_sampling_rate_hz * self.epoch_seconds)

    @property
    def wake_edge_epochs(self) -> int:
        return self.wake_edge_minutes * 60 // self.epoch_seconds

    def sleepedf_compatible_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            channel=self.source_channel,
            sampling_rate_hz=self.target_sampling_rate_hz,
            epoch_seconds=self.epoch_seconds,
            wake_edge_minutes=self.wake_edge_minutes,
            bandpass_low_hz=self.bandpass_low_hz,
            bandpass_high_hz=self.bandpass_high_hz,
            bandpass_order=self.bandpass_order,
            clip_uv=self.clip_uv,
            scale_factor=self.scale_factor,
            trim_anchor_policy=self.trim_anchor_policy,
        )


def load_locked_config(path: Path) -> tuple[dict[str, Any], str, SHHSPreprocessConfig]:
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported SHHS preprocessing config schema")
    if document.get("status") != "locked_before_pilot_preprocessing":
        raise ValueError("SHHS preprocessing config is not in its locked state")
    if document.get("dataset") != "SHHS Visit 1" or document.get("selection_seed") != 42:
        raise ValueError("SHHS preprocessing config dataset/seed differs from protocol")

    config = SHHSPreprocessConfig()
    source = document.get("source", {})
    resampling = document.get("resampling", {})
    window = resampling.get("window")
    labels = {int(key): int(value) for key, value in document.get("label_mapping", {}).items()}
    variants = tuple(document.get("checkpoint_compatible_variants", {}).keys())
    required = {
        "source_channel": source.get("edf_channel") == config.source_channel,
        "source_montage": source.get("montage") == config.source_montage,
        "source_rate": float(source.get("sampling_rate_hz", -1)) == config.source_sampling_rate_hz,
        "target_rate": float(resampling.get("target_sampling_rate_hz", -1)) == config.target_sampling_rate_hz,
        "epoch_seconds": int(source.get("epoch_seconds", -1)) == config.epoch_seconds,
        "resample_up": int(resampling.get("up", -1)) == config.resample_up,
        "resample_down": int(resampling.get("down", -1)) == config.resample_down,
        "resample_window": window == ["kaiser", config.resample_window_beta],
        "resample_padtype": resampling.get("padtype") == config.resample_padtype,
        "labels": labels == RAW_STAGE_MAP,
        "variants": variants == SHHS_VARIANTS,
        "primary_roles": document.get("roles", {}).get("primary") == PRIMARY_ROLE_COUNTS,
        "reserve_unused": document.get("roles", {}).get("reserve_used") == 0,
    }
    failed = sorted(key for key, passed in required.items() if not passed)
    if failed:
        raise ValueError(f"Locked SHHS config differs from implementation: {failed}")
    return document, hashlib.sha256(raw).hexdigest(), config


def normalize_uv_unit(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return normalized.replace("μ", "u").replace("µ", "u")


def read_raw_stages(path: Path, epoch_seconds: int = 30) -> np.ndarray:
    root = ET.parse(path).getroot()
    epoch_text = root.findtext(".//EpochLength")
    if epoch_text is None or int(epoch_text.strip()) != epoch_seconds:
        raise ValueError(f"{path.name}: EpochLength is not {epoch_seconds}")
    nodes = root.findall(".//SleepStages/SleepStage")
    if not nodes:
        raise ValueError(f"{path.name}: no SleepStage entries")
    try:
        stages = np.asarray([int((node.text or "").strip()) for node in nodes], dtype=np.int8)
    except ValueError as exc:
        raise ValueError(f"{path.name}: invalid SleepStage value") from exc
    unexpected = sorted(set(int(value) for value in stages) - set(RAW_STAGE_MAP))
    if unexpected:
        raise ValueError(f"{path.name}: unexpected raw stages {unexpected}")
    return stages


def raw_stages_to_labels(stages: Iterable[int]) -> np.ndarray:
    values = np.asarray(list(stages), dtype=np.int16)
    unexpected = sorted(set(int(value) for value in values) - set(RAW_STAGE_MAP))
    if unexpected:
        raise ValueError(f"Unexpected raw stages: {unexpected}")
    result = np.asarray([RAW_STAGE_MAP[int(value)] for value in values], dtype=np.int8)
    return result


def resample_continuous_eeg(
    signal: np.ndarray, config: SHHSPreprocessConfig
) -> np.ndarray:
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("SHHS EEG must be one-dimensional and finite")
    if signal.size % config.source_samples_per_epoch != 0:
        raise ValueError("SHHS EEG length is not divisible into source 30-second epochs")
    output = resample_poly(
        signal.astype(np.float64, copy=False),
        up=config.resample_up,
        down=config.resample_down,
        window=("kaiser", config.resample_window_beta),
        padtype=config.resample_padtype,
    )
    epochs = signal.size // config.source_samples_per_epoch
    expected = epochs * config.target_samples_per_epoch
    if output.size != expected:
        raise ValueError(f"Resampled length is {output.size}, expected {expected}")
    if not np.isfinite(output).all():
        raise ValueError("Resampled SHHS EEG contains NaN or infinity")
    return output


def select_manifest_subjects(
    manifest: dict[str, Any], scope: str
) -> list[dict[str, Any]]:
    if manifest.get("dataset") != "SHHS Visit 1" or manifest.get("selection_seed") != 42:
        raise ValueError("Selection manifest is not locked SHHS Visit 1 seed 42")
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list):
        raise ValueError("Selection manifest subjects must be a list")
    if scope == "pilot":
        selected = [item for item in subjects if bool(item.get("pilot"))]
        expected_roles = {"adaptation": 5, "validation": 5}
    elif scope == "primary":
        selected = [item for item in subjects if item.get("role") != "reserve"]
        expected_roles = PRIMARY_ROLE_COUNTS
    else:
        raise ValueError(f"Unknown preprocessing scope: {scope!r}")
    ids = [str(item["subject_id"]) for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subject in preprocessing scope")
    roles = Counter(str(item.get("role")) for item in selected)
    if dict(roles) != expected_roles:
        raise ValueError(f"Role counts differ for {scope}: {dict(roles)}")
    return sorted(selected, key=lambda item: str(item["subject_id"]))


def load_verified_sources(
    manifest_path: Path,
    audit_path: Path,
    locked_config: dict[str, Any],
    scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    manifest_raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha256 != locked_config["selection_manifest_sha256"]:
        raise ValueError("Selection manifest SHA-256 differs from locked config")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    selected = select_manifest_subjects(manifest, scope)

    audit_raw = audit_path.read_bytes()
    audit_sha256 = hashlib.sha256(audit_raw).hexdigest()
    if audit_sha256 != locked_config["technical_audit_sha256"]:
        raise ValueError("Technical audit SHA-256 differs from locked config")
    audit = json.loads(audit_raw.decode("utf-8"))
    if audit.get("status") != "passed" or audit.get("scope") != "selected":
        raise ValueError("Full selected technical audit has not passed")
    if audit.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Technical audit points to a different selection manifest")
    audit_subjects = audit.get("subjects", {})
    for item in selected:
        source = audit_subjects.get(str(item["subject_id"]))
        if not source or not source.get("passed"):
            raise ValueError(f"Subject {item['subject_id']} did not pass technical audit")
    return selected, audit, manifest_sha256, audit_sha256


def _variant_arrays(
    resampled: np.ndarray,
    raw_epochs: np.ndarray,
    start: int,
    stop: int,
    variants: tuple[str, ...],
    config: SHHSPreprocessConfig,
) -> dict[str, tuple[np.ndarray, float | None, dict[str, Any]]]:
    output: dict[str, tuple[np.ndarray, float | None, dict[str, Any]]] = {}
    sleepedf_config = config.sleepedf_compatible_config()
    for variant in variants:
        if variant == "paper_raw_v1":
            x = raw_epochs[start:stop].astype(np.float32)
            metadata = {
                "filter": "none_after_mandatory_resample_poly_antialias",
                "normalization": "none",
                "normalization_scope": "none",
                "normalization_mean": None,
                "normalization_std": None,
            }
            output[variant] = (x, None, metadata)
        else:
            processed, clip_fraction, metadata = preprocess_signal_variant(
                resampled, variant, sleepedf_config
            )
            metadata = {
                **metadata,
                "filter": (
                    f"butterworth_sosfiltfilt_order{config.bandpass_order}_"
                    f"{config.bandpass_low_hz}-{config.bandpass_high_hz}Hz"
                ),
            }
            x = processed.reshape(-1, config.target_samples_per_epoch)[start:stop]
            output[variant] = (x, clip_fraction, metadata)
    return output


def summarize_output(path: Path, variant: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as npz:
        if str(npz["preprocess_version"].item()) != variant:
            raise ValueError(f"{path.name}: existing variant metadata differs")
        x = npz["x"]
        y = npz["y"]
        counts = Counter(int(value) for value in y)
        return {
            "record_key": str(npz["record_key"].item()),
            "subject_id": str(npz["subject_id"].item()),
            "role": str(npz["role"].item()),
            "variant": variant,
            "output_path": str(path.resolve()),
            "output_sha256": sha256_file(path),
            "epochs": int(len(y)),
            "valid_epochs": int(np.sum(y >= 0)),
            "ignored_epochs": int(np.sum(y == -1)),
            "label_counts": {
                str(label): int(counts.get(label, 0)) for label in (-1, 0, 1, 2, 3, 4)
            },
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "x_mean": float(x.mean(dtype=np.float64)),
            "x_std": float(x.std(dtype=np.float64)),
        }


def process_subject(
    item: dict[str, Any],
    audit_subject: dict[str, Any],
    edf_dir: Path,
    xml_dir: Path,
    output_root: Path,
    variants: tuple[str, ...],
    config: SHHSPreprocessConfig,
    config_sha256: str,
    selection_manifest_sha256: str,
    technical_audit_sha256: str,
    resume: bool = False,
) -> list[dict[str, Any]]:
    subject_id = str(item["subject_id"])
    record_key = f"shhs1-{subject_id}"
    edf_path = edf_dir / str(item["edf_filename"])
    xml_path = xml_dir / str(item["annotation_filename"])
    output_paths = {
        variant: output_root / variant / f"{record_key}.npz" for variant in variants
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        if resume and len(existing) == len(output_paths):
            summaries = [summarize_output(output_paths[v], v) for v in variants]
            for variant, path in output_paths.items():
                with np.load(path, allow_pickle=False) as npz:
                    checks = {
                        "subject": str(npz["subject_id"].item()) == subject_id,
                        "record": str(npz["record_key"].item()) == record_key,
                        "config": str(npz["config_sha256"].item()) == config_sha256,
                        "edf_hash": str(npz["source_edf_sha256"].item()) == audit_subject["edf_sha256"],
                        "xml_hash": str(npz["source_xml_sha256"].item()) == audit_subject["xml_sha256"],
                    }
                if not all(checks.values()):
                    raise ValueError(f"{path.name}: existing output cannot be resumed: {checks}")
            return summaries
        raise FileExistsError(
            f"Existing or partial outputs for {record_key}; use --resume only when all variants exist"
        )

    if sha256_file(edf_path) != audit_subject["edf_sha256"]:
        raise ValueError(f"{edf_path.name}: SHA-256 differs from technical audit")
    if sha256_file(xml_path) != audit_subject["xml_sha256"]:
        raise ValueError(f"{xml_path.name}: SHA-256 differs from technical audit")
    raw_stages = read_raw_stages(xml_path, config.epoch_seconds)
    labels_all = raw_stages_to_labels(raw_stages)

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = reader.getSignalLabels()
        if labels.count(config.source_channel) != 1:
            raise ValueError(f"{edf_path.name}: expected exactly one EEG channel")
        channel_index = labels.index(config.source_channel)
        source_rate = float(reader.getSampleFrequency(channel_index))
        if not math.isclose(source_rate, config.source_sampling_rate_hz, abs_tol=1e-9):
            raise ValueError(f"{edf_path.name}: EEG sampling rate is {source_rate}")
        dimension = str(reader.getPhysicalDimension(channel_index)).strip()
        if normalize_uv_unit(dimension) != "uv":
            raise ValueError(f"{edf_path.name}: EEG unit is {dimension!r}")
        continuous = reader.readSignal(channel_index).astype(np.float64, copy=False)
        if not np.isfinite(continuous).all():
            raise ValueError(f"{edf_path.name}: EEG contains NaN or infinity")
        source_epochs = continuous.size // config.source_samples_per_epoch
        if continuous.size % config.source_samples_per_epoch != 0:
            raise ValueError(f"{edf_path.name}: signal is not an exact epoch multiple")
        if len(raw_stages) != source_epochs:
            raise ValueError(
                f"{record_key}: XML has {len(raw_stages)} epochs, EEG has {source_epochs}"
            )
        if not math.isclose(
            float(reader.getFileDuration()), len(raw_stages) * config.epoch_seconds
        ):
            raise ValueError(f"{record_key}: EDF and XML duration differ")
        start_datetime = reader.getStartdatetime().isoformat()
    finally:
        reader.close()

    resampled = resample_continuous_eeg(continuous, config)
    resampled_epochs = resampled.reshape(source_epochs, config.target_samples_per_epoch)
    _, labels_trimmed, original_indices, start, stop = trim_sleep_window(
        resampled_epochs, labels_all, config.wake_edge_epochs
    )
    raw_stages_trimmed = raw_stages[start:stop]
    variant_data = _variant_arrays(
        resampled, resampled_epochs, start, stop, variants, config
    )

    base = {
        "y": labels_trimmed.astype(np.int8, copy=False),
        "raw_stage": raw_stages_trimmed.astype(np.int8, copy=False),
        "valid_mask": (labels_trimmed >= 0),
        "original_epoch_index": original_indices,
        "record_key": np.array(record_key),
        "subject_id": np.array(subject_id),
        "role": np.array(str(item["role"])),
        "role_index": np.int16(int(item["role_index"])),
        "pilot": np.bool_(bool(item["pilot"])),
        "dataset": np.array("SHHS Visit 1"),
        "source_edf_name": np.array(edf_path.name),
        "source_xml_name": np.array(xml_path.name),
        "source_edf_sha256": np.array(audit_subject["edf_sha256"]),
        "source_xml_sha256": np.array(audit_subject["xml_sha256"]),
        "source_hashes_verified": np.bool_(True),
        "selection_manifest_sha256": np.array(selection_manifest_sha256),
        "technical_audit_sha256": np.array(technical_audit_sha256),
        "config_sha256": np.array(config_sha256),
        "channel": np.array(config.source_channel),
        "montage": np.array(config.source_montage),
        "physical_dimension_input": np.array(dimension),
        "source_sampling_rate_hz": np.float32(config.source_sampling_rate_hz),
        "sampling_rate_hz": np.float32(config.target_sampling_rate_hz),
        "epoch_seconds": np.int16(config.epoch_seconds),
        "samples_per_epoch": np.int16(config.target_samples_per_epoch),
        "resampling": np.array("scipy.signal.resample_poly"),
        "resample_up": np.int16(config.resample_up),
        "resample_down": np.int16(config.resample_down),
        "resample_window": np.array(f"kaiser_beta_{config.resample_window_beta}"),
        "resample_padtype": np.array(config.resample_padtype),
        "resample_scope": np.array("continuous_record_before_epoching_and_variant_processing"),
        "source_samples": np.int64(continuous.size),
        "resampled_samples": np.int64(resampled.size),
        "signal_epochs_before_trim": np.int32(source_epochs),
        "trim_start_epoch": np.int32(start),
        "trim_stop_epoch_exclusive": np.int32(stop),
        "trim_anchor_policy": np.array(config.trim_anchor_policy),
        "evaluation_window_label_dependent": np.bool_(True),
        "start_datetime": np.array(start_datetime),
        "label_mapping_json": np.array(json.dumps(RAW_STAGE_MAP, sort_keys=True)),
        "config_json": np.array(json.dumps(asdict(config), sort_keys=True)),
    }

    created: list[Path] = []
    try:
        for variant in variants:
            x, clip_fraction, metadata = variant_data[variant]
            if x.shape != (len(labels_trimmed), config.target_samples_per_epoch):
                raise AssertionError(f"{record_key}/{variant}: output shape mismatch")
            arrays = {
                "x": x,
                **base,
                "preprocess_version": np.array(variant),
                "filter": np.array(metadata["filter"]),
                "normalization": np.array(metadata["normalization"]),
                "normalization_scope": np.array(metadata["normalization_scope"]),
                "normalization_mean": np.float64(
                    np.nan if metadata["normalization_mean"] is None else metadata["normalization_mean"]
                ),
                "normalization_std": np.float64(
                    np.nan if metadata["normalization_std"] is None else metadata["normalization_std"]
                ),
                "clip_fraction": np.float64(
                    np.nan if clip_fraction is None else clip_fraction
                ),
            }
            atomic_savez(output_paths[variant], arrays)
            created.append(output_paths[variant])
        return [summarize_output(output_paths[variant], variant) for variant in variants]
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
