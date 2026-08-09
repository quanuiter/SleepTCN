"""Tiền xử lý có kiểm soát cho Sleep-EDF Expanded Sleep Cassette."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyedflib
from scipy.signal import butter, sosfiltfilt


KEY_RE = re.compile(r"^(SC\d{4}[A-Z])")
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Movement time": -1,
    "Sleep stage ?": -1,
}
VALID_VARIANTS = {
    "paper_raw_v1",
    "bandpass_v2",
    "bandpass_clip_v2",
    "filtered_v2",
    "filtered_zscore_v2",
}


@dataclass(frozen=True)
class PreprocessConfig:
    channel: str = "EEG Fpz-Cz"
    sampling_rate_hz: float = 100.0
    epoch_seconds: int = 30
    wake_edge_minutes: int = 30
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 30.0
    bandpass_order: int = 4
    clip_uv: float = 800.0
    scale_factor: float = 100.0
    trim_anchor_policy: str = "true_sleep_n1_to_rem"

    @property
    def samples_per_epoch(self) -> int:
        value = self.sampling_rate_hz * self.epoch_seconds
        if not float(value).is_integer():
            raise ValueError("samples_per_epoch must be an integer")
        return int(value)

    @property
    def wake_edge_epochs(self) -> int:
        seconds = self.wake_edge_minutes * 60
        if seconds % self.epoch_seconds != 0:
            raise ValueError("wake edge must be divisible by epoch length")
        return seconds // self.epoch_seconds


def record_key(path: Path) -> str:
    match = KEY_RE.match(path.name)
    if match is None:
        raise ValueError(f"Invalid Sleep Cassette filename: {path.name}")
    return match.group(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_hashes(manifest_path: Path) -> dict[str, dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("hashes_included"):
        raise ValueError("Raw manifest does not contain SHA-256 hashes")
    result: dict[str, dict[str, str]] = {}
    for record in manifest["records"]:
        key = record["record_key"]
        result[key] = {
            "psg_sha256": record["psg"]["sha256"],
            "hypnogram_sha256": record["hypnogram"]["sha256"],
        }
    return result


def pair_files(data_dir: Path) -> dict[str, tuple[Path, Path]]:
    def build(pattern: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for path in sorted(data_dir.glob(pattern)):
            key = record_key(path)
            if key in result:
                raise RuntimeError(f"Duplicate record key {key} for {pattern}")
            result[key] = path
        return result

    psg = build("*-PSG.edf")
    hyp = build("*-Hypnogram.edf")
    if set(psg) != set(hyp):
        raise RuntimeError(
            f"Unpaired files: missing_hyp={sorted(set(psg)-set(hyp))}, "
            f"missing_psg={sorted(set(hyp)-set(psg))}"
        )
    return {key: (psg[key], hyp[key]) for key in sorted(psg)}


def annotations_to_labels(
    onsets: Iterable[float],
    durations: Iterable[float],
    annotations: Iterable[str],
    epoch_seconds: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    expected_onset = 0.0
    for onset_raw, duration_raw, annotation_raw in zip(
        onsets, durations, annotations, strict=True
    ):
        onset = float(onset_raw)
        duration = float(duration_raw)
        annotation = str(annotation_raw)
        if annotation not in LABEL_MAP:
            raise ValueError(f"Unknown annotation: {annotation!r}")
        if not math.isclose(onset, expected_onset, abs_tol=1e-6):
            raise ValueError(
                f"Discontinuous annotation timeline: {onset} != {expected_onset}"
            )
        epochs = duration / epoch_seconds
        if duration <= 0 or not math.isclose(epochs, round(epochs), abs_tol=1e-6):
            raise ValueError(
                f"Annotation duration is not a positive {epoch_seconds}s multiple: {duration}"
            )
        pieces.append(
            np.full(int(round(epochs)), LABEL_MAP[annotation], dtype=np.int8)
        )
        expected_onset = onset + duration
    if not pieces:
        raise ValueError("No annotations")
    return np.concatenate(pieces)


def trim_sleep_window(
    signal_epochs: np.ndarray,
    labels: np.ndarray,
    edge_epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if signal_epochs.ndim != 2:
        raise ValueError("signal_epochs must have shape (epochs, samples)")
    if len(signal_epochs) != len(labels):
        raise ValueError("signal and label epoch counts differ")
    sleep_indices = np.flatnonzero((labels >= 1) & (labels <= 4))
    if sleep_indices.size == 0:
        raise ValueError("No valid sleep stage (N1-N3/REM) found")
    # Cắt theo epoch ngủ thật. Movement/Unknown được giữ nếu nằm trong
    # cửa sổ nhưng không được phép kéo cửa sổ vào vùng annotation '?' dài.
    start = max(0, int(sleep_indices[0]) - edge_epochs)
    stop = min(len(labels), int(sleep_indices[-1]) + edge_epochs + 1)
    original_indices = np.arange(start, stop, dtype=np.int32)
    return (
        signal_epochs[start:stop],
        labels[start:stop],
        original_indices,
        start,
        stop,
    )


def preprocess_signal_variant(
    signal: np.ndarray, variant: str, config: PreprocessConfig
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Apply one auditable preprocessing ablation to a continuous EEG record.

    Z-score statistics are computed from the full record after filtering and
    clipping. They never use labels or train/validation/test population data.
    """
    if signal.ndim != 1:
        raise ValueError("continuous signal must be one-dimensional")
    if variant not in VALID_VARIANTS - {"paper_raw_v1"}:
        raise ValueError(f"Unsupported filtered variant: {variant}")
    sos = butter(
        config.bandpass_order,
        [config.bandpass_low_hz, config.bandpass_high_hz],
        btype="bandpass",
        fs=config.sampling_rate_hz,
        output="sos",
    )
    bandpassed = sosfiltfilt(sos, signal.astype(np.float64, copy=False))
    clip_fraction = float(np.mean(np.abs(bandpassed) > config.clip_uv))
    metadata: dict[str, Any] = {
        "normalization_scope": "none",
        "normalization_mean": None,
        "normalization_std": None,
    }
    if variant == "bandpass_v2":
        output = bandpassed
        normalization = "none"
    else:
        clipped = np.clip(bandpassed, -config.clip_uv, config.clip_uv)
        if variant == "bandpass_clip_v2":
            output = clipped
            normalization = f"clip_{config.clip_uv}uV"
        elif variant == "filtered_v2":
            output = clipped / config.scale_factor
            normalization = (
                f"clip_{config.clip_uv}uV_then_divide_{config.scale_factor}"
            )
        elif variant == "filtered_zscore_v2":
            mean = float(clipped.mean(dtype=np.float64))
            std = float(clipped.std(dtype=np.float64))
            if not np.isfinite(std) or std <= 0.0:
                raise ValueError("Cannot z-score a record with zero/nonfinite std")
            output = (clipped - mean) / std
            normalization = f"clip_{config.clip_uv}uV_then_record_zscore"
            metadata.update(
                {
                    "normalization_scope": "full_record_after_filter_clip",
                    "normalization_mean": mean,
                    "normalization_std": std,
                }
            )
        else:  # pragma: no cover - guarded above
            raise AssertionError(variant)
    if not np.isfinite(output).all():
        raise ValueError("Processed signal contains NaN or infinity")
    metadata["normalization"] = normalization
    return output.astype(np.float32), clip_fraction, metadata


def filtered_v2(signal: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, float]:
    """Backward-compatible helper for the frozen filtered_v2 definition."""
    output, clip_fraction, _ = preprocess_signal_variant(
        signal, "filtered_v2", config
    )
    return output, clip_fraction


def atomic_savez(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def process_record(
    psg_path: Path,
    hyp_path: Path,
    variants: list[str],
    output_root: Path,
    source_hashes: dict[str, str],
    config: PreprocessConfig,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    unknown_variants = set(variants) - VALID_VARIANTS
    if unknown_variants:
        raise ValueError(f"Unknown variants: {sorted(unknown_variants)}")

    key = record_key(psg_path)
    if key != record_key(hyp_path):
        raise ValueError("PSG/Hypnogram record keys differ")
    output_paths = {variant: output_root / variant / f"{key}.npz" for variant in variants}
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Outputs already exist: {existing}")

    actual_psg_hash = sha256_file(psg_path)
    actual_hyp_hash = sha256_file(hyp_path)
    if actual_psg_hash != source_hashes["psg_sha256"]:
        raise ValueError(f"PSG SHA-256 differs from raw manifest: {psg_path.name}")
    if actual_hyp_hash != source_hashes["hypnogram_sha256"]:
        raise ValueError(
            f"Hypnogram SHA-256 differs from raw manifest: {hyp_path.name}"
        )

    psg = pyedflib.EdfReader(str(psg_path))
    hyp = pyedflib.EdfReader(str(hyp_path))
    try:
        if psg.getStartdatetime() != hyp.getStartdatetime():
            raise ValueError("PSG/Hypnogram start datetimes differ")
        channels = psg.getSignalLabels()
        if config.channel not in channels:
            raise ValueError(f"Missing channel: {config.channel}")
        channel_index = channels.index(config.channel)
        fs = float(psg.getSampleFrequency(channel_index))
        if not math.isclose(fs, config.sampling_rate_hz, abs_tol=1e-9):
            raise ValueError(f"Unexpected sampling rate: {fs}")
        dimension = str(psg.getPhysicalDimension(channel_index)).strip()
        normalized_dimension = dimension.lower().replace("μ", "u").replace("µ", "u")
        if normalized_dimension != "uv":
            raise ValueError(f"Unexpected physical dimension: {dimension!r}")

        continuous_raw = psg.readSignal(channel_index).astype(np.float64, copy=False)
        if not np.isfinite(continuous_raw).all():
            raise ValueError("Raw signal contains NaN or infinity")
        if continuous_raw.size % config.samples_per_epoch != 0:
            raise ValueError("Signal length is not divisible into 30-second epochs")

        onsets, durations, annotations = hyp.readAnnotations()
        labels_all = annotations_to_labels(
            onsets, durations, annotations, config.epoch_seconds
        )
        signal_epoch_count = continuous_raw.size // config.samples_per_epoch
        if len(labels_all) < signal_epoch_count:
            raise ValueError("Annotations are shorter than signal")
        labels_aligned = labels_all[:signal_epoch_count]
        annotation_epochs_truncated = len(labels_all) - signal_epoch_count

        raw_epochs = continuous_raw.reshape(signal_epoch_count, config.samples_per_epoch)
        _, labels_trimmed, original_indices, start, stop = trim_sleep_window(
            raw_epochs, labels_aligned, config.wake_edge_epochs
        )

        base_metadata: dict[str, Any] = {
            "y": labels_trimmed.astype(np.int8, copy=False),
            "valid_mask": (labels_trimmed >= 0),
            "original_epoch_index": original_indices,
            "record_key": np.array(key),
            "subject_id": np.array(key[:5]),
            "psg_name": np.array(psg_path.name),
            "hypnogram_name": np.array(hyp_path.name),
            "source_psg_sha256": np.array(source_hashes["psg_sha256"]),
            "source_hypnogram_sha256": np.array(source_hashes["hypnogram_sha256"]),
            "source_hashes_verified": np.bool_(True),
            "channel": np.array(config.channel),
            "physical_dimension_input": np.array(dimension),
            "sampling_rate_hz": np.float32(config.sampling_rate_hz),
            "epoch_seconds": np.int16(config.epoch_seconds),
            "samples_per_epoch": np.int16(config.samples_per_epoch),
            "trim_start_epoch": np.int32(start),
            "trim_stop_epoch_exclusive": np.int32(stop),
            "signal_epochs_before_trim": np.int32(signal_epoch_count),
            "annotation_epochs_before_alignment": np.int32(len(labels_all)),
            "annotation_epochs_truncated": np.int32(annotation_epochs_truncated),
            "start_datetime": np.array(psg.getStartdatetime().isoformat()),
            "psg_file_seconds": np.float64(psg.getFileDuration()),
            "trim_anchor_policy": np.array(config.trim_anchor_policy),
            "label_mapping_json": np.array(json.dumps(LABEL_MAP, sort_keys=True)),
        }

        results: list[dict[str, Any]] = []
        for variant in variants:
            if variant == "paper_raw_v1":
                x = raw_epochs[start:stop].astype(np.float32)
                clip_fraction = None
                variant_metadata = {
                    "filter": np.array("none"),
                    "normalization": np.array("none"),
                }
            elif variant in VALID_VARIANTS - {"paper_raw_v1"}:
                processed, clip_fraction, processing_metadata = (
                    preprocess_signal_variant(continuous_raw, variant, config)
                )
                x = processed.reshape(signal_epoch_count, config.samples_per_epoch)[start:stop]
                variant_metadata = {
                    "filter": np.array(
                        f"butterworth_sosfiltfilt_order{config.bandpass_order}_"
                        f"{config.bandpass_low_hz}-{config.bandpass_high_hz}Hz"
                    ),
                    "normalization": np.array(processing_metadata["normalization"]),
                    "normalization_scope": np.array(
                        processing_metadata["normalization_scope"]
                    ),
                    "normalization_mean": np.float64(
                        np.nan
                        if processing_metadata["normalization_mean"] is None
                        else processing_metadata["normalization_mean"]
                    ),
                    "normalization_std": np.float64(
                        np.nan
                        if processing_metadata["normalization_std"] is None
                        else processing_metadata["normalization_std"]
                    ),
                }
            else:  # pragma: no cover - protected above
                raise AssertionError(variant)

            if x.shape != (len(labels_trimmed), config.samples_per_epoch):
                raise AssertionError("Output shape mismatch")
            if not np.isfinite(x).all():
                raise ValueError("Output contains NaN or infinity")

            arrays = {
                "x": x,
                **base_metadata,
                **variant_metadata,
                "preprocess_version": np.array(variant),
                "config_json": np.array(json.dumps(asdict(config), sort_keys=True)),
            }
            atomic_savez(output_paths[variant], arrays)
            output_hash = sha256_file(output_paths[variant])
            label_counts = Counter(int(value) for value in labels_trimmed)
            results.append(
                {
                    "record_key": key,
                    "subject_id": key[:5],
                    "variant": variant,
                    "output_path": str(output_paths[variant].resolve()),
                    "output_sha256": output_hash,
                    "epochs": int(len(labels_trimmed)),
                    "samples_per_epoch": config.samples_per_epoch,
                    "label_counts": {
                        str(label): int(label_counts.get(label, 0))
                        for label in (-1, 0, 1, 2, 3, 4)
                    },
                    "trim_start_epoch": start,
                    "trim_stop_epoch_exclusive": stop,
                    "annotation_epochs_truncated": annotation_epochs_truncated,
                    "clip_fraction": clip_fraction,
                    "x_min": float(x.min()),
                    "x_max": float(x.max()),
                    "x_mean": float(x.mean(dtype=np.float64)),
                    "x_std": float(x.std(dtype=np.float64)),
                }
            )
        return results
    finally:
        psg.close()
        hyp.close()
