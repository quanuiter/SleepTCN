"""Independent validation of processed Sleep-EDF NPZ records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .hashing import sha256_file


REQUIRED_FIELDS = {
    "x",
    "y",
    "valid_mask",
    "original_epoch_index",
    "record_key",
    "subject_id",
    "source_psg_sha256",
    "source_hypnogram_sha256",
    "source_hashes_verified",
    "channel",
    "sampling_rate_hz",
    "epoch_seconds",
    "samples_per_epoch",
    "preprocess_version",
    "filter",
    "normalization",
    "trim_anchor_policy",
}


def scalar_text(npz: Any, name: str) -> str:
    value = npz[name]
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return str(value.item())


def validate_file(
    path: Path,
    variant: str,
    expected_output_hash: str | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    errors: list[str] = []
    with np.load(path, allow_pickle=False) as npz:
        missing = sorted(REQUIRED_FIELDS - set(npz.files))
        if missing:
            raise ValueError(f"{path.name}: missing fields {missing}")
        x = npz["x"]
        y = npz["y"]
        valid_mask = npz["valid_mask"]
        indices = npz["original_epoch_index"]

        if x.dtype != np.float32:
            errors.append("x_dtype")
        if y.dtype != np.int8:
            errors.append("y_dtype")
        if valid_mask.dtype != np.bool_:
            errors.append("valid_mask_dtype")
        if indices.dtype != np.int32:
            errors.append("index_dtype")
        if x.ndim != 2 or x.shape[1] != 3000:
            errors.append("x_shape")
        if y.ndim != 1 or len(y) != len(x):
            errors.append("y_shape")
        if valid_mask.shape != y.shape or not np.array_equal(valid_mask, y >= 0):
            errors.append("valid_mask")
        if indices.shape != y.shape:
            errors.append("index_shape")
        if len(indices) and not np.all(np.diff(indices) == 1):
            errors.append("nonconsecutive_indices")
        if not np.isin(y, [-1, 0, 1, 2, 3, 4]).all():
            errors.append("invalid_label")
        if not np.isfinite(x).all():
            errors.append("nonfinite_x")
        if scalar_text(npz, "preprocess_version") != variant:
            errors.append("variant_metadata")
        if scalar_text(npz, "record_key") != path.stem:
            errors.append("record_key_metadata")
        if scalar_text(npz, "subject_id") != path.stem[:5]:
            errors.append("subject_metadata")
        if float(npz["sampling_rate_hz"]) != 100.0:
            errors.append("sampling_rate")
        if int(npz["epoch_seconds"]) != 30:
            errors.append("epoch_seconds")
        if int(npz["samples_per_epoch"]) != 3000:
            errors.append("samples_per_epoch")
        if not bool(npz["source_hashes_verified"]):
            errors.append("source_hash_not_verified")
        if scalar_text(npz, "trim_anchor_policy") != "true_sleep_n1_to_rem":
            errors.append("trim_anchor_policy")

        sleep = np.flatnonzero((y >= 1) & (y <= 4))
        if sleep.size == 0:
            errors.append("no_sleep_stage")
        else:
            if indices[0] > 0 and int(sleep[0]) != 60:
                errors.append("leading_wake_window")
            signal_epochs = int(npz["signal_epochs_before_trim"])
            if indices[-1] < signal_epochs - 1 and len(y) - 1 - int(sleep[-1]) != 60:
                errors.append("trailing_wake_window")

        if variant == "paper_raw_v1":
            if scalar_text(npz, "filter") != "none":
                errors.append("paper_filter_metadata")
            if scalar_text(npz, "normalization") != "none":
                errors.append("paper_normalization_metadata")
        elif variant == "filtered_v2":
            if np.max(np.abs(x)) > 8.00001:
                errors.append("filtered_clip_bound")
            if "sosfiltfilt" not in scalar_text(npz, "filter"):
                errors.append("filtered_filter_metadata")
        elif variant == "bandpass_v2":
            if scalar_text(npz, "normalization") != "none":
                errors.append("bandpass_normalization_metadata")
            if "sosfiltfilt" not in scalar_text(npz, "filter"):
                errors.append("bandpass_filter_metadata")
        elif variant == "bandpass_clip_v2":
            if np.max(np.abs(x)) > 800.001:
                errors.append("bandpass_clip_bound")
            if "clip_800.0uV" not in scalar_text(npz, "normalization"):
                errors.append("bandpass_clip_metadata")
        elif variant == "filtered_zscore_v2":
            if "record_zscore" not in scalar_text(npz, "normalization"):
                errors.append("zscore_normalization_metadata")
            if scalar_text(npz, "normalization_scope") != "full_record_after_filter_clip":
                errors.append("zscore_scope_metadata")
            if not np.isfinite(float(npz["normalization_mean"])):
                errors.append("zscore_mean_metadata")
            if not np.isfinite(float(npz["normalization_std"])) or float(
                npz["normalization_std"]
            ) <= 0:
                errors.append("zscore_std_metadata")

        label_counts = Counter(int(value) for value in y)
        summary = {
            "record_key": path.stem,
            "variant": variant,
            "epochs": int(len(y)),
            "valid_epochs": int(np.sum(y >= 0)),
            "ignored_epochs": int(np.sum(y == -1)),
            "label_counts": {
                str(label): int(label_counts.get(label, 0))
                for label in (-1, 0, 1, 2, 3, 4)
            },
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "errors": errors,
        }
        y_copy = y.copy()
        indices_copy = indices.copy()

    if expected_output_hash is not None:
        actual_hash = sha256_file(path)
        if actual_hash != expected_output_hash:
            summary["errors"].append("output_sha256")
    return summary, y_copy, indices_copy


def validate_processed_dataset(
    processed_root: Path,
    preprocess_manifests: Iterable[Path],
    variants: list[str],
    *,
    expected_records: int = 153,
    expected_subjects: int = 78,
) -> dict[str, Any]:
    """Validate all selected variants and cross-variant label/index pairing."""

    preprocess_manifests = list(preprocess_manifests)
    process_manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in preprocess_manifests
    ]
    expected_hashes: dict[tuple[str, str], str] = {}
    canonical_hashes: dict[tuple[str, str], str] = {}
    expected_keys_by_variant: dict[str, set[str]] = {}
    for manifest in process_manifests:
        is_canonical_snapshot = (
            manifest.get("manifest_kind") == "processed_artifact_snapshot"
        )
        for record in manifest["records"]:
            key = (record["variant"], record["record_key"])
            value = record["output_sha256"]
            target = canonical_hashes if is_canonical_snapshot else expected_hashes
            if key in target and target[key] != value:
                raise ValueError(f"conflicting manifest hashes for {key}")
            target[key] = value
            expected_keys_by_variant.setdefault(record["variant"], set()).add(
                record["record_key"]
            )
    # Canonical artifact snapshots supersede legacy preprocess-container
    # hashes.  The latter are retained as provenance, but their ZIP metadata
    # may intentionally differ after canonicalization.
    expected_hashes.update(canonical_hashes)

    files_by_variant = {
        variant: sorted((processed_root / variant).glob("*.npz"))
        for variant in variants
    }
    all_keys = {
        variant: {path.stem for path in files}
        for variant, files in files_by_variant.items()
    }
    reference_keys = all_keys[variants[0]]
    global_errors: list[str] = []
    for variant, keys in all_keys.items():
        if len(keys) != expected_records:
            global_errors.append(f"{variant}:record_count={len(keys)}")
        if keys != reference_keys:
            global_errors.append(f"{variant}:record_keys_differ")
        expected_keys = expected_keys_by_variant.get(variant, set())
        if keys - expected_keys:
            global_errors.append(
                f"{variant}:unmanifested_records={sorted(keys - expected_keys)}"
            )
        if expected_keys - keys:
            global_errors.append(
                f"{variant}:missing_manifest_records={sorted(expected_keys - keys)}"
            )
        legacy_root = processed_root / f"{variant}_NoNeed"
        if not (processed_root / variant).is_dir() and legacy_root.is_dir():
            global_errors.append(f"{variant}:legacy_directory={legacy_root.name}")

    records: list[dict[str, Any]] = []
    paired_arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for variant in variants:
        for path in files_by_variant[variant]:
            expected_hash = expected_hashes.get((variant, path.stem))
            if expected_hash is None:
                global_errors.append(f"{variant}:{path.stem}:missing_manifest_hash")
            summary, y, indices = validate_file(path, variant, expected_hash)
            records.append(summary)
            paired_arrays.setdefault(path.stem, {})[variant] = (y, indices)

    if len(variants) >= 2:
        reference_variant = variants[0]
        for key, variant_arrays in paired_arrays.items():
            if set(variant_arrays) != set(variants):
                continue
            y_ref, idx_ref = variant_arrays[reference_variant]
            for variant in variants[1:]:
                y, idx = variant_arrays[variant]
                if not np.array_equal(y_ref, y):
                    global_errors.append(f"{key}:{variant}:labels_differ")
                if not np.array_equal(idx_ref, idx):
                    global_errors.append(f"{key}:{variant}:indices_differ")

    subjects = {record["record_key"][:5] for record in records}
    if len(subjects) != expected_subjects:
        global_errors.append(f"subject_count={len(subjects)}")
    file_errors = sum(bool(record["errors"]) for record in records)
    return {
        "schema_version": 2,
        "processed_root": processed_root.as_posix(),
        "preprocess_manifests": [path.as_posix() for path in preprocess_manifests],
        "variants": variants,
        "summary": {
            "files": len(records),
            "records_per_variant": {
                variant: len(files_by_variant[variant]) for variant in variants
            },
            "subjects": len(subjects),
            "files_with_errors": file_errors,
            "global_errors": global_errors,
            "total_epochs_by_variant": {
                variant: sum(r["epochs"] for r in records if r["variant"] == variant)
                for variant in variants
            },
            "valid_epochs_by_variant": {
                variant: sum(
                    r["valid_epochs"] for r in records if r["variant"] == variant
                )
                for variant in variants
            },
            "ignored_epochs_by_variant": {
                variant: sum(
                    r["ignored_epochs"] for r in records if r["variant"] == variant
                )
                for variant in variants
            },
        },
        "records": records,
    }
