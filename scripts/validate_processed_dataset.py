"""Kiểm định độc lập NPZ do preprocess_sleepedf.py tạo ra."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sleeptcn.preprocessing import sha256_file


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", default=["paper_raw_v1", "filtered_v2"]
    )
    parser.add_argument("--expected-records", type=int, default=153)
    parser.add_argument("--expected-subjects", type=int, default=78)
    args = parser.parse_args()

    process_manifest = json.loads(
        args.preprocess_manifest.read_text(encoding="utf-8")
    )
    expected_hashes = {
        (record["variant"], record["record_key"]): record["output_sha256"]
        for record in process_manifest["records"]
    }

    files_by_variant = {
        variant: sorted((args.processed_root / variant).glob("SC*.npz"))
        for variant in args.variants
    }
    all_keys = {variant: {path.stem for path in files} for variant, files in files_by_variant.items()}
    reference_keys = all_keys[args.variants[0]]
    global_errors: list[str] = []
    for variant, keys in all_keys.items():
        if len(keys) != args.expected_records:
            global_errors.append(f"{variant}:record_count={len(keys)}")
        if keys != reference_keys:
            global_errors.append(f"{variant}:record_keys_differ")

    records: list[dict[str, Any]] = []
    paired_arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for variant in args.variants:
        for path in files_by_variant[variant]:
            expected_hash = expected_hashes.get((variant, path.stem))
            if expected_hash is None:
                global_errors.append(f"{variant}:{path.stem}:missing_manifest_hash")
            summary, y, indices = validate_file(path, variant, expected_hash)
            records.append(summary)
            paired_arrays.setdefault(path.stem, {})[variant] = (y, indices)

    if len(args.variants) >= 2:
        reference_variant = args.variants[0]
        for key, variants in paired_arrays.items():
            if set(variants) != set(args.variants):
                continue
            y_ref, idx_ref = variants[reference_variant]
            for variant in args.variants[1:]:
                y, idx = variants[variant]
                if not np.array_equal(y_ref, y):
                    global_errors.append(f"{key}:{variant}:labels_differ")
                if not np.array_equal(idx_ref, idx):
                    global_errors.append(f"{key}:{variant}:indices_differ")

    subjects = {record["record_key"][:5] for record in records}
    if len(subjects) != args.expected_subjects:
        global_errors.append(f"subject_count={len(subjects)}")
    file_errors = sum(bool(record["errors"]) for record in records)
    report = {
        "schema_version": 1,
        "processed_root": str(args.processed_root.resolve()),
        "preprocess_manifest": str(args.preprocess_manifest.resolve()),
        "variants": args.variants,
        "summary": {
            "files": len(records),
            "records_per_variant": {
                variant: len(files_by_variant[variant]) for variant in args.variants
            },
            "subjects": len(subjects),
            "files_with_errors": file_errors,
            "global_errors": global_errors,
            "total_epochs_by_variant": {
                variant: sum(r["epochs"] for r in records if r["variant"] == variant)
                for variant in args.variants
            },
            "valid_epochs_by_variant": {
                variant: sum(r["valid_epochs"] for r in records if r["variant"] == variant)
                for variant in args.variants
            },
            "ignored_epochs_by_variant": {
                variant: sum(r["ignored_epochs"] for r in records if r["variant"] == variant)
                for variant in args.variants
            },
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output.resolve()}")
    passed = file_errors == 0 and not global_errors
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
