"""Kiem dinh doc lap artifact NPZ SHHS1 va can chinh giua cac bien the."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sleeptcn.preprocessing import sha256_file
from sleeptcn.shhs_preprocessing import RAW_STAGE_MAP, SHHS_VARIANTS
from sleeptcn.io.serialization import read_json


REQUIRED = {
    "x", "y", "raw_stage", "valid_mask", "original_epoch_index",
    "record_key", "subject_id", "role", "preprocess_version",
    "source_edf_sha256", "source_xml_sha256", "source_hashes_verified",
    "selection_manifest_sha256", "technical_audit_sha256", "config_sha256",
    "channel", "montage", "source_sampling_rate_hz", "sampling_rate_hz",
    "epoch_seconds", "samples_per_epoch", "resampling", "resample_up",
    "resample_down", "resample_scope", "trim_anchor_policy",
    "evaluation_window_label_dependent", "filter", "normalization",
}


def scalar(npz: Any, name: str) -> Any:
    value = npz[name]
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return value.item()


def validate_npz(path: Path, manifest_record: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    errors: list[str] = []
    variant = manifest_record["variant"]
    with np.load(path, allow_pickle=False) as npz:
        missing = sorted(REQUIRED - set(npz.files))
        if missing:
            raise ValueError(f"{path.name}: missing fields {missing}")
        x, y = npz["x"], npz["y"]
        raw_stage = npz["raw_stage"]
        valid = npz["valid_mask"]
        indices = npz["original_epoch_index"]
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 3000:
            errors.append("x_contract")
        if y.dtype != np.int8 or y.ndim != 1 or len(y) != len(x):
            errors.append("y_contract")
        if raw_stage.dtype != np.int8 or raw_stage.shape != y.shape:
            errors.append("raw_stage_contract")
        if valid.dtype != np.bool_ or valid.shape != y.shape or not np.array_equal(valid, y >= 0):
            errors.append("valid_mask")
        if indices.dtype != np.int32 or indices.shape != y.shape or (len(indices) and not np.all(np.diff(indices) == 1)):
            errors.append("original_epoch_index")
        if not np.isfinite(x).all():
            errors.append("nonfinite_x")
        expected_y = np.asarray([RAW_STAGE_MAP.get(int(value), -99) for value in raw_stage], dtype=np.int8)
        if not np.array_equal(y, expected_y):
            errors.append("raw_stage_mapping")
        metadata_checks = {
            "record_key": str(scalar(npz, "record_key")) == manifest_record["record_key"] == path.stem,
            "subject_id": str(scalar(npz, "subject_id")) == manifest_record["subject_id"],
            "role": str(scalar(npz, "role")) == manifest_record["role"],
            "variant": str(scalar(npz, "preprocess_version")) == variant,
            "source_hashes": bool(scalar(npz, "source_hashes_verified")),
            "config_hash": str(scalar(npz, "config_sha256")) == manifest["config_sha256"],
            "manifest_hash": str(scalar(npz, "selection_manifest_sha256")) == manifest["selection_manifest_sha256"],
            "audit_hash": str(scalar(npz, "technical_audit_sha256")) == manifest["technical_audit_sha256"],
            "channel": str(scalar(npz, "channel")) == "EEG",
            "montage": str(scalar(npz, "montage")) == "C4-A1",
            "source_rate": float(scalar(npz, "source_sampling_rate_hz")) == 125.0,
            "target_rate": float(scalar(npz, "sampling_rate_hz")) == 100.0,
            "epoch": int(scalar(npz, "epoch_seconds")) == 30,
            "samples": int(scalar(npz, "samples_per_epoch")) == 3000,
            "resampling": str(scalar(npz, "resampling")) == "scipy.signal.resample_poly",
            "ratio": (int(scalar(npz, "resample_up")), int(scalar(npz, "resample_down"))) == (4, 5),
            "resample_scope": str(scalar(npz, "resample_scope")) == "continuous_record_before_epoching_and_variant_processing",
            "trim": str(scalar(npz, "trim_anchor_policy")) == "true_sleep_n1_to_rem",
            "label_dependent_window": bool(scalar(npz, "evaluation_window_label_dependent")),
        }
        errors.extend(f"metadata_{key}" for key, passed in metadata_checks.items() if not passed)
        if variant == "paper_raw_v1":
            if str(scalar(npz, "normalization")) != "none" or "none_after_mandatory" not in str(scalar(npz, "filter")):
                errors.append("paper_raw_variant")
        elif variant == "filtered_v2":
            if float(np.max(np.abs(x))) > 8.00001 or "divide_100.0" not in str(scalar(npz, "normalization")):
                errors.append("filtered_v2_variant")
        elif variant == "filtered_zscore_v2":
            if "record_zscore" not in str(scalar(npz, "normalization")):
                errors.append("filtered_zscore_variant")
            if not np.isfinite(float(scalar(npz, "normalization_mean"))) or not np.isfinite(float(scalar(npz, "normalization_std"))) or float(scalar(npz, "normalization_std")) <= 0:
                errors.append("filtered_zscore_statistics")
        else:
            errors.append("unknown_variant")
        summary = {
            "record_key": path.stem,
            "variant": variant,
            "epochs": int(len(y)),
            "valid_epochs": int(valid.sum()),
            "ignored_epochs": int((y == -1).sum()),
            "errors": errors,
        }
        arrays = {"y": y.copy(), "raw_stage": raw_stage.copy(), "valid": valid.copy(), "indices": indices.copy()}
    if sha256_file(path) != manifest_record["output_sha256"]:
        summary["errors"].append("output_sha256")
    return summary, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.preprocess_manifest)
    if manifest.get("status") != "complete":
        raise ValueError("Preprocess manifest is not complete")
    variants = tuple(manifest["variants"])
    if variants != SHHS_VARIANTS:
        raise ValueError(f"Expected locked variants {SHHS_VARIANTS}, found {variants}")
    expected_subjects = 10 if manifest["scope"] == "pilot" else 200
    expected_roles = {"adaptation": 5, "validation": 5} if manifest["scope"] == "pilot" else {"adaptation": 5, "validation": 15, "test": 180}
    manifest_records = {(item["variant"], item["record_key"]): item for item in manifest["records"]}
    global_errors: list[str] = []
    records = []
    aligned: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    keys_by_variant = {}
    for variant in variants:
        paths = sorted((args.processed_root / variant).glob("shhs1-*.npz"))
        keys_by_variant[variant] = {path.stem for path in paths}
        for path in paths:
            key = (variant, path.stem)
            if key not in manifest_records:
                global_errors.append(f"{variant}/{path.stem}:not_in_manifest")
                continue
            summary, arrays = validate_npz(path, manifest_records[key], manifest)
            records.append(summary)
            aligned.setdefault(path.stem, {})[variant] = arrays
    reference = keys_by_variant[variants[0]]
    if len(reference) != expected_subjects:
        global_errors.append(f"subject_count={len(reference)}")
    for variant, keys in keys_by_variant.items():
        if keys != reference:
            global_errors.append(f"{variant}:record_keys_differ")
    for key, variant_arrays in aligned.items():
        if set(variant_arrays) != set(variants):
            global_errors.append(f"{key}:missing_variant")
            continue
        base = variant_arrays[variants[0]]
        for variant in variants[1:]:
            other = variant_arrays[variant]
            for field in ("y", "raw_stage", "valid", "indices"):
                if not np.array_equal(base[field], other[field]):
                    global_errors.append(f"{key}/{variant}:{field}_differs")
    role_counts = Counter(item["role"] for item in manifest_records.values() if item["variant"] == variants[0])
    if dict(role_counts) != expected_roles:
        global_errors.append(f"role_counts={dict(role_counts)}")
    file_errors = sum(bool(item["errors"]) for item in records)
    report = {
        "schema_version": 1,
        "status": "passed" if not global_errors and not file_errors else "failed",
        "scope": manifest["scope"],
        "preprocess_manifest_sha256": sha256_file(args.preprocess_manifest),
        "summary": {
            "subjects": len(reference),
            "roles": dict(sorted(role_counts.items())),
            "files": len(records),
            "files_with_errors": file_errors,
            "global_errors": global_errors,
            "epochs_per_variant": {
                variant: sum(item["epochs"] for item in records if item["variant"] == variant)
                for variant in variants
            },
            "valid_epochs_per_variant": {
                variant: sum(item["valid_epochs"] for item in records if item["variant"] == variant)
                for variant in variants
            },
            "ignored_epochs_per_variant": {
                variant: sum(item["ignored_epochs"] for item in records if item["variant"] == variant)
                for variant in variants
            },
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"STATUS: {report['status'].upper()}\nREPORT: {args.output.resolve()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
