"""Build portable, content-addressed manifests for processed NPZ artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .hashing import sha256_file
from .paths import portable_path
from .serialization import read_json


def build_artifact_manifest(
    processed_root: Path,
    preprocess_manifests: Iterable[Path],
    *,
    workspace: Path,
    raw_manifest: Path | None = None,
    expected_records: int = 153,
    environment_lock: Path | None = None,
    accept_legacy_container_hash_drift: bool = False,
) -> dict[str, Any]:
    """Build the v2 processed-artifact snapshot without writing it."""

    repo_root = workspace.resolve()
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    source_manifest_hashes: dict[str, str] = {}
    configs: list[dict[str, Any]] = []
    dataset = None
    for manifest_path in preprocess_manifests:
        manifest = read_json(manifest_path)
        dataset = dataset or manifest.get("dataset")
        configs.append(manifest.get("config", {}))
        source_manifest_hashes[portable_path(manifest_path, repo_root)] = sha256_file(
            manifest_path
        )
        for record in manifest.get("records", []):
            key = (record["variant"], record["record_key"])
            if key in expected and expected[key]["output_sha256"] != record[
                "output_sha256"
            ]:
                raise ValueError(f"conflicting source hashes for {key}")
            expected[key] = record

    variants = sorted({variant for variant, _ in expected})
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    legacy_hash_drift = Counter()
    for variant in variants:
        root = processed_root / variant
        legacy_root = processed_root / f"{variant}_NoNeed"
        if not root.is_dir():
            if legacy_root.is_dir():
                errors.append(f"legacy_variant_directory:{legacy_root.name}")
            else:
                errors.append(f"missing_variant_directory:{variant}")
            continue
        actual_files = {path.stem: path for path in root.glob("*.npz")}
        expected_keys = {
            record_key for variant_name, record_key in expected if variant_name == variant
        }
        missing = sorted(expected_keys - set(actual_files))
        extra = sorted(set(actual_files) - expected_keys)
        errors.extend(f"{variant}:missing:{key}" for key in missing)
        errors.extend(f"{variant}:extra:{key}" for key in extra)
        if len(actual_files) != expected_records:
            errors.append(f"{variant}:record_count={len(actual_files)}")
        for record_key in sorted(expected_keys & set(actual_files)):
            path = actual_files[record_key]
            actual_hash = sha256_file(path)
            source = expected[(variant, record_key)]
            if actual_hash != source["output_sha256"]:
                legacy_hash_drift[variant] += 1
                if not accept_legacy_container_hash_drift:
                    errors.append(f"{variant}:{record_key}:sha256")
            record = {
                "record_key": record_key,
                "subject_id": source.get("subject_id", record_key[:5]),
                "variant": variant,
                "output_path": portable_path(path, repo_root),
                "output_sha256": actual_hash,
                "size_bytes": path.stat().st_size,
            }
            if actual_hash != source["output_sha256"]:
                record["legacy_output_sha256"] = source["output_sha256"]
            for field in (
                "epochs",
                "samples_per_epoch",
                "label_counts",
                "trim_start_epoch",
                "trim_stop_epoch_exclusive",
                "annotation_epochs_truncated",
                "clip_fraction",
                "x_min",
                "x_max",
                "x_mean",
                "x_std",
            ):
                if field in source:
                    record[field] = source[field]
            records.append(record)

    if configs and any(config != configs[0] for config in configs[1:]):
        errors.append("preprocess_config_conflict")
    lock_info = None
    if environment_lock is not None:
        if not environment_lock.is_file():
            errors.append(f"missing_environment_lock:{environment_lock}")
        else:
            lock_info = {
                "path": portable_path(environment_lock, repo_root),
                "sha256": sha256_file(environment_lock),
            }

    raw_manifest_info = None
    if raw_manifest is not None:
        if not raw_manifest.is_file():
            errors.append(f"missing_raw_manifest:{raw_manifest}")
        else:
            raw_manifest_info = {
                "path": portable_path(raw_manifest, repo_root),
                "sha256": sha256_file(raw_manifest),
            }

    records.sort(key=lambda item: (item["variant"], item["record_key"]))
    counts = Counter(record["variant"] for record in records)
    return {
        "schema_version": 2,
        "manifest_kind": "processed_artifact_snapshot",
        "dataset": dataset or "sleep-edf-expanded/sleep-cassette/1.0.0",
        "processed_root": portable_path(processed_root, repo_root),
        "variants": variants,
        "artifact_serializer": "sleeptcn_deterministic_npz_v1",
        "preprocess_config": configs[0] if configs else {},
        "source_preprocess_manifests": source_manifest_hashes,
        "source_raw_manifest": raw_manifest_info,
        "environment_lock": lock_info,
        "summary": {
            "files": len(records),
            "records_per_variant": dict(sorted(counts.items())),
            "subjects": len({record["record_key"][:5] for record in records}),
            "legacy_hash_drift_by_variant": dict(sorted(legacy_hash_drift.items())),
            "legacy_hash_drift_reason": (
                "pre-canonical NPZ ZIP metadata; scientific arrays were retained"
                if legacy_hash_drift
                else None
            ),
            "errors": errors,
        },
        "records": records,
    }


def write_artifact_manifest(path: Path, report: dict[str, Any]) -> None:
    """Write a manifest preserving the existing JSON formatting contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
