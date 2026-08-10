"""Kiểm định độc lập manifest chia fold và tệp NPZ tương ứng."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.dataset import inspect_record
from sleeptcn.splits import validate_split_structure


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    errors = validate_split_structure(manifest)
    expected_records = {
        record_key: item["subject_id"]
        for item in manifest["subjects"]
        for record_key in item["record_keys"]
    }
    variant_results = {}
    expected_subjects = {
        item["subject_id"]: item for item in manifest["subjects"]
    }
    for variant in manifest["compatible_variants"]:
        files = sorted((args.processed_root / variant).glob("SC*.npz"))
        actual_keys = {path.stem for path in files}
        if actual_keys != set(expected_records):
            errors.append(f"{variant}:npz_record_keys_mismatch")
        mismatched_subjects = 0
        aggregates = {
            subject_id: {
                "record_keys": [],
                "epochs": 0,
                "valid_epochs": 0,
                "ignored_epochs": 0,
                "label_counts": Counter(),
            }
            for subject_id in expected_subjects
        }
        for path in files:
            info = inspect_record(path, variant)
            if expected_records.get(info.record_key) != info.subject_id:
                mismatched_subjects += 1
                continue
            item = aggregates[info.subject_id]
            item["record_keys"].append(info.record_key)
            item["epochs"] += info.epochs
            item["valid_epochs"] += info.valid_epochs
            item["ignored_epochs"] += info.ignored_epochs
            item["label_counts"].update(info.label_counts)
        aggregate_mismatches = 0
        for subject_id, expected in expected_subjects.items():
            actual = aggregates[subject_id]
            if sorted(actual["record_keys"]) != expected["record_keys"]:
                aggregate_mismatches += 1
                continue
            if any(
                actual[field] != expected[field]
                for field in ("epochs", "valid_epochs", "ignored_epochs")
            ):
                aggregate_mismatches += 1
                continue
            if any(
                actual["label_counts"][label] != int(expected["label_counts"][str(label)])
                for label in (-1, 0, 1, 2, 3, 4)
            ):
                aggregate_mismatches += 1
        if mismatched_subjects:
            errors.append(f"{variant}:subject_metadata_mismatch")
        if aggregate_mismatches:
            errors.append(f"{variant}:subject_aggregate_mismatch")
        variant_results[variant] = {
            "files": len(files),
            "mismatched_subjects": mismatched_subjects,
            "aggregate_mismatches": aggregate_mismatches,
        }

    report = {
        "schema_version": 1,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "summary": {
            "passed": not errors,
            "errors": sorted(set(errors)),
            "subjects": len(manifest["subjects"]),
            "folds": len(manifest["folds"]),
            "outer_runs": len(manifest["outer_runs"]),
            "variants": variant_results,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output.resolve()}")
    print("PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
