"""Sinh manifest 10-fold cố định theo đối tượng cho Sleep-EDF SC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.dataset import inspect_record
from sleeptcn.splits import (
    LABELS,
    aggregate_subjects,
    deterministic_folds,
    make_outer_runs,
    validate_split_structure,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--variant", default="paper_raw_v1")
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--expected-records", type=int, default=153)
    parser.add_argument("--expected-subjects", type=int, default=78)
    args = parser.parse_args()

    variant_dir = args.processed_root.resolve() / args.variant
    paths = sorted(variant_dir.glob("SC*.npz"))
    if len(paths) != args.expected_records:
        raise ValueError(f"Expected {args.expected_records} NPZ, found {len(paths)}")

    subject_data: dict[str, dict] = {}
    for path in paths:
        info = inspect_record(path, args.variant)
        item = subject_data.setdefault(
            info.subject_id,
            {
                "subject_id": info.subject_id,
                "record_keys": [],
                "epochs": 0,
                "valid_epochs": 0,
                "ignored_epochs": 0,
                "label_counts": {str(label): 0 for label in LABELS},
            },
        )
        item["record_keys"].append(info.record_key)
        item["epochs"] += info.epochs
        item["valid_epochs"] += info.valid_epochs
        item["ignored_epochs"] += info.ignored_epochs
        for label, count in info.label_counts.items():
            item["label_counts"][str(label)] += count

    if len(subject_data) != args.expected_subjects:
        raise ValueError(
            f"Expected {args.expected_subjects} subjects, found {len(subject_data)}"
        )
    for item in subject_data.values():
        item["record_keys"] = sorted(item["record_keys"])

    folds = deterministic_folds(subject_data, args.folds, args.seed)
    fold_items = []
    for index, subject_ids in enumerate(folds):
        fold_items.append(
            {"fold_index": index, **aggregate_subjects(subject_ids, subject_data)}
        )
    outer_runs = make_outer_runs(folds, subject_data)

    manifest = {
        "schema_version": 1,
        "dataset": "sleep-edf-expanded/sleep-cassette/1.0.0",
        "subject_id_rule": "first_5_characters_of_record_key",
        "split_method": "sorted_subjects_then_numpy_default_rng_permutation_then_array_split",
        "seed": args.seed,
        "n_folds": args.folds,
        "validation_policy": "validation_fold=(test_fold+1)%n_folds",
        "compatible_variants": ["paper_raw_v1", "filtered_v2"],
        "source_variant": args.variant,
        "source_preprocess_manifest": str(args.preprocess_manifest.resolve()),
        "source_preprocess_manifest_sha256": sha256_file(args.preprocess_manifest.resolve()),
        "source_git_commit": git_commit(args.workspace.resolve()),
        "numpy_version": np.__version__,
        "summary": {
            "subjects": len(subject_data),
            "records": len(paths),
            "fold_sizes_subjects": [len(fold) for fold in folds],
        },
        "subjects": [subject_data[key] for key in sorted(subject_data)],
        "folds": fold_items,
        "outer_runs": outer_runs,
    }
    errors = validate_split_structure(manifest)
    manifest["validation"] = {"passed": not errors, "errors": errors}
    if errors:
        raise RuntimeError(f"Split validation failed: {errors}")

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with args.csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "subject_id",
                "fold_index",
                "record_keys",
                "epochs",
                "valid_epochs",
                "ignored_epochs",
                "W",
                "N1",
                "N2",
                "N3",
                "REM",
            ]
        )
        subject_to_fold = {
            subject: index for index, fold in enumerate(folds) for subject in fold
        }
        for subject_id in sorted(subject_data):
            item = subject_data[subject_id]
            writer.writerow(
                [
                    subject_id,
                    subject_to_fold[subject_id],
                    "|".join(item["record_keys"]),
                    item["epochs"],
                    item["valid_epochs"],
                    item["ignored_epochs"],
                    *[item["label_counts"][str(label)] for label in range(5)],
                ]
            )
    sidecar = args.json_output.with_suffix(args.json_output.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256_file(args.json_output)}  {args.json_output.name}\n", encoding="ascii"
    )
    print(json.dumps(manifest["summary"], indent=2))
    print(f"JSON: {args.json_output.resolve()}")
    print(f"CSV:  {args.csv_output.resolve()}")
    print(f"SHA:  {sidecar.resolve()}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
