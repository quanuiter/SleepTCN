"""Kiểm định độc lập NPZ do preprocess_sleepedf.py tạo ra."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.processed_validation import validate_processed_dataset
from sleeptcn.io.canonical import DEFAULT_VARIANTS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument(
        "--preprocess-manifest",
        type=Path,
        action="append",
        required=True,
        help=(
            "May be repeated for source manifests and an optional "
            "processed_artifact_snapshot manifest."
        ),
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        help=(
            "Canonical processed-artifact snapshot. If omitted, the standard "
            "data/manifests snapshot next to --processed-root is used when present."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Processed variants to validate (defaults to all locked variants).",
    )
    parser.add_argument("--expected-records", type=int, default=153)
    parser.add_argument("--expected-subjects", type=int, default=78)
    args = parser.parse_args()
    manifests = list(args.preprocess_manifest)
    artifact_manifest = args.artifact_manifest
    if artifact_manifest is None:
        candidate = (
            args.processed_root.resolve().parent
            / "manifests"
            / "processed_artifact_manifest_v2.json"
        )
        if candidate.is_file():
            artifact_manifest = candidate
    if artifact_manifest is not None:
        manifests.append(artifact_manifest)
    report = validate_processed_dataset(
        args.processed_root,
        manifests,
        args.variants,
        expected_records=args.expected_records,
        expected_subjects=args.expected_subjects,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output.resolve()}")
    passed = (
        report["summary"]["files_with_errors"] == 0
        and not report["summary"]["global_errors"]
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
