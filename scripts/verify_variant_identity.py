#!/usr/bin/env python3
"""Verify byte-level equality of two preprocessed NPZ dataset variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape and bytes so equal hashes mean equal array contents."""
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def compare_array(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    same_dtype = left.dtype == right.dtype
    same_shape = left.shape == right.shape
    left_hash = array_sha256(left)
    right_hash = array_sha256(right)
    return {
        "dtype_left": left.dtype.str,
        "dtype_right": right.dtype.str,
        "shape_left": list(left.shape),
        "shape_right": list(right.shape),
        "sha256_left": left_hash,
        "sha256_right": right_hash,
        "bitwise_equal": same_dtype and same_shape and left_hash == right_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--left-variant", required=True)
    parser.add_argument("--right-variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left_files = {path.name for path in args.left_root.glob("*.npz")}
    right_files = {path.name for path in args.right_root.glob("*.npz")}
    common_files = sorted(left_files & right_files)
    missing_left = sorted(right_files - left_files)
    missing_right = sorted(left_files - right_files)

    records: list[dict[str, object]] = []
    equal_record_count = 0
    for filename in common_files:
        with np.load(args.left_root / filename, allow_pickle=False) as left, np.load(
            args.right_root / filename, allow_pickle=False
        ) as right:
            left_fields = sorted(left.files)
            right_fields = sorted(right.files)
            common_fields = sorted(set(left_fields) & set(right_fields))
            comparisons = {
                field: compare_array(left[field], right[field]) for field in common_fields
            }
            unequal_fields = [
                field for field, result in comparisons.items() if not result["bitwise_equal"]
            ]
            # The two variants deliberately encode their own preprocessing identity.
            data_fields = ["x", "y", "valid_mask", "original_epoch_index"]
            data_fields_present = [field for field in data_fields if field in comparisons]
            data_bitwise_equal = (
                len(data_fields_present) == len(data_fields)
                and all(comparisons[field]["bitwise_equal"] for field in data_fields)
            )
            record_equal = left_fields == right_fields and data_bitwise_equal
            equal_record_count += int(record_equal)
            data_hash = hashlib.sha256(
                "".join(
                    f"{field}:{comparisons[field]['sha256_left']}\n"
                    for field in data_fields_present
                ).encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "file": filename,
                    "data_fields_bitwise_equal": data_bitwise_equal,
                    "all_fields_bitwise_equal": not unequal_fields,
                    "unequal_fields": unequal_fields,
                    "data_sha256": data_hash,
                }
            )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "left": {"variant": args.left_variant, "root": str(args.left_root)},
        "right": {"variant": args.right_variant, "root": str(args.right_root)},
        "summary": {
            "left_file_count": len(left_files),
            "right_file_count": len(right_files),
            "common_file_count": len(common_files),
            "missing_from_left": missing_left,
            "missing_from_right": missing_right,
            "records_with_bitwise_equal_data": equal_record_count,
            "all_records_data_bitwise_equal": (
                not missing_left
                and not missing_right
                and equal_record_count == len(common_files)
            ),
        },
        "comparison_scope": {
            "data_fields": ["x", "y", "valid_mask", "original_epoch_index"],
            "note": (
                "Variant-identifying metadata may differ by design. Every shared field is "
                "reported, while scientific data equivalence is determined from data_fields."
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]
    print(f"Compared {summary['common_file_count']} common NPZ files.")
    print(f"Records with bitwise-identical scientific data: {equal_record_count}.")
    print(f"All records identical: {summary['all_records_data_bitwise_equal']}.")
    print(f"Report: {args.output}")
    return 0 if summary["all_records_data_bitwise_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
