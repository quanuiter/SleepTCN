#!/usr/bin/env python3
"""Compare completed Gate-5 reports without pooling seed-specific inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.seed_sensitivity import compare_seed_reports
from sleeptcn.io.hashing import sha256_file
from sleeptcn.io.serialization import read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-report", type=Path, required=True)
    parser.add_argument("--seed123-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = {42: args.seed42_report.resolve(), 123: args.seed123_report.resolve()}
    reports = {
        seed: read_json(path)
        for seed, path in inputs.items()
    }
    result = compare_seed_reports(reports)
    result["provenance"] = {
        "input_sha256": {
            str(seed): sha256_file(path) for seed, path in inputs.items()
        },
        "analysis_code_sha256": {
            "script": sha256_file(Path(__file__).resolve()),
            "module": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/sleeptcn/seed_sensitivity.py"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
