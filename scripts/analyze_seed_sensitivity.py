#!/usr/bin/env python3
"""Compare completed Gate-5 reports without pooling seed-specific inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.seed_sensitivity import compare_seed_reports


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-report", type=Path, required=True)
    parser.add_argument("--seed123-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = {42: args.seed42_report.resolve(), 123: args.seed123_report.resolve()}
    reports = {
        seed: json.loads(path.read_text(encoding="utf-8"))
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
