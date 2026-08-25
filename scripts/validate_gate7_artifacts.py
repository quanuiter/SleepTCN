"""CLI wrapper for the deterministic Gate-7 publication validator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.evaluation.publication import (  # noqa: E402
    GATE7_EXPECTED_CLAIM_STATUS as EXPECTED_CLAIM_STATUS,
    GATE7_EXPECTED_COMPARISONS as EXPECTED_COMPARISONS,
    GATE7_EXPECTED_COUNTS as EXPECTED_COUNTS,
    GATE7_EXPECTED_EXPERIMENTS as EXPECTED_EXPERIMENTS,
    GATE7_EXPECTED_OUTPUTS as EXPECTED_OUTPUTS,
    GATE7_REQUIRED_MANUSCRIPT_SECTIONS as REQUIRED_MANUSCRIPT_SECTIONS,
    read_csv,
    read_json,
    validate_gate7_claims as validate_claims,
    validate_gate7_figures as validate_figures,
    validate_gate7_manifest as validate_manifest,
    validate_gate7_manuscript as validate_manuscript,
    validate_gate7_package as validate,
    validate_gate7_tables as validate_tables,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.package_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
