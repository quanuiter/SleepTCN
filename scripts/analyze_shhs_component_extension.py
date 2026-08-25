"""Analyze E1/E2 against the locked E0 SHHS zero-shot reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sleeptcn.shhs_component_analysis import analyze_component_extension
from sleeptcn.io.hashing import sha256_file
from sleeptcn.io.serialization import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-run-manifest", type=Path, required=True)
    parser.add_argument("--component-test-gate", type=Path, required=True)
    parser.add_argument("--reference-run-manifest", type=Path, required=True)
    parser.add_argument("--reference-test-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_component_extension(
        component_manifest_path=args.component_run_manifest.resolve(),
        component_gate_path=args.component_test_gate.resolve(),
        reference_manifest_path=args.reference_run_manifest.resolve(),
        reference_gate_path=args.reference_test_gate.resolve(),
        protocol_path=args.protocol.resolve(),
    )
    atomic_write_json(args.output, report, ensure_ascii=False, sort_keys=False)
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "subjects": report["subjects"],
                "valid_epochs": report["valid_epochs"],
                "comparisons": report["primary_comparisons"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
