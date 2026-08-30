"""Run the locked post-hoc paired E3-E2 analysis on SHHS1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sleeptcn.io.hashing import sha256_file
from sleeptcn.shhs_e3_e2_analysis import analyze_e3_e2
from sleeptcn.io.serialization import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-run-manifest", type=Path, required=True)
    parser.add_argument("--e3-test-gate", type=Path, required=True)
    parser.add_argument("--e2-run-manifest", type=Path, required=True)
    parser.add_argument("--e2-test-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_e3_e2(
        e3_manifest_path=args.e3_run_manifest.resolve(),
        e3_gate_path=args.e3_test_gate.resolve(),
        e2_manifest_path=args.e2_run_manifest.resolve(),
        e2_gate_path=args.e2_test_gate.resolve(),
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
                "comparison": report["comparison"],
                "output_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
