#!/usr/bin/env python3
"""Analyze the locked SHHS1 zero-shot test campaign."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from sleeptcn.shhs_analysis import analyze_zero_shot, sha256_file


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--test-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_zero_shot(
        run_manifest_path=args.run_manifest,
        gate_path=args.test_gate,
        protocol_path=args.protocol,
    )
    atomic_json(args.output, report)
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
                "primary_comparisons": report["primary_comparisons"],
                "analysis_sha256": digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
