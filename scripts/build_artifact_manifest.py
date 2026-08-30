"""Build a portable, content-addressed manifest for processed NPZ artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.manifest_builder import (
    build_artifact_manifest,
    write_artifact_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preprocess-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--expected-records", type=int, default=153)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument(
        "--accept-legacy-container-hash-drift",
        action="store_true",
        help="accept hashes from pre-canonical NPZ containers",
    )
    args = parser.parse_args()
    report = build_artifact_manifest(
        args.processed_root,
        args.preprocess_manifest,
        workspace=Path.cwd(),
        raw_manifest=args.raw_manifest,
        expected_records=args.expected_records,
        environment_lock=args.environment_lock,
        accept_legacy_container_hash_drift=args.accept_legacy_container_hash_drift,
    )
    write_artifact_manifest(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Manifest: {args.output}")
    print("PASS" if not report["summary"]["errors"] else "FAIL")
    return 0 if not report["summary"]["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
