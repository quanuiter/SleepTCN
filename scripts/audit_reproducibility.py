"""Verify a canonical processed-artifact manifest without regenerating data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.artifact_audit import audit_artifact_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        help="audit only these variants (useful when retired E5 is not uploaded)",
    )
    args = parser.parse_args()

    report = audit_artifact_manifest(
        args.manifest, args.workspace, variants=args.variants
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.output:
        print(f"Report: {args.output}")
    print("PASS" if report["summary"]["passed"] else "FAIL")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
