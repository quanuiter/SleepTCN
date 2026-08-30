"""Check or rewrite processed NPZ files with the canonical serializer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.canonical import DEFAULT_VARIANTS, canonicalize_processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--rewrite", action="store_true")
    args = parser.parse_args()

    def progress(index: int, total: int) -> None:
        if index % 100 == 0 or index == total:
            print(f"checked {index}/{total}", flush=True)

    result = canonicalize_processed(
        args.processed_root,
        args.variants,
        rewrite=args.rewrite,
        progress=progress,
    )
    if result["missing_variants"]:
        for variant, path in result["missing_variants"]:
            print(f"MISSING_VARIANT {variant}: {path}")
        return 1

    drifted = result["drifted"]
    print(f"files={len(result['files'])} drifted={len(drifted)}")
    for path in drifted[:20]:
        print(f"DRIFT {path}")
    if args.check:
        return 1 if drifted else 0

    remaining = result["remaining"]
    if remaining:
        for path in remaining:
            print(f"REWRITE_FAILED {path}")
        return 1
    print("PASS: canonical NPZ serialization verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
