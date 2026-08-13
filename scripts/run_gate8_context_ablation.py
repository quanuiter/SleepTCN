"""Chay mot condition/fold Gate 8; mac dinh chi train va validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.gate8 import CONDITIONS, build_gate8_context, run_validation_condition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    context = build_gate8_context(
        args.workspace,
        args.condition,
        args.fold,
        args.seed,
        args.device,
        num_workers=args.num_workers,
        smoke=args.smoke,
        resume=args.resume,
    )
    result = run_validation_condition(context)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
