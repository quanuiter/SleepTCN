#!/usr/bin/env python3
"""Preflight or execute the one-time locked-test campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sleeptcn.test_gate import CONFIRMATION_PHRASE, execute_campaign, preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--confirm",
        help=f"required for --execute/--resume: {CONFIRMATION_PHRASE}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        result = preflight(args.workspace, args.seed)
    else:
        result = execute_campaign(
            args.workspace,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            confirmation=args.confirm or "",
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
