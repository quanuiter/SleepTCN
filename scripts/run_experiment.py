"""CLI chay giao thuc E0-E6; mac dinh khong mo khoa tap test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.experiment import build_context, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        choices=["E0", "E1", "E2", "E3", "E4", "E5", "E6"],
        required=True,
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    context = build_context(
        args.workspace,
        args.experiment,
        args.fold,
        args.seed,
        args.device,
        smoke=args.smoke,
        allow_test_evaluation=args.allow_test_evaluation,
        num_workers=args.num_workers,
        resume=args.resume,
    )
    result = run_experiment(context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
