"""Run one validation-only ResNet-1D tuning candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.resnet_tuning import run_resnet_tuning


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one ResNet candidate without constructing a test loader."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = run_resnet_tuning(
        args.workspace,
        args.search_config,
        args.candidate,
        args.fold,
        args.seed,
        args.device,
        args.output_root,
        num_workers=args.num_workers,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
