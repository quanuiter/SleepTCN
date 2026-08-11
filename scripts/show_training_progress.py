#!/usr/bin/env python3
"""Print checkpoint progress for one full outer-fold run."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    root = (
        args.workspace
        / "runs"
        / "v2"
        / "full"
    )
    pattern = f"E*/fold_{args.fold:02d}/seed_{args.seed}/checkpoints/**/latest.pt"
    checkpoints = sorted(root.glob(pattern))
    if not checkpoints:
        print(f"No checkpoint yet: {root / pattern}")
        return 0

    for path in checkpoints:
        try:
            progress = torch.load(
                path, map_location="cpu", weights_only=False
            )["progress"]
        except Exception as error:
            print(f"{path}: unreadable ({type(error).__name__}: {error})")
            continue
        relative = path.relative_to(root)
        component = "/".join(relative.parts[:2] + relative.parts[4:-1])
        print(
            f"{component:<30} "
            f"epoch={progress['completed_epochs']:>3} "
            f"best={progress['best_epoch']:>3} "
            f"metric={progress['best_metric']:.5f} "
            f"bad={progress['bad_validations']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
