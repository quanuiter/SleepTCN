"""Hien thi epoch hien tai cua chien dich Gate 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.workspace.resolve() / "runs" / "v2" / "gate8" / "full"
    rows = []
    for fold in range(10):
        for condition in ("CP", "CN", "C"):
            run = root / condition / f"fold_{fold:02d}" / f"seed_{args.seed}"
            manifest_path = run / "run_manifest.json"
            latest = run / "checkpoints" / "sequence" / "tcn" / "latest.pt"
            status = "pending"
            detail = ""
            if manifest_path.is_file():
                status = json.loads(manifest_path.read_text(encoding="utf-8"))["status"]
            if latest.is_file() and status == "training":
                payload = torch.load(latest, map_location="cpu", weights_only=False)
                progress = payload["progress"]
                detail = (
                    f"epoch={progress['completed_epochs']} "
                    f"best_epoch={progress['best_epoch'] + 1} "
                    f"best={progress['best_metric']:.6f} "
                    f"bad={progress['bad_validations']}"
                )
            if status != "pending" or detail:
                rows.append(
                    f"fold={fold:02d} condition={condition:<2} status={status:<19} {detail}"
                )
    if rows:
        print("\n".join(rows))
    else:
        print("Gate 8 has not started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
