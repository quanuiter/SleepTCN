"""Dieu phoi 30 run validation Gate 8 theo fold, co journal va resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.artifacts import sha256_file
from sleeptcn.gate8 import (
    CONDITIONS,
    build_gate8_context,
    clean_git_commit,
    preflight,
    run_validation_condition,
    validate_gate8_run,
)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def target_key(condition: str, fold: int) -> str:
    return f"{condition}/fold_{fold:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    journal_path = workspace / "runs" / "v2" / "gate8" / "validation_campaign_seed42.json"
    if args.resume:
        if not journal_path.is_file():
            raise FileNotFoundError("Gate 8 validation journal does not exist")
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal.get("source_git_commit") != clean_git_commit(workspace):
            raise RuntimeError("Git HEAD differs from the Gate 8 validation campaign")
    else:
        if journal_path.exists():
            raise FileExistsError("Gate 8 journal exists; use --resume")
        prepared = preflight(workspace, seed=args.seed)
        journal = {
            "schema_version": 1,
            "status": "running",
            "seed": args.seed,
            "source_git_commit": prepared["source_git_commit"],
            "gate8_config_sha256": prepared["gate8_config_sha256"],
            "target_count": 30,
            "targets": {
                target_key(condition, fold): {"state": "pending"}
                for fold in range(10)
                for condition in CONDITIONS
            },
        }
        write_json_atomic(journal_path, journal)

    index = 0
    for fold in range(10):
        for condition in CONDITIONS:
            index += 1
            key = target_key(condition, fold)
            context = build_gate8_context(
                workspace,
                condition,
                fold,
                args.seed,
                args.device,
                num_workers=args.num_workers,
                resume=(args.resume or journal["targets"][key]["state"] != "pending"),
            )
            if journal["targets"][key].get("state") == "complete":
                report = validate_gate8_run(workspace, context.run_root)
                print(f"[{index:02d}/30] {key}: da kiem dinh, bo qua", flush=True)
            else:
                print(f"[{index:02d}/30] {key}: dang chay validation", flush=True)
                journal["targets"][key]["state"] = "running"
                write_json_atomic(journal_path, journal)
                run_validation_condition(context)
                report = validate_gate8_run(workspace, context.run_root)
            validation = report["roles"]["validation"]
            journal["targets"][key] = {
                "state": "complete",
                "manifest_sha256": report["manifest_sha256"],
                "sequence_checkpoint_sha256": report[
                    "sequence_checkpoint_sha256"
                ],
                "replacement_mean_sha256": report["replacement_mean_sha256"],
                "validation_prediction_sha256": validation["prediction_sha256"],
                "validation_metrics_sha256": validation["metrics_sha256"],
                "validation_records": validation["records"],
                "validation_valid_epochs": validation["valid_epochs"],
            }
            write_json_atomic(journal_path, journal)
            print(f"[{index:02d}/30] {key}: dat", flush=True)
    journal["status"] = "validation_complete"
    write_json_atomic(journal_path, journal)
    print(json.dumps({
        "status": journal["status"],
        "completed": sum(v["state"] == "complete" for v in journal["targets"].values()),
        "journal_sha256": sha256_file(journal_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
