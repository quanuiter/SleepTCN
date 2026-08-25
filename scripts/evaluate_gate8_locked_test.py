"""Mo test Gate 8 dung mot lan sau khi 30 run validation da khoa."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.gate8 import (
    CONDITIONS,
    UNLOCK_CONFIRMATION,
    build_gate8_context,
    clean_git_commit,
    evaluate_locked_test_target,
    load_protocol,
    validate_gate8_run,
)
from sleeptcn.io.serialization import atomic_write_json


def key(condition: str, fold: int) -> str:
    return f"{condition}/fold_{fold:02d}"


def validation_preflight(workspace: Path, seed: int) -> dict[str, Any]:
    commit = clean_git_commit(workspace)
    protocol, protocol_hash = load_protocol(workspace)
    campaign_path = (
        workspace / "runs" / "v2" / "gate8" / "validation_campaign_seed42.json"
    )
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("status") != "validation_complete":
        raise ValueError("Gate 8 validation campaign is not complete")
    if campaign.get("target_count") != 30 or campaign.get("seed") != seed:
        raise ValueError("Gate 8 validation campaign count/seed mismatch")
    if campaign.get("source_git_commit") != commit:
        raise ValueError("Gate 8 validation campaign was produced from another commit")
    if campaign.get("gate8_config_sha256") != protocol_hash:
        raise ValueError("Gate 8 protocol changed after validation training")
    baselines: dict[str, Any] = {}
    for fold in range(10):
        for condition in CONDITIONS:
            target = key(condition, fold)
            entry = campaign["targets"].get(target, {})
            if entry.get("state") != "complete":
                raise ValueError(f"incomplete Gate 8 validation target: {target}")
            context = build_gate8_context(
                workspace, condition, fold, seed, "cpu", num_workers=0, resume=True
            )
            report = validate_gate8_run(workspace, context.run_root)
            if report["status"] != "validation_complete":
                raise ValueError(f"{target}: test is not in the locked pre-test state")
            if entry.get("manifest_sha256") != report["manifest_sha256"]:
                raise ValueError(f"{target}: validation manifest changed")
            observed = {
                "sequence_checkpoint_sha256": report[
                    "sequence_checkpoint_sha256"
                ],
                "replacement_mean_sha256": report["replacement_mean_sha256"],
                "validation_prediction_sha256": report["roles"]["validation"][
                    "prediction_sha256"
                ],
                "validation_metrics_sha256": report["roles"]["validation"][
                    "metrics_sha256"
                ],
            }
            for name, value in observed.items():
                if entry.get(name) != value:
                    raise ValueError(f"{target}: {name} changed after validation campaign")
            baselines[target] = observed
    return {
        "schema_version": 1,
        "status": "prepared",
        "seed": seed,
        "source_git_commit": commit,
        "gate8_config_sha256": protocol_hash,
        "target_count": 30,
        "confirmation_required": protocol["test_gate"]["unlock_confirmation"],
        "baselines": baselines,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    if args.dry_run == args.execute and not args.resume:
        raise ValueError("choose exactly one of --dry-run or --execute")
    workspace = args.workspace.resolve()
    journal_path = workspace / "runs" / "v2" / "gate8" / "test_campaign_seed42.json"
    if args.dry_run:
        if journal_path.exists():
            raise FileExistsError("Gate 8 test journal already exists")
        report = validation_preflight(workspace, args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.confirm != UNLOCK_CONFIRMATION:
        raise ValueError(f"confirmation must equal {UNLOCK_CONFIRMATION!r}")
    if args.resume:
        if not journal_path.is_file():
            raise FileNotFoundError("Gate 8 test journal does not exist")
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal.get("source_git_commit") != clean_git_commit(workspace):
            raise RuntimeError("Git HEAD differs from the Gate 8 test campaign")
    else:
        if journal_path.exists():
            raise FileExistsError("Gate 8 test journal exists; use --resume")
        prepared = validation_preflight(workspace, args.seed)
        journal = {
            **prepared,
            "status": "running",
            "targets": {
                key(condition, fold): {"state": "pending"}
                for fold in range(10)
                for condition in CONDITIONS
            },
        }
        atomic_write_json(journal_path, journal)

    index = 0
    for fold in range(10):
        for condition in CONDITIONS:
            index += 1
            target = key(condition, fold)
            context = build_gate8_context(
                workspace,
                condition,
                fold,
                args.seed,
                args.device,
                num_workers=args.num_workers,
                resume=True,
            )
            if journal["targets"][target].get("state") == "complete":
                report = validate_gate8_run(workspace, context.run_root)
                print(f"[{index:02d}/30] {target}: da kiem dinh, bo qua", flush=True)
            else:
                print(f"[{index:02d}/30] {target}: dang suy luan test", flush=True)
                journal["targets"][target]["state"] = "running"
                atomic_write_json(journal_path, journal)
                report = evaluate_locked_test_target(context)
            baseline = journal["baselines"][target]
            current = {
                "sequence_checkpoint_sha256": report[
                    "sequence_checkpoint_sha256"
                ],
                "replacement_mean_sha256": report["replacement_mean_sha256"],
                "validation_prediction_sha256": report["roles"]["validation"][
                    "prediction_sha256"
                ],
                "validation_metrics_sha256": report["roles"]["validation"][
                    "metrics_sha256"
                ],
            }
            if current != baseline:
                raise ValueError(f"{target}: locked validation artifacts changed")
            journal["targets"][target] = {
                "state": "complete",
                **current,
                "final_manifest_sha256": report["manifest_sha256"],
                "test_prediction_sha256": report["roles"]["test"][
                    "prediction_sha256"
                ],
                "test_metrics_sha256": report["roles"]["test"]["metrics_sha256"],
                "test_records": report["roles"]["test"]["records"],
                "test_valid_epochs": report["roles"]["test"]["valid_epochs"],
            }
            atomic_write_json(journal_path, journal)
            print(f"[{index:02d}/30] {target}: dat", flush=True)
    journal["status"] = "complete"
    atomic_write_json(journal_path, journal)
    print(json.dumps({"status": "complete", "completed_targets": 30}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
