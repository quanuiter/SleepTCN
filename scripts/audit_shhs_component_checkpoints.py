"""Audit the locked E1/E2 checkpoints for the secondary SHHS extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_shhs_zero_shot_checkpoints import read_json, validate_checkpoint
from sleeptcn.io.hashing import combined_sha256, sha256_file
from sleeptcn.features import expected_15cnn_keys


EXPERIMENTS = {
    "E1": {"variant": "paper_raw_v1", "extractor": "cnn15", "sequence": "tcn"},
    "E2": {"variant": "paper_raw_v1", "extractor": "resnet1d", "sequence": "tcn"},
}


def run_root(workspace: Path, experiment: str, fold: int, seed: int) -> Path:
    return (
        workspace
        / "runs"
        / "v2"
        / "full"
        / experiment
        / f"fold_{fold:02d}"
        / f"seed_{seed}"
    )


def validate_manifest(
    workspace: Path, experiment: str, fold: int, seed: int
) -> tuple[Path, dict[str, Any]]:
    root = run_root(workspace, experiment, fold, seed)
    manifest = read_json(root / "run_manifest.json")
    expected = {
        "experiment_id": experiment,
        "outer_fold": fold,
        "seed": seed,
        "status": "complete",
        "smoke": False,
        "data_variant": "paper_raw_v1",
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{experiment}/fold_{fold:02d}: manifest mismatch {mismatches}")
    if manifest.get("git_dirty") is not False:
        raise ValueError(f"{experiment}/fold_{fold:02d}: training worktree was dirty")
    return root, manifest


def audit_e1(workspace: Path, fold: int, seed: int) -> dict[str, Any]:
    root, manifest = validate_manifest(workspace, "E1", fold, seed)
    e0_root, e0_manifest = validate_manifest(workspace, "E0", fold, seed)
    for key in ("config_sha256", "split_sha256"):
        if manifest[key] != e0_manifest[key]:
            raise ValueError(f"E1/fold_{fold:02d}: E0 source {key} differs")

    checkpoints = []
    extractor_hashes: dict[str, str] = {}
    for index, key in enumerate(expected_15cnn_keys()):
        directory = e0_root / "checkpoints" / "cnn15" / key
        entry = validate_checkpoint(
            directory / "best.pt",
            directory / "complete.json",
            experiment="E0",
            fold=fold,
            component_seed=seed + index,
            stage=f"cnn15/{key}",
            variant="paper_raw_v1",
            config_sha256=e0_manifest["config_sha256"],
            split_sha256=e0_manifest["split_sha256"],
            selection_metric="validation_loss",
        )
        entry["checkpoint_owner"] = "E0"
        entry["use"] = "E1_reused_extractor"
        checkpoints.append(entry)
        extractor_hashes[key] = entry["sha256"]
    extractor_sha256 = combined_sha256(extractor_hashes)
    if extractor_sha256 != manifest["extractor_sha256"]:
        raise ValueError(f"E1/fold_{fold:02d}: reused E0 extractor digest differs")

    directory = root / "checkpoints" / "sequence" / "tcn"
    sequence = validate_checkpoint(
        directory / "best.pt",
        directory / "complete.json",
        experiment="E1",
        fold=fold,
        component_seed=seed,
        stage="sequence/tcn",
        variant="paper_raw_v1",
        config_sha256=manifest["config_sha256"],
        split_sha256=manifest["split_sha256"],
        selection_metric="validation_macro_f1",
    )
    sequence["checkpoint_owner"] = "E1"
    sequence["use"] = "E1_sequence"
    checkpoints.append(sequence)
    if sequence["sha256"] != manifest["sequence_checkpoint_sha256"]:
        raise ValueError(f"E1/fold_{fold:02d}: sequence digest differs")
    component_hashes = {
        **{f"E0/{key}": value for key, value in extractor_hashes.items()},
        "E1/sequence/tcn": sequence["sha256"],
    }
    return {
        "experiment": "E1",
        "outer_fold": fold,
        "seed": seed,
        "data_variant": "paper_raw_v1",
        "config_sha256": manifest["config_sha256"],
        "split_sha256": manifest["split_sha256"],
        "extractor_sha256": extractor_sha256,
        "sequence_checkpoint_sha256": sequence["sha256"],
        "fold_checkpoint_set_sha256": combined_sha256(component_hashes),
        "checkpoints": checkpoints,
    }


def audit_e2(workspace: Path, fold: int, seed: int) -> dict[str, Any]:
    root, manifest = validate_manifest(workspace, "E2", fold, seed)
    directory = root / "checkpoints" / "resnet1d"
    extractor = validate_checkpoint(
        directory / "best.pt",
        directory / "complete.json",
        experiment="E2",
        fold=fold,
        component_seed=seed,
        stage="resnet1d",
        variant="paper_raw_v1",
        config_sha256=manifest["config_sha256"],
        split_sha256=manifest["split_sha256"],
        selection_metric="validation_macro_f1",
    )
    directory = root / "checkpoints" / "sequence" / "tcn"
    sequence = validate_checkpoint(
        directory / "best.pt",
        directory / "complete.json",
        experiment="E2",
        fold=fold,
        component_seed=seed,
        stage="sequence/tcn",
        variant="paper_raw_v1",
        config_sha256=manifest["config_sha256"],
        split_sha256=manifest["split_sha256"],
        selection_metric="validation_macro_f1",
    )
    if extractor["sha256"] != manifest["extractor_sha256"]:
        raise ValueError(f"E2/fold_{fold:02d}: extractor digest differs")
    if sequence["sha256"] != manifest["sequence_checkpoint_sha256"]:
        raise ValueError(f"E2/fold_{fold:02d}: sequence digest differs")
    return {
        "experiment": "E2",
        "outer_fold": fold,
        "seed": seed,
        "data_variant": "paper_raw_v1",
        "config_sha256": manifest["config_sha256"],
        "split_sha256": manifest["split_sha256"],
        "extractor_sha256": extractor["sha256"],
        "sequence_checkpoint_sha256": sequence["sha256"],
        "fold_checkpoint_set_sha256": combined_sha256(
            {"E2/resnet1d": extractor["sha256"], "E2/sequence/tcn": sequence["sha256"]}
        ),
        "checkpoints": [extractor, sequence],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw.decode("utf-8"))
    if protocol.get("status") != "locked_before_component_inference":
        raise ValueError("Component extension protocol is not locked")
    if tuple(protocol.get("experiments", {})) != tuple(EXPERIMENTS):
        raise ValueError("Component extension experiment order differs")

    folds = []
    for experiment in EXPERIMENTS:
        for fold in range(10):
            folds.append(
                audit_e1(workspace, fold, args.seed)
                if experiment == "E1"
                else audit_e2(workspace, fold, args.seed)
            )
    checkpoint_entries = [entry for fold in folds for entry in fold["checkpoints"]]
    hash_map = {
        f"{fold['experiment']}/fold_{fold['outer_fold']:02d}/{index:02d}/{entry['stage']}": entry["sha256"]
        for fold in folds
        for index, entry in enumerate(fold["checkpoints"])
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "seed": args.seed,
        "selection_policy": "all_10_folds_no_ranking_probability_ensemble",
        "summary": {
            "experiments": 2,
            "folds_per_experiment": 10,
            "fold_sets": len(folds),
            "best_checkpoints": len(checkpoint_entries),
            "unique_checkpoint_hashes": len(
                {entry["sha256"] for entry in checkpoint_entries}
            ),
            "checkpoint_bytes_referenced": sum(
                entry["bytes"] for entry in checkpoint_entries
            ),
            "campaign_checkpoint_sha256": combined_sha256(hash_map),
        },
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"STATUS: PASSED\nREPORT: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
