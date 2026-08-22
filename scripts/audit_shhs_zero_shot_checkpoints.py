"""Audit fold-specific SHHS zero-shot checkpoints before inference.

By default this preserves the locked E0/E3/E6 audit.  ``--experiments`` can
select the same generic audit for a separately locked extension campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from sleeptcn.artifacts import combined_sha256, sha256_file
from sleeptcn.features import expected_15cnn_keys


EXPERIMENTS = {
    "E0": {"variant": "paper_raw_v1", "extractor": "cnn15", "sequence": "bilstm"},
    "E2": {"variant": "paper_raw_v1", "extractor": "resnet1d", "sequence": "tcn"},
    "E3": {"variant": "filtered_v2", "extractor": "resnet1d", "sequence": "tcn"},
    "E4": {"variant": "bandpass_v2", "extractor": "resnet1d", "sequence": "tcn"},
    "E6": {"variant": "filtered_zscore_v2", "extractor": "resnet1d", "sequence": "tcn"},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_checkpoint(
    path: Path,
    marker_path: Path,
    *,
    experiment: str,
    fold: int,
    component_seed: int,
    stage: str,
    variant: str,
    config_sha256: str,
    split_sha256: str,
    selection_metric: str,
) -> dict[str, Any]:
    if path.name != "best.pt" or not path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"Missing locked best checkpoint/marker: {path}")
    digest = sha256_file(path)
    marker = read_json(marker_path)
    expected_marker = {
        "stage": stage,
        "outer_fold": fold,
        "component_seed": component_seed,
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "data_variant": variant,
        "smoke": False,
        "best_checkpoint_sha256": digest,
    }
    marker_mismatch = {
        key: (marker.get(key), value)
        for key, value in expected_marker.items()
        if marker.get(key) != value
    }
    if marker_mismatch:
        raise ValueError(f"{path}: completion marker mismatch {marker_mismatch}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    expected_metadata = {
        "experiment_id": experiment,
        "stage": stage,
        "outer_fold": fold,
        "seed": component_seed,
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "data_variant": variant,
        "selection_metric": selection_metric,
    }
    metadata_mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if metadata_mismatch:
        raise ValueError(f"{path}: checkpoint metadata mismatch {metadata_mismatch}")
    if not isinstance(payload.get("model_state"), dict) or not payload["model_state"]:
        raise ValueError(f"{path}: missing model state")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "stage": stage,
        "component_seed": component_seed,
        "selection_metric": selection_metric,
        "model_class": metadata.get("model_class"),
    }


def audit_fold(
    workspace: Path,
    experiment: str,
    fold: int,
    seed: int,
    spec: dict[str, str],
) -> dict[str, Any]:
    root = workspace / "runs" / "v2" / "full" / experiment / f"fold_{fold:02d}" / f"seed_{seed}"
    manifest = read_json(root / "run_manifest.json")
    expected_manifest = {
        "experiment_id": experiment,
        "outer_fold": fold,
        "seed": seed,
        "status": "complete",
        "smoke": False,
        "data_variant": spec["variant"],
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{experiment}/fold_{fold:02d}: run manifest mismatch {mismatches}")
    config_sha256 = manifest["config_sha256"]
    split_sha256 = manifest["split_sha256"]
    checkpoints = []
    extractor_hashes: dict[str, str] = {}
    if experiment == "E0":
        for index, key in enumerate(expected_15cnn_keys()):
            directory = root / "checkpoints" / "cnn15" / key
            entry = validate_checkpoint(
                directory / "best.pt",
                directory / "complete.json",
                experiment=experiment,
                fold=fold,
                component_seed=seed + index,
                stage=f"cnn15/{key}",
                variant=spec["variant"],
                config_sha256=config_sha256,
                split_sha256=split_sha256,
                selection_metric="validation_loss",
            )
            checkpoints.append(entry)
            extractor_hashes[key] = entry["sha256"]
        observed_extractor_sha256 = combined_sha256(extractor_hashes)
    else:
        directory = root / "checkpoints" / "resnet1d"
        entry = validate_checkpoint(
            directory / "best.pt",
            directory / "complete.json",
            experiment=experiment,
            fold=fold,
            component_seed=seed,
            stage="resnet1d",
            variant=spec["variant"],
            config_sha256=config_sha256,
            split_sha256=split_sha256,
            selection_metric="validation_macro_f1",
        )
        checkpoints.append(entry)
        observed_extractor_sha256 = entry["sha256"]
    if observed_extractor_sha256 != manifest["extractor_sha256"]:
        raise ValueError(f"{experiment}/fold_{fold:02d}: extractor digest differs")

    sequence_dir = root / "checkpoints" / "sequence" / spec["sequence"]
    sequence = validate_checkpoint(
        sequence_dir / "best.pt",
        sequence_dir / "complete.json",
        experiment=experiment,
        fold=fold,
        component_seed=seed,
        stage=f"sequence/{spec['sequence']}",
        variant=spec["variant"],
        config_sha256=config_sha256,
        split_sha256=split_sha256,
        selection_metric="validation_macro_f1",
    )
    checkpoints.append(sequence)
    if sequence["sha256"] != manifest["sequence_checkpoint_sha256"]:
        raise ValueError(f"{experiment}/fold_{fold:02d}: sequence digest differs")
    hashes = {entry["stage"]: entry["sha256"] for entry in checkpoints}
    return {
        "experiment": experiment,
        "outer_fold": fold,
        "seed": seed,
        "data_variant": spec["variant"],
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "extractor_sha256": observed_extractor_sha256,
        "sequence_checkpoint_sha256": sequence["sha256"],
        "fold_checkpoint_set_sha256": combined_sha256(hashes),
        "checkpoints": checkpoints,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=("E0", "E3", "E6"),
        help="Experiments to audit; defaults to the locked primary E0/E3/E6 set.",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw.decode("utf-8"))
    if protocol.get("status") != "locked_before_validation_inference":
        raise ValueError("Zero-shot protocol is not locked")
    selected_experiments = tuple(args.experiments)
    if len(selected_experiments) != len(set(selected_experiments)):
        raise ValueError("Duplicate experiment in --experiments")
    if tuple(protocol["experiments"]) != selected_experiments:
        raise ValueError("Zero-shot experiment order differs from implementation")
    if protocol["checkpoint_policy"]["outer_folds"] != list(range(10)):
        raise ValueError("Zero-shot protocol does not require all ten folds")
    folds = [
        audit_fold(workspace, experiment, fold, args.seed, EXPERIMENTS[experiment])
        for experiment in selected_experiments
        for fold in range(10)
    ]
    checkpoint_entries = [entry for fold in folds for entry in fold["checkpoints"]]
    hash_map = {
        f"{fold['experiment']}/fold_{fold['outer_fold']:02d}/{entry['stage']}": entry["sha256"]
        for fold in folds
        for entry in fold["checkpoints"]
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "seed": args.seed,
        "selection_policy": "all_10_folds_no_ranking_probability_ensemble",
        "summary": {
            "experiments": len(selected_experiments),
            "folds_per_experiment": 10,
            "fold_sets": len(folds),
            "best_checkpoints": len(checkpoint_entries),
            "best_checkpoints_by_experiment": {
                experiment: sum(
                    len(fold["checkpoints"])
                    for fold in folds
                    if fold["experiment"] == experiment
                )
                for experiment in selected_experiments
            },
            "checkpoint_bytes": sum(entry["bytes"] for entry in checkpoint_entries),
            "unique_checkpoint_hashes": len({entry["sha256"] for entry in checkpoint_entries}),
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
