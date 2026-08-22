"""Independently validate fold and ensemble SHHS zero-shot predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleeptcn.artifacts import sha256_file
from sleeptcn.shhs_zero_shot import (
    EXPERIMENT_VARIANTS,
    FOLDS,
    ensemble_probabilities,
    load_prediction_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("role") not in {"validation", "test"}:
        raise ValueError("Zero-shot run manifest is not complete")
    role = run["role"]
    expected_subjects = 15 if role == "validation" else 180
    experiments = tuple(run.get("metrics", {}))
    allowed_experiment_sets = {
        tuple(EXPERIMENT_VARIANTS),
        ("E1", "E2"),
        ("E0", "E2", "E3", "E4", "E6"),
    }
    if experiments not in allowed_experiment_sets:
        raise ValueError(f"Unsupported zero-shot experiment set: {experiments}")
    errors = []
    cross_experiment_alignment = {}
    fold_map = {
        (item["experiment"], int(item["outer_fold"]), item["record_key"]): item
        for item in run["fold_records"]
    }
    ensemble_map = {
        (item["experiment"], item["record_key"]): item
        for item in run["ensemble_records"]
    }
    for experiment in experiments:
        keys = sorted(key for exp, key in ensemble_map if exp == experiment)
        if len(keys) != expected_subjects:
            errors.append(f"{experiment}:ensemble_subject_count={len(keys)}")
        for key in keys:
            parts = []
            reference = None
            for fold in FOLDS:
                item = fold_map.get((experiment, fold, key))
                if item is None:
                    errors.append(f"{experiment}/fold_{fold:02d}/{key}:missing")
                    continue
                path = Path(item["path"])
                if sha256_file(path) != item["sha256"]:
                    errors.append(f"{experiment}/fold_{fold:02d}/{key}:sha256")
                _, probabilities, y, valid, indices = load_prediction_artifact(
                    path,
                    {
                        "protocol_sha256": run["protocol_sha256"],
                        "checkpoint_inventory_sha256": run["checkpoint_inventory_sha256"],
                        "experiment": experiment,
                        "outer_fold": fold,
                        "role": role,
                        "record_key": key,
                    },
                )
                if reference is None:
                    reference = (y, valid, indices)
                elif not all(np.array_equal(a, b) for a, b in zip(reference, (y, valid, indices), strict=True)):
                    errors.append(f"{experiment}/fold_{fold:02d}/{key}:alignment")
                parts.append(probabilities)
            if len(parts) != 10:
                continue
            expected_probability = ensemble_probabilities(parts)
            ensemble_item = ensemble_map[(experiment, key)]
            ensemble_path = Path(ensemble_item["path"])
            if sha256_file(ensemble_path) != ensemble_item["sha256"]:
                errors.append(f"{experiment}/{key}:ensemble_sha256")
            with np.load(ensemble_path, allow_pickle=False) as npz:
                metadata = json.loads(str(npz["metadata_json"].item()))
                observed = npz["probabilities"]
                prediction = npz["prediction"]
                y = npz["y"]
                valid = npz["valid_mask"]
                indices = npz["original_epoch_index"]
            expected_metadata = {
                "artifact_type": "shhs_zero_shot_ensemble_prediction",
                "protocol_sha256": run["protocol_sha256"],
                "checkpoint_inventory_sha256": run["checkpoint_inventory_sha256"],
                "experiment": experiment,
                "role": role,
                "record_key": key,
                "folds": list(FOLDS),
                "aggregation": "arithmetic_mean_probability_float64_accumulator",
            }
            mismatches = {
                name: (metadata.get(name), value)
                for name, value in expected_metadata.items()
                if metadata.get(name) != value
            }
            if mismatches:
                errors.append(f"{experiment}/{key}:ensemble_metadata")
            if not np.array_equal(observed, expected_probability):
                errors.append(f"{experiment}/{key}:ensemble_probability")
            if not np.array_equal(prediction, np.argmax(observed, axis=1).astype(np.int8)):
                errors.append(f"{experiment}/{key}:argmax")
            if reference is not None and not all(np.array_equal(a, b) for a, b in zip(reference, (y, valid, indices), strict=True)):
                errors.append(f"{experiment}/{key}:ensemble_alignment")
            if key not in cross_experiment_alignment:
                cross_experiment_alignment[key] = (y.copy(), valid.copy(), indices.copy())
            elif not all(
                np.array_equal(a, b)
                for a, b in zip(
                    cross_experiment_alignment[key], (y, valid, indices), strict=True
                )
            ):
                errors.append(f"{experiment}/{key}:cross_experiment_alignment")
    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "role": role,
        "protocol_sha256": run["protocol_sha256"],
        "checkpoint_inventory_sha256": run["checkpoint_inventory_sha256"],
        "run_manifest_sha256": sha256_file(args.run_manifest),
        "summary": {
            "subjects": expected_subjects,
            "fold_artifacts": len(run["fold_records"]),
            "ensemble_artifacts": len(run["ensemble_records"]),
            "errors": len(errors),
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"STATUS: {report['status'].upper()}\nREPORT: {args.output.resolve()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
