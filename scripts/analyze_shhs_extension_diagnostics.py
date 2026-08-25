"""Diagnostic metrics from completed SHHS seed-123 ensemble predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleeptcn.io.hashing import sha256_file
from sleeptcn.gate8_analysis import transition_mask
from sleeptcn.metrics import compute_metrics
from sleeptcn.shhs_analysis import load_ensemble_predictions


EXPERIMENTS = ("E0", "E2", "E3", "E4", "E6")
PAIRS = (("E4", "E2"), ("E3", "E4"), ("E3", "E6"), ("E3", "E0"))
STAGES = ("W", "N1", "N2", "N3", "REM")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--test-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    gate = json.loads(args.test_gate.read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("role") != "test":
        raise ValueError("test run is not complete")
    if gate.get("status") != "passed":
        raise ValueError("test gate has not passed")
    if tuple(run.get("metrics", {})) != EXPERIMENTS:
        raise ValueError("unexpected experiment order")

    predictions = {
        experiment: load_ensemble_predictions(run, experiment).sorted()
        for experiment in EXPERIMENTS
    }
    descriptions: dict[str, object] = {}
    subject_macro: dict[str, dict[str, float]] = {}
    subject_n1: dict[str, dict[str, float]] = {}
    for experiment, prediction in predictions.items():
        metrics = compute_metrics(prediction.true_label, prediction.predicted_label)
        transition = transition_mask(prediction, radius=1)
        transition_metrics = compute_metrics(
            prediction.true_label[transition], prediction.predicted_label[transition]
        )
        descriptions[experiment] = {
            "overall": metrics,
            "transition_radius_1": {
                "definition": "epochs within one epoch of a contiguous true-label transition",
                "metrics": transition_metrics,
            },
        }
        subjects = np.unique(prediction.subject_id)
        macro_values: dict[str, float] = {}
        n1_values: dict[str, float] = {}
        for subject in subjects:
            selected = prediction.subject_id == subject
            subject_metrics = compute_metrics(
                prediction.true_label[selected], prediction.predicted_label[selected]
            )
            macro_values[str(subject)] = float(subject_metrics["macro_f1"])
            n1_values[str(subject)] = float(subject_metrics["per_class"]["N1"]["f1"])
        subject_macro[experiment] = macro_values
        subject_n1[experiment] = n1_values

    subject_distributions = {}
    for experiment in EXPERIMENTS:
        values = np.asarray(list(subject_macro[experiment].values()), dtype=np.float64)
        n1 = np.asarray(list(subject_n1[experiment].values()), dtype=np.float64)
        subject_distributions[experiment] = {
            "macro_f1": {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            },
            "n1_f1": {
                "mean": float(n1.mean()),
                "median": float(np.median(n1)),
                "q25": float(np.quantile(n1, 0.25)),
                "q75": float(np.quantile(n1, 0.75)),
                "min": float(n1.min()),
                "max": float(n1.max()),
            },
        }

    paired_deltas = {}
    for proposed, reference in PAIRS:
        common = sorted(set(subject_macro[proposed]) & set(subject_macro[reference]))
        delta = np.asarray(
            [subject_macro[proposed][key] - subject_macro[reference][key] for key in common],
            dtype=np.float64,
        )
        paired_deltas[f"{proposed}-{reference}"] = {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
            "q25": float(np.quantile(delta, 0.25)),
            "q75": float(np.quantile(delta, 0.75)),
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta < 0)),
        }

    report = {
        "schema_version": 1,
        "status": "complete",
        "analysis_scope": "same_cohort_seed123_shhs_extension_diagnostics",
        "subjects": 180,
        "experiments": list(EXPERIMENTS),
        "descriptive": descriptions,
        "subject_distributions": subject_distributions,
        "paired_subject_macro_f1_deltas": paired_deltas,
        "source_sha256": {
            "run_manifest": sha256_file(args.run_manifest),
            "test_gate": sha256_file(args.test_gate),
        },
        "interpretation_boundary": {
            "purpose": "diagnostic explanation of completed zero-shot predictions",
            "not_a_new_training_or_confirmation_experiment": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps({"status": "complete", "output_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
