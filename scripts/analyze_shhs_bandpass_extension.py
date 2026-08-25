"""Paired analysis for the seed-123 SHHS E4 extension.

This is a supplemental same-cohort analysis.  It does not modify the locked
primary seed-42 SHHS results.  The unit of inference is the subject, not the
individual epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from sleeptcn.io.hashing import sha256_file
from sleeptcn.shhs_analysis import (
    _subject_confusions,
    _subject_macro_f1,
    _pooled_metric_bootstrap,
    _mean_paired_bootstrap,
    load_ensemble_predictions,
)
from sleeptcn.statistics import assert_paired, holm_adjust


EXPERIMENTS = ("E0", "E2", "E3", "E4", "E6")
COMPARISONS = (("E4", "E2"), ("E3", "E4"), ("E3", "E6"), ("E3", "E0"))


def subject_wilcoxon(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    difference = left - right
    if np.all(difference == 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            left,
            right,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "median_difference": float(np.median(difference)),
        "wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "losses": int(np.sum(difference < 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--test-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2031)
    args = parser.parse_args()

    run = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    gate = json.loads(args.test_gate.read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("role") != "test":
        raise ValueError("test run manifest is not complete")
    if gate.get("status") != "passed" or gate.get("role") != "test":
        raise ValueError("test gate has not passed")
    observed = tuple(run.get("metrics", {}))
    if observed != EXPERIMENTS:
        raise ValueError(f"Unexpected experiment order: {observed}")

    predictions = {
        experiment: load_ensemble_predictions(run, experiment).sorted()
        for experiment in EXPERIMENTS
    }
    reference = predictions["E0"]
    for experiment in EXPERIMENTS[1:]:
        assert_paired(reference, predictions[experiment])

    subject_ids = None
    values: dict[str, np.ndarray] = {}
    confusions: dict[str, np.ndarray] = {}
    descriptive: dict[str, dict[str, float | int]] = {}
    for experiment, prediction in predictions.items():
        subjects, matrices = _subject_confusions(prediction)
        if subject_ids is None:
            subject_ids = subjects
        elif not np.array_equal(subject_ids, subjects):
            raise ValueError("subject membership differs between experiments")
        confusions[experiment] = matrices
        values[experiment] = _subject_macro_f1(matrices)
        pooled = matrices.sum(axis=0)
        # Metrics are already present in the run manifest; retain the primary
        # subject-level estimator here and the pooled macro-F1 for convenience.
        descriptive[experiment] = {
            "subject_macro_f1_mean": float(values[experiment].mean()),
            "pooled_macro_f1": float(run["metrics"][experiment]["pooled"]["macro_f1"]),
            "pooled_accuracy": float(run["metrics"][experiment]["pooled"]["accuracy"]),
            "pooled_cohen_kappa": float(run["metrics"][experiment]["pooled"]["cohen_kappa"]),
            "valid_epochs": int(run["metrics"][experiment]["pooled"]["valid_epochs"]),
        }

    assert subject_ids is not None
    rng = np.random.default_rng(args.bootstrap_seed)
    samples = rng.integers(
        0,
        len(subject_ids),
        size=(args.bootstrap_resamples, len(subject_ids)),
        dtype=np.int32,
    )
    comparisons = []
    raw_p = []
    for proposed, reference_name in COMPARISONS:
        primary = _mean_paired_bootstrap(
            values[proposed], values[reference_name], samples
        )
        wilcoxon_summary = subject_wilcoxon(
            values[proposed], values[reference_name]
        )
        raw_p.append(float(wilcoxon_summary["p_value"]))
        comparisons.append(
            {
                "comparison": f"{proposed}-{reference_name}",
                "proposed": proposed,
                "reference": reference_name,
                "mean_subject_macro_f1": primary,
                "subject_wilcoxon": wilcoxon_summary,
                "pooled_macro_f1": _pooled_metric_bootstrap(
                    confusions[proposed], confusions[reference_name], samples, "macro_f1"
                ),
                "pooled_accuracy": _pooled_metric_bootstrap(
                    confusions[proposed], confusions[reference_name], samples, "accuracy"
                ),
                "pooled_cohen_kappa": _pooled_metric_bootstrap(
                    confusions[proposed], confusions[reference_name], samples, "cohen_kappa"
                ),
            }
        )
    adjusted = holm_adjust(raw_p)
    for item, adjusted_p in zip(comparisons, adjusted, strict=True):
        item["subject_wilcoxon"]["holm_adjusted_p_value"] = float(adjusted_p)
        item["subject_wilcoxon"]["holm_family_size"] = len(comparisons)

    report = {
        "schema_version": 1,
        "status": "complete",
        "analysis_scope": "same_cohort_seed123_shhs_bandpass_extension",
        "subjects": int(len(subject_ids)),
        "valid_epochs": int(len(reference.true_label)),
        "experiments": list(EXPERIMENTS),
        "descriptive": descriptive,
        "comparisons": comparisons,
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "unit": "subject",
        },
        "source_sha256": {
            "run_manifest": sha256_file(args.run_manifest),
            "test_gate": sha256_file(args.test_gate),
        },
        "claim_boundary": {
            "allowed": [
                "paired performance differences on this locked 180-subject SHHS1 sample",
                "descriptive evidence about preprocessing contrasts under seed 123",
            ],
            "forbidden": [
                "clinical validation",
                "equivalence or non-inferiority",
                "causal attribution to preprocessing alone",
                "generalization to all SHHS records or other montages",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps({"status": report["status"], "subjects": report["subjects"], "comparisons": report["comparisons"], "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
