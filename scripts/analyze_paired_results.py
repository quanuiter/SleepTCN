"""Run the locked Gate-5 analysis on completed out-of-fold test predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.artifacts import combined_sha256, sha256_file
from sleeptcn.metrics import STAGE_NAMES, compute_metrics
from sleeptcn.statistics import (
    PredictionArrays,
    assert_paired,
    holm_adjust,
    paired_cluster_bootstrap,
    paired_subject_wilcoxon,
)


ACTIVE_EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E6")
SECONDARY_COMPARISONS = (("E4", "E2"),)
EXPECTED_SUBJECTS = 78
EXPECTED_RECORDS = 153
EXPECTED_VALID_EPOCHS = 195_469


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_git_commit(workspace: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode or status.returncode:
        raise RuntimeError("analysis workspace must be a readable Git repository")
    if status.stdout.strip():
        raise RuntimeError("official Gate-5 analysis requires a clean Git worktree")
    return commit.stdout.strip()


def _parse_comparison(value: str) -> tuple[str, str]:
    pieces = value.split("-")
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f"invalid locked comparison: {value!r}")
    proposed, reference = pieces
    if proposed not in ACTIVE_EXPERIMENTS or reference not in ACTIVE_EXPERIMENTS:
        raise ValueError(f"comparison uses an inactive experiment: {value}")
    return proposed, reference


def locked_comparisons(config: dict[str, Any]) -> tuple[
    tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]
]:
    stats = config["statistical_analysis"]
    primary = tuple(_parse_comparison(item) for item in stats["primary_comparisons"])
    expected_primary = (
        ("E1", "E0"),
        ("E2", "E1"),
        ("E3", "E2"),
        ("E3", "E6"),
    )
    if primary != expected_primary:
        raise ValueError(
            f"primary comparisons differ from the frozen protocol: {primary}"
        )
    if any(item in primary for item in SECONDARY_COMPARISONS):
        raise ValueError("a secondary comparison entered the Holm family")
    if stats.get("primary_metric") != "macro_f1":
        raise ValueError("the frozen primary metric must be macro_f1")
    if stats.get("multiple_testing_correction") != "holm":
        raise ValueError("the frozen multiplicity correction must be Holm")
    return primary, SECONDARY_COMPARISONS


def _validate_campaign(campaign: dict[str, Any], seed: int) -> None:
    if campaign.get("status") != "complete":
        raise ValueError("test campaign is not complete")
    if campaign.get("seed") != seed or campaign.get("target_count") != 60:
        raise ValueError("test campaign seed/count differs from the frozen protocol")
    targets = campaign.get("targets", {})
    expected_keys = {
        f"{experiment}/fold_{fold:02d}"
        for experiment in ACTIVE_EXPERIMENTS
        for fold in range(10)
    }
    if set(targets) != expected_keys:
        raise ValueError("test campaign target set is not exactly the locked 60 runs")
    if any(target.get("state") != "complete" for target in targets.values()):
        raise ValueError("at least one test campaign target is incomplete")


def load_experiment(
    workspace: Path,
    experiment: str,
    seed: int,
    campaign: dict[str, Any],
    config: dict[str, Any],
) -> tuple[PredictionArrays, dict[str, Any]]:
    if experiment not in ACTIVE_EXPERIMENTS:
        raise ValueError(f"inactive experiment: {experiment}")
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "subject_id",
            "record_key",
            "original_epoch_index",
            "true_label",
            "predicted_label",
        )
    }
    seen_subjects: set[str] = set()
    prediction_hashes: dict[str, str] = {}
    for fold in range(10):
        key = f"{experiment}/fold_{fold:02d}"
        path = (
            workspace
            / "runs"
            / "v2"
            / "full"
            / experiment
            / f"fold_{fold:02d}"
            / f"seed_{seed}"
            / "predictions"
            / "test.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing completed test prediction: {path}")
        prediction_hash = sha256_file(path)
        expected_hash = campaign["targets"][key]["test_prediction_sha256"]
        if prediction_hash != expected_hash:
            raise ValueError(f"{key}: prediction SHA-256 differs from campaign")
        prediction_hashes[f"fold_{fold:02d}"] = prediction_hash
        with np.load(path, allow_pickle=False) as npz:
            metadata = json.loads(str(npz["metadata_json"].item()))
            expected = {
                "experiment_id": experiment,
                "outer_fold": fold,
                "seed": seed,
                "role": "test",
                "smoke": False,
                "data_variant": config["experiments"][experiment]["data_variant"],
            }
            mismatches = {
                name: (metadata.get(name), value)
                for name, value in expected.items()
                if metadata.get(name) != value
            }
            if mismatches:
                raise ValueError(f"prediction metadata mismatch: {mismatches}")
            fold_subjects = set(npz["subject_id"].tolist())
            overlap = seen_subjects & fold_subjects
            if overlap:
                raise ValueError(f"subjects tested in multiple folds: {sorted(overlap)}")
            seen_subjects.update(fold_subjects)
            for name in parts:
                parts[name].append(npz[name].copy())
    predictions = PredictionArrays(
        **{name: np.concatenate(values) for name, values in parts.items()}
    ).sorted()
    subjects = np.unique(predictions.subject_id)
    records = np.unique(predictions.record_key)
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(f"{experiment}: expected 78 subjects, found {len(subjects)}")
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"{experiment}: expected 153 records, found {len(records)}")
    if len(predictions.true_label) != EXPECTED_VALID_EPOCHS:
        raise ValueError(
            f"{experiment}: expected 195469 epochs, found {len(predictions.true_label)}"
        )
    if not np.isin(predictions.true_label, np.arange(5)).all():
        raise ValueError(f"{experiment}: invalid test truth label")
    if not np.isin(predictions.predicted_label, np.arange(5)).all():
        raise ValueError(f"{experiment}: invalid predicted label")
    if not np.all(
        predictions.subject_id
        == np.asarray([record[:5] for record in predictions.record_key])
    ):
        raise ValueError(f"{experiment}: subject/record identity mismatch")
    return predictions, {
        "subjects": int(len(subjects)),
        "records": int(len(records)),
        "valid_epochs": int(len(predictions.true_label)),
        "fold_prediction_sha256": prediction_hashes,
        "combined_prediction_sha256": combined_sha256(prediction_hashes),
    }


def _metric_view(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "cohen_kappa": metrics["cohen_kappa"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def _descriptive_effect(
    proposed: PredictionArrays, reference: PredictionArrays
) -> dict[str, Any]:
    proposed_metrics = compute_metrics(proposed.true_label, proposed.predicted_label)
    reference_metrics = compute_metrics(reference.true_label, reference.predicted_label)
    return {
        "proposed": _metric_view(proposed_metrics),
        "reference": _metric_view(reference_metrics),
        "difference_proposed_minus_reference": {
            "accuracy": proposed_metrics["accuracy"] - reference_metrics["accuracy"],
            "macro_f1": proposed_metrics["macro_f1"] - reference_metrics["macro_f1"],
            "cohen_kappa": (
                proposed_metrics["cohen_kappa"] - reference_metrics["cohen_kappa"]
            ),
            "per_class_f1": {
                stage: (
                    proposed_metrics["per_class"][stage]["f1"]
                    - reference_metrics["per_class"][stage]["f1"]
                )
                for stage in STAGE_NAMES
            },
        },
    }


def _comparison_result(
    proposed_id: str,
    reference_id: str,
    predictions: dict[str, PredictionArrays],
    *,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    proposed = predictions[proposed_id]
    reference = predictions[reference_id]
    assert_paired(proposed, reference)
    return {
        "comparison": f"{proposed_id}-{reference_id}",
        "proposed": proposed_id,
        "reference": reference_id,
        "descriptive": _descriptive_effect(proposed, reference),
        "cluster_bootstrap_macro_f1": paired_cluster_bootstrap(
            proposed,
            reference,
            resamples=resamples,
            seed=bootstrap_seed,
        ),
        "subject_wilcoxon_macro_f1": paired_subject_wilcoxon(
            proposed, reference
        ),
    }


def build_report(
    workspace: Path,
    *,
    seed: int,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    analysis_git_commit = _clean_git_commit(workspace)
    config_path = workspace / "configs" / "experiments_v2.json"
    split_path = workspace / "data" / "splits" / "sleepedf_sc_10fold_seed42_v2.json"
    campaign_path = workspace / "runs" / "v2" / f"test_campaign_seed{seed}.json"
    config = _read_json(config_path)
    campaign = _read_json(campaign_path)
    _validate_campaign(campaign, seed)
    primary_comparisons, secondary_comparisons = locked_comparisons(config)
    frozen = config["statistical_analysis"]
    if resamples != frozen["bootstrap_resamples"]:
        raise ValueError("bootstrap resamples differ from the frozen protocol")
    if bootstrap_seed != frozen["bootstrap_seed"]:
        raise ValueError("bootstrap seed differs from the frozen protocol")

    predictions: dict[str, PredictionArrays] = {}
    inputs: dict[str, Any] = {}
    for experiment in ACTIVE_EXPERIMENTS:
        predictions[experiment], inputs[experiment] = load_experiment(
            workspace, experiment, seed, campaign, config
        )
    reference = predictions[ACTIVE_EXPERIMENTS[0]]
    for experiment in ACTIVE_EXPERIMENTS[1:]:
        assert_paired(reference, predictions[experiment])

    primary_results = [
        _comparison_result(
            proposed,
            reference_id,
            predictions,
            resamples=resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for proposed, reference_id in primary_comparisons
    ]
    adjusted = holm_adjust(
        [item["subject_wilcoxon_macro_f1"]["p_value"] for item in primary_results]
    )
    for item, value in zip(primary_results, adjusted, strict=True):
        item["subject_wilcoxon_macro_f1"]["holm_adjusted_p_value"] = value
        item["subject_wilcoxon_macro_f1"]["holm_family_size"] = len(primary_results)

    secondary_results = [
        _comparison_result(
            proposed,
            reference_id,
            predictions,
            resamples=resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for proposed, reference_id in secondary_comparisons
    ]
    for item in secondary_results:
        item["inference_scope"] = "prespecified_secondary_mechanism_analysis"
        item["subject_wilcoxon_macro_f1"]["holm_adjusted_p_value"] = None
        item["subject_wilcoxon_macro_f1"]["holm_family_size"] = 0

    return {
        "schema_version": 2,
        "status": "complete",
        "seed": seed,
        "statistical_unit": "subject",
        "primary_effect": "paired_difference_in_pooled_out_of_fold_macro_f1",
        "primary_uncertainty": "paired_subject_cluster_bootstrap_percentile_95ci",
        "supporting_test": "two_sided_paired_subject_wilcoxon",
        "multiplicity": "holm_on_four_primary_comparisons_only",
        "provenance": {
            "analysis_git_commit": analysis_git_commit,
            "analysis_code_sha256": combined_sha256(
                {
                    "scripts/analyze_paired_results.py": sha256_file(
                        workspace / "scripts" / "analyze_paired_results.py"
                    ),
                    "src/sleeptcn/statistics.py": sha256_file(
                        workspace / "src" / "sleeptcn" / "statistics.py"
                    ),
                    "src/sleeptcn/metrics.py": sha256_file(
                        workspace / "src" / "sleeptcn" / "metrics.py"
                    ),
                }
            ),
            "config_sha256": sha256_file(config_path),
            "split_sha256": sha256_file(split_path),
            "campaign_sha256": sha256_file(campaign_path),
            "campaign_source_git_commit": campaign["source_git_commit"],
            "bootstrap_resamples": resamples,
            "bootstrap_seed": bootstrap_seed,
        },
        "input_coverage": inputs,
        "primary_results": primary_results,
        "secondary_results": secondary_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.workspace,
        seed=args.seed,
        resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
