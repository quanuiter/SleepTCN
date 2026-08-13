"""Phan tich Gate 8 sau khi chien dich test da hoan tat."""

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
from sleeptcn.gate8 import CONDITIONS, load_protocol
from sleeptcn.gate8_analysis import (
    descriptive_views,
    paired_cluster_bootstrap_subset,
    paired_subject_subset_test,
    transition_mask,
)
from sleeptcn.metrics import compute_metrics
from sleeptcn.statistics import PredictionArrays, assert_paired, holm_adjust


EXPECTED_SUBJECTS = 78
EXPECTED_RECORDS = 153
EXPECTED_EPOCHS = 195_469
PAIR_NAMES = {
    "W_N1": (0, 1),
    "N1_N2": (1, 2),
    "N1_REM": (1, 4),
    "N2_N3": (2, 3),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_commit(workspace: Path) -> str:
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
    if commit.returncode or status.returncode or status.stdout.strip():
        raise RuntimeError("official Gate 8 analysis requires a clean Git worktree")
    return commit.stdout.strip()


def load_npz(path: Path) -> PredictionArrays:
    with np.load(path, allow_pickle=False) as npz:
        return PredictionArrays(
            subject_id=npz["subject_id"].copy(),
            record_key=npz["record_key"].copy(),
            original_epoch_index=npz["original_epoch_index"].copy(),
            true_label=npz["true_label"].copy(),
            predicted_label=npz["predicted_label"].copy(),
        )


def load_all_folds(
    workspace: Path,
    condition: str,
    seed: int,
    source_campaign: dict[str, Any],
    gate8_campaign: dict[str, Any],
) -> tuple[PredictionArrays, dict[str, str]]:
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
    hashes: dict[str, str] = {}
    seen_subjects: set[str] = set()
    for fold in range(10):
        if condition == "FULL_CPN":
            path = (
                workspace / "runs" / "v2" / "full" / "E1"
                / f"fold_{fold:02d}" / f"seed_{seed}" / "predictions" / "test.npz"
            )
            expected_hash = source_campaign["targets"][f"E1/fold_{fold:02d}"][
                "test_prediction_sha256"
            ]
        else:
            path = (
                workspace / "runs" / "v2" / "gate8" / "full" / condition
                / f"fold_{fold:02d}" / f"seed_{seed}" / "predictions" / "test.npz"
            )
            expected_hash = gate8_campaign["targets"][f"{condition}/fold_{fold:02d}"][
                "test_prediction_sha256"
            ]
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"{condition}/fold_{fold:02d}: test prediction hash mismatch")
        hashes[f"fold_{fold:02d}"] = observed_hash
        fold_predictions = load_npz(path)
        fold_subjects = set(fold_predictions.subject_id.tolist())
        if seen_subjects & fold_subjects:
            raise ValueError(f"{condition}: subject occurs in multiple test folds")
        seen_subjects.update(fold_subjects)
        for name in parts:
            parts[name].append(getattr(fold_predictions, name))
    predictions = PredictionArrays(
        **{name: np.concatenate(values) for name, values in parts.items()}
    ).sorted()
    if len(np.unique(predictions.subject_id)) != EXPECTED_SUBJECTS:
        raise ValueError(f"{condition}: expected {EXPECTED_SUBJECTS} subjects")
    if len(np.unique(predictions.record_key)) != EXPECTED_RECORDS:
        raise ValueError(f"{condition}: expected {EXPECTED_RECORDS} records")
    if len(predictions.true_label) != EXPECTED_EPOCHS:
        raise ValueError(f"{condition}: expected {EXPECTED_EPOCHS} valid epochs")
    return predictions, hashes


def subset_macro_f1(predictions: PredictionArrays, selected: np.ndarray) -> float:
    return float(
        compute_metrics(
            predictions.true_label[selected], predictions.predicted_label[selected]
        )["macro_f1"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    commit = clean_commit(workspace)
    protocol, protocol_hash = load_protocol(workspace)
    analysis = protocol["analysis"]
    source_campaign_path = workspace / "runs" / "v2" / "test_campaign_seed42.json"
    gate8_campaign_path = workspace / "runs" / "v2" / "gate8" / "test_campaign_seed42.json"
    source_campaign = read_json(source_campaign_path)
    gate8_campaign = read_json(gate8_campaign_path)
    if source_campaign.get("status") != "complete":
        raise ValueError("source E1 test campaign is incomplete")
    if gate8_campaign.get("status") != "complete":
        raise ValueError("Gate 8 test campaign is incomplete")
    if gate8_campaign.get("gate8_config_sha256") != protocol_hash:
        raise ValueError("Gate 8 protocol hash differs from the test campaign")
    predictions: dict[str, PredictionArrays] = {}
    input_hashes: dict[str, Any] = {}
    for condition in ("FULL_CPN", *CONDITIONS):
        predictions[condition], hashes = load_all_folds(
            workspace,
            condition,
            args.seed,
            source_campaign,
            gate8_campaign,
        )
        input_hashes[condition] = {
            "fold_prediction_sha256": hashes,
            "combined_prediction_sha256": combined_sha256(hashes),
        }
    reference = predictions["FULL_CPN"]
    for condition in CONDITIONS:
        assert_paired(reference, predictions[condition])
        for name in ("subject_id", "record_key", "original_epoch_index", "true_label"):
            if not np.array_equal(getattr(reference, name), getattr(predictions[condition], name)):
                raise ValueError(f"{condition}: aligned prediction order differs in {name}")

    radius1 = transition_mask(reference, radius=1)
    radius2 = transition_mask(reference, radius=2)
    if not radius1.any() or not radius2.any():
        raise ValueError("Gate 8 transition selection is empty")
    descriptive = {
        condition: descriptive_views(value, radius1)
        for condition, value in predictions.items()
    }
    for condition, value in predictions.items():
        descriptive[condition]["transition_radius_2"] = {
            "epochs": int(radius2.sum()),
            "macro_f1": subset_macro_f1(value, radius2),
        }
        descriptive[condition]["transition_types"] = {}
        for name, pair in PAIR_NAMES.items():
            selected = transition_mask(reference, radius=1, stage_pair=pair)
            descriptive[condition]["transition_types"][name] = {
                "epochs": int(selected.sum()),
                "macro_f1": subset_macro_f1(value, selected),
            }

    comparison_order = ("C", "CP", "CN")
    comparisons = []
    for condition in comparison_order:
        proposed, comparator = reference, predictions[condition]
        overall_left = compute_metrics(proposed.true_label, proposed.predicted_label)
        overall_right = compute_metrics(comparator.true_label, comparator.predicted_label)
        item = {
            "comparison": f"FULL_CPN-{condition}",
            "role": "primary" if condition == "C" else "key_secondary",
            "direction": "FULL_CPN_minus_ablation",
            "transition_radius_1_macro_f1": {
                "cluster_bootstrap": paired_cluster_bootstrap_subset(
                    proposed,
                    comparator,
                    radius1,
                    resamples=analysis["bootstrap_resamples"],
                    seed=analysis["bootstrap_seed"],
                ),
                "subject_wilcoxon": paired_subject_subset_test(
                    proposed, comparator, radius1
                ),
            },
            "supporting_overall": {
                "macro_f1_difference": (
                    overall_left["macro_f1"] - overall_right["macro_f1"]
                ),
                "accuracy_difference": (
                    overall_left["accuracy"] - overall_right["accuracy"]
                ),
                "n1_f1_difference": (
                    overall_left["per_class"]["N1"]["f1"]
                    - overall_right["per_class"]["N1"]["f1"]
                ),
                "n1_recall_difference": (
                    overall_left["per_class"]["N1"]["recall"]
                    - overall_right["per_class"]["N1"]["recall"]
                ),
            },
        }
        comparisons.append(item)
    adjusted = holm_adjust([
        item["transition_radius_1_macro_f1"]["subject_wilcoxon"]["p_value"]
        for item in comparisons
    ])
    for item, value in zip(comparisons, adjusted, strict=True):
        test = item["transition_radius_1_macro_f1"]["subject_wilcoxon"]
        test["holm_adjusted_p_value"] = value
        test["holm_family_size"] = 3

    interaction = {}
    for name, selected in (("overall", np.ones(EXPECTED_EPOCHS, dtype=bool)), ("transition_radius_1", radius1)):
        scores = {
            condition: subset_macro_f1(value, selected)
            for condition, value in predictions.items()
        }
        interaction[name] = {
            "macro_f1_scores": scores,
            "interaction_full_minus_cp_minus_cn_plus_c": (
                scores["FULL_CPN"] - scores["CP"] - scores["CN"] + scores["C"]
            ),
            "scope": "descriptive_nonadditivity_not_a_percentage_of_information",
        }

    report = {
        "schema_version": 1,
        "status": "complete",
        "gate": "GATE_8_CONTEXT_GROUP_ABLATION",
        "inference_scope": "posthoc_mechanism_analysis_single_training_seed",
        "claim_boundary": protocol["claim_boundary"],
        "seed": args.seed,
        "statistical_unit": "subject",
        "transition_definition": {
            "anchor": analysis["transition_anchor"],
            "primary_radius_epochs": analysis["primary_transition_radius_epochs"],
            "sensitivity_radius_epochs": analysis["sensitivity_transition_radius_epochs"],
            "radius_1_epochs": int(radius1.sum()),
            "radius_2_epochs": int(radius2.sum()),
        },
        "provenance": {
            "analysis_git_commit": commit,
            "gate8_config_sha256": protocol_hash,
            "source_campaign_sha256": sha256_file(source_campaign_path),
            "gate8_campaign_sha256": sha256_file(gate8_campaign_path),
            "analysis_code_sha256": combined_sha256({
                "scripts/analyze_gate8_results.py": sha256_file(
                    workspace / "scripts" / "analyze_gate8_results.py"
                ),
                "src/sleeptcn/gate8_analysis.py": sha256_file(
                    workspace / "src" / "sleeptcn" / "gate8_analysis.py"
                ),
            }),
            "bootstrap_resamples": analysis["bootstrap_resamples"],
            "bootstrap_seed_base": analysis["bootstrap_seed"],
        },
        "input_coverage": {
            "subjects": EXPECTED_SUBJECTS,
            "records": EXPECTED_RECORDS,
            "valid_epochs": EXPECTED_EPOCHS,
            "predictions": input_hashes,
        },
        "descriptive": descriptive,
        "comparisons": comparisons,
        "group_interaction": interaction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
