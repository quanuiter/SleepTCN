"""Run the independent transition-region analysis on Sleep-EDF fold outputs.

The source branch is read after a narrow, non-working-tree extraction of the
locked prediction artifacts.  This script does not edit reports or PDFs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from analyze_transition_regions_complete import (  # noqa: E402
    EXPERIMENTS,
    _group_result,
    _metrics,
    _pair_summary,
    _transition_masks,
)
from sleeptcn.io.hashing import combined_sha256, sha256_file  # noqa: E402
from sleeptcn.io.serialization import read_json  # noqa: E402
from sleeptcn.metrics import STAGE_NAMES  # noqa: E402
from sleeptcn.shhs_transition_analysis import EnsemblePredictions  # noqa: E402


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def load_experiment(
    root: Path, campaign_path: Path, experiment: str, seed: int
) -> tuple[EnsemblePredictions, dict[str, Any]]:
    campaign = read_json(campaign_path)
    if campaign.get("status") != "complete" or campaign.get("seed") != seed:
        raise ValueError("campaign is not the expected complete seed campaign")
    arrays: dict[str, list[np.ndarray]] = {
        "subject_id": [],
        "record_key": [],
        "original_epoch_index": [],
        "true_label": [],
        "predicted_label": [],
        "probabilities": [],
    }
    seen_subjects: set[str] = set()
    hashes: dict[str, str] = {}
    for fold in range(10):
        key = f"{experiment}/fold_{fold:02d}"
        path = (
            root
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
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        expected_hash = campaign["targets"][key]["test_prediction_sha256"]
        if observed_hash != expected_hash:
            raise ValueError(f"{key}: prediction hash mismatch")
        hashes[f"fold_{fold:02d}"] = observed_hash
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            expected_metadata = {
                "experiment_id": experiment,
                "outer_fold": fold,
                "seed": seed,
                "role": "test",
                "smoke": False,
            }
            mismatches = {
                name: (metadata.get(name), expected)
                for name, expected in expected_metadata.items()
                if metadata.get(name) != expected
            }
            if mismatches:
                raise ValueError(f"{key}: metadata mismatch {mismatches}")
            subjects = artifact["subject_id"].astype("U32", copy=True)
            overlap = seen_subjects & set(subjects.tolist())
            if overlap:
                raise ValueError(f"{key}: subjects occur in multiple folds")
            seen_subjects.update(subjects.tolist())
            true = artifact["true_label"].astype(np.int8, copy=True)
            predicted = artifact["predicted_label"].astype(np.int8, copy=True)
            logits = artifact["logits"].copy()
            probabilities = _softmax(logits)
            if not np.array_equal(predicted, probabilities.argmax(axis=1)):
                raise ValueError(f"{key}: logits disagree with stored predictions")
            arrays["subject_id"].append(subjects)
            arrays["record_key"].append(artifact["record_key"].astype("U32", copy=True))
            arrays["original_epoch_index"].append(
                artifact["original_epoch_index"].astype(np.int32, copy=True)
            )
            arrays["true_label"].append(true)
            arrays["predicted_label"].append(predicted)
            arrays["probabilities"].append(probabilities)
    predictions = EnsemblePredictions(
        **{name: np.concatenate(values) for name, values in arrays.items()}
    ).sorted()
    return predictions, {
        "records": int(len(np.unique(predictions.record_key))),
        "subjects": int(len(np.unique(predictions.subject_id))),
        "valid_epochs": int(len(predictions.true_label)),
        "fold_prediction_sha256": hashes,
        "combined_prediction_sha256": combined_sha256(hashes),
    }


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _write_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for definition, pairs in (
        ("raw", result["transition_pairs"]),
        (
            "persistent",
            result["persistent_transition_pairs"],
        ),
    ):
        for pair, models in pairs.items():
            for experiment, summary in models.items():
                rows.append(
                    {
                        "definition": definition,
                        "pair": pair,
                        "model": experiment,
                        "boundary_events": summary["boundary_events"],
                        "support": summary["support"],
                        "accuracy": summary["accuracy"],
                        "macro_f1": summary["macro_f1"],
                        "source_recall": summary["source_recall"],
                        "target_recall": summary["target_recall"],
                        "source_as_target_rate": summary["source_as_target_rate"],
                        "target_as_source_rate": summary["target_as_source_rate"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    overall = result["overall"]
    lines = [
        f"# Sleep-EDF transition-region analysis (independent seed-{result['seed']} artifact)",
        "",
        "This is a post-hoc analysis of the locked Sleep-EDF test predictions from the run-in-docker branch. It is not a manuscript edit.",
        "Reference labels define transition regions before predictions are inspected.",
        "",
        "## Scope",
        "",
        f"- Records: **{overall['records']:,}**; subjects: **{overall['subjects']:,}**; valid epochs: **{overall['all_valid_epochs']:,}**.",
        f"- Raw reference-label changes: **{overall['raw_reference_changes']:,}**.",
        f"- Radius-one transition neighbourhood: **{overall['transition_neighbourhood_epochs']:,}** epochs ({overall['transition_neighbourhood_fraction']:.2%}).",
        f"- Persistent changes (at least 3 epochs on each side): **{overall['persistent_reference_changes']:,}**, covering **{overall['persistent_transition_neighbourhood_epochs']:,}** epochs ({overall['persistent_transition_neighbourhood_fraction']:.2%}).",
        f"- Stable interior (distance >= {overall['stable_distance']} from any raw change): **{overall['stable_epochs']:,}** epochs ({overall['stable_fraction']:.2%}).",
        "",
        "## Raw transition neighbourhood versus stable interior",
        "",
        "| Model | Region | n | Accuracy | Macro-F1 | N3 recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        for label, key in (("Transition", "transition_neighbourhood"), ("Stable", "stable")):
            summary = overall["transition_vs_stable"][key][experiment]
            lines.append(
                f"| {experiment} | {label} | {_fmt(summary['support'], 0)} | {_fmt(summary['accuracy'])} | {_fmt(summary['macro_f1'])} | {_fmt(summary['per_class_recall']['N3'])} |"
            )
    lines += [
        "",
        "## Persistent transition neighbourhood",
        "",
        "| Model | n | Accuracy | Macro-F1 | N3 recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        summary = overall["persistent_transition_neighbourhood"][experiment]
        lines.append(
            f"| {experiment} | {_fmt(summary['support'], 0)} | {_fmt(summary['accuracy'])} | {_fmt(summary['macro_f1'])} | {_fmt(summary['per_class_recall']['N3'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The raw table includes every reference-label change, including short runs. The persistent table is a sensitivity analysis, not proof that a boundary is physiologically real. Pair-specific results must be read together with support because rare transitions can produce unstable rates.",
        "",
        "## All raw directed transitions",
        "",
        "| Pair | Boundaries | Model | Window n | Accuracy | Macro-F1 | Source recall | Target recall | Source→target | Target→source |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair, models in result["transition_pairs"].items():
        for experiment in EXPERIMENTS:
            summary = models[experiment]
            lines.append(
                f"| {pair} | {_fmt(summary['boundary_events'], 0)} | {experiment} | {_fmt(summary['support'], 0)} | {_fmt(summary['accuracy'])} | {_fmt(summary['macro_f1'])} | {_fmt(summary['source_recall'])} | {_fmt(summary['target_recall'])} | {_fmt(summary['source_as_target_rate'])} | {_fmt(summary['target_as_source_rate'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--source-ref",
        default="external checkout",
        help="Portable Git ref or artifact identifier recorded in the output",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    # Load once per experiment so the hashes and metadata are captured in the
    # output without doing a second full read of the binary artifacts.
    loaded = {
        experiment: load_experiment(
            args.root.resolve(), args.campaign.resolve(), experiment, args.seed
        )
        for experiment in EXPERIMENTS
    }
    predictions = {experiment: item[0] for experiment, item in loaded.items()}
    inputs = {experiment: item[1] for experiment, item in loaded.items()}
    result = analyze_from_loaded(
        predictions, inputs, args.campaign.resolve(), args.seed, args.source_ref
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(result, args.output_csv)
    _write_markdown(result, args.output_md)
    print(json.dumps({"status": result["status"], "output_json": str(args.output_json)}))
    return 0


def analyze_from_loaded(
    predictions: dict[str, EnsemblePredictions],
    inputs: dict[str, dict[str, Any]],
    campaign_path: Path,
    seed: int,
    source_ref: str,
) -> dict[str, Any]:
    reference = predictions["E0"]
    for experiment in EXPERIMENTS[1:]:
        for name in ("subject_id", "record_key", "original_epoch_index", "true_label"):
            if not np.array_equal(getattr(reference, name), getattr(predictions[experiment], name)):
                raise ValueError(f"paired predictions differ in {name}")
    (
        any_mask,
        distance,
        pair_masks,
        persistent_mask,
        persistent_pair_masks,
        events,
        persistent_events,
    ) = _transition_masks(reference, radius=1)
    stable_mask = distance >= 3
    distances = {
        "distance_0": distance == 0,
        "distance_1": distance == 1,
        "distance_2": distance == 2,
        "distance_3_or_more": distance >= 3,
    }
    overall: dict[str, Any] = {
        "records": int(len(np.unique(reference.record_key))),
        "subjects": int(len(np.unique(reference.subject_id))),
        "all_valid_epochs": int(len(reference.true_label)),
        "raw_reference_changes": int(len(events)),
        "transition_neighbourhood_epochs": int(any_mask.sum()),
        "transition_neighbourhood_fraction": float(any_mask.mean()),
        "stable_distance": 3,
        "stable_epochs": int(stable_mask.sum()),
        "stable_fraction": float(stable_mask.mean()),
        "persistent_reference_changes": int(len(persistent_events)),
        "persistent_transition_neighbourhood_epochs": int(persistent_mask.sum()),
        "persistent_transition_neighbourhood_fraction": float(persistent_mask.mean()),
        "transition_vs_stable": {
            "transition_neighbourhood": _group_result(predictions, any_mask),
            "stable": _group_result(predictions, stable_mask),
            "transition_minus_stable": {},
        },
        "persistent_transition_neighbourhood": _group_result(predictions, persistent_mask),
        "distance_profile": {
            name: {"support": int(selected.sum()), "models": _group_result(predictions, selected)}
            for name, selected in distances.items()
        },
    }
    for experiment in EXPERIMENTS:
        near = overall["transition_vs_stable"]["transition_neighbourhood"][experiment]
        stable = overall["transition_vs_stable"]["stable"][experiment]
        overall["transition_vs_stable"]["transition_minus_stable"][experiment] = {
            "accuracy": near["accuracy"] - stable["accuracy"],
            "macro_f1": near["macro_f1"] - stable["macro_f1"],
            "n3_recall": near["per_class_recall"]["N3"] - stable["per_class_recall"]["N3"],
        }
    pairs = {
        pair: {
            experiment: _pair_summary(predictions[experiment], mask, events, pair)
            for experiment in EXPERIMENTS
        }
        for pair, mask in sorted(pair_masks.items())
    }
    persistent_pairs = {
        pair: {
            experiment: _pair_summary(predictions[experiment], mask, persistent_events, pair)
            for experiment in EXPERIMENTS
        }
        for pair, mask in sorted(persistent_pair_masks.items())
    }
    event_counts: dict[str, int] = {}
    for event in events:
        event_counts[event["pair"]] = event_counts.get(event["pair"], 0) + 1
    persistent_counts: dict[str, int] = {}
    for event in persistent_events:
        persistent_counts[event["pair"]] = persistent_counts.get(event["pair"], 0) + 1
    return {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": "TRANSITION-REGIONS-EDF-T2",
        "seed": seed,
        "source": {
            "source_ref": source_ref,
            "campaign_path": f"runs/v2/{campaign_path.name}",
            "campaign_sha256": sha256_file(campaign_path),
            "inputs": inputs,
            "reference_for_transition_definition": "E0 true labels",
        },
        "experiments": {"E0": "15-CNN + BiLSTM", "E3": "ResNet-1D + TCN"},
        "definition": {
            "transition_window": "radius-one union around each reference-label change",
            "persistent_transition": "at least three contiguous epochs on both sides of the boundary",
            "stable_window": "at least three contiguous epoch positions from every raw reference-label change",
            "post_hoc_only": True,
        },
        "overall": overall,
        "transition_pairs": pairs,
        "persistent_transition_pairs": persistent_pairs,
        "transition_pair_event_counts": dict(sorted(event_counts.items())),
        "persistent_transition_pair_event_counts": dict(sorted(persistent_counts.items())),
    }


if __name__ == "__main__":
    raise SystemExit(main())
