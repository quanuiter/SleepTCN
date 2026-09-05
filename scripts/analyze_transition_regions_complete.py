"""Complete post-hoc transition-region analysis for locked prediction artifacts.

This script is intentionally separate from the manuscript builders.  It does
not modify a report or PDF.  A transition is defined from the reference-label
sequence only; predictions are inspected after the transition masks have been
fixed.

The primary window is the legacy-compatible radius-one neighbourhood: the
epoch immediately before a reference change, the first epoch of the new
stage, and the following epoch.  Stable epochs are at least three epoch
positions away from every reference-label change in the same contiguous
record segment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.hashing import sha256_file
from sleeptcn.io.serialization import read_json
from sleeptcn.metrics import STAGE_NAMES, compute_metrics
from sleeptcn.shhs_transition_analysis import (
    EnsemblePredictions,
    load_ensemble_predictions,
)


EXPERIMENTS = ("E0", "E3")
MAX_DISTANCE = np.iinfo(np.int32).max


def _empty_metrics() -> dict[str, Any]:
    return {
        "support": 0,
        "accuracy": None,
        "macro_f1": None,
        "cohen_kappa": None,
        "per_class_recall": {name: None for name in STAGE_NAMES},
        "per_class_f1": {name: None for name in STAGE_NAMES},
    }


def _metrics(predictions: EnsemblePredictions, selected: np.ndarray) -> dict[str, Any]:
    if selected.dtype != bool or selected.shape != predictions.true_label.shape:
        raise ValueError("invalid transition mask")
    if not selected.any():
        return _empty_metrics()
    result = compute_metrics(
        predictions.true_label[selected], predictions.predicted_label[selected]
    )
    return {
        "support": int(selected.sum()),
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "cohen_kappa": float(result["cohen_kappa"]),
        "per_class_recall": {
            name: float(result["per_class"][name]["recall"])
            for name in STAGE_NAMES
        },
        "per_class_f1": {
            name: float(result["per_class"][name]["f1"])
            for name in STAGE_NAMES
        },
    }


def _assert_paired(left: EnsemblePredictions, right: EnsemblePredictions) -> None:
    for name in ("subject_id", "record_key", "original_epoch_index", "true_label"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"paired predictions differ in {name}")


def _transition_masks(
    reference: EnsemblePredictions, radius: int = 1
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Return raw and persistent transition masks plus the distance profile."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    length = len(reference.true_label)
    distance = np.full(length, MAX_DISTANCE, dtype=np.int32)
    any_mask = np.zeros(length, dtype=bool)
    pair_masks: dict[str, np.ndarray] = {}
    persistent_mask = np.zeros(length, dtype=bool)
    persistent_pair_masks: dict[str, np.ndarray] = {}
    events: list[dict[str, Any]] = []
    persistent_events: list[dict[str, Any]] = []

    for record in np.unique(reference.record_key):
        positions = np.flatnonzero(reference.record_key == record)
        order = np.argsort(reference.original_epoch_index[positions])
        positions = positions[order]
        indices = reference.original_epoch_index[positions]
        labels = reference.true_label[positions]
        starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
        stops = np.r_[starts[1:], len(indices)]
        for start, stop in zip(starts, stops, strict=True):
            local_positions = positions[start:stop]
            local_labels = labels[start:stop]
            if len(local_positions) < 2:
                continue
            grid = np.arange(len(local_positions))
            changes = np.flatnonzero(local_labels[:-1] != local_labels[1:])
            for boundary in changes:
                left = int(boundary)
                right = left + 1
                source = int(local_labels[left])
                target = int(local_labels[right])
                pair_name = f"{STAGE_NAMES[source]}->{STAGE_NAMES[target]}"
                mask_local = np.abs(grid - right) <= radius
                selected = local_positions[mask_local]
                any_mask[selected] = True
                local_distance = np.minimum(
                    np.abs(grid - left), np.abs(grid - right)
                ).astype(np.int32)
                # Keep the distance for every position in the contiguous
                # segment, not only for the primary window.  This makes the
                # distance profile a real decay analysis (0, 1, 2, >=3).
                distance[local_positions] = np.minimum(
                    distance[local_positions], local_distance
                )
                if pair_name not in pair_masks:
                    pair_masks[pair_name] = np.zeros(length, dtype=bool)
                pair_masks[pair_name][selected] = True
                event = {
                    "record_key": str(record),
                    "epoch_before": int(indices[left]),
                    "epoch_after": int(indices[right]),
                    "source": STAGE_NAMES[source],
                    "target": STAGE_NAMES[target],
                    "pair": pair_name,
                }
                events.append(event)

                left_run = 1
                while left - left_run >= 0 and local_labels[left - left_run] == source:
                    left_run += 1
                right_run = 1
                while right + right_run < len(local_labels) and local_labels[right + right_run] == target:
                    right_run += 1
                if left_run >= 3 and right_run >= 3:
                    persistent_mask[selected] = True
                    if pair_name not in persistent_pair_masks:
                        persistent_pair_masks[pair_name] = np.zeros(length, dtype=bool)
                    persistent_pair_masks[pair_name][selected] = True
                    persistent_events.append(event)
    return (
        any_mask,
        distance,
        pair_masks,
        persistent_mask,
        persistent_pair_masks,
        events,
        persistent_events,
    )


def _pair_summary(
    predictions: EnsemblePredictions,
    selected: np.ndarray,
    events: list[dict[str, Any]],
    pair: str,
) -> dict[str, Any]:
    source_name, target_name = pair.split("->")
    source = STAGE_NAMES.index(source_name)
    target = STAGE_NAMES.index(target_name)
    truth = predictions.true_label[selected]
    predicted = predictions.predicted_label[selected]
    result = _metrics(predictions, selected)
    result.update(
        {
            "pair": pair,
            "boundary_events": int(sum(item["pair"] == pair for item in events)),
            "source_support": int(np.sum(truth == source)),
            "target_support": int(np.sum(truth == target)),
            "source_recall": float(np.mean(predicted[truth == source] == source))
            if np.any(truth == source)
            else None,
            "target_recall": float(np.mean(predicted[truth == target] == target))
            if np.any(truth == target)
            else None,
            "source_as_target_rate": float(
                np.mean(predicted[truth == source] == target)
            )
            if np.any(truth == source)
            else None,
            "target_as_source_rate": float(
                np.mean(predicted[truth == target] == source)
            )
            if np.any(truth == target)
            else None,
        }
    )
    return result


def _group_result(
    predictions: dict[str, EnsemblePredictions],
    selected: np.ndarray,
) -> dict[str, Any]:
    return {experiment: _metrics(predictions[experiment], selected) for experiment in EXPERIMENTS}


def _finite_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def analyze(manifest_path: Path, radius: int = 1, stable_distance: int = 3) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("role") != "test":
        raise ValueError("expected a complete locked test run manifest")
    predictions = {
        experiment: load_ensemble_predictions(manifest, experiment)
        for experiment in EXPERIMENTS
    }
    reference = predictions["E0"]
    for experiment in EXPERIMENTS[1:]:
        _assert_paired(reference, predictions[experiment])

    (
        any_mask,
        distance,
        pair_masks,
        persistent_mask,
        persistent_pair_masks,
        events,
        persistent_events,
    ) = _transition_masks(reference, radius=radius)
    stable_mask = distance >= stable_distance
    distances: dict[str, np.ndarray] = {
        "distance_0": distance == 0,
        "distance_1": distance == 1,
        "distance_2": distance == 2,
        f"distance_{stable_distance}_or_more": distance >= stable_distance,
    }

    overall: dict[str, Any] = {
        "all_valid_epochs": int(len(reference.true_label)),
        "raw_reference_changes": int(len(events)),
        "transition_radius": int(radius),
        "transition_neighbourhood_epochs": int(any_mask.sum()),
        "transition_neighbourhood_fraction": float(any_mask.mean()),
        "stable_distance": int(stable_distance),
        "stable_epochs": int(stable_mask.sum()),
        "stable_fraction": float(stable_mask.mean()),
        "persistent_transition_neighbourhood_epochs": int(persistent_mask.sum()),
        "persistent_transition_neighbourhood_fraction": float(persistent_mask.mean()),
        "persistent_reference_changes": int(len(persistent_events)),
        "persistent_transition_neighbourhood": _group_result(predictions, persistent_mask),
        "transition_vs_stable": {
            "transition_neighbourhood": _group_result(predictions, any_mask),
            "stable": _group_result(predictions, stable_mask),
            "transition_minus_stable": {},
        },
        "distance_profile": {},
    }
    for experiment in EXPERIMENTS:
        near = overall["transition_vs_stable"]["transition_neighbourhood"][experiment]
        stable = overall["transition_vs_stable"]["stable"][experiment]
        overall["transition_vs_stable"]["transition_minus_stable"][experiment] = {
            "accuracy": _finite_delta(near["accuracy"], stable["accuracy"]),
            "macro_f1": _finite_delta(near["macro_f1"], stable["macro_f1"]),
            "n3_recall": _finite_delta(
                near["per_class_recall"]["N3"], stable["per_class_recall"]["N3"]
            ),
        }
    for name, selected in distances.items():
        overall["distance_profile"][name] = {
            "support": int(selected.sum()),
            "models": _group_result(predictions, selected),
        }

    pairs: dict[str, Any] = {}
    for pair in sorted(pair_masks):
        pairs[pair] = {
            experiment: _pair_summary(predictions[experiment], pair_masks[pair], events, pair)
            for experiment in EXPERIMENTS
        }

    persistent_pairs: dict[str, Any] = {}
    for pair in sorted(persistent_pair_masks):
        persistent_pairs[pair] = {
            experiment: _pair_summary(
                predictions[experiment], persistent_pair_masks[pair], persistent_events, pair
            )
            for experiment in EXPERIMENTS
        }

    event_counts: dict[str, int] = defaultdict(int)
    for event in events:
        event_counts[event["pair"]] += 1
    return {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": "TRANSITION-REGIONS-T2",
        "source": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "artifact_hashes_verified_by_loader": True,
            "reference_for_transition_definition": "E0 reference labels",
        },
        "experiments": {
            "E0": "15-CNN + BiLSTM",
            "E3": "ResNet-1D + TCN",
        },
        "definition": {
            "transition_window": (
                "for each reference-label change A->B, select the epoch before the "
                "boundary, the first B epoch, and the following epoch"
            ),
            "stable_window": (
                "epochs at least three contiguous epoch positions from every "
                "reference-label change; record gaps break contiguity"
            ),
            "post_hoc_only": True,
            "confidence_not_used_to_define_regions": True,
        },
        "overall": overall,
        "transition_pairs": pairs,
        "transition_pair_event_counts": dict(sorted(event_counts.items())),
        "persistent_transition_pairs": persistent_pairs,
        "persistent_transition_pair_event_counts": {
            pair: int(sum(item["pair"] == pair for item in persistent_events))
            for pair in sorted({item["pair"] for item in persistent_events})
        },
    }


def _write_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for pair, models in result["transition_pairs"].items():
        for experiment, summary in models.items():
            rows.append(
                {
                    "pair": pair,
                    "model": experiment,
                    "boundary_events": summary["boundary_events"],
                    "support": summary["support"],
                    "source_support": summary["source_support"],
                    "target_support": summary["target_support"],
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


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    overall = result["overall"]
    lines = [
        "# Complete transition-region analysis (independent T2 artifact)",
        "",
        "This file is a post-hoc analysis of locked SHHS1 test predictions. It is not a manuscript edit.",
        "Reference labels define transition regions before predictions are inspected.",
        "",
        "## Scope and definition",
        "",
        f"- Valid epochs: **{overall['all_valid_epochs']:,}**.",
        f"- Reference-label changes: **{overall['raw_reference_changes']:,}**.",
        f"- Primary transition window: radius **{overall['transition_radius']}**, with {2 * overall['transition_radius'] + 1} selected positions around each boundary (union of overlapping windows).",
        f"- Transition-neighbourhood support: **{overall['transition_neighbourhood_epochs']:,}** ({overall['transition_neighbourhood_fraction']:.2%}).",
        f"- Stable support: **{overall['stable_epochs']:,}** ({overall['stable_fraction']:.2%}), at least {overall['stable_distance']} contiguous positions from any reference change.",
        f"- Persistent transition support: **{overall['persistent_transition_neighbourhood_epochs']:,}** ({overall['persistent_transition_neighbourhood_fraction']:.2%}) from **{overall['persistent_reference_changes']:,}** changes whose adjacent runs each last at least three epochs.",
        "",
        "## Transition neighbourhood versus stable interiors",
        "",
        "| Model | Region | n | Accuracy | Macro-F1 | N3 recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        for region_name, key in (("Transition", "transition_neighbourhood"), ("Stable", "stable")):
            summary = overall["transition_vs_stable"][key][experiment]
            lines.append(
                f"| {experiment} | {region_name} | {_fmt(summary['support'], 0)} | {_fmt(summary['accuracy'])} | {_fmt(summary['macro_f1'])} | {_fmt(summary['per_class_recall']['N3'])} |"
            )
    lines += [
        "",
        "### Persistent transitions only",
        "",
        "| Model | Region | n | Accuracy | Macro-F1 | N3 recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        summary = overall["persistent_transition_neighbourhood"][experiment]
        lines.append(
            f"| {experiment} | Persistent transition | {_fmt(summary['support'], 0)} | {_fmt(summary['accuracy'])} | {_fmt(summary['macro_f1'])} | {_fmt(summary['per_class_recall']['N3'])} |"
        )
    lines += [
        "",
        "The transition-minus-stable difference is descriptive; it does not by itself establish a new confirmatory hypothesis.",
        "",
        "## All directed reference transitions",
        "",
        "The table reports the union of radius-one windows for each directed pair. `Target recall` is recall for the new stage B inside the local window; the two `as` columns show direct source↔target confusions inside that window.",
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
    lines += [
        "",
        "## Reading rule",
        "",
        "A low score for one pair does not mean every transition is equally difficult. The useful conclusion must be pair-specific and should also be checked against the stable interior baseline. Because the transition mask is defined using the reference sequence, this is an evaluation diagnostic, not a deployable online transition detector.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(result, args.output_csv)
    _write_markdown(result, args.output_md)
    print(json.dumps({"status": result["status"], "output_json": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
