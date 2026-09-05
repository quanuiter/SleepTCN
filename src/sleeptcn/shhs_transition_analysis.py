"""Post-hoc SHHS transition-neighbourhood and uncertainty analysis.

This module deliberately treats a transition as a property of the reference
label sequence.  Model confidence is evaluated after the transition groups are
fixed; it is never used to define a reference transition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .io.hashing import sha256_file
from .io.serialization import read_json


EXPERIMENTS = ("E0", "E3", "E6")
N2 = 2
N3 = 3


@dataclass(frozen=True)
class EnsemblePredictions:
    subject_id: np.ndarray
    record_key: np.ndarray
    original_epoch_index: np.ndarray
    true_label: np.ndarray
    predicted_label: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.true_label)
        vectors = (
            self.subject_id,
            self.record_key,
            self.original_epoch_index,
            self.predicted_label,
        )
        if any(value.ndim != 1 or len(value) != length for value in vectors):
            raise ValueError("prediction arrays are not aligned")
        if self.probabilities.shape != (length, 5):
            raise ValueError("probability matrix is not aligned")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("probability matrix contains non-finite values")
        if not np.allclose(
            self.probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("probabilities do not sum to one")
        if not np.array_equal(
            self.predicted_label,
            np.argmax(self.probabilities, axis=1).astype(np.int8),
        ):
            raise ValueError("stored predictions disagree with probabilities")

    def sorted(self) -> "EnsemblePredictions":
        order = np.lexsort(
            (self.original_epoch_index, self.record_key, self.subject_id)
        )
        return EnsemblePredictions(
            subject_id=self.subject_id[order],
            record_key=self.record_key[order],
            original_epoch_index=self.original_epoch_index[order],
            true_label=self.true_label[order],
            predicted_label=self.predicted_label[order],
            probabilities=self.probabilities[order],
        )


@dataclass(frozen=True)
class TransitionAnnotations:
    distance_any_change: np.ndarray
    distance_persistent_n2_n3: np.ndarray
    n2_n3_direction: np.ndarray
    legacy_radius_1: np.ndarray
    raw_change_count: int
    persistent_change_count: int
    persistent_n2_n3_count: int


def _assert_paired(left: EnsemblePredictions, right: EnsemblePredictions) -> None:
    left, right = left.sorted(), right.sorted()
    for name in ("subject_id", "record_key", "original_epoch_index", "true_label"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"paired predictions differ in {name}")


def load_ensemble_predictions(
    manifest: dict[str, Any], experiment: str
) -> EnsemblePredictions:
    entries = [
        item
        for item in manifest["ensemble_records"]
        if item["experiment"] == experiment
    ]
    if not entries:
        raise ValueError(f"{experiment}: no ensemble records")
    arrays: dict[str, list[np.ndarray]] = {
        "subject_id": [],
        "record_key": [],
        "original_epoch_index": [],
        "true_label": [],
        "predicted_label": [],
        "probabilities": [],
    }
    seen: set[str] = set()
    for entry in entries:
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{experiment}: ensemble artifact hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            expected = {
                "experiment": experiment,
                "subject_id": entry["subject_id"],
                "record_key": entry["record_key"],
                "role": "test",
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"{path}: metadata mismatch {mismatches}")
            record_key = str(entry["record_key"])
            if record_key in seen:
                raise ValueError(f"{experiment}: duplicate record key {record_key}")
            seen.add(record_key)
            valid = artifact["valid_mask"].astype(bool, copy=False)
            true = artifact["y"][valid].astype(np.int8, copy=False)
            predicted = artifact["prediction"][valid].astype(np.int8, copy=False)
            probabilities = artifact["probabilities"][valid].astype(
                np.float32, copy=False
            )
            indices = artifact["original_epoch_index"][valid].astype(
                np.int32, copy=False
            )
        count = len(true)
        arrays["subject_id"].append(
            np.full(count, str(entry["subject_id"]), dtype="U32")
        )
        arrays["record_key"].append(np.full(count, record_key, dtype="U64"))
        arrays["original_epoch_index"].append(indices)
        arrays["true_label"].append(true)
        arrays["predicted_label"].append(predicted)
        arrays["probabilities"].append(probabilities)
    return EnsemblePredictions(
        **{key: np.concatenate(value) for key, value in arrays.items()}
    ).sorted()


def _run_length_left(labels: np.ndarray, position: int) -> int:
    stage = labels[position]
    start = position
    while start > 0 and labels[start - 1] == stage:
        start -= 1
    return position - start + 1


def _run_length_right(labels: np.ndarray, position: int) -> int:
    stage = labels[position]
    stop = position
    while stop + 1 < len(labels) and labels[stop + 1] == stage:
        stop += 1
    return stop - position + 1


def annotate_transitions(
    reference: EnsemblePredictions, *, persistence_epochs: int = 3
) -> TransitionAnnotations:
    """Annotate distances to reference-label changes without using predictions.

    Distance zero comprises the two epochs directly adjacent to a boundary.
    A persistent A->B boundary requires at least ``persistence_epochs`` epochs
    in the A run before it and in the B run after it.  The legacy radius-one
    mask exactly reproduces the existing report definition: anchor at the first
    epoch of the new stage, with anchor-1 through anchor+1 selected.
    """

    if persistence_epochs < 1:
        raise ValueError("persistence_epochs must be positive")
    length = len(reference.true_label)
    distance_any = np.full(length, np.iinfo(np.int32).max, dtype=np.int32)
    distance_n23 = np.full(length, np.iinfo(np.int32).max, dtype=np.int32)
    direction = np.zeros(length, dtype=np.int8)
    legacy = np.zeros(length, dtype=bool)
    raw_count = 0
    persistent_count = 0
    n23_count = 0

    for record in np.unique(reference.record_key):
        record_positions = np.flatnonzero(reference.record_key == record)
        order = np.argsort(reference.original_epoch_index[record_positions])
        record_positions = record_positions[order]
        indices = reference.original_epoch_index[record_positions]
        labels = reference.true_label[record_positions]
        segment_starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
        segment_stops = np.r_[segment_starts[1:], len(indices)]
        for start, stop in zip(segment_starts, segment_stops, strict=True):
            local_positions = record_positions[start:stop]
            local_labels = labels[start:stop]
            if len(local_positions) < 2:
                continue
            changes = np.flatnonzero(local_labels[:-1] != local_labels[1:])
            local_grid = np.arange(len(local_positions))
            for left in changes:
                right = left + 1
                raw_count += 1
                symmetric_distance = np.minimum(
                    np.abs(local_grid - left), np.abs(local_grid - right)
                ).astype(np.int32)
                current = distance_any[local_positions]
                distance_any[local_positions] = np.minimum(current, symmetric_distance)
                legacy_local = np.abs(local_grid - right) <= 1
                legacy[local_positions[legacy_local]] = True

                left_run = _run_length_left(local_labels, left)
                right_run = _run_length_right(local_labels, right)
                if left_run < persistence_epochs or right_run < persistence_epochs:
                    continue
                persistent_count += 1
                before, after = int(local_labels[left]), int(local_labels[right])
                if {before, after} != {N2, N3}:
                    continue
                n23_count += 1
                previous = distance_n23[local_positions]
                update = symmetric_distance < previous
                distance_n23[local_positions] = np.minimum(previous, symmetric_distance)
                code = 1 if (before, after) == (N2, N3) else 2
                direction[local_positions[update]] = code

    return TransitionAnnotations(
        distance_any_change=distance_any,
        distance_persistent_n2_n3=distance_n23,
        n2_n3_direction=direction,
        legacy_radius_1=legacy,
        raw_change_count=raw_count,
        persistent_change_count=persistent_count,
        persistent_n2_n3_count=n23_count,
    )


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"n": 0, "mean": None, "q25": None, "median": None, "q75": None}
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
    }


def _group_summary(
    predictions: EnsemblePredictions, selected: np.ndarray
) -> dict[str, Any]:
    if selected.dtype != bool or selected.shape != predictions.true_label.shape:
        raise ValueError("invalid group mask")
    support = int(np.sum(selected))
    predicted = predictions.predicted_label[selected]
    probabilities = predictions.probabilities[selected]
    if support == 0:
        return {"support": 0, "n3_recall": None, "n3_to_n2_rate": None}
    max_confidence = probabilities.max(axis=1)
    return {
        "support": support,
        "predicted_n3": int(np.sum(predicted == N3)),
        "predicted_n2": int(np.sum(predicted == N2)),
        "n3_recall": float(np.mean(predicted == N3)),
        "n3_to_n2_rate": float(np.mean(predicted == N2)),
        "max_confidence": _quantiles(max_confidence),
        "reference_n3_probability": _quantiles(probabilities[:, N3]),
    }


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = target.astype(bool, copy=False)
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(score, method="average")
    auc = (ranks[target].sum() - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    return float(auc)


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def _uncertainty_summary(predictions: EnsemblePredictions) -> dict[str, Any]:
    n3 = predictions.true_label == N3
    predicted = predictions.predicted_label[n3]
    probabilities = predictions.probabilities[n3]
    confidence = probabilities.max(axis=1)
    correct = predicted == N3
    n3_to_n2 = predicted == N2
    thresholds: dict[str, Any] = {}
    for threshold in (0.50, 0.60, 0.70):
        flagged = confidence < threshold
        thresholds[f"{threshold:.2f}"] = {
            "fraction_of_true_n3_flagged": float(flagged.mean()),
            "fraction_of_n3_to_n2_errors_flagged": (
                float(flagged[n3_to_n2].mean()) if n3_to_n2.any() else None
            ),
            "n3_to_n2_fraction_among_flagged": (
                float(n3_to_n2[flagged].mean()) if flagged.any() else None
            ),
        }
    all_confidence = predictions.probabilities.max(axis=1)
    all_correct = predictions.predicted_label == predictions.true_label
    clipped = np.clip(predictions.probabilities, 1e-12, 1.0)
    one_hot = np.eye(5, dtype=np.float64)[predictions.true_label]
    return {
        "true_n3": {
            "correct_n3_max_confidence": _quantiles(confidence[correct]),
            "n3_to_n2_max_confidence": _quantiles(confidence[n3_to_n2]),
            "auc_low_confidence_detects_any_n3_error": _binary_auc(
                ~correct, 1.0 - confidence
            ),
            "auc_low_confidence_detects_n3_to_n2": _binary_auc(
                n3_to_n2, 1.0 - confidence
            ),
            "fixed_threshold_descriptive_only": thresholds,
        },
        "overall_calibration_descriptive": {
            "ece_15_equal_width": _ece(all_confidence, all_correct, bins=15),
            "multiclass_brier": float(
                np.mean(np.sum((predictions.probabilities - one_hot) ** 2, axis=1))
            ),
            "negative_log_likelihood": float(
                -np.mean(np.log(clipped[np.arange(len(clipped)), predictions.true_label]))
            ),
        },
    }


def _bootstrap_rate_difference(
    predictions: EnsemblePredictions,
    near: np.ndarray,
    stable: np.ndarray,
    *,
    outcome_stage: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    subjects = np.unique(predictions.subject_id)
    counts = np.zeros((len(subjects), 4), dtype=np.int64)
    for index, subject in enumerate(subjects):
        own = predictions.subject_id == subject
        near_subject = own & near
        stable_subject = own & stable
        counts[index] = (
            int(near_subject.sum()),
            int(np.sum(near_subject & (predictions.predicted_label == outcome_stage))),
            int(stable_subject.sum()),
            int(np.sum(stable_subject & (predictions.predicted_label == outcome_stage))),
        )
    if counts[:, 0].sum() == 0 or counts[:, 2].sum() == 0:
        raise ValueError("bootstrap comparison group is empty")
    observed = counts[:, 1].sum() / counts[:, 0].sum() - counts[:, 3].sum() / counts[:, 2].sum()
    rng = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        selected = rng.integers(0, len(subjects), size=(stop - start, len(subjects)))
        sampled = counts[selected].sum(axis=1)
        distribution[start:stop] = sampled[:, 1] / sampled[:, 0] - sampled[:, 3] / sampled[:, 2]
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "near_minus_stable": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def analyze_transition_uncertainty(
    *,
    run_manifest_path: Path,
    persistence_epochs: int = 3,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 2031,
) -> tuple[dict[str, Any], dict[str, EnsemblePredictions], TransitionAnnotations]:
    manifest = read_json(run_manifest_path)
    if manifest.get("status") != "complete" or manifest.get("role") != "test":
        raise ValueError("expected a complete locked test run manifest")
    predictions = {
        experiment: load_ensemble_predictions(manifest, experiment)
        for experiment in EXPERIMENTS
    }
    reference = predictions["E0"]
    for experiment in EXPERIMENTS[1:]:
        _assert_paired(reference, predictions[experiment])
    annotation = annotate_transitions(
        reference, persistence_epochs=persistence_epochs
    )
    n3 = reference.true_label == N3
    groups = {
        "all_true_n3": n3,
        "legacy_any_change_radius_1": n3 & annotation.legacy_radius_1,
        "persistent_n2_n3_distance_0": n3
        & (annotation.distance_persistent_n2_n3 == 0),
        "persistent_n2_n3_distance_1": n3
        & (annotation.distance_persistent_n2_n3 == 1),
        "persistent_n2_n3_distance_2": n3
        & (annotation.distance_persistent_n2_n3 == 2),
        "near_persistent_n2_n3_distance_le_1": n3
        & (annotation.distance_persistent_n2_n3 <= 1),
        "stable_n3_distance_ge_3_from_any_change": n3
        & (annotation.distance_any_change >= 3),
    }
    distance_groups = {
        "0": n3 & (annotation.distance_any_change == 0),
        "1": n3 & (annotation.distance_any_change == 1),
        "2": n3 & (annotation.distance_any_change == 2),
        "3_or_more_or_no_change": n3 & (annotation.distance_any_change >= 3),
    }
    direction_groups = {
        "N2_to_N3_near_distance_le_1": groups[
            "near_persistent_n2_n3_distance_le_1"
        ]
        & (annotation.n2_n3_direction == 1),
        "N3_to_N2_near_distance_le_1": groups[
            "near_persistent_n2_n3_distance_le_1"
        ]
        & (annotation.n2_n3_direction == 2),
    }
    model_results: dict[str, Any] = {}
    for model_index, experiment in enumerate(EXPERIMENTS):
        value = predictions[experiment]
        model_results[experiment] = {
            "groups": {
                name: _group_summary(value, selected)
                for name, selected in groups.items()
            },
            "distance_to_any_reference_change": {
                name: _group_summary(value, selected)
                for name, selected in distance_groups.items()
            },
            "persistent_n2_n3_direction": {
                name: _group_summary(value, selected)
                for name, selected in direction_groups.items()
            },
            "uncertainty": _uncertainty_summary(value),
            "near_vs_stable_cluster_bootstrap": {
                "n3_recall_difference": _bootstrap_rate_difference(
                    value,
                    groups["near_persistent_n2_n3_distance_le_1"],
                    groups["stable_n3_distance_ge_3_from_any_change"],
                    outcome_stage=N3,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + model_index * 2,
                ),
                "n3_to_n2_rate_difference": _bootstrap_rate_difference(
                    value,
                    groups["near_persistent_n2_n3_distance_le_1"],
                    groups["stable_n3_distance_ge_3_from_any_change"],
                    outcome_stage=N2,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + model_index * 2 + 1,
                ),
            },
        }

    return (
        {
            "schema_version": 1,
            "status": "complete",
            "analysis_id": "T1",
            "analysis_scope": "exploratory_post_hoc_locked_shhs1_test",
            "subjects": int(len(np.unique(reference.subject_id))),
            "valid_epochs": int(len(reference.true_label)),
            "reference_n3_epochs": int(n3.sum()),
            "source": {
                "run_manifest_filename": run_manifest_path.name,
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "record_artifacts_sha256_verified": True,
            },
            "definitions": {
                "transition_source": "reference expert label sequence only",
                "confidence_role": "evaluated after groups are fixed; never defines a transition",
                "boundary_distance_0": "the two epochs directly adjacent to a reference label change",
                "persistence_epochs_each_side": persistence_epochs,
                "persistence_seconds_each_side": persistence_epochs * 30,
                "near_n2_n3": "distance <= 1 from a persistent reference N2<->N3 boundary",
                "stable_n3": "true N3 at least 3 epochs from any contiguous reference label change",
                "legacy_radius_1": "first epoch of new stage plus one epoch before and after; reproduces the existing report",
            },
            "transition_inventory": {
                "raw_reference_changes": annotation.raw_change_count,
                "persistent_reference_changes": annotation.persistent_change_count,
                "persistent_n2_n3_changes": annotation.persistent_n2_n3_count,
            },
            "models": model_results,
            "claim_boundary": {
                "allowed": [
                    "descriptive concentration of errors near reference scoring boundaries",
                    "descriptive ability of softmax confidence to rank errors on this locked sample",
                    "subject-cluster bootstrap uncertainty for pre-defined near-versus-stable contrasts",
                ],
                "forbidden": [
                    "physiological transition detection",
                    "clinical validation",
                    "a deployment confidence threshold selected from this test sample",
                    "causal attribution to expert disagreement, physiology, architecture, or preprocessing",
                    "using future reference labels during inference",
                ],
            },
        },
        predictions,
        annotation,
    )


__all__ = [
    "EnsemblePredictions",
    "TransitionAnnotations",
    "analyze_transition_uncertainty",
    "annotate_transitions",
    "load_ensemble_predictions",
]
