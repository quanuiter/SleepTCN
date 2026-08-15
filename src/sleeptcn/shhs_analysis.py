"""Paired statistical analysis for the locked SHHS zero-shot campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from .gate8_analysis import transition_mask
from .metrics import confusion_matrix_5, metrics_from_confusion
from .statistics import PredictionArrays, assert_paired, holm_adjust


EXPERIMENTS = ("E0", "E3", "E6")
PRIMARY_COMPARISONS = (("E3", "E0"), ("E3", "E6"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ensemble_predictions(
    run_manifest: dict[str, Any], experiment: str
) -> PredictionArrays:
    entries = [
        item
        for item in run_manifest["ensemble_records"]
        if item["experiment"] == experiment
    ]
    if len(entries) != 180:
        raise ValueError(f"{experiment}: expected 180 ensemble records")
    arrays: dict[str, list[np.ndarray]] = {
        "subject_id": [],
        "record_key": [],
        "original_epoch_index": [],
        "true_label": [],
        "predicted_label": [],
    }
    seen: set[str] = set()
    for entry in entries:
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{experiment}: ensemble artifact hash mismatch")
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            if (
                metadata["experiment"] != experiment
                or metadata["subject_id"] != entry["subject_id"]
                or metadata["record_key"] != entry["record_key"]
                or metadata["role"] != "test"
            ):
                raise ValueError(f"{experiment}: inconsistent ensemble metadata")
            record_key = str(entry["record_key"])
            if record_key in seen:
                raise ValueError(f"{experiment}: duplicate record key")
            seen.add(record_key)
            valid = artifact["valid_mask"].astype(bool, copy=False)
            true = artifact["y"][valid].astype(np.int8, copy=False)
            predicted = artifact["prediction"][valid].astype(np.int8, copy=False)
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
    return PredictionArrays(**{key: np.concatenate(value) for key, value in arrays.items()})


def _subject_confusions(predictions: PredictionArrays) -> tuple[np.ndarray, np.ndarray]:
    subjects = np.unique(predictions.subject_id)
    confusions = np.stack(
        [
            confusion_matrix_5(
                predictions.true_label[predictions.subject_id == subject],
                predictions.predicted_label[predictions.subject_id == subject],
            )
            for subject in subjects
        ]
    )
    return subjects, confusions


def _subject_macro_f1(confusions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [metrics_from_confusion(matrix)["macro_f1"] for matrix in confusions],
        dtype=np.float64,
    )


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _absolute_subject_bootstrap(
    values: np.ndarray, samples: np.ndarray
) -> dict[str, Any]:
    distribution = values[samples].mean(axis=1)
    low, high = _percentile_interval(distribution)
    return {
        "estimate": float(values.mean()),
        "ci95_low": low,
        "ci95_high": high,
    }


def _mean_paired_bootstrap(
    left: np.ndarray, right: np.ndarray, samples: np.ndarray
) -> dict[str, Any]:
    difference = left - right
    distribution = difference[samples].mean(axis=1)
    low, high = _percentile_interval(distribution)
    return {
        "observed_difference": float(difference.mean()),
        "ci95_low": low,
        "ci95_high": high,
    }


def _pooled_metric_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    samples: np.ndarray,
    metric: str,
) -> dict[str, Any]:
    observed = (
        metrics_from_confusion(left.sum(axis=0))[metric]
        - metrics_from_confusion(right.sum(axis=0))[metric]
    )
    distribution = np.empty(len(samples), dtype=np.float64)
    for index, selected in enumerate(samples):
        distribution[index] = (
            metrics_from_confusion(left[selected].sum(axis=0))[metric]
            - metrics_from_confusion(right[selected].sum(axis=0))[metric]
        )
    low, high = _percentile_interval(distribution)
    return {
        "observed_difference": float(observed),
        "ci95_low": low,
        "ci95_high": high,
    }


def _wilcoxon_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
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
        "median_subject_difference": float(np.median(difference)),
        "wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "losses": int(np.sum(difference < 0)),
    }


def _transition_confusions(
    predictions: PredictionArrays, selected: np.ndarray, subjects: np.ndarray
) -> np.ndarray:
    matrices = []
    for subject in subjects:
        positions = selected & (predictions.subject_id == subject)
        if positions.any():
            matrix = confusion_matrix_5(
                predictions.true_label[positions],
                predictions.predicted_label[positions],
            )
        else:
            matrix = np.zeros((5, 5), dtype=np.int64)
        matrices.append(matrix)
    return np.stack(matrices)


def analyze_zero_shot(
    *,
    run_manifest_path: Path,
    gate_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    manifest = _load_json(run_manifest_path)
    gate = _load_json(gate_path)
    protocol = _load_json(protocol_path)
    manifest_hash = sha256_file(run_manifest_path)
    protocol_hash = sha256_file(protocol_path)
    if manifest.get("status") != "complete" or manifest.get("role") != "test":
        raise ValueError("test run manifest is not complete")
    if gate.get("status") != "passed" or gate.get("role") != "test":
        raise ValueError("test gate has not passed")
    if gate.get("run_manifest_sha256") != manifest_hash:
        raise ValueError("test gate points to another run manifest")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("run manifest points to another zero-shot protocol")
    if protocol["metrics"]["primary"] != "subject_macro_f1":
        raise ValueError("unexpected primary metric")
    if protocol["metrics"]["primary_comparisons"] != ["E3-E0", "E3-E6"]:
        raise ValueError("unexpected primary comparisons")

    resamples = int(protocol["metrics"]["bootstrap_resamples"])
    seed = int(protocol["metrics"]["bootstrap_seed"])
    predictions = {
        experiment: load_ensemble_predictions(manifest, experiment)
        for experiment in EXPERIMENTS
    }
    reference = predictions["E0"]
    for experiment in EXPERIMENTS[1:]:
        assert_paired(reference, predictions[experiment])
    predictions = {key: value.sorted() for key, value in predictions.items()}

    subject_ids: np.ndarray | None = None
    confusions: dict[str, np.ndarray] = {}
    subject_values: dict[str, np.ndarray] = {}
    transition_confusions: dict[str, np.ndarray] = {}
    transition_reference = transition_mask(reference.sorted(), radius=1)
    for experiment, value in predictions.items():
        subjects, matrices = _subject_confusions(value)
        if subject_ids is None:
            subject_ids = subjects
        elif not np.array_equal(subject_ids, subjects):
            raise ValueError("subject membership differs between experiments")
        confusions[experiment] = matrices
        subject_values[experiment] = _subject_macro_f1(matrices)
        selected = transition_mask(value, radius=1)
        if not np.array_equal(selected, transition_reference):
            raise ValueError("transition masks differ between experiments")
        transition_confusions[experiment] = _transition_confusions(
            value, selected, subjects
        )
    assert subject_ids is not None
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0, len(subject_ids), size=(resamples, len(subject_ids)), dtype=np.int32
    )

    descriptive: dict[str, Any] = {}
    for experiment in EXPERIMENTS:
        pooled = metrics_from_confusion(confusions[experiment].sum(axis=0))
        transition = metrics_from_confusion(
            transition_confusions[experiment].sum(axis=0)
        )
        descriptive[experiment] = {
            "subject_macro_f1": _absolute_subject_bootstrap(
                subject_values[experiment], samples
            ),
            "pooled": pooled,
            "transition_radius_1": {
                "definition": "epochs within one epoch of a contiguous true-label transition",
                "macro_f1": transition["macro_f1"],
                "epochs": transition["n_valid_epochs"],
            },
        }

    comparisons = []
    for proposed, reference_name in PRIMARY_COMPARISONS:
        primary = _mean_paired_bootstrap(
            subject_values[proposed], subject_values[reference_name], samples
        )
        supporting = _wilcoxon_summary(
            subject_values[proposed], subject_values[reference_name]
        )
        comparisons.append(
            {
                "comparison": f"{proposed}-{reference_name}",
                "proposed": proposed,
                "reference": reference_name,
                "primary_mean_subject_macro_f1": primary,
                "supporting_subject_wilcoxon": supporting,
                "secondary_pooled_macro_f1": _pooled_metric_bootstrap(
                    confusions[proposed],
                    confusions[reference_name],
                    samples,
                    "macro_f1",
                ),
                "secondary_pooled_accuracy": _pooled_metric_bootstrap(
                    confusions[proposed],
                    confusions[reference_name],
                    samples,
                    "accuracy",
                ),
                "secondary_pooled_cohen_kappa": _pooled_metric_bootstrap(
                    confusions[proposed],
                    confusions[reference_name],
                    samples,
                    "cohen_kappa",
                ),
                "supporting_transition_radius_1_macro_f1": _pooled_metric_bootstrap(
                    transition_confusions[proposed],
                    transition_confusions[reference_name],
                    samples,
                    "macro_f1",
                ),
            }
        )
    adjusted = holm_adjust(
        [item["supporting_subject_wilcoxon"]["p_value"] for item in comparisons]
    )
    for item, p_value in zip(comparisons, adjusted, strict=True):
        item["supporting_subject_wilcoxon"]["holm_adjusted_p_value"] = p_value
        item["supporting_subject_wilcoxon"]["holm_family_size"] = 2

    return {
        "schema_version": 1,
        "status": "complete",
        "analysis_scope": "locked_shhs1_zero_shot_test",
        "subjects": int(len(subject_ids)),
        "valid_epochs": int(len(reference.true_label)),
        "experiments": list(EXPERIMENTS),
        "primary_estimand": "mean paired subject-level macro_f1 difference",
        "primary_uncertainty": "paired subject-cluster percentile bootstrap 95% CI",
        "supporting_test": "two-sided paired-subject Wilcoxon signed-rank",
        "multiplicity": "Holm correction over E3-E0 and E3-E6 only",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "source_sha256": {
            "run_manifest": manifest_hash,
            "test_gate": sha256_file(gate_path),
            "protocol": protocol_hash,
        },
        "descriptive": descriptive,
        "primary_comparisons": comparisons,
        "claim_boundary": {
            "allowed": [
                "performance on the locked 180-subject SHHS1 sample",
                "paired E3-E0 and E3-E6 differences under the locked zero-shot protocol",
            ],
            "forbidden": [
                "clinical validation",
                "equivalence or non-inferiority",
                "causal attribution to architecture or preprocessing alone",
                "generalization to all SHHS records or other montages",
            ],
        },
    }
