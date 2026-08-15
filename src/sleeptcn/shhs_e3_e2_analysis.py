"""Post-hoc paired E3-E2 analysis on the fixed SHHS1 test cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .gate8_analysis import transition_mask
from .metrics import metrics_from_confusion
from .shhs_analysis import (
    _mean_paired_bootstrap,
    _subject_confusions,
    _subject_macro_f1,
    _transition_confusions,
    _wilcoxon_summary,
    load_ensemble_predictions,
    sha256_file,
)
from .statistics import assert_paired


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def supporting_evidence_decision(ci95_low: float, p_value: float) -> str:
    if ci95_low > 0.0 and p_value < 0.05:
        return "supporting_evidence_on_opened_shhs_sample"
    return "not_supported_on_opened_shhs_sample"


def _validate_source(
    *,
    manifest_path: Path,
    gate_path: Path,
    expected_manifest_hash: str,
    expected_gate_hash: str,
) -> tuple[dict[str, Any], str, str]:
    manifest = _load_json(manifest_path)
    gate = _load_json(gate_path)
    manifest_hash = sha256_file(manifest_path)
    gate_hash = sha256_file(gate_path)
    if manifest_hash != expected_manifest_hash:
        raise ValueError("source run manifest hash differs from locked protocol")
    if gate_hash != expected_gate_hash:
        raise ValueError("source test gate hash differs from locked protocol")
    if manifest.get("status") != "complete" or manifest.get("role") != "test":
        raise ValueError("source test run is not complete")
    if gate.get("status") != "passed" or gate.get("role") != "test":
        raise ValueError("source test gate has not passed")
    if gate.get("run_manifest_sha256") != manifest_hash:
        raise ValueError("source test gate points to another run manifest")
    return manifest, manifest_hash, gate_hash


def _pooled_metrics_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    samples: np.ndarray,
    *,
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    left_observed = metrics_from_confusion(left.sum(axis=0))
    right_observed = metrics_from_confusion(right.sum(axis=0))
    distributions = {
        metric: np.empty(len(samples), dtype=np.float64) for metric in metrics
    }
    for index, selected in enumerate(samples):
        left_metrics = metrics_from_confusion(left[selected].sum(axis=0))
        right_metrics = metrics_from_confusion(right[selected].sum(axis=0))
        for metric in metrics:
            distributions[metric][index] = float(left_metrics[metric]) - float(
                right_metrics[metric]
            )
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        low, high = np.quantile(distributions[metric], [0.025, 0.975])
        result[metric] = {
            "observed_difference": float(left_observed[metric])
            - float(right_observed[metric]),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return result


def _pooled_per_class_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    samples: np.ndarray,
    *,
    stage: str,
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    left_observed = metrics_from_confusion(left.sum(axis=0))["per_class"][stage]
    right_observed = metrics_from_confusion(right.sum(axis=0))["per_class"][stage]
    distributions = {
        metric: np.empty(len(samples), dtype=np.float64) for metric in metrics
    }
    for index, selected in enumerate(samples):
        left_metrics = metrics_from_confusion(left[selected].sum(axis=0))[
            "per_class"
        ][stage]
        right_metrics = metrics_from_confusion(right[selected].sum(axis=0))[
            "per_class"
        ][stage]
        for metric in metrics:
            distributions[metric][index] = float(left_metrics[metric]) - float(
                right_metrics[metric]
            )
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        low, high = np.quantile(distributions[metric], [0.025, 0.975])
        result[metric] = {
            "observed_difference": float(left_observed[metric])
            - float(right_observed[metric]),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return result


def analyze_e3_e2(
    *,
    e3_manifest_path: Path,
    e3_gate_path: Path,
    e2_manifest_path: Path,
    e2_gate_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if protocol.get("status") != "locked_posthoc_before_paired_computation":
        raise ValueError("E3-E2 post-hoc protocol is not locked")
    if protocol.get("comparison") != "E3-E2":
        raise ValueError("unexpected comparison")
    if protocol["metrics"]["primary"] != "subject_macro_f1":
        raise ValueError("unexpected primary metric")

    sources = protocol["source_artifacts"]
    e3_manifest, e3_manifest_hash, e3_gate_hash = _validate_source(
        manifest_path=e3_manifest_path,
        gate_path=e3_gate_path,
        expected_manifest_hash=sources["E3_run_manifest_sha256"],
        expected_gate_hash=sources["E3_test_gate_sha256"],
    )
    e2_manifest, e2_manifest_hash, e2_gate_hash = _validate_source(
        manifest_path=e2_manifest_path,
        gate_path=e2_gate_path,
        expected_manifest_hash=sources["E2_run_manifest_sha256"],
        expected_gate_hash=sources["E2_test_gate_sha256"],
    )

    predictions = {
        "E3": load_ensemble_predictions(e3_manifest, "E3").sorted(),
        "E2": load_ensemble_predictions(e2_manifest, "E2").sorted(),
    }
    assert_paired(predictions["E3"], predictions["E2"])

    subject_ids: np.ndarray | None = None
    confusions: dict[str, np.ndarray] = {}
    subject_values: dict[str, np.ndarray] = {}
    transition_confusions: dict[str, np.ndarray] = {}
    reference_transition = transition_mask(predictions["E2"], radius=1)
    for experiment, values in predictions.items():
        subjects, matrices = _subject_confusions(values)
        if subject_ids is None:
            subject_ids = subjects
        elif not np.array_equal(subject_ids, subjects):
            raise ValueError("subject membership differs between E3 and E2")
        selected = transition_mask(values, radius=1)
        if not np.array_equal(selected, reference_transition):
            raise ValueError("transition masks differ between E3 and E2")
        confusions[experiment] = matrices
        subject_values[experiment] = _subject_macro_f1(matrices)
        transition_confusions[experiment] = _transition_confusions(
            values, selected, subjects
        )
    assert subject_ids is not None
    if len(subject_ids) != int(protocol["subjects"]):
        raise ValueError("subject count differs from locked protocol")
    if len(predictions["E2"].true_label) != int(protocol["valid_epochs"]):
        raise ValueError("valid epoch count differs from locked protocol")

    resamples = int(protocol["metrics"]["bootstrap_resamples"])
    seed = int(protocol["metrics"]["bootstrap_seed"])
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0, len(subject_ids), size=(resamples, len(subject_ids)), dtype=np.int32
    )

    descriptive = {
        experiment: {
            "subject_macro_f1_mean": float(subject_values[experiment].mean()),
            "pooled": metrics_from_confusion(confusions[experiment].sum(axis=0)),
            "transition_radius_1": metrics_from_confusion(
                transition_confusions[experiment].sum(axis=0)
            ),
        }
        for experiment in ("E3", "E2")
    }
    primary = _mean_paired_bootstrap(
        subject_values["E3"], subject_values["E2"], samples
    )
    wilcoxon = _wilcoxon_summary(subject_values["E3"], subject_values["E2"])
    pooled = _pooled_metrics_bootstrap(
        confusions["E3"],
        confusions["E2"],
        samples,
        metrics=("macro_f1", "accuracy", "cohen_kappa"),
    )
    n1 = _pooled_per_class_bootstrap(
        confusions["E3"],
        confusions["E2"],
        samples,
        stage="N1",
        metrics=("f1", "recall"),
    )
    transition = _pooled_metrics_bootstrap(
        transition_confusions["E3"],
        transition_confusions["E2"],
        samples,
        metrics=("macro_f1",),
    )
    comparison = {
        "comparison": "E3-E2",
        "primary_mean_subject_macro_f1": primary,
        "supporting_subject_wilcoxon": wilcoxon,
        "secondary_pooled_macro_f1": pooled["macro_f1"],
        "secondary_pooled_accuracy": pooled["accuracy"],
        "secondary_pooled_cohen_kappa": pooled["cohen_kappa"],
        "supporting_n1_f1": n1["f1"],
        "supporting_n1_recall": n1["recall"],
        "supporting_transition_radius_1_macro_f1": transition["macro_f1"],
    }
    comparison["decision"] = supporting_evidence_decision(
        primary["ci95_low"], wilcoxon["p_value"]
    )

    return {
        "schema_version": 1,
        "status": "complete",
        "analysis_scope": protocol["analysis_scope"],
        "subjects": int(len(subject_ids)),
        "valid_epochs": int(len(predictions["E2"].true_label)),
        "experiments": ["E3", "E2"],
        "primary_estimand": protocol["estimand"],
        "primary_uncertainty": protocol["metrics"]["uncertainty"],
        "supporting_test": protocol["metrics"]["supporting_test"],
        "multiplicity": protocol["metrics"]["multiple_testing"],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "E3_run_manifest": e3_manifest_hash,
            "E3_test_gate": e3_gate_hash,
            "E2_run_manifest": e2_manifest_hash,
            "E2_test_gate": e2_gate_hash,
        },
        "descriptive": descriptive,
        "comparison": comparison,
        "claim_boundary": protocol["claim_boundary"],
    }
