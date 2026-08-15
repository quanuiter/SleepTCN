"""Paired analysis for the pre-locked E1/E2 SHHS component extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .gate8_analysis import transition_mask
from .metrics import metrics_from_confusion
from .shhs_analysis import (
    _mean_paired_bootstrap,
    _pooled_metric_bootstrap,
    _subject_confusions,
    _subject_macro_f1,
    _transition_confusions,
    _wilcoxon_summary,
    load_ensemble_predictions,
    sha256_file,
)
from .statistics import assert_paired, holm_adjust


EXPERIMENTS = ("E0", "E1", "E2")
PRIMARY_COMPARISONS = (("E1", "E0"), ("E2", "E1"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def superiority_decision(
    ci95_low: float, holm_adjusted_p_value: float
) -> str:
    return (
        "supported_on_locked_shhs_sample"
        if ci95_low > 0.0 and holm_adjusted_p_value < 0.05
        else "not_supported_on_locked_shhs_sample"
    )


def analyze_component_extension(
    *,
    component_manifest_path: Path,
    component_gate_path: Path,
    reference_manifest_path: Path,
    reference_gate_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    component_manifest = _load_json(component_manifest_path)
    component_gate = _load_json(component_gate_path)
    reference_manifest = _load_json(reference_manifest_path)
    reference_gate = _load_json(reference_gate_path)
    protocol_hash = sha256_file(protocol_path)
    component_manifest_hash = sha256_file(component_manifest_path)
    reference_manifest_hash = sha256_file(reference_manifest_path)

    if protocol.get("status") != "locked_before_component_inference":
        raise ValueError("component protocol is not locked")
    if protocol["metrics"]["primary_comparisons"] != ["E1-E0", "E2-E1"]:
        raise ValueError("unexpected primary comparisons")
    if component_manifest.get("status") != "complete" or component_manifest.get("role") != "test":
        raise ValueError("component test run is not complete")
    if component_manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("component run points to another protocol")
    if component_gate.get("status") != "passed" or component_gate.get("role") != "test":
        raise ValueError("component test gate has not passed")
    if component_gate.get("run_manifest_sha256") != component_manifest_hash:
        raise ValueError("component gate points to another run")

    source = protocol["reference_artifacts"]
    if reference_manifest_hash != source["run_manifest_sha256"]:
        raise ValueError("E0 reference run hash differs from locked protocol")
    if sha256_file(reference_gate_path) != source["test_gate_sha256"]:
        raise ValueError("E0 reference gate hash differs from locked protocol")
    if reference_gate.get("status") != "passed" or reference_gate.get("role") != "test":
        raise ValueError("E0 reference test gate has not passed")
    if reference_gate.get("run_manifest_sha256") != reference_manifest_hash:
        raise ValueError("E0 reference gate points to another run")
    if reference_manifest.get("protocol_sha256") != source["protocol_sha256"]:
        raise ValueError("E0 reference protocol hash differs")

    predictions = {
        "E0": load_ensemble_predictions(reference_manifest, "E0"),
        "E1": load_ensemble_predictions(component_manifest, "E1"),
        "E2": load_ensemble_predictions(component_manifest, "E2"),
    }
    predictions = {key: value.sorted() for key, value in predictions.items()}
    for experiment in EXPERIMENTS[1:]:
        assert_paired(predictions["E0"], predictions[experiment])

    subject_ids: np.ndarray | None = None
    confusions: dict[str, np.ndarray] = {}
    subject_values: dict[str, np.ndarray] = {}
    transition_confusions: dict[str, np.ndarray] = {}
    reference_transition = transition_mask(predictions["E0"], radius=1)
    for experiment, values in predictions.items():
        subjects, matrices = _subject_confusions(values)
        if subject_ids is None:
            subject_ids = subjects
        elif not np.array_equal(subject_ids, subjects):
            raise ValueError("subject membership differs between experiments")
        selected = transition_mask(values, radius=1)
        if not np.array_equal(selected, reference_transition):
            raise ValueError("transition masks differ between experiments")
        confusions[experiment] = matrices
        subject_values[experiment] = _subject_macro_f1(matrices)
        transition_confusions[experiment] = _transition_confusions(
            values, selected, subjects
        )
    assert subject_ids is not None

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
        for experiment in EXPERIMENTS
    }

    comparisons = []
    for proposed, reference in PRIMARY_COMPARISONS:
        comparisons.append(
            {
                "comparison": f"{proposed}-{reference}",
                "proposed": proposed,
                "reference": reference,
                "primary_mean_subject_macro_f1": _mean_paired_bootstrap(
                    subject_values[proposed], subject_values[reference], samples
                ),
                "supporting_subject_wilcoxon": _wilcoxon_summary(
                    subject_values[proposed], subject_values[reference]
                ),
                "secondary_pooled_macro_f1": _pooled_metric_bootstrap(
                    confusions[proposed], confusions[reference], samples, "macro_f1"
                ),
                "supporting_transition_radius_1_macro_f1": _pooled_metric_bootstrap(
                    transition_confusions[proposed],
                    transition_confusions[reference],
                    samples,
                    "macro_f1",
                ),
            }
        )
    adjusted = holm_adjust(
        [item["supporting_subject_wilcoxon"]["p_value"] for item in comparisons]
    )
    for item, adjusted_p in zip(comparisons, adjusted, strict=True):
        item["supporting_subject_wilcoxon"]["holm_adjusted_p_value"] = adjusted_p
        item["supporting_subject_wilcoxon"]["holm_family_size"] = 2
        item["decision"] = superiority_decision(
            item["primary_mean_subject_macro_f1"]["ci95_low"], adjusted_p
        )

    return {
        "schema_version": 1,
        "status": "complete",
        "analysis_scope": protocol["analysis_scope"],
        "subjects": int(len(subject_ids)),
        "valid_epochs": int(len(predictions["E0"].true_label)),
        "experiments": list(EXPERIMENTS),
        "primary_estimand": "mean paired subject-level macro_f1 difference",
        "primary_uncertainty": "paired subject-cluster percentile bootstrap 95% CI",
        "supporting_test": "two-sided paired-subject Wilcoxon signed-rank",
        "multiplicity": "Holm correction over E1-E0 and E2-E1",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "source_sha256": {
            "protocol": protocol_hash,
            "component_run_manifest": component_manifest_hash,
            "component_test_gate": sha256_file(component_gate_path),
            "reference_run_manifest": reference_manifest_hash,
            "reference_test_gate": sha256_file(reference_gate_path),
        },
        "descriptive": descriptive,
        "primary_comparisons": comparisons,
        "claim_boundary": protocol["interpretation"],
    }
