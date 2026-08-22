"""Descriptive comparison of locked result reports across training seeds.

This module intentionally does not pool p-values or treat a small, fixed seed set as
a random sample from a population of initializations.  It summarizes replication of
effect direction and inferential decisions while preserving each seed-specific test.
"""

from __future__ import annotations

from math import isclose
from typing import Any, Mapping


EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E6")
PRIMARY_COMPARISONS = ("E1-E0", "E2-E1", "E3-E2", "E3-E6")
SECONDARY_COMPARISONS = ("E4-E2",)
EXPECTED_SUBJECTS = 78
EXPECTED_RECORDS = 153
EXPECTED_VALID_EPOCHS = 195_469


def _results_by_comparison(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report["primary_results"] + report["secondary_results"]
    mapped = {row["comparison"]: row for row in rows}
    expected = set(PRIMARY_COMPARISONS + SECONDARY_COMPARISONS)
    if set(mapped) != expected or len(mapped) != len(rows):
        raise ValueError("comparison set differs from the locked Gate-5 contract")
    return mapped


def _metrics_by_experiment(report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    found: dict[str, dict[str, float]] = {}
    for row in _results_by_comparison(report).values():
        descriptive = row["descriptive"]
        for experiment, side in (
            (row["proposed"], "proposed"),
            (row["reference"], "reference"),
        ):
            source = descriptive[side]
            metrics = {
                "macro_f1": float(source["macro_f1"]),
                "accuracy": float(source["accuracy"]),
                "cohen_kappa": float(source["cohen_kappa"]),
            }
            if experiment in found:
                if any(
                    not isclose(found[experiment][key], value, rel_tol=0.0, abs_tol=0.0)
                    for key, value in metrics.items()
                ):
                    raise ValueError(f"inconsistent metrics for {experiment}")
            else:
                found[experiment] = metrics
    if set(found) != set(EXPERIMENTS):
        raise ValueError("experiment metrics are incomplete")
    return found


def validate_gate5_report(report: Mapping[str, Any], expected_seed: int) -> None:
    if report.get("schema_version") != 2 or report.get("status") != "complete":
        raise ValueError("Gate-5 report is not complete schema version 2")
    if report.get("seed") != expected_seed:
        raise ValueError("Gate-5 report seed differs from its declared input seed")
    if tuple(row["comparison"] for row in report["primary_results"]) != PRIMARY_COMPARISONS:
        raise ValueError("primary comparison order differs from the locked protocol")
    if tuple(row["comparison"] for row in report["secondary_results"]) != SECONDARY_COMPARISONS:
        raise ValueError("secondary comparison set differs from the locked protocol")
    coverage = report.get("input_coverage", {})
    if set(coverage) != set(EXPERIMENTS):
        raise ValueError("input coverage does not contain the six active experiments")
    for experiment, row in coverage.items():
        observed = (row.get("subjects"), row.get("records"), row.get("valid_epochs"))
        expected = (EXPECTED_SUBJECTS, EXPECTED_RECORDS, EXPECTED_VALID_EPOCHS)
        if observed != expected:
            raise ValueError(f"unexpected coverage for {experiment}: {observed}")
    for row in report["primary_results"]:
        wilcoxon = row["subject_wilcoxon_macro_f1"]
        if wilcoxon.get("holm_family_size") != len(PRIMARY_COMPARISONS):
            raise ValueError("primary Holm family size must remain four")
        adjusted = wilcoxon.get("holm_adjusted_p_value")
        if adjusted is None or not 0.0 <= float(adjusted) <= 1.0:
            raise ValueError("invalid Holm-adjusted p-value")
    _metrics_by_experiment(report)


def compare_seed_reports(reports: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("seed sensitivity requires at least two reports")
    seeds = tuple(sorted(reports))
    for seed in seeds:
        validate_gate5_report(reports[seed], seed)
    split_hashes = {reports[seed]["provenance"]["split_sha256"] for seed in seeds}
    config_hashes = {reports[seed]["provenance"]["config_sha256"] for seed in seeds}
    if len(split_hashes) != 1 or len(config_hashes) != 1:
        raise ValueError("seed reports do not share the same split and configuration")

    model_metrics = {seed: _metrics_by_experiment(reports[seed]) for seed in seeds}
    experiments: dict[str, Any] = {}
    for experiment in EXPERIMENTS:
        by_seed = {
            str(seed): model_metrics[seed][experiment]
            for seed in seeds
        }
        macro_f1 = [by_seed[str(seed)]["macro_f1"] for seed in seeds]
        experiments[experiment] = {
            "by_seed": by_seed,
            "macro_f1_mean_across_fixed_seeds": sum(macro_f1) / len(macro_f1),
            "macro_f1_min": min(macro_f1),
            "macro_f1_max": max(macro_f1),
        }

    result_maps = {seed: _results_by_comparison(reports[seed]) for seed in seeds}
    comparisons: dict[str, Any] = {}
    for comparison in PRIMARY_COMPARISONS + SECONDARY_COMPARISONS:
        by_seed: dict[str, Any] = {}
        effects: list[float] = []
        for seed in seeds:
            row = result_maps[seed][comparison]
            bootstrap = row["cluster_bootstrap_macro_f1"]
            wilcoxon = row["subject_wilcoxon_macro_f1"]
            effect = float(bootstrap["observed_difference"])
            effects.append(effect)
            by_seed[str(seed)] = {
                "observed_difference": effect,
                "ci95_low": float(bootstrap["ci95_low"]),
                "ci95_high": float(bootstrap["ci95_high"]),
                "wilcoxon_p": float(wilcoxon["p_value"]),
                "holm_adjusted_p": (
                    None
                    if wilcoxon.get("holm_adjusted_p_value") is None
                    else float(wilcoxon["holm_adjusted_p_value"])
                ),
                "wins": int(wilcoxon["wins"]),
                "ties": int(wilcoxon["ties"]),
                "losses": int(wilcoxon["losses"]),
            }
        nonzero_signs = {1 if effect > 0 else -1 for effect in effects if effect != 0}
        primary = comparison in PRIMARY_COMPARISONS
        comparisons[comparison] = {
            "by_seed": by_seed,
            "effect_mean_across_fixed_seeds": sum(effects) / len(effects),
            "effect_min": min(effects),
            "effect_max": max(effects),
            "same_nonzero_direction_in_all_seeds": len(nonzero_signs) == 1,
            "positive_ci_in_all_seeds": all(
                row["ci95_low"] > 0.0 for row in by_seed.values()
            ),
            "holm_significant_seeds": (
                [
                    seed
                    for seed in seeds
                    if by_seed[str(seed)]["holm_adjusted_p"] is not None
                    and by_seed[str(seed)]["holm_adjusted_p"] < 0.05
                ]
                if primary
                else []
            ),
            "holm_significant_in_all_seeds": (
                all(
                    by_seed[str(seed)]["holm_adjusted_p"] is not None
                    and by_seed[str(seed)]["holm_adjusted_p"] < 0.05
                    for seed in seeds
                )
                if primary
                else None
            ),
        }

    return {
        "schema_version": 1,
        "status": "complete",
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "scope": "post_protocol_fixed_seed_sensitivity_analysis",
        "statistical_boundary": (
            "Seed-specific confidence intervals and tests are retained. No p-values are "
            "pooled, and the fixed seeds are not treated as a random sample of training seeds."
        ),
        "shared_split_sha256": next(iter(split_hashes)),
        "shared_config_sha256": next(iter(config_hashes)),
        "experiments": experiments,
        "comparisons": comparisons,
    }
