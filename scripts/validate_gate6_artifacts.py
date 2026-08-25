"""Validate the locked Gate-6 benchmark and feature-space artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.serialization import read_json  # noqa: E402


ACTIVE = ("E0", "E1", "E2", "E3", "E4", "E6")
EXPECTED_PARAMETERS = {
    "E0": 248_630,
    "E1": 640_950,
    "E2": 1_085_578,
    "E3": 1_085_578,
    "E4": 1_085_578,
    "E6": 1_085_578,
}


def validate_parameters(path: Path) -> dict[str, Any]:
    report = read_json(path)
    if report.get("schema_version") != 2 or report.get("status") != "complete":
        raise ValueError("invalid parameter report schema/status")
    if report.get("mode") != "parameters" or set(report.get("models", {})) != set(ACTIVE):
        raise ValueError("parameter report model set/mode mismatch")
    observed = {
        experiment: int(report["models"][experiment]["parameters"])
        for experiment in ACTIVE
    }
    if observed != EXPECTED_PARAMETERS:
        raise ValueError(f"parameter counts differ: {observed}")
    if any(
        report["models"][experiment]["trainable_parameters"] != observed[experiment]
        for experiment in ACTIVE
    ):
        raise ValueError("unexpected frozen parameters")
    return {"passed": True, "parameters": observed}


def validate_latency(path: Path) -> dict[str, Any]:
    report = read_json(path)
    expected_top = {
        "schema_version": 2,
        "status": "complete",
        "mode": "latency",
        "fold": 0,
        "seed": 42,
        "batch_records": 1,
        "sequence_length": 100,
        "warmup": 20,
        "repeats": 100,
        "rounds": 3,
    }
    mismatches = {
        key: (report.get(key), value)
        for key, value in expected_top.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"latency protocol mismatch: {mismatches}")
    if set(report.get("models", {})) != set(ACTIVE):
        raise ValueError("latency model set mismatch")
    environment = report.get("environment", {})
    if not environment.get("cuda_available") or not environment.get("gpu_name"):
        raise ValueError("latency report lacks CUDA/GPU provenance")
    if len(report.get("experiment_execution_orders", [])) != 3:
        raise ValueError("latency report lacks three execution rounds")
    for order in report["experiment_execution_orders"]:
        if set(order) != set(ACTIVE) or len(order) != len(ACTIVE):
            raise ValueError("invalid randomized execution order")
    summary: dict[str, Any] = {}
    for experiment in ACTIVE:
        model = report["models"][experiment]
        if model["parameters"] != EXPECTED_PARAMETERS[experiment]:
            raise ValueError(f"{experiment}: parameter count mismatch in latency report")
        latency = model.get("latency", {})
        if len(latency.get("rounds", [])) != 3:
            raise ValueError(f"{experiment}: missing timing rounds")
        samples = [
            value
            for round_report in latency["rounds"]
            for value in round_report.get("latency_samples_ms", [])
        ]
        if len(samples) != 300 or not np.isfinite(samples).all() or min(samples) <= 0:
            raise ValueError(f"{experiment}: invalid timing samples")
        if latency.get("total_timed_forward_passes") != 300:
            raise ValueError(f"{experiment}: timed pass count mismatch")
        for key in (
            "all_samples_ms_median",
            "all_samples_ms_p95",
            "throughput_epochs_per_second_from_all_sample_median",
            "maximum_peak_allocated_bytes",
            "maximum_peak_reserved_bytes",
        ):
            if latency.get(key, 0) <= 0:
                raise ValueError(f"{experiment}: invalid {key}")
        summary[experiment] = {
            "latency_ms_median": latency["all_samples_ms_median"],
            "latency_ms_p95": latency["all_samples_ms_p95"],
            "throughput_epochs_per_second": latency[
                "throughput_epochs_per_second_from_all_sample_median"
            ],
            "peak_allocated_bytes": latency["maximum_peak_allocated_bytes"],
        }
    return {"passed": True, "gpu": environment["gpu_name"], "models": summary}


def validate_feature_space(
    sample_manifest_path: Path, output_dir: Path
) -> dict[str, Any]:
    sample_manifest = read_json(sample_manifest_path)
    report_path = output_dir / "feature_space_report.json"
    csv_path = output_dir / "tsne_points.csv"
    png_path = output_dir / "tsne_E1_vs_E2.png"
    report = read_json(report_path)
    if sample_manifest.get("schema_version") != 3:
        raise ValueError("invalid feature sample manifest")
    if sample_manifest.get("folds") != list(range(10)):
        raise ValueError("feature sample manifest does not cover 10 folds")
    if sample_manifest.get("sample_per_class_per_fold") != 200:
        raise ValueError("feature sample size differs from frozen protocol")
    if sample_manifest.get("total_sample_count") != 10_000:
        raise ValueError("feature total sample count mismatch")
    if sample_manifest.get("subjects_represented_across_folds") != 78:
        raise ValueError("feature sample does not cover 78 subjects")
    if report.get("schema_version") != 3 or report.get("status") != "complete":
        raise ValueError("invalid feature report schema/status")
    if report.get("folds") != list(range(10)) or report.get("role") != "test":
        raise ValueError("feature report fold/role mismatch")
    folds = report.get("fold_results", {})
    if set(folds) != {f"fold_{fold:02d}" for fold in range(10)}:
        raise ValueError("feature report fold set mismatch")
    differences = []
    for fold in range(10):
        item = folds[f"fold_{fold:02d}"]
        if item.get("sample_count") != 1000:
            raise ValueError(f"fold {fold:02d}: sample count mismatch")
        if set(item.get("representations", {})) != {"E1", "E2"}:
            raise ValueError(f"fold {fold:02d}: representation set mismatch")
        for experiment in ("E1", "E2"):
            value = item["representations"][experiment].get("silhouette_score_pca")
            if value is None or not np.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"fold {fold:02d}/{experiment}: invalid silhouette")
        differences.append(item["silhouette_difference_E2_minus_E1"])
    expected_positive = int(np.sum(np.asarray(differences) > 0))
    if report["silhouette_difference_E2_minus_E1_summary"][
        "folds_E2_greater_than_E1"
    ] != expected_positive:
        raise ValueError("feature difference summary mismatch")
    if not csv_path.is_file() or not png_path.is_file() or png_path.stat().st_size <= 0:
        raise FileNotFoundError("missing feature visualization artifact")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = sum(1 for _ in csv.reader(handle)) - 1
    if rows != 2000:
        raise ValueError(f"t-SNE CSV must contain 2000 paired points, found {rows}")
    return {
        "passed": True,
        "folds": 10,
        "total_silhouette_samples": 10_000,
        "tsne_rows": rows,
        "E2_greater_than_E1_folds": expected_positive,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--feature-samples", type=Path, required=True)
    parser.add_argument("--feature-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "status": "passed",
        "parameters": validate_parameters(args.parameters),
        "latency": validate_latency(args.latency),
        "feature_space": validate_feature_space(
            args.feature_samples, args.feature_output_dir
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
