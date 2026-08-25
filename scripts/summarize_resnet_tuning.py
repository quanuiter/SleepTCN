"""Rank validation-only ResNet tuning candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.serialization import atomic_write_json, read_json


def summarize(output_root: Path) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metrics_path in sorted(
        output_root.glob("*/fold_*/seed_*/validation_metrics.json")
    ):
        run_root = metrics_path.parent
        manifest_path = run_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest beside {metrics_path}")
        metrics = read_json(metrics_path)
        manifest = read_json(manifest_path)
        if manifest.get("test_records_loaded") is not False:
            raise ValueError(f"candidate run is not validation-only: {run_root}")
        subject_level = metrics.get("subject_level")
        if not isinstance(subject_level, dict):
            raise ValueError(f"missing subject-level metrics: {metrics_path}")
        subjects = subject_level.get("subjects")
        if not isinstance(subjects, dict) or not subjects:
            raise ValueError(f"missing per-subject metrics: {metrics_path}")
        subject_scores = {
            subject: float(value["macro_f1"])
            for subject, value in subjects.items()
        }
        rows[str(manifest["candidate_id"])].append(
            {
                "fold": int(manifest["outer_fold"]),
                "seed": int(manifest["seed"]),
                "mean_macro_f1": float(subject_level["mean_macro_f1"]),
                "std_macro_f1": float(subject_level["std_macro_f1"]),
                "subject_scores": subject_scores,
                "run_root": str(run_root),
            }
        )

    ranking = []
    for candidate_id, values in rows.items():
        scores = [
            score
            for row in values
            for score in row["subject_scores"].values()
        ]
        mean = sum(scores) / len(scores)
        variance = (
            sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
            if len(scores) > 1
            else 0.0
        )
        ranking.append(
            {
                "candidate_id": candidate_id,
                "n_runs": len(values),
                "n_subject_evaluations": len(scores),
                "mean_validation_subject_macro_f1": mean,
                "std_across_runs": variance**0.5,
                "runs": values,
            }
        )
    ranking.sort(
        key=lambda row: row["mean_validation_subject_macro_f1"], reverse=True
    )
    return {
        "schema_version": 1,
        "selection_role": "validation_only",
        "test_records_loaded": False,
        "ranking": ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.output_root.resolve())
    atomic_write_json(args.output, summary)
    for index, row in enumerate(summary["ranking"], start=1):
        print(
            f"{index}. {row['candidate_id']}: "
            f"{row['mean_validation_subject_macro_f1']:.6f} "
            f"(n={row['n_runs']})"
        )


if __name__ == "__main__":
    main()
