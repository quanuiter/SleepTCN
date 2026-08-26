"""Summarize validation-only ResNet tuning runs.

The global ranking is deliberately descriptive. A candidate used for an
outer-fold test run must be selected from that same fold's validation results;
otherwise a candidate selected after pooling all outer folds can indirectly
see subjects that are test subjects in another fold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.serialization import atomic_write_json, read_json


SUMMARY_SCHEMA_VERSION = 2


def _load_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
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
        try:
            candidate_id = str(manifest["candidate_id"])
            fold = int(manifest["outer_fold"])
            seed = int(manifest["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid tuning manifest: {manifest_path}") from exc
        key = (candidate_id, fold, seed)
        if key in seen:
            raise ValueError(f"duplicate candidate/fold/seed run: {key}")
        seen.add(key)
        subject_scores = {
            subject: float(value["macro_f1"])
            for subject, value in subjects.items()
        }
        rows.append(
            {
                "candidate_id": candidate_id,
                "fold": fold,
                "seed": seed,
                "mean_macro_f1": float(subject_level["mean_macro_f1"]),
                "std_macro_f1": float(subject_level["std_macro_f1"]),
                "subject_scores": subject_scores,
                "run_root": str(run_root),
            }
        )
    if not rows:
        raise ValueError(f"no validation metrics found below {output_root}")
    return rows


def _global_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    ranking = []
    for candidate_id, values in grouped.items():
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
                "std_across_subject_evaluations": variance**0.5,
                "runs": sorted(values, key=lambda row: (row["seed"], row["fold"])),
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["mean_validation_subject_macro_f1"],
            row["candidate_id"],
        )
    )
    return ranking


def _validate_coverage(
    rows: list[dict[str, Any]],
    *,
    expected_candidate_ids: list[str] | None,
    expected_folds: list[int] | None,
    seed: int | None,
) -> tuple[list[str], list[int], list[int]]:
    seeds = sorted({int(row["seed"]) for row in rows})
    if seed is not None:
        if seed not in seeds:
            raise ValueError(f"no tuning runs found for seed {seed}")
        rows[:] = [row for row in rows if row["seed"] == seed]
        seeds = [seed]
    candidates = sorted({str(row["candidate_id"]) for row in rows})
    if expected_candidate_ids is not None:
        expected = list(expected_candidate_ids)
        missing = sorted(set(expected) - set(candidates))
        unexpected = sorted(set(candidates) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"candidate coverage mismatch; missing={missing}, unexpected={unexpected}"
            )
        candidates = expected
    observed_folds = sorted({int(row["fold"]) for row in rows})
    if expected_folds is not None and observed_folds != sorted(expected_folds):
        raise ValueError(
            "incomplete outer-fold coverage; "
            f"expected={sorted(expected_folds)}, observed={observed_folds}"
        )
    for current_seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == current_seed]
        for candidate_id in candidates:
            candidate_folds = sorted(
                row["fold"]
                for row in seed_rows
                if row["candidate_id"] == candidate_id
            )
            if candidate_folds != observed_folds:
                raise ValueError(
                    "candidate runs do not cover the same folds; "
                    f"seed={current_seed}, candidate={candidate_id}, "
                    f"folds={candidate_folds}, expected={observed_folds}"
                )
    return candidates, observed_folds, seeds


def _per_fold_selections(
    rows: list[dict[str, Any]],
    candidates: list[str],
    folds: list[int],
    seeds: list[int],
) -> dict[str, dict[str, dict[str, Any]]]:
    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        seed_selections: dict[str, dict[str, Any]] = {}
        for fold in folds:
            fold_rows = [
                row
                for row in rows
                if row["seed"] == seed and row["fold"] == fold
            ]
            fold_rows.sort(
                key=lambda row: (-row["mean_macro_f1"], row["candidate_id"])
            )
            if {row["candidate_id"] for row in fold_rows} != set(candidates):
                raise ValueError(
                    f"missing candidate result for seed={seed}, fold={fold}"
                )
            seed_selections[str(fold)] = {
                "selected_candidate_id": fold_rows[0]["candidate_id"],
                "selection_metric": "validation_subject_macro_f1",
                "ranking": [
                    {
                        "candidate_id": row["candidate_id"],
                        "validation_subject_macro_f1": row["mean_macro_f1"],
                        "n_subjects": len(row["subject_scores"]),
                        "run_root": row["run_root"],
                    }
                    for row in fold_rows
                ],
            }
        selections[str(seed)] = seed_selections
    return selections


def summarize(
    output_root: Path,
    *,
    search_config: Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    rows = _load_rows(output_root)
    expected_candidate_ids: list[str] | None = None
    expected_folds: list[int] | None = None
    campaign_id: str | None = None
    if search_config is not None:
        document = read_json(search_config)
        candidates = document.get("candidates")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError("search config must define candidates")
        expected_candidate_ids = [str(candidate_id) for candidate_id in candidates]
        n_folds = int(document.get("n_folds", 10))
        if n_folds <= 0:
            raise ValueError("search config n_folds must be positive")
        expected_folds = list(range(n_folds))
        campaign_id = str(document.get("campaign_id"))
    candidates, folds, seeds = _validate_coverage(
        rows,
        expected_candidate_ids=expected_candidate_ids,
        expected_folds=expected_folds,
        seed=seed,
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "selection_role": "validation_only",
        "selection_policy": "per_outer_fold_validation",
        "global_ranking_role": "descriptive_only",
        "test_records_loaded": False,
        "candidate_ids": candidates,
        "folds": folds,
        "seeds": seeds,
        "ranking": _global_ranking(rows),
        "selections": _per_fold_selections(rows, candidates, folds, seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        args.output_root.resolve(),
        search_config=args.search_config.resolve(),
        seed=args.seed,
    )
    atomic_write_json(args.output, summary)
    print("Global validation ranking (descriptive; do not use for same-split test):")
    for index, row in enumerate(summary["ranking"], start=1):
        print(
            f"{index}. {row['candidate_id']}: "
            f"{row['mean_validation_subject_macro_f1']:.6f} "
            f"(n={row['n_runs']})"
        )
    for current_seed, fold_rows in summary["selections"].items():
        selected = ", ".join(
            f"fold_{fold}={value['selected_candidate_id']}"
            for fold, value in sorted(fold_rows.items(), key=lambda item: int(item[0]))
        )
        print(f"Per-fold selections for seed {current_seed}: {selected}")


if __name__ == "__main__":
    main()
