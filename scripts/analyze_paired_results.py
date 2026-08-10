"""Analyze paired test predictions after the protocol has unlocked test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.statistics import (
    PredictionArrays,
    assert_paired,
    holm_adjust,
    paired_cluster_bootstrap,
    paired_subject_wilcoxon,
)


def load_experiment(workspace: Path, experiment: str, seed: int) -> PredictionArrays:
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "subject_id",
            "record_key",
            "original_epoch_index",
            "true_label",
            "predicted_label",
        )
    }
    seen_subjects: set[str] = set()
    for fold in range(10):
        path = (
            workspace
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
            raise FileNotFoundError(f"missing completed test prediction: {path}")
        with np.load(path, allow_pickle=False) as npz:
            metadata = json.loads(str(npz["metadata_json"].item()))
            expected = {
                "experiment_id": experiment,
                "outer_fold": fold,
                "seed": seed,
                "role": "test",
                "smoke": False,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"prediction metadata mismatch: {mismatches}")
            fold_subjects = set(npz["subject_id"].tolist())
            overlap = seen_subjects & fold_subjects
            if overlap:
                raise ValueError(f"subjects tested in multiple folds: {sorted(overlap)}")
            seen_subjects.update(fold_subjects)
            for name in parts:
                parts[name].append(npz[name].copy())
    return PredictionArrays(**{name: np.concatenate(value) for name, value in parts.items()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="proposed:reference, for example E1:E0",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache: dict[str, PredictionArrays] = {}
    results = []
    for item in args.comparison:
        pieces = item.split(":")
        if len(pieces) != 2 or not all(pieces):
            raise ValueError("comparison must use proposed:reference")
        proposed_id, reference_id = pieces
        for experiment in pieces:
            if experiment not in cache:
                cache[experiment] = load_experiment(
                    args.workspace.resolve(), experiment, args.seed
                )
        proposed, reference = cache[proposed_id], cache[reference_id]
        assert_paired(proposed, reference)
        results.append(
            {
                "comparison": f"{proposed_id}-{reference_id}",
                "cluster_bootstrap_macro_f1": paired_cluster_bootstrap(
                    proposed,
                    reference,
                    resamples=args.bootstrap_resamples,
                    seed=args.bootstrap_seed,
                ),
                "subject_wilcoxon_macro_f1": paired_subject_wilcoxon(
                    proposed, reference
                ),
            }
        )
    adjusted = holm_adjust(
        [item["subject_wilcoxon_macro_f1"]["p_value"] for item in results]
    )
    for item, value in zip(results, adjusted, strict=True):
        item["subject_wilcoxon_macro_f1"]["holm_adjusted_p_value"] = value
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "statistical_unit": "subject",
        "primary_effect": "paired_difference_in_global_macro_f1",
        "supporting_test": "two_sided_paired_subject_wilcoxon",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
