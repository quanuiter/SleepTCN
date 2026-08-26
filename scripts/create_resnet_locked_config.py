"""Create a full E0-E6 config with a validation-selected ResNet candidate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.hashing import sha256_file
from sleeptcn.io.serialization import atomic_write_json, read_json


def _candidate_from_selection_report(
    report: dict[str, Any], *, outer_fold: int, seed: int
) -> str:
    if report.get("selection_role") != "validation_only":
        raise ValueError("selection report is not validation-only")
    if report.get("selection_policy") != "per_outer_fold_validation":
        raise ValueError(
            "selection report must select one candidate per outer fold"
        )
    if report.get("test_records_loaded") is not False:
        raise ValueError("selection report must not contain test results")
    selections = report.get("selections")
    if not isinstance(selections, dict):
        raise ValueError("selection report has no per-fold selections")
    seed_selections = selections.get(str(seed))
    if not isinstance(seed_selections, dict):
        raise ValueError(f"selection report has no seed {seed}")
    fold_selection = seed_selections.get(str(outer_fold))
    if not isinstance(fold_selection, dict):
        raise ValueError(
            f"selection report has no result for seed={seed}, fold={outer_fold}"
        )
    candidate_id = fold_selection.get("selected_candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("selection report has no selected candidate")
    return candidate_id


def build_locked_config(
    base_config: dict[str, object],
    search_config: dict[str, object],
    candidate_id: str | None = None,
    *,
    selection_report: dict[str, Any] | None = None,
    outer_fold: int | None = None,
    seed: int | None = None,
    selection_report_sha256: str | None = None,
) -> dict[str, object]:
    if selection_report is not None:
        if search_config.get("selection_policy") != "per_outer_fold_validation":
            raise ValueError(
                "search config must use per_outer_fold_validation selection"
            )
        if candidate_id is not None:
            raise ValueError("pass either candidate_id or selection_report, not both")
        if outer_fold is None or seed is None:
            raise ValueError(
                "outer_fold and seed are required with a selection report"
            )
        candidate_id = _candidate_from_selection_report(
            selection_report, outer_fold=outer_fold, seed=seed
        )
    elif not candidate_id:
        raise ValueError("candidate_id is required without a selection report")

    assert candidate_id is not None
    candidates = search_config.get("candidates")
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        raise ValueError(f"unknown tuning candidate: {candidate_id}")
    candidate = candidates[candidate_id]
    if not isinstance(candidate, dict) or not isinstance(
        candidate.get("resnet1d"), dict
    ):
        raise ValueError("candidate must contain a resnet1d mapping")
    components = base_config.get("components")
    if not isinstance(components, dict) or not isinstance(
        components.get("resnet1d"), dict
    ):
        raise ValueError("base config has no components.resnet1d mapping")

    locked = copy.deepcopy(base_config)
    locked_components = locked["components"]
    assert isinstance(locked_components, dict)
    resnet_config = copy.deepcopy(locked_components["resnet1d"])
    assert isinstance(resnet_config, dict)
    # Search candidates keep architecture under ``resnet1d`` and optimizer
    # hyperparameters at the candidate root.  Merge the architecture into the
    # component itself; copying the candidate root verbatim would create a
    # dead nested ``components.resnet1d.resnet1d`` key and silently leave the
    # baseline architecture in use.
    resnet_config.update(copy.deepcopy(candidate["resnet1d"]))
    resnet_config.update(
        {
            key: copy.deepcopy(value)
            for key, value in candidate.items()
            if key != "resnet1d"
        }
    )
    locked_components["resnet1d"] = resnet_config
    locked["status"] = "resnet_v3_locked_candidate"
    tuning_metadata: dict[str, object] = {
        "campaign_id": search_config.get("campaign_id"),
        "candidate_id": candidate_id,
        "selection_role": "validation_only",
        "selection_policy": (
            search_config.get("selection_policy", "per_outer_fold_validation")
            if selection_report is not None
            else "manual_candidate_external_confirmation"
        ),
        "test_policy": (
            "outer_fold_test_unseen_during_selection"
            if selection_report is not None
            else "external_confirmation_only"
        ),
        "source_search_config": "configs/tuning/resnet_v3_search.json",
    }
    if selection_report is not None:
        assert outer_fold is not None and seed is not None
        tuning_metadata.update(
            {
                "selection_report_sha256": selection_report_sha256,
                "selected_outer_fold": outer_fold,
                "selected_seed": seed,
            }
        )
    locked["resnet_tuning"] = tuning_metadata
    return locked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    candidate_group = parser.add_mutually_exclusive_group(required=False)
    candidate_group.add_argument("--candidate")
    candidate_group.add_argument("--selection-report", type=Path)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = None
    report_sha256 = None
    if args.selection_report is not None:
        report = read_json(args.selection_report)
        report_sha256 = sha256_file(args.selection_report)
    locked = build_locked_config(
        read_json(args.base_config),
        read_json(args.search_config),
        args.candidate,
        selection_report=report,
        outer_fold=args.outer_fold,
        seed=args.seed,
        selection_report_sha256=report_sha256,
    )
    atomic_write_json(args.output, locked)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate": locked["resnet_tuning"]["candidate_id"],
                "outer_fold": locked["resnet_tuning"].get("selected_outer_fold"),
                "seed": locked["resnet_tuning"].get("selected_seed"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
