import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_resnet_locked_config import build_locked_config
from scripts.summarize_resnet_tuning import summarize
from sleeptcn.experiment import build_context


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_run(root: Path, candidate: str, fold: int, score: float) -> None:
    run_root = root / candidate / f"fold_{fold:02d}" / "seed_42"
    _write_json(
        run_root / "run_manifest.json",
        {
            "candidate_id": candidate,
            "outer_fold": fold,
            "seed": 42,
            "test_records_loaded": False,
        },
    )
    _write_json(
        run_root / "validation_metrics.json",
        {
            "subject_level": {
                "mean_macro_f1": score,
                "std_macro_f1": 0.0,
                "subjects": {f"subject_{fold}": {"macro_f1": score}},
            }
        },
    )


def test_summary_selects_candidate_per_outer_fold(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    _write_run(output_root, "candidate_a", 0, 0.80)
    _write_run(output_root, "candidate_b", 0, 0.70)
    _write_run(output_root, "candidate_a", 1, 0.60)
    _write_run(output_root, "candidate_b", 1, 0.90)
    search_path = tmp_path / "search.json"
    _write_json(
        search_path,
        {"campaign_id": "test", "n_folds": 2, "candidates": {"candidate_a": {}, "candidate_b": {}}},
    )

    summary = summarize(output_root, search_config=search_path, seed=42)

    assert summary["selection_policy"] == "per_outer_fold_validation"
    assert summary["global_ranking_role"] == "descriptive_only"
    assert summary["selections"]["42"]["0"]["selected_candidate_id"] == "candidate_a"
    assert summary["selections"]["42"]["1"]["selected_candidate_id"] == "candidate_b"


def test_summary_rejects_incomplete_campaign(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    _write_run(output_root, "candidate_a", 0, 0.80)
    search_path = tmp_path / "search.json"
    _write_json(
        search_path,
        {"campaign_id": "test", "n_folds": 2, "candidates": {"candidate_a": {}, "candidate_b": {}}},
    )

    with pytest.raises(ValueError, match="candidate coverage mismatch"):
        summarize(output_root, search_config=search_path, seed=42)


def test_locked_config_records_selection_provenance() -> None:
    base = {"components": {"resnet1d": {"feature_dim": 128}}}
    search = {
        "campaign_id": "test",
        "selection_policy": "per_outer_fold_validation",
        "candidates": {"candidate_a": {"resnet1d": {"feature_dim": 64}}},
    }
    report = {
        "selection_role": "validation_only",
        "selection_policy": "per_outer_fold_validation",
        "test_records_loaded": False,
        "selections": {"42": {"0": {"selected_candidate_id": "candidate_a"}}},
    }

    locked = build_locked_config(
        base,
        search,
        selection_report=report,
        outer_fold=0,
        seed=42,
        selection_report_sha256="a" * 64,
    )

    assert locked["components"]["resnet1d"]["feature_dim"] == 64
    assert locked["resnet_tuning"]["selected_outer_fold"] == 0
    assert locked["resnet_tuning"]["selected_seed"] == 42
    assert locked["resnet_tuning"]["test_policy"] == "outer_fold_test_unseen_during_selection"


def test_context_rejects_locked_config_for_another_fold(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = deepcopy(json.loads((root / "configs/experiments_v2.json").read_text()))
    config["resnet_tuning"] = {
        "selection_policy": "per_outer_fold_validation",
        "selected_outer_fold": 1,
        "selected_seed": 42,
    }
    config_path = tmp_path / "locked.json"
    _write_json(config_path, config)

    with pytest.raises(ValueError, match="different fold/seed"):
        build_context(
            root,
            "E2",
            0,
            42,
            "cpu",
            smoke=True,
            allow_test_evaluation=False,
            num_workers=0,
            config_path=config_path,
        )
