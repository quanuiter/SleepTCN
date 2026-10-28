"""Checks for the Amendment A2 config and run planning (no training, no data)."""

from __future__ import annotations

import copy
from types import SimpleNamespace
import json
import shutil
from pathlib import Path

import pytest

from sleeptcn import amendment_a2 as a2
from sleeptcn.preprocessing import VALID_VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / a2.A2_CONFIG).read_text(encoding="utf-8"))


def _workspace(tmp_path: Path, config: dict) -> Path:
    (tmp_path / a2.A2_CONFIG).parent.mkdir(parents=True)
    (tmp_path / a2.A2_CONFIG).write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_config_arms_use_known_experiments_and_variants():
    config = _config()
    experiments = json.loads((ROOT / a2.EXPERIMENT_CONFIG).read_text(encoding="utf-8"))["experiments"]
    for arm, spec in config["training"]["arms"].items():
        assert spec["experiment"] in experiments, arm
        assert spec["data_variant"] in VALID_VARIANTS, arm
    assert config["shhs"]["new_variants"] == ["filtered_zscore_rolling90_v3"]


def test_minimal_package_targets_and_order():
    targets = a2.training_targets(_config())
    assert len(targets) == 50
    assert len(set(targets)) == 50
    pairs = list(dict.fromkeys((t.arm, t.seed) for t in targets))
    assert pairs == [("E6R", 42), ("E6R", 123), ("E4", 2025), ("E4", 2027), ("E4", 2029)]
    assert [t.fold for t in targets[:10]] == list(range(10))
    # New E4 seeds never collide with the archived seed-42/123 E4 runs or the reproduction run.
    assert not {t.seed for t in targets if t.arm == "E4"} & {42, 123}


def test_run_order_must_match_declared_arms():
    config = _config()
    broken = copy.deepcopy(config)
    broken["training"]["run_order"] = broken["training"]["run_order"][:-1]
    with pytest.raises(ValueError, match="run_order"):
        a2.training_targets(broken)
    duplicated = copy.deepcopy(config)
    duplicated["training"]["run_order"].append(["E4", 2025])
    with pytest.raises(ValueError, match="run_order"):
        a2.training_targets(duplicated)
    reserved = copy.deepcopy(config)
    reserved["training"]["arms"][a2.REPRO_ARM] = {"experiment": "E4", "data_variant": "bandpass_v2",
                                                 "seeds": [7]}
    reserved["training"]["run_order"].append([a2.REPRO_ARM, 7])
    with pytest.raises(ValueError, match="reserved"):
        a2.training_targets(reserved)


def test_unlocked_config_is_refused(tmp_path):
    config = _config()
    config["status"] = "draft"
    workspace = _workspace(tmp_path, config)
    with pytest.raises(ValueError, match="lock"):
        a2.load_a2_config(workspace)
    loaded, sha = a2.load_a2_config(workspace, require_locked=False)
    assert loaded["status"] == "draft" and len(sha) == 64
    config["status"] = a2.LOCKED_STATUS
    workspace = _workspace(tmp_path / "locked", config)
    assert a2.load_a2_config(workspace)[0]["status"] == a2.LOCKED_STATUS


def test_run_paths_are_separate_per_arm_seed_and_fold(tmp_path):
    targets = a2.training_targets(_config()) + [a2.repro_target()]
    roots = {a2.run_root(tmp_path, t) for t in targets}
    assert len(roots) == len(targets)
    assert a2.run_root(tmp_path, a2.repro_target()).parts[-3:] == ("REPRO_E4", "fold_00", "seed_123")


def test_training_state_transitions(tmp_path):
    target = a2.A2Target("E4", 2025, 3)
    assert a2.training_state(tmp_path, target) == "missing"
    root = a2.run_root(tmp_path, target)
    root.mkdir(parents=True)
    assert a2.training_state(tmp_path, target) == "partial"
    (root / "run_manifest.json").write_text(json.dumps({"status": "complete", "metrics_roles": ["validation"]}))
    assert a2.training_state(tmp_path, target) == "partial"  # run validation report still missing
    (root / "validation_report.json").write_text("{}")
    assert a2.training_state(tmp_path, target) == "trained"
    (root / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "metrics_roles": ["validation", "test"]}))
    assert a2.training_state(tmp_path, target) == "tested"
    shutil.rmtree(root)
    assert a2.training_state(tmp_path, target) == "missing"


def test_frozen_split_accepts_a2_variants_but_no_other_new_variant(tmp_path):
    from sleeptcn.dataset import POST_SPLIT_VARIANTS, paths_for_role

    assert set(_config()["shhs"]["new_variants"]) <= POST_SPLIT_VARIANTS
    assert {spec["data_variant"] for spec in _config()["training"]["arms"].values()} <= (
        POST_SPLIT_VARIANTS | {"bandpass_v2"})
    split = json.loads((ROOT / a2.SPLIT).read_text(encoding="utf-8"))
    assert not POST_SPLIT_VARIANTS & set(split["compatible_variants"])  # file not edited
    manifest = {"compatible_variants": ["filtered_v2"],
                "outer_runs": [{"outer_fold": 0, "train": {"record_keys": ["SC4001E"]}}]}
    for variant in ("filtered_zscore_rolling90_v3", "filtered_v2"):
        (tmp_path / variant).mkdir()
        (tmp_path / variant / "SC4001E.npz").write_bytes(b"")
        assert paths_for_role(tmp_path, manifest, 0, "train", variant) == [
            tmp_path / variant / "SC4001E.npz"]
    with pytest.raises(ValueError, match="not permitted"):
        paths_for_role(tmp_path, manifest, 0, "train", "filtered_zscore_v2")


def test_test_campaign_requires_confirmation_phrase(tmp_path):
    with pytest.raises(ValueError, match="confirmation"):
        a2.execute_test(tmp_path, _config(), [], device="cpu", num_workers=0, confirmation="yes")


def test_gpu_notebook_cells_compile_and_detect_training_processes(monkeypatch):
    import ast
    import subprocess

    notebook = json.loads((ROOT / "notebooks" / "a2_gpu_run.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    setup = code[0].split("assert Path(PY).is_file()")[0]
    namespace: dict = {}
    exec(setup, namespace)
    for name in ("states", "start_training", "wait_for_training", "test_campaign", "holdout", "step_benchmark",
                 "export"):
        assert callable(namespace[name]), name
    listings = {
        True: "/workspace/SleepTCN/.venv/bin/python scripts/run_amendment_a2.py train --device cuda --workers 3\n",
        False: "/workspace/SleepTCN/.venv/bin/python scripts/run_amendment_a2.py status\npython -m ipykernel\n",
    }
    for expected, listing in listings.items():
        monkeypatch.setattr(subprocess, "run", lambda *a, _l=listing, **k: SimpleNamespace(stdout=_l))
        assert namespace["training_running"]() is expected
    worker = "/x/python scripts/run_amendment_a2.py train-worker --index 0 --count 3 --device cuda\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=worker))
    assert namespace["training_running"]() is True
