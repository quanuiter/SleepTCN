import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adast_pilot_test", ROOT / "scripts/run_adast_cpu_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
UPSTREAM = ROOT / "runs/teacher_revision_cpu_20261001/adast_upstream"


@pytest.mark.skipif(not (UPSTREAM / "models/models.py").exists(), reason="optional pinned upstream checkout absent")
def test_matched_initialization_training_resume_and_no_target_source_only(tmp_path):
    torch.set_num_threads(1)
    module = pilot.load_module("test_adast_original_models", UPSTREAM / "models/models.py")
    upstream_cfg = pilot.load_module("test_adast_original_cfg", UPSTREAM / "config_files/configs.py").Config()
    utils = pilot.load_module("test_adast_original_utils", UPSTREAM / "utils.py")
    cfg = json.loads(pilot.PROTOCOL.read_bytes())
    cfg.update(rounds=2, epochs_per_round=1, steps_per_epoch=1, batch_size=4)
    x = np.random.default_rng(1).normal(size=(8, 3000)).astype(np.float32)
    y = np.arange(8, dtype=np.int64) % 5
    target = np.random.default_rng(2).normal(size=(8, 3000)).astype(np.float32)
    models = pilot.build_models(module, upstream_cfg, 123)
    initial = pilot.model_digest(models)
    torch.manual_seed(987)
    other = pilot.build_models(module, upstream_cfg, 123)
    assert initial == pilot.model_digest(other)

    def forbidden(*args, **kwargs):
        raise AssertionError("source-only must never forward through target attention")

    handle = models["target_attention"].register_forward_pre_hook(forbidden)
    left = pilot.train_arm("source_only", models, utils, x, y, target, cfg, tmp_path / "source_only")
    handle.remove()
    right = pilot.train_arm("adast", other, utils, x, y, target, cfg, tmp_path / "adast")
    assert left["initial_state_sha256"] == right["initial_state_sha256"]
    assert left["updates"] == right["updates"] == 2
    assert [r["source_order_sha256"] for r in left["history"]] == [r["source_order_sha256"] for r in right["history"]]
    assert len(list((tmp_path / "adast").glob("pseudo_round_*.npy"))) == 2
    fresh = pilot.build_models(module, upstream_cfg, 123)
    replay = pilot.train_arm("adast", fresh, utils, x, y, target, cfg, tmp_path / "adast")
    assert replay == right
    assert pilot.model_digest(fresh) == pilot.model_digest(other)
    np.testing.assert_array_equal(pilot.predict(fresh, target, "target"), pilot.predict(other, target, "target"))


def test_fixed_protocol_has_matched_update_budget_and_no_target_selection():
    cfg = json.loads(pilot.PROTOCOL.read_bytes())
    assert cfg["total_updates_per_arm"] == cfg["rounds"] * cfg["epochs_per_round"] * cfg["steps_per_epoch"] == 1140
    assert cfg["steps_per_epoch"] == 4989 // cfg["batch_size"]
    assert cfg["selection"] == "final_fixed_budget_checkpoint_no_source_or_target_score_selection"


@pytest.mark.skipif(not (UPSTREAM / "models/models.py").exists(), reason="optional pinned upstream checkout absent")
def test_epoch_boundary_resume_replays_uninterrupted_training(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    module = pilot.load_module("test_resume_adast_models", UPSTREAM / "models/models.py")
    upstream_cfg = pilot.load_module("test_resume_adast_cfg", UPSTREAM / "config_files/configs.py").Config()
    utils = pilot.load_module("test_resume_adast_utils", UPSTREAM / "utils.py")
    cfg = json.loads(pilot.PROTOCOL.read_bytes())
    cfg.update(rounds=2, epochs_per_round=2, steps_per_epoch=1, batch_size=4)
    x = np.random.default_rng(1).normal(size=(8, 3000)).astype(np.float32)
    y = np.arange(8, dtype=np.int64) % 5
    target = np.random.default_rng(2).normal(size=(8, 3000)).astype(np.float32)
    full = pilot.build_models(module, upstream_cfg, 123)
    pilot.train_arm("adast", full, utils, x, y, target, cfg, tmp_path / "full")
    partial = pilot.build_models(module, upstream_cfg, 123)
    update = pilot.update
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption at round boundary")
        return update(*args, **kwargs)

    monkeypatch.setattr(pilot, "update", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pilot.train_arm("adast", partial, utils, x, y, target, cfg, tmp_path / "resumed")
    monkeypatch.setattr(pilot, "update", update)
    resumed = pilot.build_models(module, upstream_cfg, 123)
    result = pilot.train_arm("adast", resumed, utils, x, y, target, cfg, tmp_path / "resumed")
    assert result["updates"] == 4
    assert pilot.model_digest(full) == pilot.model_digest(resumed)
    for round_index in [0, 1]:
        np.testing.assert_array_equal(np.load(tmp_path / "full" / f"pseudo_round_{round_index}.npy"),
                                      np.load(tmp_path / "resumed" / f"pseudo_round_{round_index}.npy"))
