import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from sleeptcn.revision_weighted import train_arm, state_digest

loader = importlib.util.spec_from_file_location("revision_cpu_trial", Path(__file__).resolve().parents[1] / "scripts/run_revision_weighted_cpu_trial.py")
trial = importlib.util.module_from_spec(loader)
loader.loader.exec_module(trial)


def test_wall_clock_and_projection_limits(monkeypatch):
    now = [0.]
    monkeypatch.setattr(trial.time, "perf_counter", lambda: now[0])
    guard = trial.RuntimeGuard(10)
    now[0] = 11
    with pytest.raises(trial.ResourceStop, match="wall-clock"):
        guard()
    now[0] = 0
    guard = trial.RuntimeGuard(18000)
    guard.feature_seconds = 180
    with pytest.raises(trial.ResourceStop, match="projects"):
        guard({"phase": "epoch_saved", "history": [{"seconds": 300}] * 5})
    assert guard.last_projection["projected_remaining_feature_train_validation_hours"] > 5
    with pytest.raises(ValueError):
        trial.RuntimeGuard(18001)


def test_budget_stop_retains_checkpoint_and_resumes_exactly(tmp_path):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(5)
    train = [(torch.randn(8, 128, generator=generator), torch.arange(8) % 5)]
    val = [(torch.randn(10, 128, generator=generator), torch.arange(10) % 5)]
    cfg = {"seed": 123, "learning_rate": .0005, "max_epochs": 3, "patience": 3,
           "batch_size_records": 2, "gradient_clip_norm": 1.}
    identity = {"fold": 1, "synthetic": True}
    expected, _ = train_arm(train, val, np.ones(5), cfg, tmp_path / "baseline", identity)

    def stop(progress):
        if progress["phase"] == "epoch_saved":
            raise trial.ResourceStop("test budget exceeded")

    with pytest.raises(trial.ResourceStop):
        train_arm(train, val, np.ones(5), cfg, tmp_path / "stopped", identity, runtime_guard=stop)
    assert (tmp_path / "stopped/latest.pt").exists()
    assert not (tmp_path / "stopped/selection.json").exists()
    saved = torch.load(tmp_path / "stopped/latest.pt", weights_only=True)
    assert saved["epoch"] == 1
    actual, _ = train_arm(train, val, np.ones(5), cfg, tmp_path / "stopped", identity)
    assert state_digest(actual) == state_digest(expected)
