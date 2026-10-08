import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "weighted_pilot", Path(__file__).resolve().parents[1] / "scripts/run_weighted_tcn_cpu_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_matched_initialization_and_selected_checkpoint_replay(tmp_path):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    train = [(torch.randn(12, 128, generator=generator), torch.arange(12) % 5)]
    validation = [(torch.randn(10, 128, generator=generator), torch.arange(10) % 5)]
    cfg = {"seed": 123, "learning_rate": .0005, "max_epochs": 2, "patience": 1,
           "batch_size_records": 8, "gradient_clip_norm": 1.}
    model, left = pilot.train_arm(train, validation, np.ones(5), cfg, tmp_path / "left")
    torch.manual_seed(987)
    other, right = pilot.train_arm(train, validation, np.ones(5), cfg, tmp_path / "right")
    assert left["initial_state_sha256"] == right["initial_state_sha256"]
    assert pilot.state_digest(model) == pilot.state_digest(other)
    score = pilot.evaluate_source(model, validation, 8)["macro_f1"]
    assert score == pytest.approx(left["selected_validation_macro_f1"])
    payload = torch.load(tmp_path / "left/best.pt", weights_only=True)
    assert payload["epoch"] == left["selected_epoch"]
    assert payload["selection"] == "source_validation_macro_f1"
