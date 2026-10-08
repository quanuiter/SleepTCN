import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from sleeptcn.revision_campaign import mean_ten, write_once_json
from sleeptcn import revision_weighted as training


@pytest.fixture
def inputs():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(19)
    train = [(torch.randn(n, 128, generator=generator), torch.arange(n) % 5) for n in [9, 11, 7]]
    val = [(torch.randn(10, 128, generator=generator), torch.arange(10) % 5)]
    cfg = {"seed": 123, "learning_rate": .0005, "max_epochs": 4, "patience": 4,
           "batch_size_records": 2, "gradient_clip_norm": 1.}
    return train, val, cfg


def test_ten_fold_membership_and_float64_accumulation():
    parts = {i: np.full((3, 5), .2, dtype=np.float32) for i in range(10)}
    np.testing.assert_array_equal(mean_ten(parts), np.full((3, 5), .2, dtype=np.float32))
    with pytest.raises(ValueError):
        mean_ten({k: v for k, v in parts.items() if k != 9})
    parts[9] = np.ones((2, 5))
    with pytest.raises(ValueError):
        mean_ten(parts)


def test_write_once_rejects_mutated_manifest(tmp_path):
    path = tmp_path / "spec.json"
    write_once_json(path, {"fold": 0})
    write_once_json(path, {"fold": 0})
    with pytest.raises(ValueError):
        write_once_json(path, {"fold": 1})


def test_restart_matches_uninterrupted_training(inputs, tmp_path, monkeypatch):
    train, val, cfg = inputs
    identity = {"fold": 1, "source_only_selection": True}
    baseline, expected = training.train_arm(train, val, np.ones(5), cfg, tmp_path / "baseline", identity)
    evaluate = training.evaluate_source
    calls = 0

    def interrupt(model, records, batch_size, device="cpu"):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption mid-epoch")
        return evaluate(model, records, batch_size, device)

    monkeypatch.setattr(training, "evaluate_source", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train_arm(train, val, np.ones(5), cfg, tmp_path / "resumed", identity)
    monkeypatch.setattr(training, "evaluate_source", evaluate)
    actual_model, actual = training.train_arm(train, val, np.ones(5), cfg, tmp_path / "resumed", identity)
    assert training.state_digest(actual_model) == training.state_digest(baseline)
    assert actual["selected_epoch"] == expected["selected_epoch"]
    assert [{k: v for k, v in r.items() if k != "seconds"} for r in actual["history"]] == [
        {k: v for k, v in r.items() if k != "seconds"} for r in expected["history"]]
    cached, cached_selection = training.train_arm(train, val, np.ones(5), cfg, tmp_path / "resumed", identity)
    assert training.state_digest(cached) == training.state_digest(baseline)
    assert cached_selection == actual
    with pytest.raises(ValueError, match="manifest differs"):
        training.train_arm(train, val, np.ones(5), {**cfg, "learning_rate": .001}, tmp_path / "resumed", identity)


def test_training_recipe_matches_original_fold_zero(inputs, tmp_path):
    loader = importlib.util.spec_from_file_location("original_weighted_pilot", Path(__file__).resolve().parents[1] / "scripts/run_weighted_tcn_cpu_pilot.py")
    original = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(original)
    train, val, cfg = inputs
    weights = np.array([.6, 1.8, .6, 3.2, 1.5])
    old_model, old = original.train_arm(train, val, weights, cfg, tmp_path / "old")
    new_model, new = training.train_arm(train, val, weights, cfg, tmp_path / "new", {"fold": 0})
    assert training.state_digest(old_model) == training.state_digest(new_model)
    assert old["selected_epoch"] == new["selected_epoch"]
    assert old["selected_validation_macro_f1"] == new["selected_validation_macro_f1"]


def test_calibration_auditor_rejects_corrupt_ensemble(tmp_path):
    import json
    from sleeptcn.calibration import apply_temperature, adjust_prior
    loader = importlib.util.spec_from_file_location("calibration_auditor", Path(__file__).resolve().parents[1] / "scripts/verify_revision_calibration_10fold.py")
    auditor = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(auditor)
    raw = np.random.default_rng(4).dirichlet(np.ones(5), size=(10, 3)).astype(np.float32)
    fits = [{"temperature": 1.2, "source_prior": np.ones(5)/5, "target_prior": np.array([.1, .2, .3, .1, .3])} for _ in range(10)]
    cal = [apply_temperature(p, 1.2) for p in raw]
    em = [adjust_prior(p, fits[0]["source_prior"], fits[0]["target_prior"]) for p in cal]
    entry = {"subject_id": "synthetic", "source_edf_sha256": "edf", "epochs": 3}
    meta = {"subject_id": "synthetic", "source_edf_sha256": "edf", "fitted_sha256": "fit", "specification_sha256": "spec"}
    values = {"fold_raw": raw, "raw": np.mean(raw.astype(np.float64), axis=0).astype(np.float32),
              "calibrated": np.mean(cal, axis=0).astype(np.float32), "calibrated_em": np.mean(em, axis=0).astype(np.float32),
              "original_epoch_index": np.arange(3), "metadata_json": np.array(json.dumps(meta))}
    path = tmp_path / "prediction.npz"
    np.savez(path, **values)
    auditor.audit_prediction(path, entry, fits, "fit", "spec")
    values["calibrated_em"][0] = .2
    np.savez(path, **values)
    with pytest.raises(AssertionError):
        auditor.audit_prediction(path, entry, fits, "fit", "spec")
