"""Orchestration checks for scripts/run_holdout_inference.py (models replaced by fakes)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_holdout_inference as rhi  # noqa: E402


def write_record(path: Path, key: str, variant: str, n: int, seed: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.savez(path, x=rng.standard_normal((n, 3000)).astype(np.float32), y=np.zeros(n, dtype=np.int8),
             record_key=np.array(key), subject_id=np.array(key.split("-")[1]), role=np.array("holdout_test"),
             preprocess_version=np.array(variant), original_epoch_index=np.arange(10, 10 + n, dtype=np.int32))
    return {"record_key": key, "subject_id": key.split("-")[1], "variant": variant, "output_path": str(path),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture()
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    protocol_path = ROOT / "configs" / "shhs_holdout_confirmatory_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["cohort"]["n_target"] = 3
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    processed = tmp_path / "processed"
    keys = ["shhs1-900001", "shhs1-900002", "shhs1-900003"]
    records = [write_record(processed / "filtered_v2" / f"{k}.npz", k, "filtered_v2", 40 + i, i) for i, k in enumerate(keys)]
    pre = {"status": "complete", "protocol_sha256": protocol_sha, "analysis_cohort": [k.split("-")[1] for k in keys],
           "records": records}
    pre_file = tmp_path / "pre.json"
    pre_file.write_text(json.dumps(pre), encoding="utf-8")

    calls = {"n": 0}

    def fake_zero_shot(args, spec, fold):
        def predict(record):
            calls["n"] += 1
            rng = np.random.default_rng(fold * 1000 + len(record["x"]))
            logits = rng.standard_normal((len(record["x"]), 5))
            e = np.exp(logits - logits.max(1, keepdims=True))
            return (e / e.sum(1, keepdims=True)).astype(np.float32)
        return predict, {"extractor_sha256": "e", "sequence_checkpoint_sha256": "s"}

    monkeypatch.setattr(rhi, "zero_shot_predictor", fake_zero_shot)
    args = argparse.Namespace(protocol=protocol_file, preprocess_manifest=pre_file, processed_root=processed,
                              output_root=tmp_path / "out", batch_size=8, adast_batch_size=8)
    return args, calls


def test_run_arm_writes_label_free_artifacts_and_resumes(setup) -> None:
    args, calls = setup
    ctx = rhi.load_inputs(args)
    summary = rhi.run_arm(args, ctx, "E3_s42")
    assert summary["fold_artifacts"] == 30 and summary["ensemble_artifacts"] == 3
    assert calls["n"] == 30
    with np.load(rhi.ensemble_path(args.output_root, "E3_s42", "shhs1-900002")) as z:
        assert "y" not in z.files and "valid_mask" not in z.files
        folds = []
        for fold in range(10):
            with np.load(rhi.fold_path(args.output_root, "E3_s42", fold, "shhs1-900002")) as f:
                assert "y" not in f.files
                folds.append(f["probabilities"].astype(np.float64))
        np.testing.assert_allclose(z["probabilities"], np.mean(folds, axis=0), atol=1e-6)
        np.testing.assert_array_equal(z["prediction"], z["probabilities"].argmax(1))
        np.testing.assert_array_equal(z["original_epoch_index"], np.arange(10, 51))
    # Resume: all fold artifacts exist, so no model is rebuilt and no prediction is recomputed.
    rhi.run_arm(args, ctx, "E3_s42")
    assert calls["n"] == 30


def test_run_arm_rejects_changed_input(setup) -> None:
    args, _ = setup
    ctx = rhi.load_inputs(args)
    rhi.run_arm(args, ctx, "E3_s42")
    entry = ctx["entries"]["filtered_v2"][0]
    np.savez(entry["output_path"], x=np.zeros((40, 3000), np.float32))
    rhi.fold_path(args.output_root, "E3_s42", 3, entry["record_key"]).unlink()
    with pytest.raises(ValueError, match="input hash differs"):
        rhi.run_arm(args, ctx, "E3_s42")


def test_inputs_reject_other_protocol(setup, tmp_path: Path) -> None:
    args, _ = setup
    pre = json.loads(args.preprocess_manifest.read_text(encoding="utf-8"))
    pre["protocol_sha256"] = "0" * 64
    args.preprocess_manifest.write_text(json.dumps(pre), encoding="utf-8")
    with pytest.raises(ValueError, match="another protocol"):
        rhi.load_inputs(args)


def test_workers_partition_all_units_without_overlap() -> None:
    units = rhi.all_units()
    assert len(units) == 12 * 10
    split = [rhi.worker_units(i, 4) for i in range(4)]
    flat = [u for part in split for u in part]
    assert sorted(flat) == sorted(units) and len(flat) == len(set(flat))
    assert units[0] == ("E3_s42", 0)


def test_finalize_requires_every_fold(setup) -> None:
    args, _ = setup
    ctx = rhi.load_inputs(args)
    for fold in range(9):
        rhi.run_unit(args, ctx, "E3_s42", fold)
    with pytest.raises(FileNotFoundError):
        rhi.finalize_arm(args, ctx, "E3_s42")
