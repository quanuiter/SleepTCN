"""Checks for scripts/run_holdout_amendment_a1.py: AdaBN statistics, input scaling and the full
prepare -> gate -> worker -> finalize -> analyze sequence on a tiny synthetic cohort (random models)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_holdout_amendment_a1 as a1  # noqa: E402
import run_holdout_inference as rhi  # noqa: E402
from sleeptcn.models import EEGResNet1D, SleepTCN  # noqa: E402


def random_models(seed: int = 0):
    torch.manual_seed(seed)
    extractor = EEGResNet1D()
    for m in extractor.modules():  # non-trivial stored statistics
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
    return extractor.eval(), SleepTCN(128).eval()


def test_batchnorm_order_starts_at_stem_and_has_ten_layers():
    extractor, _ = random_models()
    order = a1.batchnorm_order(extractor)
    assert order[0] == "stem.1" and len(order) == 10 and len(set(order)) == 10


def test_adabn_sets_exact_layer_input_statistics():
    extractor, _ = random_models()
    sample = (np.random.default_rng(1).standard_normal((48, 3000)) * 30).astype(np.float32)
    a1.adapt_batchnorm(extractor, sample, batch_size=16)  # chunked accumulation
    recorded = {}
    hooks = [m.register_forward_pre_hook(lambda _m, i, name=name: recorded.__setitem__(name, i[0].double()))
             for name, m in extractor.named_modules() if isinstance(m, torch.nn.BatchNorm1d)]
    with torch.no_grad():
        extractor.eval().extract_features(torch.from_numpy(sample).unsqueeze(1))
    for h in hooks:
        h.remove()
    modules = dict(extractor.named_modules())
    for name, x in recorded.items():
        torch.testing.assert_close(modules[name].running_mean.double(), x.mean((0, 2)), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(modules[name].running_var.double(), x.var((0, 2), unbiased=False), rtol=1e-4, atol=1e-5)


def test_adabn_removes_a_global_input_scale():
    sample = (np.random.default_rng(2).standard_normal((32, 3000)) * 30).astype(np.float32)
    x = (np.random.default_rng(3).standard_normal((20, 3000)) * 30).astype(np.float32)
    e1, s1 = random_models(5)
    e2, s2 = random_models(5)
    a1.adapt_batchnorm(e1, sample, 16)
    a1.adapt_batchnorm(e2, sample * 2, 16)
    p1 = a1.predict(e1, s1, x, 1.0, 8)
    p2 = a1.predict(e2, s2, x, 2.0, 8)
    np.testing.assert_allclose(p1, p2, atol=1e-4)


def test_predict_factor_one_matches_locked_inference_path():
    from sleeptcn.shhs_zero_shot import ExternalRecord, infer_probabilities

    extractor, sequence = random_models(7)
    x = (np.random.default_rng(4).standard_normal((15, 3000)) * 30).astype(np.float32)
    record = ExternalRecord(path=Path("r"), input_sha256="x", record_key="r", subject_id="1", role="holdout_test",
                            variant="bandpass_v2", x=x, y=np.zeros(15, dtype=np.int8), valid_mask=np.ones(15, bool),
                            original_epoch_index=np.arange(15))
    locked = infer_probabilities(record, "resnet1d", extractor, sequence, "tcn", 8)
    np.testing.assert_allclose(a1.predict(extractor, sequence, x, 1.0, 8), locked, atol=1e-6)
    assert not np.allclose(a1.predict(extractor, sequence, x, 2.0, 8), locked, atol=1e-4)


def test_sample_indices_are_deterministic_sorted_and_unique():
    a = a1.sample_indices("200157", 900, 40)
    assert np.array_equal(a, a1.sample_indices("200157", 900, 40))
    assert len(np.unique(a)) == 40 and np.all(np.diff(a) > 0) and a.max() < 900
    assert not np.array_equal(a, a1.sample_indices("200158", 900, 40))


# --------------------------------------------------------------------------- end to end
def write_npz(path: Path, x: np.ndarray, y: np.ndarray, key: str, variant: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, y=y, record_key=np.array(key), subject_id=np.array(key.split("-")[1]),
             role=np.array("holdout_test"), preprocess_version=np.array(variant),
             original_epoch_index=np.arange(5, 5 + len(x), dtype=np.int32))
    return {"record_key": key, "subject_id": key.split("-")[1], "variant": variant, "output_path": str(path),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture()
def cohort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    protocol = json.loads((ROOT / "configs" / "shhs_holdout_confirmatory_v1.json").read_text(encoding="utf-8"))
    protocol["cohort"]["n_target"] = 3
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    keys = ["shhs1-900001", "shhs1-900002", "shhs1-900003"]
    processed = tmp_path / "processed"
    records = []
    for i, key in enumerate(keys):
        rng = np.random.default_rng(i)
        n = 45 + i
        x = (rng.standard_normal((n, 3000)) * 30).astype(np.float32)
        y = rng.integers(0, 5, n).astype(np.int8)
        y[0] = -1
        for variant in ("filtered_v2", "bandpass_v2"):
            records.append(write_npz(processed / variant / f"{key}.npz", x, y, key, variant))
    pre = {"status": "complete", "protocol_sha256": protocol_sha, "analysis_cohort": [k.split("-")[1] for k in keys],
           "records": records}
    pre_file = tmp_path / "pre.json"
    pre_file.write_text(json.dumps(pre), encoding="utf-8")
    pre_sha = hashlib.sha256(pre_file.read_bytes()).hexdigest()

    amendment = json.loads((ROOT / "configs" / "shhs_holdout_amendment_a1.json").read_text(encoding="utf-8"))
    amendment["parent_protocol"]["config_sha256"] = protocol_sha
    amendment["cohort"]["preprocess_manifest_sha256"] = pre_sha
    amendment["cohort"]["participants"] = 3
    amendment["adabn"]["sample"]["total_epochs"] = 120
    amendment_file = tmp_path / "amendment.json"
    amendment_file.write_text(json.dumps(amendment), encoding="utf-8")

    edf = tmp_path / "edf"
    edf.mkdir()
    for i in range(153):
        np.savez(edf / f"SC4{i:03d}E.npz", x=(np.random.default_rng(100 + i).standard_normal((2, 3000)) * 20).astype(np.float32))

    def fake_models(args, fold):
        extractor, sequence = random_models(fold)
        return extractor, sequence, {"extractor_sha256": f"e{fold}", "sequence_checkpoint_sha256": f"s{fold}"}

    def fake_gate(args, ctx, arms):
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "checkpoint_gate.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
        return {"status": "passed"}

    monkeypatch.setattr(a1, "load_models", fake_models)
    monkeypatch.setattr(rhi, "checkpoint_gate", fake_gate)

    # Parent run: stored E4_s42 fold predictions (from the same fake models) and parent ensembles.
    parent = tmp_path / "parent"
    bandpass = {r["record_key"]: r for r in records if r["variant"] == "bandpass_v2"}
    filtered = {r["record_key"]: r for r in records if r["variant"] == "filtered_v2"}
    for arm, variant_records, shift in (("E4_s42", bandpass, 0), ("ADAST_full_e30", filtered, 1),
                                        ("SourceOnly_full_e30", filtered, 2)):
        ensemble_records = []
        for key, rec in variant_records.items():
            with np.load(rec["output_path"]) as z:
                x = z["x"]
            if arm == "E4_s42":
                for fold in a1.FOLDS:
                    e, s, _ = fake_models(None, fold)
                    p = a1.predict(e, s, x, 1.0, 16)
                    path = rhi.fold_path(parent, arm, fold, key)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, probabilities=p)
            pred = (np.arange(len(x)) + shift) % 5
            path = rhi.ensemble_path(parent, arm, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = {"arm": arm, "protocol_sha256": protocol_sha, "input_npz_sha256": rec["output_sha256"]}
            np.savez(path, metadata_json=np.array(json.dumps(meta)), prediction=pred.astype(np.int8),
                     original_epoch_index=np.arange(5, 5 + len(x), dtype=np.int32))
            ensemble_records.append({"record_key": key, "subject_id": rec["subject_id"],
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (parent / "arm_manifests").mkdir(parents=True, exist_ok=True)
        (parent / "arm_manifests" / f"{arm}.json").write_text(json.dumps({"ensemble_records": ensemble_records}),
                                                              encoding="utf-8")

    args = argparse.Namespace(amendment=amendment_file, protocol=protocol_file, preprocess_manifest=pre_file,
                              processed_root=processed, sleep_edf_root=edf, restore_workspace=tmp_path / "restore",
                              adast_backup=tmp_path / "adast", parent_inference_root=parent,
                              output_root=tmp_path / "out", batch_size=16, deviations_file=None,
                              worker_index=0, num_workers=1)
    return args


def test_end_to_end_sequence(cohort):
    args = cohort
    ctx = a1.load_context(args)
    with pytest.raises(ValueError):
        a1.require_gate(args, ctx)
    prep = a1.prepare(args, ctx)
    assert prep["adabn_sample_epochs"] == 120 and prep["labels_read"] is False
    assert a1.prepare(args, ctx) == prep  # idempotent
    gate = a1.gate(args, ctx)
    assert gate["status"] == "passed"
    assert all(c["max_abs_difference"] <= 1e-5 for c in gate["reproduction_of_parent_E4_s42"])
    units = a1.all_units(ctx)
    assert len(units) == 50
    for condition, fold in units[1::2] + units[0::2]:  # two "workers"
        a1.run_unit(args, ctx, condition, fold)
    assert a1.progress(args, ctx) == (150, 150)
    assert a1.run_unit(args, ctx, "adabn", 0) == 0  # resumable, nothing redone
    a1.finalize(args, ctx)
    report = a1.analyze(args, ctx)
    assert set(report["contrasts_vs_E4_s42"]) == set(ctx["condition_names"])
    assert set(report["descriptive"]) == {"E4_s42", "ADAST_full_e30", "SourceOnly_full_e30", *ctx["condition_names"]}
    assert len(report["scale_curve"]["n3_recall"]) == 5
    with pytest.raises(FileExistsError):
        a1.analyze(args, ctx)
