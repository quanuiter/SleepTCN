"""Checks for scripts/analyze_holdout_confirmatory.py on synthetic predictions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_holdout_confirmatory as ah  # noqa: E402
from sleeptcn.metrics import metrics_from_confusion  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs" / "shhs_holdout_confirmatory_v1.json").read_text(encoding="utf-8"))
VARIANT = {a: ("filtered_v2" if a.startswith(("ADAST", "Source")) else PROTOCOL["arms"][a]["data_variant"]) for a in ah.ARMS}


def test_vectorised_metrics_match_reference() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        conf = rng.integers(0, 30, size=(5, 5))
        conf[rng.integers(0, 5)] = 0  # an absent reference class
        ref = metrics_from_confusion(conf)
        vec = ah.pooled_stats(conf)
        assert np.isclose(vec["macro_f1"], ref["macro_f1"]) and np.isclose(vec["cohen_kappa"], ref["cohen_kappa"])
        assert np.isclose(vec["accuracy"], ref["accuracy"])
        for i, s in enumerate(ah.STAGES):
            assert np.isclose(vec[f"{s.lower()}_recall"], ref["per_class"][s]["recall"])
            assert np.isclose(vec[f"{s.lower()}_f1"], ref["per_class"][s]["f1"])


def test_weights_reproduce_resampled_sums() -> None:
    rng = np.random.default_rng(1)
    conf = rng.integers(0, 9, size=(7, 5, 5))
    samples = rng.integers(0, 7, size=(5, 7))
    weights = ah.draw_weights(samples, 7)
    expected = np.stack([conf[s].sum(0) for s in samples])
    np.testing.assert_allclose((weights @ conf.reshape(7, 25)).reshape(-1, 5, 5), expected)


def make_world(tmp: Path, n: int = 30) -> dict[str, Path]:
    rng = np.random.default_rng(2)
    protocol = json.loads(json.dumps(PROTOCOL))
    protocol["cohort"]["n_target"] = n
    protocol["statistics"]["bootstrap"]["resamples"] = 200
    protocol_path = tmp / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    keys = [f"shhs1-9{i:05d}" for i in range(n)]
    records, truth = [], {}
    for key in keys:
        m = int(rng.integers(40, 60))
        y = rng.integers(0, 5, m).astype(np.int8)
        y[0] = -1
        truth[key] = y
        for variant in sorted(set(VARIANT.values())):
            path = tmp / "pre" / variant / f"{key}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, y=y, subject_id=np.array(key[6:]), original_epoch_index=np.arange(m, dtype=np.int32))
            records.append({"variant": variant, "record_key": key, "subject_id": key[6:], "output_path": str(path),
                            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pre = {"records": records, "summary": {"s6_zscore_input_std_in_window": {"median": 0.99, "q1": 0.85, "q3": 1.0, "n": n}}}
    pre_path = tmp / "pre.json"
    pre_path.write_text(json.dumps(pre), encoding="utf-8")
    pre_index = {(r["variant"], r["record_key"]): r["output_sha256"] for r in records}
    root = tmp / "inf"
    arm_hash = {}
    for arm in ah.ARMS:
        accuracy = {"E3_s42": 0.8, "E0_s42": 0.5}.get(arm, 0.6)
        entries = []
        for key in keys:
            y = truth[key]
            pred = np.where(rng.random(len(y)) < accuracy, np.maximum(y, 0), rng.integers(0, 5, len(y))).astype(np.int8)
            path = root / "ensemble" / arm / f"{key}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = {"arm": arm, "protocol_sha256": protocol_sha, "input_npz_sha256": pre_index[(VARIANT[arm], key)]}
            np.savez(path, metadata_json=np.array(json.dumps(meta)), prediction=pred,
                     original_epoch_index=np.arange(len(y), dtype=np.int32))
            entries.append({"record_key": key, "subject_id": key[6:], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = root / "arm_manifests" / f"{arm}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"ensemble_records": entries}), encoding="utf-8")
        arm_hash[arm] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "run_manifest.json").write_text(json.dumps({
        "status": "complete_all_arms_no_metrics_computed", "protocol_sha256": protocol_sha,
        "preprocess_manifest_sha256": hashlib.sha256(pre_path.read_bytes()).hexdigest(),
        "arm_manifest_sha256": arm_hash}), encoding="utf-8")
    return {"protocol": protocol_path, "pre": pre_path, "root": root}


def test_end_to_end_report_is_write_once(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    out = tmp_path / "out" / "analysis.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "analyze_holdout_confirmatory.py"), "--protocol", str(world["protocol"]),
           "--preprocess-manifest", str(world["pre"]), "--inference-root", str(world["root"]), "--output", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    h1 = report["confirmatory_family"][0]
    assert h1["hypothesis"] == "H1" and h1["contrast"] == "E3_s42 - E0_s42"
    assert h1["subject_mean_macro_f1_difference"] > 0 and h1["confirmed"]
    assert h1["wilcoxon"]["holm_family_size"] == 3
    assert len(report["S2_null_pair_and_run_to_run"]) == 6 and "wilcoxon" not in report["S2_null_pair_and_run_to_run"][0]
    assert set(report["S4_n3_under_detection"]) == {"E0_s42", "E3_s42", "E4_s42", "E6_s42"}
    assert report["subjects"] == 30 and out.with_suffix(".md").exists()
    again = subprocess.run(cmd, capture_output=True, text=True)
    assert again.returncode != 0 and "exactly once" in again.stderr


def test_rejects_tampered_prediction(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    victim = next((world["root"] / "ensemble" / "E3_s42").glob("*.npz"))
    with np.load(victim) as z:
        data = dict(z)
    data["prediction"] = np.zeros_like(data["prediction"])
    np.savez(victim, **data)
    cmd = [sys.executable, str(ROOT / "scripts" / "analyze_holdout_confirmatory.py"), "--protocol", str(world["protocol"]),
           "--preprocess-manifest", str(world["pre"]), "--inference-root", str(world["root"]),
           "--output", str(tmp_path / "o" / "a.json")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and "ensemble hash differs" in result.stderr
