"""Synthetic checks for the A2 data checks, cost decomposition, holdout ensembling and analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import analyze_amendment_a2 as an  # noqa: E402
from sleeptcn import amendment_a2 as a2  # noqa: E402
from sleeptcn import amendment_a2_cost as cost  # noqa: E402
from sleeptcn import amendment_a2_holdout as hold  # noqa: E402
from sleeptcn.amendment_a2_data import VARIANT, compare_with_reference  # noqa: E402


def _config() -> dict:
    return json.loads((ROOT / a2.A2_CONFIG).read_text(encoding="utf-8"))


def _npz(path: Path, **fields) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fields)
    return path


def _record(n: int, variant: str, scope: str | None = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y = rng.integers(-1, 5, n).astype(np.int8)
    fields = {"x": rng.standard_normal((n, 3000)).astype(np.float32), "y": y, "valid_mask": y >= 0,
              "original_epoch_index": np.arange(10, 10 + n, dtype=np.int32), "record_key": np.array("SC4001E"),
              "subject_id": np.array("SC400"), "preprocess_version": np.array(variant)}
    if scope:
        fields["normalization_scope"] = np.array(scope)
    return fields


# --------------------------------------------------------------------------- data identity check
def test_identity_check_accepts_same_epochs_and_rejects_shifted_labels(tmp_path):
    reference = _npz(tmp_path / "ref.npz", **_record(20, "filtered_v2"))
    good = _npz(tmp_path / "new.npz", **{**_record(20, VARIANT, "centred_rolling_window_after_filter_clip"),
                                         "x": np.ones((20, 3000), np.float32)})
    result = compare_with_reference(good, reference)
    assert result["passed"] and result["x_std_in_window"] == 0.0
    shifted = _record(20, VARIANT, "centred_rolling_window_after_filter_clip")
    shifted["y"] = np.roll(shifted["y"], 1)
    bad = _npz(tmp_path / "bad.npz", **shifted)
    assert not compare_with_reference(bad, reference)["checks"]["y"]
    unscoped = _npz(tmp_path / "unscoped.npz", **_record(20, VARIANT))
    assert not compare_with_reference(unscoped, reference)["checks"]["scope"]


# --------------------------------------------------------------------------- analysis E
def test_wall_clock_parsing_and_decomposition(monkeypatch):
    def fake_git(_workspace, path):
        lines = []
        clock = 0
        for config, minutes in (("E0", 200), ("E1", 1), ("E2", 15), ("E3", 20), ("E4", 18), ("E6", 14)):
            lines.append(f"START {config} 2026-08-20T{clock // 60:02d}:{clock % 60:02d}:00+00:00")
            clock += minutes
            lines.append(f"DONE {config} 2026-08-20T{clock // 60:02d}:{clock % 60:02d}:00+00:00")
        return "\n".join(lines).encode()

    monkeypatch.setattr(cost, "_git_bytes", fake_git)
    wall = cost.archived_wall_clock(ROOT)
    assert wall["total_seconds"]["E0"] == 10 * 200 * 60
    assert wall["total_seconds"]["E3"] == 10 * 20 * 60
    rows = [{"config": c, "fold": f, "stage": s, "kind": cost.stage_kind(s), "completed_epochs": 3,
             "validation_events": 3, "optimizer_steps": 100 if c == "E0" else 50, "best_epoch": 1}
            for c in cost.CONFIGS for f in cost.FOLDS for s in cost.stage_names(c)]
    bench = {"gpu": "Tesla V100-SXM2-16GB", "kinds": {k: {"mean_seconds_per_step": t} for k, t in
                                                       (("cnn15", 0.01), ("bilstm", 0.02), ("resnet1d", 0.05),
                                                        ("tcn", 0.03))}}
    result = cost.decompose({"rows": rows}, wall, bench)
    e0_steps = 10 * (15 * 100 + 100)
    assert result["E0"]["optimizer_steps"] == e0_steps
    assert result["E0"]["step_compute_seconds_on_benchmark_device"] == pytest.approx(10 * (15 * 100 * 0.01 + 100 * 0.02))
    assert result["E3"]["ratios_vs_E0"]["observed_wall_clock"] == pytest.approx(10.0)
    assert result["E3"]["ratios_vs_E0"]["optimizer_steps"] == pytest.approx(e0_steps / 1000)
    assert "remainder_seconds" in result["E3"]
    assert "remainder_seconds" not in cost.decompose({"rows": rows}, wall, {**bench, "gpu": "A100"})["E3"]


def test_step_benchmark_runs_the_training_loop_on_cpu(monkeypatch):
    monkeypatch.setattr(cost, "WARMUP_STEPS", 1)
    monkeypatch.setattr(cost, "TIMED_STEPS", 2)
    monkeypatch.setattr(cost, "ROUNDS", 2)
    monkeypatch.setattr(cost, "fold0_training_lengths", lambda _workspace: [7, 5, 9, 6, 8, 4, 10, 3, 11])
    report = cost.benchmark_step_times(ROOT, "cpu")
    assert set(report["kinds"]) == {"cnn15", "resnet1d", "bilstm", "tcn"}
    for values in report["kinds"].values():
        assert len(values["round_means"]) == 2 and values["mean_seconds_per_step"] > 0


# --------------------------------------------------------------------------- holdout ensembling
def test_holdout_arms_follow_run_order():
    names = [arm.name for arm in hold.holdout_arms(_config())]
    assert names == ["E6R_s42", "E6R_s123", "E4_s2025", "E4_s2027", "E4_s2029"]
    assert len(hold.units(_config())) == 50


def test_finalize_averages_fold_probabilities(tmp_path, monkeypatch):
    config = _config()
    config["status"] = a2.LOCKED_STATUS
    (tmp_path / a2.A2_CONFIG).parent.mkdir(parents=True)
    (tmp_path / a2.A2_CONFIG).write_text(json.dumps(config), encoding="utf-8")
    _, sha = a2.load_a2_config(tmp_path)
    rng = np.random.default_rng(1)
    entries = {v: [{"record_key": "shhs1-200001", "subject_id": "200001", "sha256": f"in-{v}"}]
               for v in ("bandpass_v2", VARIANT)}
    inputs = {"entries": entries, "parent_manifest_sha256": "p", "a2_manifest_sha256": "a"}
    stored = {}
    for arm in hold.holdout_arms(config):
        for fold in hold.FOLDS:
            p = rng.dirichlet(np.ones(5), size=6).astype(np.float32)
            stored[(arm.name, fold)] = p
            meta = {"input_npz_sha256": f"in-{arm.variant}", "arm": arm.name, "outer_fold": fold,
                    "a2_config_sha256": sha, "device": "cpu"}
            _npz(hold.fold_path(tmp_path, arm.name, fold, "shhs1-200001"), metadata_json=np.array(json.dumps(meta)),
                 probabilities=p, original_epoch_index=np.arange(6, dtype=np.int32))
    manifest = hold.finalize(tmp_path, inputs)
    assert set(manifest["arm_manifest_sha256"]) == {a.name for a in hold.holdout_arms(config)}
    with np.load(hold.ensemble_path(tmp_path, "E4_s2025", "shhs1-200001")) as z:
        expected = np.mean([stored[("E4_s2025", f)].astype(np.float64) for f in hold.FOLDS], axis=0)
        np.testing.assert_allclose(z["probabilities"], expected.astype(np.float32))
        np.testing.assert_array_equal(z["prediction"], expected.argmax(1))


# --------------------------------------------------------------------------- analyses
class FakeHoldout:
    def __init__(self, n_records: int = 12, epochs: int = 40, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.keys = [f"shhs1-2{i:05d}" for i in range(n_records)]
        self.subjects = np.asarray([k.split("-")[1] for k in self.keys])
        self.y = {k: rng.integers(0, 5, epochs).astype(np.int8) for k in self.keys}
        self.index = {k: np.arange(epochs, dtype=np.int32) for k in self.keys}

    def confusion(self, key, prediction):
        return an.confusion_matrix_5(self.y[key], prediction)


def test_analysis_d_on_synthetic_probabilities(tmp_path, monkeypatch):
    holdout = FakeHoldout()
    rng = np.random.default_rng(3)
    folds = [{k: rng.dirichlet(np.ones(5), size=40) for k in holdout.keys} for _ in an.FOLDS]
    fits = tmp_path / "fits"
    fits.mkdir()
    for f in an.FOLDS:
        (fits / f"fold_{f:02d}.json").write_text(json.dumps({
            "fold": f, "temperature": 1.3, "source_prior": [0.3, 0.15, 0.33, 0.07, 0.15],
            "target_prior": [0.3, 0.15, 0.33, 0.07, 0.15], "target_labels_used": False}))
    a1 = tmp_path / "a1"
    a1.mkdir()
    keys = np.repeat(holdout.keys, 2)
    np.savez(a1 / "adabn_sample.npz", record_key=keys, original_epoch_index=np.tile([3, 7], len(holdout.keys)))
    (a1 / "prepare_manifest.json").write_text(json.dumps({"adabn_sample_sha256": an.sha256_file(a1 / "adabn_sample.npz")}))
    monkeypatch.setattr(an, "CALIBRATION_FITS", fits)
    monkeypatch.setattr(an, "A1_ROOT", a1)
    monkeypatch.setattr(an, "load_fold_probabilities", lambda _h, arm: folds)
    monkeypatch.setattr(an, "spec_total", lambda _c: 2 * len(holdout.keys))
    monkeypatch.setattr(an, "PARENT_INFERENCE", tmp_path)
    monkeypatch.setattr(an, "SUBSET_SIZES", (2, 3, 5))
    monkeypatch.setattr(an, "SUBSET_DRAWS", 4)
    (tmp_path / "arm_manifests").mkdir()
    (tmp_path / "arm_manifests" / "E3_s123.json").write_text("{}")
    report = an.analysis_d(holdout, _config(), "sha", 100, 1)
    conditions = report["conditions"]
    assert {"raw_ensemble", "temperature_only", "original_five", "adabn_sample", "all_500", "per_record",
            "random_subsets"} <= set(conditions)
    # A target prior equal to the source prior is no adjustment at all.
    for stat in ("subject_mean_macro_f1", "n3_recall", "n3_prediction_share"):
        assert conditions["original_five"][stat] == pytest.approx(conditions["temperature_only"][stat])
    assert set(conditions["random_subsets"]["by_size"]) == {"2", "3", "5"}
    assert an.SUBSET_SIZES == (2, 3, 5)
    assert report["interpretation"]["outcome"] in {"label_shift_assumption_fails", "small_adaptation_sample",
                                                   "neither_rule_met"}
    assert 0 < report["observed_holdout_n3_share"] < 1


def _diagonal_confusions(n: int, accuracy: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 5, 5), dtype=np.int64)
    for i in range(n):
        for c in range(5):
            correct = rng.binomial(40, accuracy)
            out[i, c, c] = correct
            out[i, c, (c + 1) % 5] = 40 - correct
    return out


def test_interpretation_rules_for_a_and_b(monkeypatch):
    holdout = FakeHoldout(n_records=30)
    levels = {"E4": 0.80, "E3": 0.79, "E6": 0.70, "E6R": 0.60, "E0": 0.55, "E2": 0.54}
    monkeypatch.setattr(an, "load_ensemble", lambda _h, arm: _diagonal_confusions(
        30, levels[arm.split("_s")[0]] + (0.001 * (hash(arm) % 3) if arm.startswith("E4") else 0), seed=len(arm)))
    monkeypatch.setattr(an, "sleepedf_confusion", lambda arm, seed: {"pooled": {"macro_f1": np.array(0.78)}})
    monkeypatch.setattr(an, "read_json", lambda path: {"input_scale_in_window": {}})
    report_a = an.analysis_a(holdout, _config(), "sha", 200, 1)
    assert report_a["interpretation"]["keep_calibrated_amplitude_statement"] is True
    assert set(report_a["contrasts"]) == {f"{l}_s{s} - {r}_s{s}" for s in (42, 123)
                                          for l, r in (("E4", "E6R"), ("E3", "E6R"), ("E6R", "E6"))}
    report_b = an.analysis_b(holdout, _config(), "sha", 200, 1)
    assert len(report_b["between_seed_differences"]) == 10
    assert len(report_b["configuration_contrasts"]) == 20
    assert report_b["interpretation"]["configuration_contrasts_exceed_training_variability"] is True
    assert report_b["summary"]["holdout_subject_mean_macro_f1"]["sd"] >= 0
