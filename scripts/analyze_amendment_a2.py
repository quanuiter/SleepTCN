"""Descriptive analyses A-E of Amendment A2 (configs/shhs_holdout_amendment_a2.json).

Each analysis runs once, when its inputs exist, and writes a write-once report to
``runs/a2/analysis/<name>.json``. The pre-set interpretation rule of each analysis is
evaluated in code and stored with the numbers.

  d   EM sample size (stored parent E3_s123 fold probabilities; no A2 training needed)
  e   training-cost decomposition (needs runs/a2_step_time_benchmark.json from the GPU)
  c   ensemble versus single fold models (needs the A2 holdout inference)
  a   normalisation scope (needs the A2 holdout inference and Sleep-EDF test campaign)
  b   five E4 training seeds (same inputs as a)

Statistics reuse the parent holdout analysis: subject macro-F1 over five fixed classes,
one matrix of paired subject-cluster bootstrap draws (A2 seed 2043, 10,000 resamples),
percentile 95% intervals. No hypothesis test is made.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

import analyze_holdout_confirmatory as ahc  # noqa: E402
from sleeptcn.amendment_a2 import A2_ROOT, A2Target, load_a2_config, run_root, test_journal_path  # noqa: E402
from sleeptcn.amendment_a2_cost import archived_steps, archived_wall_clock, decompose  # noqa: E402
from sleeptcn.amendment_a2_cost import BENCHMARK_FILE  # noqa: E402
from sleeptcn.amendment_a2_data import (SHHS_MANIFEST, SLEEPEDF_MANIFEST, load_parent_manifest,  # noqa: E402
                                        parent_entries, verified_parent_path)
from sleeptcn.amendment_a2_holdout import HOLDOUT_ROOT, holdout_arms  # noqa: E402
from sleeptcn.calibration import apply_temperature, estimate_target_prior  # noqa: E402
from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.io.serialization import atomic_write_json, read_json  # noqa: E402
from sleeptcn.metrics import confusion_matrix_5  # noqa: E402
from sleeptcn.shhs_analysis import (_absolute_subject_bootstrap, _mean_paired_bootstrap,  # noqa: E402
                                    _subject_macro_f1)

PARENT_INFERENCE = ROOT / "runs" / "holdout_inference_v1"
A2_INFERENCE = ROOT / HOLDOUT_ROOT
OUTPUT = ROOT / A2_ROOT / "analysis"
GATE5 = {42: ROOT / "runs/v2/analysis/gate5_paired_results_seed42.json",
         123: ROOT / "runs/v2/analysis/gate5_paired_results_seed123.json"}
CALIBRATION_FITS = ROOT / "runs/teacher_revision_cpu_20261003/calibration_10fold_seed123/fits"
A1_ROOT = ROOT / "runs" / "holdout_amendment_a1"
PARENT_ZERO_SHOT = [f"{e}_s{s}" for e in ("E0", "E2", "E3", "E4", "E6") for s in (42, 123)]
PARENT_VARIANT = {"E0": "paper_raw_v1", "E2": "paper_raw_v1", "E3": "filtered_v2", "E4": "bandpass_v2",
                  "E6": "filtered_zscore_v2"}
FOLDS = tuple(range(10))
N3 = 3
SUBSET_SIZES = (5, 20, 100)
SUBSET_DRAWS = 200
SUBSET_SEED = 2044


# --------------------------------------------------------------------------- holdout labels
class Holdout:
    """Reference labels of the 500 holdout records (hash-verified filtered_v2 inputs), subject order."""

    def __init__(self, parent: dict[str, Any], processed_root: Path) -> None:
        self.parent = parent
        self.processed_root = processed_root
        rows = sorted(parent_entries(parent, "filtered_v2").values(), key=lambda r: str(r["subject_id"]))
        self.keys = [r["record_key"] for r in rows]
        self.subjects = np.asarray([str(r["subject_id"]) for r in rows])
        self.y, self.index = {}, {}
        for row in rows:
            with np.load(verified_parent_path(processed_root, "filtered_v2", row), allow_pickle=False) as z:
                self.y[row["record_key"]] = z["y"].astype(np.int8)
                self.index[row["record_key"]] = z["original_epoch_index"].astype(np.int32)

    def input_sha(self, variant: str) -> dict[str, str]:
        return {k: r["output_sha256"] for k, r in parent_entries(self.parent, variant).items()}

    def confusion(self, key: str, prediction: np.ndarray) -> np.ndarray:
        valid = self.y[key] >= 0
        return confusion_matrix_5(self.y[key][valid], prediction[valid])


def expected_inputs(holdout: Holdout, arm: str) -> dict[str, str]:
    if arm.split("_s")[0] in PARENT_VARIANT:
        return holdout.input_sha(PARENT_VARIANT[arm.split("_s")[0]])
    if arm.startswith("E4_"):
        return holdout.input_sha("bandpass_v2")
    a2 = read_json(ROOT / SHHS_MANIFEST)
    return {r["record_key"]: r["output_sha256"] for r in a2["records"]}


def inference_root(arm: str) -> Path:
    return PARENT_INFERENCE if arm in PARENT_ZERO_SHOT else A2_INFERENCE


def load_ensemble(holdout: Holdout, arm: str) -> np.ndarray:
    root, inputs = inference_root(arm), expected_inputs(holdout, arm)
    manifest = read_json(root / "arm_manifests" / f"{arm}.json")
    by_key = {e["record_key"]: e for e in manifest["ensemble_records"]}
    confusions = []
    for key in holdout.keys:
        path = root / "ensemble" / arm / f"{key}.npz"
        if sha256_file(path) != by_key[key]["sha256"]:
            raise ValueError(f"{arm}/{key}: ensemble hash differs from its arm manifest")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"].item()))
            prediction, index = z["prediction"].astype(np.int8), z["original_epoch_index"]
        if meta["arm"] != arm or meta["input_npz_sha256"] != inputs[key] or not np.array_equal(index, holdout.index[key]):
            raise ValueError(f"{arm}/{key}: ensemble provenance or alignment differs")
        confusions.append(holdout.confusion(key, prediction))
    return np.stack(confusions)


def load_fold_probabilities(holdout: Holdout, arm: str) -> list[dict[str, np.ndarray]]:
    root, inputs = inference_root(arm), expected_inputs(holdout, arm)
    manifest = read_json(root / "arm_manifests" / f"{arm}.json")
    hashes = {(r["fold"], r["record_key"]): r["sha256"] for r in manifest["fold_records"]}
    folds = []
    for fold in FOLDS:
        out = {}
        for key in holdout.keys:
            path = root / "fold_predictions" / arm / f"fold_{fold:02d}" / f"{key}.npz"
            if sha256_file(path) != hashes[(fold, key)]:
                raise ValueError(f"{path}: hash differs from its arm manifest")
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata_json"].item()))
                if (meta["arm"] != arm or meta["outer_fold"] != fold or meta["input_npz_sha256"] != inputs[key]
                        or not np.array_equal(z["original_epoch_index"], holdout.index[key])):
                    raise ValueError(f"{path}: provenance or alignment differs")
                out[key] = z["probabilities"].astype(np.float64)
        folds.append(out)
    return folds


# --------------------------------------------------------------------------- Sleep-EDF out-of-fold
def _git_bytes(path: str) -> tuple[bytes, str]:
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "--all", "-1", "--format=%H", "--", path],
                            capture_output=True, text=True, check=True).stdout.strip()
    if not commit:
        raise FileNotFoundError(f"{path} is not in the Git history")
    data = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    return data, commit


def sleepedf_confusion(arm: str, seed: int) -> dict[str, Any]:
    """Pooled out-of-fold Sleep-EDF confusion; archived runs from Git, A2 runs from runs/a2."""
    total, sources = np.zeros((5, 5), dtype=np.int64), []
    for fold in FOLDS:
        if arm in ("E0", "E2", "E3", "E4", "E6") and seed in (42, 123):
            path = f"runs/v2/full/{arm}/fold_{fold:02d}/seed_{seed}/predictions/test.npz"
            data, commit = _git_bytes(path)
            expected = read_json(GATE5[seed])["input_coverage"][arm]["fold_prediction_sha256"][f"fold_{fold:02d}"]
            digest = hashlib.sha256(data).hexdigest()
            if digest != expected:
                raise ValueError(f"{path}@{commit}: hash differs from the archived Gate 5 analysis")
            stream = io.BytesIO(data)
            sources.append({"fold": fold, "git_commit": commit, "sha256": digest})
        else:
            journal = read_json(test_journal_path(ROOT))
            key = A2Target(arm, seed, fold).key
            path = run_root(ROOT, A2Target(arm, seed, fold)) / "predictions" / "test.npz"
            if journal.get("status") != "complete" or sha256_file(path) != journal["targets"][key]["test_prediction_sha256"]:
                raise ValueError(f"{key}: test prediction differs from the A2 test campaign journal")
            stream = path
            sources.append({"fold": fold, "sha256": sha256_file(path)})
        with np.load(stream, allow_pickle=False) as z:
            true, predicted = z["true_label"].astype(np.int64), z["predicted_label"].astype(np.int64)
        valid = true >= 0
        total += confusion_matrix_5(true[valid], predicted[valid])
    return {"confusion": total, "pooled": ahc.pooled_stats(total), "sources": sources}


# --------------------------------------------------------------------------- helpers
def config_and_bootstrap() -> tuple[dict[str, Any], str, int, int]:
    config, sha = load_a2_config(ROOT)
    boot = config["analyses"]["bootstrap"]
    return config, sha, int(boot["resamples"]), int(boot["seed"])


def scalar(stats: dict[str, Any], key: str) -> float:
    return float(np.asarray(stats[key]))


def write_report(name: str, report: dict[str, Any]) -> Path:
    path = OUTPUT / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; A2 analysis reports are write-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, report, ensure_ascii=False, sort_keys=False)
    return path


def a2_arm_names(config: dict[str, Any]) -> list[str]:
    return [arm.name for arm in holdout_arms(config)]


# --------------------------------------------------------------------------- D: EM sample size
def analysis_d(holdout: Holdout, config: dict[str, Any], config_sha: str, resamples: int, boot_seed: int) -> dict[str, Any]:
    spec = config["analyses"]["D_em_sample_size"]
    folds = load_fold_probabilities(holdout, "E3_s123")
    fits = [read_json(CALIBRATION_FITS / f"fold_{f:02d}.json") for f in FOLDS]
    if [fit["fold"] for fit in fits] != list(FOLDS) or any(fit["target_labels_used"] for fit in fits):
        raise ValueError("Archived calibration fits are incomplete or used target labels")
    offsets = np.cumsum([0] + [len(holdout.y[k]) for k in holdout.keys])
    rows_of = {k: np.arange(offsets[i], offsets[i + 1]) for i, k in enumerate(holdout.keys)}
    y = np.concatenate([holdout.y[k] for k in holdout.keys]).astype(np.int64)
    subject_row = np.concatenate([np.full(len(holdout.y[k]), i) for i, k in enumerate(holdout.keys)])
    valid = y >= 0
    source = [np.asarray(fit["source_prior"], dtype=np.float64) for fit in fits]
    calibrated = [apply_temperature(np.concatenate([folds[f][k] for k in holdout.keys]), fits[f]["temperature"])
                  for f in FOLDS]
    log_q = [np.log(q) for q in calibrated]
    log_raw = [np.log(np.maximum(np.concatenate([folds[f][k] for k in holdout.keys]), 1e-12)) for f in FOLDS]
    observed_n3 = float((y[valid] == N3).mean())

    def predict(priors: list[np.ndarray] | None, rows: np.ndarray | None = None, logs=log_q) -> np.ndarray:
        rows = np.arange(len(y)) if rows is None else rows
        acc = np.zeros((len(rows), 5))
        for f in FOLDS:
            score = logs[f][rows] + (0.0 if priors is None else np.log(priors[f]) - np.log(source[f]))
            acc += np.exp(score - logsumexp(score, axis=1, keepdims=True))
        return acc.argmax(1)

    def summary(prediction: np.ndarray) -> dict[str, float]:
        flat = subject_row[valid] * 25 + y[valid] * 5 + prediction[valid]
        conf = np.bincount(flat, minlength=len(holdout.keys) * 25).reshape(-1, 5, 5)
        pooled = ahc.pooled_stats(conf.sum(0))
        return {"subject_mean_macro_f1": float(_subject_macro_f1(conf).mean()),
                "pooled_macro_f1": scalar(pooled, "macro_f1"), "n3_recall": scalar(pooled, "n3_recall"),
                "n3_prediction_share": float((prediction[valid] == N3).mean())}

    def em(rows: np.ndarray) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        priors, info = [], []
        for f in FOLDS:
            result = estimate_target_prior(calibrated[f][rows], source[f])
            priors.append(result.target_prior)
            info.append({"iterations": result.iterations, "converged": result.converged})
        return priors, info

    def condition(priors: list[np.ndarray], info: Any = None) -> dict[str, Any]:
        out = {"fold_mean_n3_prior": float(np.mean([p[N3] for p in priors])),
               "fold_n3_priors": [float(p[N3]) for p in priors], **summary(predict(priors))}
        if info is not None:
            out["em"] = info
        return out

    report: dict[str, Any] = {"observed_holdout_n3_share": observed_n3, "conditions": {}}
    report["conditions"]["raw_ensemble"] = summary(predict(None, logs=log_raw))
    report["conditions"]["temperature_only"] = summary(predict(None))
    report["conditions"]["original_five"] = condition([np.asarray(fit["target_prior"]) for fit in fits])
    sample = np.load(A1_ROOT / "adabn_sample.npz", allow_pickle=False)
    a1_manifest = read_json(A1_ROOT / "prepare_manifest.json")
    if sha256_file(A1_ROOT / "adabn_sample.npz") != a1_manifest["adabn_sample_sha256"]:
        raise ValueError("A1 AdaBN sample differs from its prepare manifest")
    position = {(k, int(i)): r for k in holdout.keys for i, r in zip(holdout.index[k], rows_of[k])}
    adabn_rows = np.asarray([position[(str(k), int(i))] for k, i in zip(sample["record_key"], sample["original_epoch_index"])])
    if len(adabn_rows) != int(spec_total(config)):
        raise ValueError("AdaBN sample size differs")
    report["conditions"]["adabn_sample"] = condition(*em(adabn_rows))
    report["conditions"]["all_500"] = condition(*em(np.arange(len(y))))
    per_record = np.empty(len(y), dtype=np.int64)
    record_priors = []
    for key in holdout.keys:
        priors, _ = em(rows_of[key])
        record_priors.append(float(np.mean([p[N3] for p in priors])))
        per_record[rows_of[key]] = predict(priors, rows_of[key])
    report["conditions"]["per_record"] = {"fold_mean_n3_prior_median_over_records": float(np.median(record_priors)),
                                          **summary(per_record)}
    generator = np.random.default_rng(SUBSET_SEED)
    subsets = {}
    for n in SUBSET_SIZES:
        draws = []
        for _ in range(SUBSET_DRAWS):
            chosen = generator.choice(len(holdout.keys), size=n, replace=False)
            rows = np.concatenate([rows_of[holdout.keys[i]] for i in chosen])
            draws.append(condition(em(rows)[0]))
        subsets[str(n)] = {stat: {"median": float(np.median([d[stat] for d in draws])),
                                  "p2_5": float(np.percentile([d[stat] for d in draws], 2.5)),
                                  "p97_5": float(np.percentile([d[stat] for d in draws], 97.5))}
                           for stat in ("fold_mean_n3_prior", "n3_prediction_share", "n3_recall",
                                        "pooled_macro_f1", "subject_mean_macro_f1")}
    report["conditions"]["random_subsets"] = {"draws_per_size": SUBSET_DRAWS, "generator_seed": SUBSET_SEED, "by_size": subsets}

    baseline_recall = report["conditions"]["temperature_only"]["n3_recall"]
    large = [report["conditions"][c] for c in ("adabn_sample", "all_500")]
    fails = all(c["fold_mean_n3_prior"] < 0.5 * observed_n3 and c["n3_recall"] < baseline_recall for c in large)
    small = all(c["fold_mean_n3_prior"] >= 0.5 * observed_n3 and c["n3_recall"] >= baseline_recall for c in large)
    report["interpretation"] = {
        "rule": spec["interpretation_rule"],
        "outcome": "label_shift_assumption_fails" if fails else "small_adaptation_sample" if small else "neither_rule_met",
    }
    report["inputs"] = {"a2_config_sha256": config_sha, "calibration_fit_sha256":
                        {f"fold_{f:02d}": sha256_file(CALIBRATION_FITS / f"fold_{f:02d}.json") for f in FOLDS},
                        "adabn_sample_sha256": a1_manifest["adabn_sample_sha256"],
                        "e3_s123_arm_manifest_sha256": sha256_file(PARENT_INFERENCE / "arm_manifests/E3_s123.json")}
    return report


def spec_total(config: dict[str, Any]) -> int:
    a1 = read_json(ROOT / "configs" / "shhs_holdout_amendment_a1.json")
    return int(a1["adabn"]["sample"]["total_epochs"])


# --------------------------------------------------------------------------- E: training cost
def analysis_e(config: dict[str, Any], config_sha: str) -> dict[str, Any]:
    benchmark_path = ROOT / BENCHMARK_FILE
    benchmark = read_json(benchmark_path)
    steps, wall = archived_steps(ROOT), archived_wall_clock(ROOT)
    result = decompose(steps, wall, benchmark)
    return {"inputs": {"a2_config_sha256": config_sha, "benchmark_sha256": sha256_file(benchmark_path)},
            "benchmark": benchmark, "archived_steps": steps, "archived_wall_clock": wall, "decomposition": result,
            "interpretation": {"rule": config["analyses"]["E_training_cost"]["interpretation_rule"],
                               "E0_vs_E3": result["E3"]["ratios_vs_E0"]}}


# --------------------------------------------------------------------------- C: ensemble asymmetry
def analysis_c(holdout: Holdout, config: dict[str, Any], config_sha: str, resamples: int, boot_seed: int) -> dict[str, Any]:
    samples = np.random.default_rng(boot_seed).integers(0, len(holdout.keys), size=(resamples, len(holdout.keys)),
                                                       dtype=np.int32)
    arms = PARENT_ZERO_SHOT + a2_arm_names(config)
    out: dict[str, Any] = {}
    single_pooled: dict[str, float] = {}
    ensemble_pooled: dict[str, float] = {}
    for arm in arms:
        folds = load_fold_probabilities(holdout, arm)
        fold_conf = np.stack([np.stack([holdout.confusion(k, folds[f][k].argmax(1)) for k in holdout.keys])
                              for f in FOLDS])
        ensemble = load_ensemble(holdout, arm)
        fold_values = np.stack([_subject_macro_f1(c) for c in fold_conf])
        ens_values = _subject_macro_f1(ensemble)
        rows = []
        for f in FOLDS:
            pooled = ahc.pooled_stats(fold_conf[f].sum(0))
            rows.append({"fold": f, "subject_mean_macro_f1": float(fold_values[f].mean()),
                         "pooled_macro_f1": scalar(pooled, "macro_f1"), "n3_recall": scalar(pooled, "n3_recall"),
                         "n3_to_n2_rate": scalar(pooled, "n3_to_n2_rate")})
        ens_pooled = ahc.pooled_stats(ensemble.sum(0))
        difference = _mean_paired_bootstrap(ens_values, fold_values.mean(0), samples)
        single_pooled[arm] = float(np.mean([r["pooled_macro_f1"] for r in rows]))
        ensemble_pooled[arm] = scalar(ens_pooled, "macro_f1")
        out[arm] = {"single_models": rows,
                    "single_model_summary": {s: {"mean": float(np.mean([r[s] for r in rows])),
                                                 "min": float(np.min([r[s] for r in rows])),
                                                 "max": float(np.max([r[s] for r in rows]))}
                                             for s in ("subject_mean_macro_f1", "pooled_macro_f1", "n3_recall",
                                                       "n3_to_n2_rate")},
                    "ensemble": {"subject_mean_macro_f1": float(ens_values.mean()),
                                 "pooled_macro_f1": ensemble_pooled[arm],
                                 "n3_recall": scalar(ens_pooled, "n3_recall"),
                                 "n3_to_n2_rate": scalar(ens_pooled, "n3_to_n2_rate")},
                    "ensemble_minus_mean_single_subject_mean_macro_f1": {
                        "estimate": difference["observed_difference"],
                        "ci95": [difference["ci95_low"], difference["ci95_high"]]}}
    drop = {}
    for experiment in ("E3", "E4"):
        for seed in (42, 123):
            arm = f"{experiment}_s{seed}"
            edf = scalar(sleepedf_confusion(experiment, seed)["pooled"], "macro_f1")
            drop[arm] = {"sleep_edf_pooled_oof_macro_f1": edf,
                         "drop_vs_mean_single_model": edf - single_pooled[arm],
                         "drop_vs_ensemble": edf - ensemble_pooled[arm]}
    return {"inputs": {"a2_config_sha256": config_sha, "bootstrap": {"resamples": resamples, "seed": boot_seed}},
            "arms": out, "cross_cohort_drop": drop}


# --------------------------------------------------------------------------- A and B
def holdout_analysis(holdout: Holdout, arms: list[str], resamples: int, boot_seed: int) -> ahc.Analysis:
    return ahc.Analysis(holdout.subjects, {arm: load_ensemble(holdout, arm) for arm in arms}, resamples, boot_seed)


def contrast_row(analysis: ahc.Analysis, left: str, right: str) -> dict[str, Any]:
    row = analysis.contrast(left, right, with_test=False)
    for stat in ("n3_recall", "n3_to_n2_rate"):
        row[f"pooled_{stat}"] = analysis.pooled_difference(left, right, stat)
    return row


def analysis_a(holdout: Holdout, config: dict[str, Any], config_sha: str, resamples: int, boot_seed: int) -> dict[str, Any]:
    spec = config["analyses"]["A_normalisation_scope"]
    seeds = [int(s) for s in spec["seeds"]]
    arms = [f"{e}_s{s}" for s in seeds for e in ("E4", "E3", "E6", "E6R")]
    analysis = holdout_analysis(holdout, arms, resamples, boot_seed)
    contrasts = {}
    for seed in seeds:
        for text in spec["contrasts_per_seed"]:
            left, right = (f"{part.strip()}_s{seed}" for part in text.split(" - "))
            contrasts[f"{left} - {right}"] = contrast_row(analysis, left, right)
    sleep_edf = {f"{arm}_s{seed}": scalar(sleepedf_confusion(arm, seed)["pooled"], "macro_f1")
                 for seed in seeds for arm in ("E3", "E4", "E6", "E6R")}
    keep = all(contrasts[f"E4_s{s} - E6R_s{s}"]["ci95"][0] > 0 for s in seeds)
    return {"inputs": {"a2_config_sha256": config_sha, "bootstrap": {"resamples": resamples, "seed": boot_seed}},
            "descriptive": {arm: analysis.describe(arm) for arm in arms}, "contrasts": contrasts,
            "sleep_edf_pooled_oof_macro_f1": sleep_edf,
            "input_scale_in_window": {"sleep_edf": read_json(ROOT / SLEEPEDF_MANIFEST)["input_scale_in_window"],
                                      "holdout": read_json(ROOT / SHHS_MANIFEST)["input_scale_in_window"]},
            "interpretation": {"rule": spec["interpretation_rule"],
                               "keep_calibrated_amplitude_statement": keep}}


def analysis_b(holdout: Holdout, config: dict[str, Any], config_sha: str, resamples: int, boot_seed: int) -> dict[str, Any]:
    spec = config["analyses"]["B_training_seeds"]
    seeds = [int(s) for s in spec["seeds"]]
    e4 = [f"E4_s{s}" for s in seeds]
    references = ["E0_s42", "E0_s123", "E2_s42", "E2_s123"]
    analysis = holdout_analysis(holdout, e4 + references, resamples, boot_seed)
    per_seed = {}
    for seed, arm in zip(seeds, e4, strict=True):
        absolute = _absolute_subject_bootstrap(analysis.values[arm], analysis.samples)
        per_seed[arm] = {"holdout_subject_mean_macro_f1": absolute["estimate"],
                         "ci95": [absolute["ci95_low"], absolute["ci95_high"]],
                         "sleep_edf_pooled_oof_macro_f1": scalar(sleepedf_confusion("E4", seed)["pooled"], "macro_f1")}

    def spread(values: list[float]) -> dict[str, float]:
        return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
                "min": float(np.min(values)), "max": float(np.max(values))}

    between = [analysis.contrast(e4[i], e4[j], with_test=False) for i in range(len(e4)) for j in range(i + 1, len(e4))]
    configuration = [analysis.contrast(arm, ref, with_test=False) for arm in e4 for ref in references]
    largest = max(abs(r["subject_mean_macro_f1_difference"]) for r in between)
    exceeds = all(r["subject_mean_macro_f1_difference"] > largest for r in configuration)
    return {"inputs": {"a2_config_sha256": config_sha, "bootstrap": {"resamples": resamples, "seed": boot_seed}},
            "per_seed": per_seed,
            "summary": {"holdout_subject_mean_macro_f1": spread([v["holdout_subject_mean_macro_f1"] for v in per_seed.values()]),
                        "sleep_edf_pooled_oof_macro_f1": spread([v["sleep_edf_pooled_oof_macro_f1"] for v in per_seed.values()])},
            "between_seed_differences": between, "largest_absolute_between_seed_difference": largest,
            "configuration_contrasts": configuration,
            "interpretation": {"rule": spec["interpretation_rule"],
                               "configuration_contrasts_exceed_training_variability": exceeds}}


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=("a", "b", "c", "d", "e"))
    parser.add_argument("--parent-manifest", type=Path,
                        default=ROOT / A2_ROOT / "manifests" / "parent_preprocess_holdout_v1.json",
                        help="parent holdout preprocess manifest (byte-identical copy kept by prepare-shhs)")
    parser.add_argument("--parent-processed-root", type=Path, default=ROOT / "data/processed/shhs_holdout_v1")
    args = parser.parse_args()
    config, config_sha, resamples, boot_seed = config_and_bootstrap()
    name = {"a": "A_normalisation_scope", "b": "B_training_seeds", "c": "C_ensemble_asymmetry",
            "d": "D_em_sample_size", "e": "E_training_cost"}[args.analysis]
    if args.analysis == "e":
        report = analysis_e(config, config_sha)
    else:
        holdout = Holdout(load_parent_manifest(args.parent_manifest, config), args.parent_processed_root)
        report = {"a": analysis_a, "b": analysis_b, "c": analysis_c, "d": analysis_d}[args.analysis](
            holdout, config, config_sha, resamples, boot_seed)
    report = {"schema_version": 1, "amendment": "A2", "analysis": name, "hypothesis_tests": "none", **report}
    path = write_report(name, report)
    print(f"WROTE {path}")
    print(json.dumps(report.get("interpretation", {}), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
