"""Amendment A3, step 3: amplitude-guided N2/N3 correction (F) and second target cohort (G).

Everything the rules need is estimated on reference N2 and N3 epochs of Sleep-EDF; no target data and
no model performance enter them. The rules are then applied to the stored ensemble probabilities of the
SHHS1 holdout sample, to the telemetry cohort and, with fold-wise estimation, to the Sleep-EDF
out-of-fold predictions. Descriptive; no hypothesis tests. See docs/AMENDMENT_A3.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/amendment_a3.json").read_text(encoding="utf8"))
SPLIT = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
STAGES = ("W", "N1", "N2", "N3", "REM")
N2, N3 = 2, 3
RESAMPLES = CONFIG["bootstrap"]["resamples"]
SEED = CONFIG["bootstrap"]["seed"]


# --------------------------------------------------------------------------- data
def load_record(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        x = z["x"].astype(np.float64)
        return {"y": z["y"].astype(np.int64), "valid": z["valid_mask"].astype(bool),
                "index": z["original_epoch_index"].copy(), "sd": x.std(axis=1)}


def load_cohort(bandpass_folder: Path, subject_rule) -> dict[str, dict]:
    records = {}
    for path in sorted(bandpass_folder.glob("*.npz")):
        record = load_record(path)
        record["subject"] = subject_rule(path.stem)
        records[path.stem] = record
    return records


def load_probabilities(folder: Path, records: dict[str, dict], keys=None) -> dict[str, np.ndarray]:
    result = {}
    for key in (keys if keys is not None else records):
        with np.load(folder / f"{key}.npz", allow_pickle=False) as z:
            if not np.array_equal(z["original_epoch_index"], records[key]["index"]):
                raise ValueError(f"{folder.name}/{key}: epoch index differs from the band-pass input")
            result[key] = z["probabilities"].astype(np.float64)
    return result


# --------------------------------------------------------------------------- rules
def fit_source(records: dict[str, dict], keys) -> dict:
    sd = np.concatenate([records[k]["sd"][records[k]["valid"]] for k in keys])
    y = np.concatenate([records[k]["y"][records[k]["valid"]] for k in keys])
    n3, n2 = sd[y == N3], sd[y == N2]
    grid = np.quantile(np.concatenate([n3, n2]), np.linspace(0.001, 0.999, 999))
    sensitivity = 1.0 - np.searchsorted(np.sort(n3), grid, side="left") / n3.size
    specificity = np.searchsorted(np.sort(n2), grid, side="left") / n2.size
    threshold = float(grid[int(np.argmax(sensitivity + specificity))])
    features = np.log(np.concatenate([n3, n2]))[:, None]
    target = np.concatenate([np.ones(n3.size), np.zeros(n2.size)])
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(features, target)
    return {"threshold_uv": threshold, "slope": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]), "log_prior_odds": float(np.log(n3.size / n2.size)),
            "n3_epochs": int(n3.size), "n2_epochs": int(n2.size)}


def rule_hard(probabilities: np.ndarray, sd: np.ndarray, fit: dict) -> np.ndarray:
    prediction = probabilities.argmax(axis=1)
    prediction[(prediction == N2) & (sd >= fit["threshold_uv"])] = N3
    return prediction


def rule_soft(probabilities: np.ndarray, sd: np.ndarray, fit: dict) -> np.ndarray:
    adjusted = probabilities.copy()
    log_ratio = fit["intercept"] + fit["slope"] * np.log(sd) - fit["log_prior_odds"]
    tiny = 1e-300
    log_odds = np.log(adjusted[:, N3] + tiny) - np.log(adjusted[:, N2] + tiny) + log_ratio
    mass = adjusted[:, N2] + adjusted[:, N3]
    share = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -700, 700)))
    adjusted[:, N3] = mass * share
    adjusted[:, N2] = mass * (1.0 - share)
    return adjusted.argmax(axis=1)


# --------------------------------------------------------------------------- metrics
def subject_confusions(records: dict[str, dict], predictions: dict[str, np.ndarray]) -> tuple[list, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for key, prediction in predictions.items():
        record = records[key]
        valid = record["valid"]
        matrix = np.zeros((5, 5), dtype=np.int64)
        np.add.at(matrix, (record["y"][valid], prediction[valid]), 1)
        table[record["subject"]] = table.get(record["subject"], 0) + matrix
    subjects = sorted(table)
    return subjects, np.stack([table[s] for s in subjects])


def safe_divide(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def class_scores(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    precision = safe_divide(diagonal, matrices.sum(axis=-2))
    recall = safe_divide(diagonal, matrices.sum(axis=-1))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def pooled_values(matrices: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics of one or many pooled 5x5 matrices (leading axes are kept)."""
    precision, recall, f1 = class_scores(matrices)
    total = matrices.sum(axis=(-2, -1))
    accuracy = safe_divide(np.trace(matrices, axis1=-2, axis2=-1), total)
    expected = safe_divide((matrices.sum(axis=-1) * matrices.sum(axis=-2)).sum(axis=-1), total * total)
    values = {"pooled_macro_f1": f1.mean(axis=-1), "accuracy": accuracy,
              "kappa": safe_divide(accuracy - expected, 1.0 - expected),
              "n3_to_n2_rate": safe_divide(matrices[..., N3, N2], matrices[..., N3, :].sum(axis=-1))}
    for index, stage in enumerate(STAGES):
        values[f"precision_{stage}"] = precision[..., index]
        values[f"recall_{stage}"] = recall[..., index]
        values[f"f1_{stage}"] = f1[..., index]
    return values


def all_values(matrices: np.ndarray, weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Subject-mean macro-F1 and pooled metrics; with ``weights`` (resamples x subjects) per resample."""
    subject_macro = class_scores(matrices)[2].mean(axis=-1)
    if weights is None:
        values = pooled_values(matrices.sum(axis=0))
        values["subject_mean_macro_f1"] = subject_macro.mean()
        return values
    pooled = (weights @ matrices.reshape(len(matrices), 25).astype(np.float64)).reshape(-1, 5, 5)
    values = pooled_values(pooled)
    values["subject_mean_macro_f1"] = weights @ subject_macro / weights.sum(axis=1)
    return values


def bootstrap_weights(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, n, size=(RESAMPLES, n))
    weights = np.zeros((RESAMPLES, n), dtype=np.float64)
    np.add.at(weights, (np.arange(RESAMPLES)[:, None], sampled), 1.0)
    return weights


REPORTED = ("subject_mean_macro_f1", "pooled_macro_f1", "accuracy", "kappa", "recall_N3", "precision_N3",
            "f1_N3", "n3_to_n2_rate", "recall_N2", "precision_N2", "f1_N2", "f1_W", "f1_N1", "f1_REM")


def summary(matrices: np.ndarray, weights: np.ndarray) -> dict:
    point, resampled = all_values(matrices), all_values(matrices, weights)
    return {name: {"value": float(point[name]), "ci95": [float(v) for v in np.quantile(resampled[name], [0.025, 0.975])]}
            for name in REPORTED}


def difference(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> dict:
    """left minus right, paired over the same subjects and the same resamples."""
    point_l, point_r = all_values(left), all_values(right)
    boot_l, boot_r = all_values(left, weights), all_values(right, weights)
    return {name: {"value": float(point_l[name] - point_r[name]),
                   "ci95": [float(v) for v in np.quantile(boot_l[name] - boot_r[name], [0.025, 0.975])]}
            for name in REPORTED}


def verdict(delta: dict) -> str:
    macro, recall = delta["subject_mean_macro_f1"]["ci95"], delta["recall_N3"]["ci95"]
    if macro[0] > 0:
        return "improves"
    if macro[1] < 0:
        return "harms"
    return "trade-off" if recall[0] > 0 else "no detectable change"


# --------------------------------------------------------------------------- analyses
def analyse_arm(records, probabilities, fits_by_key, weights_cache) -> tuple[dict, dict]:
    predictions = {"unmodified": {}, "R1_hard": {}, "R2_soft": {}}
    for key, p in probabilities.items():
        sd, fit = records[key]["sd"], fits_by_key[key]
        predictions["unmodified"][key] = p.argmax(axis=1)
        predictions["R1_hard"][key] = rule_hard(p, sd, fit)
        predictions["R2_soft"][key] = rule_soft(p, sd, fit)
    matrices = {}
    for name, prediction in predictions.items():
        subjects, matrices[name] = subject_confusions(records, prediction)
    n = len(subjects)
    if n not in weights_cache:
        weights_cache[n] = bootstrap_weights(n)
    weights = weights_cache[n]
    result = {"subjects": n, "confusion_unmodified": matrices["unmodified"].sum(axis=0).tolist()}
    for name in predictions:
        result[name] = summary(matrices[name], weights)
        result[f"confusion_{name}"] = matrices[name].sum(axis=0).tolist()
    for name in ("R1_hard", "R2_soft"):
        delta = difference(matrices[name], matrices["unmodified"], weights)
        result[f"{name}_minus_unmodified"] = delta
        result[f"{name}_verdict"] = verdict(delta)
    return result, matrices


def separation(records: dict[str, dict], fit: dict) -> dict:
    sd = np.concatenate([r["sd"][r["valid"]] for r in records.values()])
    y = np.concatenate([r["y"][r["valid"]] for r in records.values()])
    counts = {stage: int((y == i).sum()) for i, stage in enumerate(STAGES)}
    n3, n2 = sd[y == N3], sd[y == N2]
    return {"scored_epochs": int(y.size), "stage_counts": counts,
            "source_threshold_uv": fit["threshold_uv"],
            "n3_at_or_above": float((n3 >= fit["threshold_uv"]).mean()) if n3.size else None,
            "n2_below": float((n2 < fit["threshold_uv"]).mean()) if n2.size else None}


def contrasts(matrices_by_arm: dict[str, dict], weights_cache: dict) -> dict:
    result = {}
    for seed in (42, 123):
        for name in CONFIG["contrasts_new_cohort"]:
            left, right = (f"{part}_s{seed}" for part in name.split("-"))
            if left in matrices_by_arm and right in matrices_by_arm:
                a, b = matrices_by_arm[left]["unmodified"], matrices_by_arm[right]["unmodified"]
                delta = difference(a, b, weights_cache[len(a)])
                keep = ("subject_mean_macro_f1", "pooled_macro_f1", "accuracy", "recall_N3", "n3_to_n2_rate")
                result[f"{name}_s{seed}"] = {k: delta[k] for k in keep}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-root", type=Path, default=ROOT / "runs/a3")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/a3/analysis/a3_summary.json")
    parser.add_argument("--cohorts", nargs="+", default=["sleep_edf_oof", "shhs1_holdout", "telemetry"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing summary; do not overwrite")
    weights_cache: dict[int, np.ndarray] = {}
    source = load_cohort(ROOT / "data/processed/bandpass_v2", lambda key: key[:5])
    full_fit = fit_source(source, sorted(source))
    report = {"schema_version": 1, "amendment": "A3", "role": "post-confirmatory descriptive",
              "bootstrap": CONFIG["bootstrap"], "source_fit_all_recordings": full_fit, "cohorts": {}}
    arms = [arm["arm"] for arm in CONFIG["arms"]]

    if "sleep_edf_oof" in args.cohorts:
        split = json.loads(SPLIT.read_text(encoding="utf8"))
        fits_by_key, fold_fits = {}, []
        for fold in split["folds"]:
            held_out = set(fold["record_keys"])
            fit = fit_source(source, [k for k in sorted(source) if k not in held_out])
            fold_fits.append({"fold": fold["fold_index"], **fit})
            fits_by_key.update({key: fit for key in held_out})
        cohort = {"fold_fits": fold_fits, "arms": {}}
        for arm in arms:
            folder = args.a3_root / "sleep_edf_oof" / arm
            if folder.exists():
                cohort["arms"][arm], _ = analyse_arm(source, load_probabilities(folder, source), fits_by_key, weights_cache)
        report["cohorts"]["sleep_edf_oof"] = cohort

    targets = {
        "shhs1_holdout": (ROOT / "data/processed/shhs_holdout_v1/bandpass_v2", lambda key: key,
                          [ROOT / "runs/holdout_inference_v1/ensemble", ROOT / "runs/a2/holdout/ensemble"],
                          arms + ["E4_s2025", "E4_s2027", "E4_s2029", "E6R_s42", "E6R_s123"]),
        "telemetry": (ROOT / CONFIG["cohorts"]["sleep_edf_telemetry"]["processed_root"] / "bandpass_v2",
                      lambda key: key[:5], [args.a3_root / "telemetry" / "ensemble"], arms),
    }
    for name, (folder, subject_rule, roots, cohort_arms) in targets.items():
        if name not in args.cohorts:
            continue
        records = load_cohort(folder, subject_rule)
        fits_by_key = {key: full_fit for key in records}
        cohort = {"recordings": len(records), "separation_by_source_threshold": separation(records, full_fit),
                  "arms": {}}
        matrices_by_arm = {}
        for arm in cohort_arms:
            found = [root / arm for root in roots if (root / arm).exists()]
            if not found:
                continue
            cohort["arms"][arm], matrices_by_arm[arm] = analyse_arm(
                records, load_probabilities(found[0], records), fits_by_key, weights_cache)
        cohort["contrasts_unmodified"] = contrasts(matrices_by_arm, weights_cache)
        report["cohorts"][name] = cohort

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    for name, cohort in report["cohorts"].items():
        for arm, result in cohort["arms"].items():
            row = [f"{result[v]['subject_mean_macro_f1']['value']:.4f}/{result[v]['recall_N3']['value']:.3f}"
                   for v in ("unmodified", "R1_hard", "R2_soft")]
            print(name, arm, *row, result["R1_hard_verdict"], result["R2_soft_verdict"], sep="  ")
    return None


if __name__ == "__main__":
    sys.exit(main())
