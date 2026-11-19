"""Post-hoc descriptive analyses for the manuscript revision (no training, no new inference).

1. telemetry_by_night: telemetry results split by placebo and temazepam night (PhysioNet ST-subjects.xls),
   for the archived A3 arms and the A4 one-edge arms at seeds 42 and 123.
2. sleep_parameters: whole-night sleep parameters (TST, N1/N2/N3/REM minutes, WASO, sleep efficiency and REM
   latency within the evaluation window) from the ten-model ensemble against the reference hypnogram on the
   SHHS1 holdout sample and telemetry: bias, 95% limits of agreement and ICC(A,1).

Both analyses are post hoc and descriptive; they reuse the locked predictions and the metric and bootstrap
functions of the A3 analysis with their own bootstrap seed (2048).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a3_analyze", ROOT / "scripts/a3_analyze.py")
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3)
a3.SEED = 2048

TELEMETRY = ROOT / "data/processed/telemetry_a3/bandpass_v2"
HOLDOUT = ROOT / "data/processed/shhs_holdout_v1/bandpass_v2"
A3_TELEMETRY = ROOT / "runs/a3/telemetry/ensemble"
HOLDOUT_ENSEMBLE = ROOT / "runs/holdout_inference_v1/ensemble"
A4_FINAL = ROOT / "runs/final/a4"
SEEDS = (42, 123)
ARMS_A3 = ("E0", "E2", "E3", "E4", "E6")
ARMS_A4 = ("E4HP", "E4LP")
KEEP = ("subject_mean_macro_f1", "pooled_macro_f1", "accuracy", "recall_N3", "n3_to_n2_rate")
CONTRASTS = ("E3-E0", "E4-E0", "E4-E2", "E4HP-E2", "E4LP-E2", "E4-E4HP", "E4-E4LP")


def night_types(xls: Path) -> dict[str, str]:
    """record key (ST7ssnJ) -> 'placebo' or 'temazepam' from PhysioNet ST-subjects.xls."""
    import xlrd

    sheet = xlrd.open_workbook(str(xls)).sheet_by_index(0)
    result = {}
    for row in range(2, sheet.nrows):
        values = sheet.row_values(row)
        if values[0] in ("", None):
            continue
        subject = int(values[0])
        result[f"ST7{subject:02d}{int(values[3])}J"] = "placebo"
        result[f"ST7{subject:02d}{int(values[5])}J"] = "temazepam"
    return result


def predictions(folder: Path, records: dict) -> dict[str, np.ndarray]:
    return {k: p.argmax(axis=1) for k, p in a3.load_probabilities(folder, records).items()}


def telemetry_by_night(xls: Path) -> dict:
    records_all = a3.load_cohort(TELEMETRY, lambda k: k[:5])
    nights = night_types(xls)
    missing = sorted(set(records_all) - set(nights))
    if missing:
        raise ValueError(f"no night type for {missing}")
    out = {"source": "PhysioNet sleep-edfx 1.0.0 ST-subjects.xls", "conditions": {}}
    for condition in ("placebo", "temazepam"):
        records = {k: r for k, r in records_all.items() if nights[k] == condition}
        matrices = {}
        for seed in SEEDS:
            for arm in ARMS_A3:
                _, matrices[f"{arm}_s{seed}"] = a3.subject_confusions(
                    records, predictions(A3_TELEMETRY / f"{arm}_s{seed}", records))
            for arm in ARMS_A4:
                _, matrices[f"{arm}_s{seed}"] = a3.subject_confusions(
                    records, predictions(A4_FINAL / "telemetry" / f"{arm}_s{seed}", records))
        weights = a3.bootstrap_weights(len(next(iter(matrices.values()))))
        result = {"subjects": int(weights.shape[1]), "records": sorted(records), "arms": {}, "contrasts": {}}
        for name, matrix in matrices.items():
            summary = a3.summary(matrix, weights)
            result["arms"][name] = {k: summary[k] for k in KEEP}
        for seed in SEEDS:
            for contrast in CONTRASTS:
                left, right = contrast.split("-")
                delta = a3.difference(matrices[f"{left}_s{seed}"], matrices[f"{right}_s{seed}"], weights)
                result["contrasts"][f"{contrast}_s{seed}"] = {k: delta[k] for k in KEEP}
        out["conditions"][condition] = result
    return out


# --------------------------------------------------------------------------- sleep parameters
SLEEP = (1, 2, 3, 4)  # N1, N2, N3, REM in the 0-4 label coding (W=0)


def parameters(stages: np.ndarray) -> dict[str, float]:
    """Sleep parameters (minutes or %) of one hypnogram restricted to scored epochs of the window."""
    sleep = np.isin(stages, SLEEP)
    minutes = lambda mask: 0.5 * float(np.count_nonzero(mask))
    result = {"TST": minutes(sleep), "N1": minutes(stages == 1), "N2": minutes(stages == 2),
              "N3": minutes(stages == 3), "REM": minutes(stages == 4),
              "SE": 100.0 * np.count_nonzero(sleep) / len(stages)}
    if sleep.any():
        first, last = np.flatnonzero(sleep)[[0, -1]]
        result["WASO"] = minutes(stages[first:last + 1] == 0)
        rem = np.flatnonzero(stages[first:] == 4)
        result["REM_latency"] = 0.5 * float(rem[0]) if rem.size else float("nan")
    else:
        result["WASO"] = result["REM_latency"] = float("nan")
    return result


def icc_a1(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Two-way random, absolute agreement, single measurement ICC(A,1) (McGraw and Wong)."""
    data = np.column_stack([reference, predicted])
    n, k = data.shape
    grand = data.mean()
    ms_rows = k * np.sum((data.mean(axis=1) - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((data.mean(axis=0) - grand) ** 2) / (k - 1)
    ms_err = (np.sum((data - grand) ** 2) - (n - 1) * ms_rows - (k - 1) * ms_cols) / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n))


def agreement(records: dict, folder: Path) -> dict:
    predicted = predictions(folder, records)
    ref_rows, pred_rows = [], []
    for key, record in records.items():
        valid = record["valid"]
        ref_rows.append(parameters(record["y"][valid]))
        pred_rows.append(parameters(predicted[key][valid]))
    result = {}
    for name in ref_rows[0]:
        ref = np.array([r[name] for r in ref_rows])
        pred = np.array([p[name] for p in pred_rows])
        keep = np.isfinite(ref) & np.isfinite(pred)
        diff = pred[keep] - ref[keep]
        sd = float(diff.std(ddof=1))
        result[name] = {"n": int(keep.sum()), "reference_mean": float(ref[keep].mean()),
                        "predicted_mean": float(pred[keep].mean()), "bias": float(diff.mean()),
                        "loa": [float(diff.mean() - 1.96 * sd), float(diff.mean() + 1.96 * sd)],
                        "icc_a1": icc_a1(ref[keep], pred[keep])}
    return result


def sleep_parameters() -> dict:
    out = {"window": "evaluation window (30 min before the first to 30 min after the last scored sleep epoch); "
                     "scored epochs only", "cohorts": {}}
    holdout = a3.load_cohort(HOLDOUT, lambda k: k)
    telemetry = a3.load_cohort(TELEMETRY, lambda k: k)
    out["cohorts"]["shhs1_holdout"] = {f"{arm}_s{seed}": agreement(holdout, HOLDOUT_ENSEMBLE / f"{arm}_s{seed}")
                                       for arm in ("E0", "E3", "E4") for seed in SEEDS}
    out["cohorts"]["telemetry"] = {f"{arm}_s{seed}": agreement(telemetry, A3_TELEMETRY / f"{arm}_s{seed}")
                                   for arm in ("E0", "E3", "E4") for seed in SEEDS}
    return out


AGE_BANDS = ((0, 55, "<55"), (55, 65, "55-64"), (65, 75, "65-74"), (75, 200, ">=75"))
AHI_BANDS = ((0, 5, "<5"), (5, 15, "5-15"), (15, 30, "15-30"), (30, 1e9, ">=30"))


def covariates(csv: Path, keys) -> dict[str, dict]:
    """Age, sex and AHI (ahi_a0h3: apnoeas and hypopnoeas with >=3% desaturation) of holdout participants."""
    import pandas as pd

    table = pd.read_csv(csv, usecols=["nsrrid", "age_s1", "gender", "ahi_a0h3"], encoding="latin1")
    table = table.set_index("nsrrid")
    result = {}
    for key in keys:
        row = table.loc[int(key.split("-")[1])]
        result[key] = {"age": float(row["age_s1"]), "sex": "male" if int(row["gender"]) == 1 else "female",
                       "ahi": float(row["ahi_a0h3"])}
    return result


def band(value: float, bands) -> str:
    return next(label for low, high, label in bands if low <= value < high)


def subgroups(csv: Path) -> dict:
    records_all = a3.load_cohort(HOLDOUT, lambda k: k)
    cov = covariates(csv, records_all)
    groups = {"age": {k: band(c["age"], AGE_BANDS) for k, c in cov.items()},
              "sex": {k: c["sex"] for k, c in cov.items()},
              "ahi": {k: band(c["ahi"], AHI_BANDS) for k, c in cov.items()}}
    arms = [f"{arm}_s{seed}" for arm in ("E0", "E3", "E4") for seed in SEEDS]
    preds = {arm: predictions(HOLDOUT_ENSEMBLE / arm, records_all) for arm in arms}
    out = {"ahi_variable": "ahi_a0h3", "groups": {}}
    for factor, assignment in groups.items():
        out["groups"][factor] = {}
        for level in sorted(set(assignment.values())):
            records = {k: r for k, r in records_all.items() if assignment[k] == level}
            matrices = {arm: a3.subject_confusions(records, {k: preds[arm][k] for k in records})[1] for arm in arms}
            weights = a3.bootstrap_weights(len(records))
            entry = {"participants": len(records), "arms": {}, "contrasts": {}}
            for arm in arms:
                summary = a3.summary(matrices[arm], weights)
                n3_ref = sum(0.5 * np.count_nonzero(r["y"][r["valid"]] == 3) for r in records.values())
                n3_pred = sum(0.5 * np.count_nonzero(preds[arm][k][r["valid"]] == 3) for k, r in records.items())
                entry["arms"][arm] = {**{m: summary[m] for m in KEEP},
                                      "n3_minutes_reference_mean": n3_ref / len(records),
                                      "n3_minutes_bias_mean": (n3_pred - n3_ref) / len(records)}
            for seed in SEEDS:
                for left in ("E3", "E4"):
                    delta = a3.difference(matrices[f"{left}_s{seed}"], matrices[f"E0_s{seed}"], weights)
                    entry["contrasts"][f"{left}-E0_s{seed}"] = {m: delta[m] for m in KEEP}
            out["groups"][factor][level] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xls", type=Path, default=ROOT / "data/raw/sleep-telemetry/ST-subjects.xls")
    parser.add_argument("--shhs-csv", type=Path,
                        default=Path("E:/research/Dataset/SHHS_v1/shhs/datasets/shhs1-dataset-0.21.0.csv"))
    parser.add_argument("--output", type=Path, default=ROOT / "runs/revision/extra_analyses.json")
    args = parser.parse_args()
    report = {"schema_version": 1, "role": "post-hoc descriptive", "bootstrap_seed": a3.SEED,
              "telemetry_by_night": telemetry_by_night(args.xls), "sleep_parameters": sleep_parameters(),
              "holdout_subgroups": subgroups(args.shhs_csv)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print("written", args.output)


if __name__ == "__main__":
    sys.exit(main())
