"""Post-hoc descriptive stage-wise amplitude and delta power of the band-pass input in both cohorts.

Reads the processed evaluation-window epochs (bandpass_v2) of the 153 Sleep-EDF recordings and the 500
holdout SHHS1 recordings together with their reference stages. Loads no model and uses no prediction.
For every valid epoch it computes the standard deviation (microvolts) and the 0.5-4 Hz power, and asks
how well these single quantities separate reference N3 from reference N2 within each cohort and when a
threshold chosen in one cohort is applied to the other. Not part of the confirmatory protocol or of
amendments A1 and A2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "sleep_edf": ROOT / "data/processed/bandpass_v2",
    "shhs1_holdout": ROOT / "data/processed/shhs_holdout_v1/bandpass_v2",
}
STAGES = ("W", "N1", "N2", "N3", "REM")
FS = 100.0
NPERSEG = 400  # 4 s Hann segments, 0.25 Hz resolution
DELTA = (0.5, 4.0)
MIN_EPOCHS = 10  # per stage, for a recording to enter the recording-level summaries


def record_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        valid = z["valid_mask"]
        x = z["x"][valid].astype(np.float64)
        y = z["y"][valid].astype(np.int64)
    frequencies, power = welch(x, fs=FS, window="hann", nperseg=NPERSEG, noverlap=NPERSEG // 2,
                               detrend="constant", axis=-1)
    band = (frequencies >= DELTA[0]) & (frequencies <= DELTA[1])
    delta_db = 10.0 * np.log10(power[:, band].sum(axis=1) * (frequencies[1] - frequencies[0]))
    return y, x.std(axis=1), delta_db


def quartiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(q50), "q25": float(q25), "q75": float(q75), "n": int(values.size)}


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    ranks = rankdata(np.concatenate([positive, negative]))
    n_pos, n_neg = positive.size, negative.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_threshold(positive: np.ndarray, negative: np.ndarray) -> float:
    """Threshold maximising balanced accuracy for the rule 'value >= threshold -> N3'."""
    grid = np.quantile(np.concatenate([positive, negative]), np.linspace(0.001, 0.999, 999))
    sensitivity = 1.0 - np.searchsorted(np.sort(positive), grid, side="left") / positive.size
    specificity = np.searchsorted(np.sort(negative), grid, side="left") / negative.size
    return float(grid[int(np.argmax(sensitivity + specificity))])


def apply_threshold(threshold: float, positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    return {"n3_at_or_above": float((positive >= threshold).mean()),
            "n2_below": float((negative < threshold).mean())}


def collect(folder: Path) -> dict:
    paths = sorted(folder.glob("*.npz"))
    per_record = {name: {stage: [] for stage in STAGES} for name in ("sd_uv", "delta_db")}
    contrast = {"sd_ratio_n3_over_n2": [], "delta_db_n3_minus_n2": []}
    pooled = {name: {"N2": [], "N3": []} for name in ("sd_uv", "delta_db")}
    for path in paths:
        y, sd, delta_db = record_features(path)
        medians = {}
        for index, stage in enumerate(STAGES):
            mask = y == index
            if stage in ("N2", "N3"):
                pooled["sd_uv"][stage].append(sd[mask])
                pooled["delta_db"][stage].append(delta_db[mask])
            if mask.sum() >= MIN_EPOCHS:
                medians[stage] = (float(np.median(sd[mask])), float(np.median(delta_db[mask])))
                per_record["sd_uv"][stage].append(medians[stage][0])
                per_record["delta_db"][stage].append(medians[stage][1])
        if "N2" in medians and "N3" in medians:
            contrast["sd_ratio_n3_over_n2"].append(medians["N3"][0] / medians["N2"][0])
            contrast["delta_db_n3_minus_n2"].append(medians["N3"][1] - medians["N2"][1])
    pooled = {name: {stage: np.concatenate(parts) for stage, parts in stages.items()}
              for name, stages in pooled.items()}
    return {"recordings": len(paths), "per_record": per_record, "contrast": contrast, "pooled": pooled}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/cohort_psd/stage_spectra_summary.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing summary; do not overwrite")
    data = {cohort: collect(folder) for cohort, folder in COHORTS.items()}
    result = {
        "schema_version": 1,
        "role": "post-hoc descriptive; reference stages and band-pass input only, no model",
        "settings": {"input": "bandpass_v2", "fs_hz": FS, "welch_nperseg": NPERSEG,
                     "welch_noverlap": NPERSEG // 2, "delta_band_hz": list(DELTA),
                     "delta_unit": "dB re 1 uV^2", "min_epochs_per_stage": MIN_EPOCHS},
        "cohorts": {},
        "separation_n3_vs_n2": {},
    }
    for cohort, d in data.items():
        result["cohorts"][cohort] = {
            "recordings": d["recordings"],
            "recording_median_by_stage": {
                name: {stage: quartiles(values) for stage, values in stages.items()}
                for name, stages in d["per_record"].items()},
            "within_recording_contrast": {name: quartiles(values)
                                          for name, values in d["contrast"].items()},
            "pooled_epochs": {stage: int(d["pooled"]["sd_uv"][stage].size) for stage in ("N2", "N3")},
            "pooled_epoch_quartiles": {
                name: {stage: quartiles(values) for stage, values in stages.items()}
                for name, stages in d["pooled"].items()},
        }
    for name in ("sd_uv", "delta_db"):
        entry = {}
        thresholds = {}
        for cohort, d in data.items():
            n3, n2 = d["pooled"][name]["N3"], d["pooled"][name]["N2"]
            thresholds[cohort] = best_threshold(n3, n2)
            entry[cohort] = {"auc": auc(n3, n2), "own_threshold": thresholds[cohort],
                             "own_threshold_rates": apply_threshold(thresholds[cohort], n3, n2)}
        for source, target in (("sleep_edf", "shhs1_holdout"), ("shhs1_holdout", "sleep_edf")):
            n3, n2 = data[target]["pooled"][name]["N3"], data[target]["pooled"][name]["N2"]
            entry[f"{source}_threshold_on_{target}"] = apply_threshold(thresholds[source], n3, n2)
        result["separation_n3_vs_n2"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
