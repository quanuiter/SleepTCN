"""Post-hoc descriptive power spectra of the unfiltered and band-pass inputs in both cohorts.

Reads the processed evaluation-window epochs (paper_raw_v1 and bandpass_v2) of the 153 Sleep-EDF
recordings and the 500 holdout SHHS1 recordings. Uses no stage labels beyond the evaluation window and
the valid mask already stored with the inputs, loads no model and computes no performance metric.
Not part of the confirmatory protocol or of amendments A1 and A2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "sleep_edf": ROOT / "data/processed",
    "shhs1_holdout": ROOT / "data/processed/shhs_holdout_v1",
}
VARIANTS = ("paper_raw_v1", "bandpass_v2")
FS = 100.0
NPERSEG = 2000  # 20 s Hann segments, 0.05 Hz resolution
BANDS = {"below_0.5_hz": (0.0, 0.5), "0.5_to_30_hz": (0.5, 30.0), "above_30_hz": (30.0, 50.0)}
PLOT_FREQUENCIES = np.unique(np.round(np.geomspace(0.05, 50.0, 90) / 0.05) * 0.05)


def record_psd(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as z:
        x = z["x"][z["valid_mask"]].astype(np.float64)
    offset = float(x.mean())
    frequencies, power = welch(x, fs=FS, window="hann", nperseg=NPERSEG, noverlap=NPERSEG // 2,
                               detrend="constant", axis=-1)
    return frequencies, power.mean(axis=0), offset


def quartiles(values: np.ndarray) -> dict[str, float]:
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(q50), "q25": float(q25), "q75": float(q75)}


def summarise(folder: Path) -> dict:
    paths = sorted(folder.glob("*.npz"))
    spectra, offsets = [], []
    for path in paths:
        frequencies, power, offset = record_psd(path)
        spectra.append(power)
        offsets.append(abs(offset))
    spectra = np.asarray(spectra)
    positive = frequencies > 0  # the zero bin is removed by per-segment mean subtraction
    total = spectra[:, positive].sum(axis=1)
    fractions = {}
    for name, (low, high) in BANDS.items():
        band = (frequencies > low if low == 0.0 else frequencies >= low) & (frequencies < high)
        if name == "above_30_hz":
            band = (frequencies >= low) & (frequencies <= high)
        fractions[name] = quartiles(spectra[:, band].sum(axis=1) / total)
    index = np.searchsorted(frequencies, PLOT_FREQUENCIES - 1e-9)
    decibel = 10.0 * np.log10(spectra[:, index])
    q25, q50, q75 = np.percentile(decibel, [25, 50, 75], axis=0)
    return {
        "recordings": len(paths),
        "absolute_record_mean_uv": quartiles(np.asarray(offsets)),
        "power_fraction": fractions,
        "frequency_hz": [round(float(f), 2) for f in frequencies[index]],
        "psd_db_median": [round(float(v), 3) for v in q50],
        "psd_db_q25": [round(float(v), 3) for v in q25],
        "psd_db_q75": [round(float(v), 3) for v in q75],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cohort_psd/psd_summary.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing summary; do not overwrite")
    result = {
        "schema_version": 1,
        "role": "post-hoc descriptive; no labels beyond the stored evaluation window, no model",
        "welch": {"fs_hz": FS, "window": "hann", "nperseg": NPERSEG, "noverlap": NPERSEG // 2,
                  "detrend": "constant", "unit": "dB re 1 uV^2/Hz"},
        "cohorts": {cohort: {variant: summarise(folder / variant) for variant in VARIANTS}
                    for cohort, folder in COHORTS.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    for cohort, variants in result["cohorts"].items():
        for variant, summary in variants.items():
            print(cohort, variant, summary["recordings"], summary["power_fraction"],
                  summary["absolute_record_mean_uv"])


if __name__ == "__main__":
    main()
