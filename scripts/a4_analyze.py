"""Amendment A4: which edge of the band-pass carries the transfer gain (descriptive, no tests).

Compares the one-edge arms E4HP (high-pass 0.5 Hz) and E4LP (low-pass 30 Hz) with the archived E2
(unfiltered) and E4 (band-pass) at the same seeds on Sleep-EDF (out-of-fold), the SHHS1 holdout and the
telemetry cohort, using the metric and bootstrap functions of the A3 analysis. See docs/AMENDMENT_A4.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/amendment_a4.json").read_text(encoding="utf8"))
BOOT = CONFIG["analysis"]["bootstrap"]

spec = importlib.util.spec_from_file_location("a3_analyze", ROOT / "scripts/a3_analyze.py")
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3)
a3.SEED, a3.RESAMPLES = BOOT["seed"], BOOT["resamples"]  # A4 uses its own locked bootstrap seed

COHORTS = {
    "sleep_edf_oof": (ROOT / "data/processed/bandpass_v2", lambda k: k[:5]),
    "shhs1_holdout": (ROOT / "data/processed/shhs_holdout_v1/bandpass_v2", lambda k: k),
    "telemetry": (ROOT / "data/processed/telemetry_a3/bandpass_v2", lambda k: k[:5]),
}
REFERENCE = {
    "sleep_edf_oof": ROOT / "runs/a3/sleep_edf_oof",
    "shhs1_holdout": ROOT / "runs/holdout_inference_v1/ensemble",
    "telemetry": ROOT / "runs/a3/telemetry/ensemble",
}
KEEP = ("subject_mean_macro_f1", "pooled_macro_f1", "accuracy", "kappa", "recall_N3", "n3_to_n2_rate",
        "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM")


def reference_predictions(cohort: str, arm: str, records: dict) -> dict[str, np.ndarray]:
    probabilities = a3.load_probabilities(REFERENCE[cohort] / arm, records)
    return {key: p.argmax(axis=1) for key, p in probabilities.items()}


def oof_predictions(arm: str, seed: int, records: dict) -> dict[str, np.ndarray]:
    """Sleep-EDF out-of-fold labels of an A4 arm from the locked test campaign (valid epochs only)."""
    predictions = {key: np.full(len(r["y"]), -1, dtype=np.int64) for key, r in records.items()}
    for fold in CONFIG["training"]["folds"]:
        path = ROOT / "runs/a4/full" / arm / f"fold_{fold:02d}" / f"seed_{seed}" / "predictions" / "test.npz"
        with np.load(path, allow_pickle=False) as z:
            for key in np.unique(z["record_key"]):
                rows = z["record_key"] == key
                position = np.searchsorted(records[key]["index"], z["original_epoch_index"][rows])
                if not np.array_equal(records[key]["index"][position], z["original_epoch_index"][rows]):
                    raise ValueError(f"{arm} s{seed} {key}: epoch index mismatch")
                if not np.array_equal(records[key]["y"][position], z["true_label"][rows]):
                    raise ValueError(f"{arm} s{seed} {key}: label mismatch")
                predictions[key][position] = z["predicted_label"][rows]
    for key, record in records.items():
        if np.any(predictions[key][record["valid"]] < 0):
            raise ValueError(f"{arm} s{seed} {key}: missing out-of-fold predictions")
    return predictions


def a4_predictions(cohort: str, arm: str, seed: int, records: dict) -> dict[str, np.ndarray]:
    if cohort == "sleep_edf_oof":
        return oof_predictions(arm, seed, records)
    folder = ROOT / "runs/a4/zero_shot" / cohort / "ensemble" / f"{arm}_s{seed}"
    return {key: p.argmax(axis=1) for key, p in a3.load_probabilities(folder, records).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/a4/analysis/a4_summary.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing summary; do not overwrite")
    report = {"schema_version": 1, "amendment": "A4", "role": "descriptive", "bootstrap": BOOT, "cohorts": {}}
    for cohort, (folder, subject_rule) in COHORTS.items():
        records = a3.load_cohort(folder, subject_rule)
        matrices = {}
        for seed in (42, 123):
            for arm in ("E2", "E4"):
                _, matrices[f"{arm}_s{seed}"] = a3.subject_confusions(
                    records, reference_predictions(cohort, f"{arm}_s{seed}", records))
            for arm in CONFIG["training"]["arms"]:
                _, matrices[f"{arm}_s{seed}"] = a3.subject_confusions(
                    records, a4_predictions(cohort, arm, seed, records))
        weights = a3.bootstrap_weights(len(next(iter(matrices.values()))))
        result = {"subjects": int(weights.shape[1]), "arms": {}, "contrasts": {}, "share_of_gain": {}}
        for name, matrix in matrices.items():
            summary = a3.summary(matrix, weights)
            result["arms"][name] = {k: summary[k] for k in KEEP}
        for seed in (42, 123):
            for contrast in CONFIG["analysis"]["contrasts_per_seed"]:
                left, right = contrast.split("-")
                delta = a3.difference(matrices[f"{left}_s{seed}"], matrices[f"{right}_s{seed}"], weights)
                result["contrasts"][f"{contrast}_s{seed}"] = {k: delta[k] for k in KEEP}
            gain = a3.difference(matrices[f"E4_s{seed}"], matrices[f"E2_s{seed}"], weights)["subject_mean_macro_f1"]
            if gain["ci95"][0] > 0:
                for arm in CONFIG["training"]["arms"]:
                    edge = result["contrasts"][f"{arm}-E2_s{seed}"]["subject_mean_macro_f1"]["value"]
                    result["share_of_gain"][f"{arm}_s{seed}"] = edge / gain["value"]
        verdicts = {}
        for arm in CONFIG["training"]["arms"]:
            above = all(result["contrasts"][f"{arm}-E2_s{s}"]["subject_mean_macro_f1"]["ci95"][0] > 0
                        for s in (42, 123))
            not_worse = all(result["contrasts"][f"E4-{arm}_s{s}"]["subject_mean_macro_f1"]["ci95"][0] <= 0
                            for s in (42, 123))
            verdicts[arm] = {"differs_from_E2_at_both_seeds": above, "E4_not_better_at_both_seeds": not_worse,
                             "carries_gain": above and not_worse}
        result["verdicts"] = verdicts
        report["cohorts"][cohort] = result
        print(cohort, json.dumps({a: round(v["subject_mean_macro_f1"]["value"], 4)
                                  for a, v in result["arms"].items()}), verdicts, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
