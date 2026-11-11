"""A5 development diagnostic (exploratory; development data only, no independent validation).

On Sleep-EDF out-of-fold predictions, the SHHS1 holdout and the telemetry cohort, compares two label-free
recording statistics as detectors of N3 under-detection (amplitude disagreement versus mean prediction
entropy), and the effect of the A5 correction gated by each. The gate and statistic were developed on these
same cohorts, so the results describe the idea and are not evidence for it (docs/AMENDMENT_A5.md, sections 2
and 7). Writes runs/a5_dev/development_diagnostic.json once.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a5_analyze", ROOT / "scripts/a5_analyze.py")
a5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a5)
a3 = a5.a3
ARMS = ("E0_s42", "E2_s42", "E3_s42", "E4_s42", "E0_s123", "E2_s123", "E3_s123", "E4_s123")
COHORTS = {
    "sleep_edf_oof": (ROOT / "data/processed/bandpass_v2", lambda k: k[:5], ROOT / "runs/a3/sleep_edf_oof"),
    "shhs1_holdout": (ROOT / "data/processed/shhs_holdout_v1/bandpass_v2", lambda k: k,
                      ROOT / "runs/holdout_inference_v1/ensemble"),
    "telemetry": (ROOT / "data/processed/telemetry_a3/bandpass_v2", lambda k: k[:5], ROOT / "runs/a3/telemetry/ensemble"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/a5_dev/development_diagnostic.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing summary; do not overwrite")
    records = {name: a3.load_cohort(folder, rule) for name, (folder, rule, _) in COHORTS.items()}
    fit = a3.fit_source(records["sleep_edf_oof"], sorted(records["sleep_edf_oof"]))
    tau = fit["threshold_uv"]
    report = {"schema_version": 1, "role": "exploratory development diagnostic; not evidence",
              "threshold_uv": tau, "bootstrap": {"resamples": a3.RESAMPLES, "seed": a3.SEED}, "arms": {}}
    for arm in ARMS:
        probs = {name: a3.load_probabilities(COHORTS[name][2] / arm, recs) for name, recs in records.items()}
        source = records["sleep_edf_oof"]
        gates = {"amplitude": float(np.nanquantile([a5.amplitude_disagreement(p, source[k]["sd"], tau)
                                                     for k, p in probs["sleep_edf_oof"].items()], 0.95)),
                 "entropy": float(np.quantile([a5.mean_entropy(p) for p in probs["sleep_edf_oof"].values()], 0.95))}
        entry = {"gates": gates, "cohorts": {}}
        for name, recs in records.items():
            weights = a3.bootstrap_weights(len({r["subject"] for r in recs.values()}))
            predictions, fired = a5.predictions_for(recs, probs[name], fit, gates)
            result, _ = a5.compare(recs, predictions, weights)
            within = a5.detector(recs, probs[name], tau)
            entry["cohorts"][name] = {"recordings": len(recs), "fired": fired, "detector_within_cohort": within,
                                      **{k: v for k, v in result.items() if k != "arms"},
                                      "unmodified": result["arms"]["unmodified"]}
        report["arms"][arm] = entry
        shhs = entry["cohorts"]["shhs1_holdout"]
        print(arm, "SHHS1 rho amp/entropy", round(shhs["detector_within_cohort"]["spearman_amplitude_vs_n3_recall"], 3),
              round(shhs["detector_within_cohort"]["spearman_entropy_vs_n3_recall"], 3),
              {n: round(c["amplitude_gated_minus_unmodified"]["subject_mean_macro_f1"]["value"], 4)
               for n, c in entry["cohorts"].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
