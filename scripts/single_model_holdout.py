"""Single fold models versus the ten-model ensemble on the SHHS1 holdout sample (post hoc, descriptive).

Reuses the archived per-fold probabilities of the confirmatory holdout inference; no model is rerun.
For every arm and seed it reports the point subject-mean macro-F1 and N3 recall of each of the ten
fold models, the ensemble value and the fold-matched E3-E0 and E4-E0 differences.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a3_analyze", ROOT / "scripts/a3_analyze.py")
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3)

HOLDOUT = ROOT / "data/processed/shhs_holdout_v1/bandpass_v2"
RUN = ROOT / "runs/holdout_inference_v1"
ARMS = ("E0", "E2", "E3", "E4", "E6")
SEEDS = (42, 123)
METRICS = ("subject_mean_macro_f1", "recall_N3")


def point(records: dict, folder: Path) -> dict[str, float]:
    probabilities = a3.load_probabilities(folder, records)
    preds = {k: p.argmax(axis=1) for k, p in probabilities.items()}
    values = a3.all_values(a3.subject_confusions(records, preds)[1])
    return {m: float(values[m]) for m in METRICS}


def describe(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/revision/single_model_holdout.json")
    args = parser.parse_args()
    records = a3.load_cohort(HOLDOUT, lambda k: k)
    folds: dict[str, list[dict]] = {}
    out: dict = {"participants": len(records), "arms": {}, "contrasts": {}}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            folds[name] = [point(records, RUN / "fold_predictions" / name / f"fold_{f:02d}") for f in range(10)]
            ensemble = point(records, RUN / "ensemble" / name)
            out["arms"][name] = {m: {"ensemble": ensemble[m], "single": describe([v[m] for v in folds[name]])}
                                 for m in METRICS}
            print(name, {m: round(out["arms"][name][m]["single"]["mean"], 4) for m in METRICS},
                  {m: round(ensemble[m], 4) for m in METRICS})
    for left in ("E3", "E4"):
        for seed in SEEDS:
            diffs = [l["subject_mean_macro_f1"] - r["subject_mean_macro_f1"]
                     for l, r in zip(folds[f"{left}_s{seed}"], folds[f"E0_s{seed}"])]
            out["contrasts"][f"{left}-E0_s{seed}"] = {**describe(diffs), "positive_folds": int(sum(d > 0 for d in diffs))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["contrasts"], indent=2))


if __name__ == "__main__":
    main()
