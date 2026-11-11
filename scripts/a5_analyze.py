"""Amendment A5 analysis: amplitude-gated N3 correction and channel contrast on ISRUC (descriptive).

Implements configs/amendment_a5.json with the metric and bootstrap functions of the A3 analysis. Gates and
thresholds come from Sleep-EDF only. Writes runs/final/a5_summary.json once.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/amendment_a5.json").read_text(encoding="utf8"))
spec = importlib.util.spec_from_file_location("a3_analyze", ROOT / "scripts/a3_analyze.py")
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3)
a3.SEED = CONFIG["analysis"]["bootstrap"]["seed"]
a3.RESAMPLES = CONFIG["analysis"]["bootstrap"]["resamples"]
MIN_EPOCHS = 20
KEEP = ("subject_mean_macro_f1", "pooled_macro_f1", "accuracy", "kappa", "recall_N3", "precision_N3",
        "n3_to_n2_rate", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM")


def amplitude_disagreement(p: np.ndarray, sd: np.ndarray, tau: float) -> float:
    pred = p.argmax(1)
    high = (sd >= tau) & np.isin(pred, (2, 3))
    return float((pred[high] == 2).mean()) if high.sum() >= MIN_EPOCHS else float("nan")


def mean_entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1).mean())


def load_isruc(folder: Path, scorer: int) -> dict[str, dict]:
    records = {}
    for path in sorted(folder.glob("*.npz")):
        record = a3.load_record(path)
        if scorer == 2:
            with np.load(path, allow_pickle=False) as z:
                y2 = z["y_scorer2"].astype(np.int64)
            record["y"], record["valid"] = np.where(y2 < 0, 0, y2), y2 >= 0
        record["subject"] = path.stem
        records[path.stem] = record
    return records


def predictions_for(records: dict, probabilities: dict, fit: dict, gates: dict) -> dict[str, dict]:
    out = {"unmodified": {}, "ungated_R1": {}, "amplitude_gated": {}, "entropy_gated": {}}
    fired = {"amplitude_gated": 0, "entropy_gated": 0}
    for key, p in probabilities.items():
        sd = records[key]["sd"]
        base, corrected = p.argmax(1), a3.rule_hard(p, sd, fit)
        out["unmodified"][key], out["ungated_R1"][key] = base, corrected
        statistic = amplitude_disagreement(p, sd, fit["threshold_uv"])
        amp = not np.isnan(statistic) and statistic > gates["amplitude"]
        ent = mean_entropy(p) > gates["entropy"]
        out["amplitude_gated"][key] = corrected if amp else base
        out["entropy_gated"][key] = corrected if ent else base
        fired["amplitude_gated"] += int(amp)
        fired["entropy_gated"] += int(ent)
    return out, fired


def compare(records: dict, predictions: dict, weights) -> dict:
    matrices = {name: a3.subject_confusions(records, pred)[1] for name, pred in predictions.items()}
    result = {"arms": {name: {k: v for k, v in a3.summary(m, weights).items() if k in KEEP}
                       for name, m in matrices.items()}}
    for name in ("ungated_R1", "amplitude_gated", "entropy_gated"):
        delta = a3.difference(matrices[name], matrices["unmodified"], weights)
        result[f"{name}_minus_unmodified"] = {k: delta[k] for k in KEEP}
    delta = a3.difference(matrices["amplitude_gated"], matrices["entropy_gated"], weights)
    result["amplitude_minus_entropy_gated"] = {k: delta[k] for k in KEEP}
    return result, matrices


def detector(records: dict, probabilities: dict, tau: float) -> dict:
    rows = []
    for key, p in probabilities.items():
        r = records[key]
        y, pred = r["y"][r["valid"]], p.argmax(1)[r["valid"]]
        if (y == 3).sum() >= MIN_EPOCHS:
            rows.append(((pred[y == 3] == 3).mean(), amplitude_disagreement(p, r["sd"], tau), mean_entropy(p)))
    rows = np.array(rows, dtype=float)
    ok = ~np.isnan(rows[:, 1])
    return {"recordings": int(len(rows)),
            "spearman_amplitude_vs_n3_recall": float(spearmanr(rows[ok, 1], rows[ok, 0]).correlation),
            "spearman_entropy_vs_n3_recall": float(spearmanr(rows[:, 2], rows[:, 0]).correlation)}


def stage_ratio(records: dict) -> dict:
    ratios = []
    for r in records.values():
        y, sd, v = r["y"], r["sd"], r["valid"]
        n3, n2 = sd[v & (y == 3)], sd[v & (y == 2)]
        if n3.size >= 10 and n2.size >= 10:
            ratios.append(np.median(n3) / np.median(n2))
    q = np.percentile(ratios, [25, 50, 75])
    return {"recordings": len(ratios), "median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isruc-root", type=Path, default=ROOT / "data/processed/isruc_a5")
    parser.add_argument("--predictions", type=Path, default=ROOT / "runs/final/a5")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/final/a5_summary.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing summary; do not overwrite")
    arms = CONFIG["arms"]
    source = a3.load_cohort(ROOT / "data/processed/bandpass_v2", lambda k: k[:5])
    fit = a3.fit_source(source, sorted(source))
    report = {"schema_version": 1, "amendment": "A5", "threshold_uv": fit["threshold_uv"], "gates": {},
              "sleep_edf_oof": {}, "isruc": {}, "channel_contrast": {}, "verdicts": {}}

    # gates and source cost, from Sleep-EDF out-of-fold predictions of each arm
    source_weights = a3.bootstrap_weights(len({r["subject"] for r in source.values()}))
    for arm in arms:
        probabilities = a3.load_probabilities(ROOT / "runs/a3/sleep_edf_oof" / arm, source)
        amp = [amplitude_disagreement(p, source[k]["sd"], fit["threshold_uv"]) for k, p in probabilities.items()]
        ent = [mean_entropy(p) for p in probabilities.values()]
        gates = {"amplitude": float(np.nanquantile(amp, 0.95)), "entropy": float(np.quantile(ent, 0.95))}
        report["gates"][arm] = gates
        predictions, fired = predictions_for(source, probabilities, fit, gates)
        result, _ = compare(source, predictions, source_weights)
        report["sleep_edf_oof"][arm] = {"fired": fired, **result}

    matrices_by = {}
    for channel in ("C4", "F4"):
        for scorer in (1, 2):
            records = load_isruc(args.isruc_root / channel / "bandpass_v2", scorer)
            weights = a3.bootstrap_weights(len(records))
            block = {"recordings": len(records), "stage_sd_ratio_n3_over_n2": stage_ratio(records), "arms": {}}
            for arm in (arms if scorer == 1 else [CONFIG["primary_arm"]]):
                probabilities = a3.load_probabilities(args.predictions / channel / arm, records)
                predictions, fired = predictions_for(records, probabilities, fit, report["gates"][arm])
                result, matrices = compare(records, predictions, weights)
                result["fired"] = fired
                result["detector"] = detector(records, probabilities, fit["threshold_uv"])
                block["arms"][arm] = result
                if scorer == 1:
                    matrices_by[(channel, arm)] = matrices["unmodified"]
            if scorer == 1:
                block["transfer_contrasts"] = {}
                for seed in (42, 123):
                    for contrast in CONFIG["analysis"]["transfer_contrasts"]:
                        left, right = contrast.split("-")
                        delta = a3.difference(matrices_by[(channel, f"{left}_s{seed}")],
                                              matrices_by[(channel, f"{right}_s{seed}")], weights)
                        block["transfer_contrasts"][f"{contrast}_s{seed}"] = {k: delta[k] for k in KEEP}
            report["isruc"][f"{channel}_scorer{scorer}"] = block
    weights = a3.bootstrap_weights(report["isruc"]["C4_scorer1"]["recordings"])
    for arm in arms:
        delta = a3.difference(matrices_by[("F4", arm)], matrices_by[("C4", arm)], weights)
        report["channel_contrast"][f"F4_minus_C4_{arm}"] = {k: delta[k] for k in KEEP}

    # locked interpretation rules
    primary = CONFIG["primary_arm"]
    c4 = report["isruc"]["C4_scorer1"]["arms"][primary]
    f4 = report["isruc"]["F4_scorer1"]["arms"][primary]
    gain = c4["amplitude_gated_minus_unmodified"]["subject_mean_macro_f1"]["ci95"]
    recall = c4["arms"]["unmodified"]["recall_N3"]["value"]
    if recall >= 0.7:
        rule = "not tested (no N3 under-detection on ISRUC C4)"
    elif (gain[0] > 0 and f4["amplitude_gated_minus_unmodified"]["subject_mean_macro_f1"]["ci95"][0] > -0.01
          and report["sleep_edf_oof"][primary]["amplitude_gated_minus_unmodified"]["subject_mean_macro_f1"]["ci95"][0]
          > -0.01):
        rule = "supported"
    elif recall < 0.5:
        rule = "not supported"
    else:
        rule = "inconclusive (N3 recall between 0.5 and 0.7 and the support criteria not met)"
    versus = c4["amplitude_minus_entropy_gated"]["subject_mean_macro_f1"]["ci95"]
    derivation = all(report["channel_contrast"][f"F4_minus_C4_{e}_s{s}"]["recall_N3"]["ci95"][0] > 0
                     for e in ("E3", "E4") for s in (42, 123))
    report["verdicts"] = {
        "rule": rule, "primary_unmodified_n3_recall_c4": recall,
        "amplitude_gate_better_than_entropy_gate": bool(versus[0] > 0),
        "derivation_related_n3_deficit": derivation,
        "transfer_replicates_c4_seed42": {c: report["isruc"]["C4_scorer1"]["transfer_contrasts"][f"{c}_s42"]
                                          ["subject_mean_macro_f1"]["ci95"][0] > 0
                                          for c in CONFIG["analysis"]["transfer_contrasts"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report["verdicts"], indent=2))


if __name__ == "__main__":
    main()
