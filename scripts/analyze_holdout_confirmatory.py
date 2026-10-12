"""Pre-specified analysis of the SHHS1 holdout (configs/shhs_holdout_confirmatory_v1.json, sections 6-8).

Run exactly once, after ``run_holdout_inference.py --mode finalize`` has produced
``run_manifest.json`` for all twelve arms. The report is write-once.

Statistical machinery is reused from the v1 SHHS analysis (``sleeptcn.shhs_analysis``):
subject-level macro-F1 with five fixed classes, one shared matrix of paired subject-cluster
bootstrap draws (10,000 resamples, protocol seed), percentile 95% intervals, two-sided
Wilcoxon signed-rank on subject macro-F1 and Holm adjustment within each family.
Labels are read from the preprocessed holdout inputs (hash-verified); prediction artifacts
contain no labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion  # noqa: E402
from sleeptcn.shhs_analysis import (  # noqa: E402
    _absolute_subject_bootstrap,
    _mean_paired_bootstrap,
    _subject_macro_f1,
    _wilcoxon_summary,
    load_ensemble_predictions,
)
from sleeptcn.statistics import holm_adjust  # noqa: E402

ARMS = ["E3_s42", "E0_s42", "E2_s42", "E6_s42", "E4_s42", "ADAST_full_e30", "SourceOnly_full_e30",
        "E3_s123", "E0_s123", "E2_s123", "E6_s123", "E4_s123"]
STAGES = ("W", "N1", "N2", "N3", "REM")
DEVIATIONS = [
    "Inference used 6 parallel CPU worker processes with 2 threads each instead of one process "
    "(device, deterministic algorithms, checkpoints, ensembling and code path unchanged; the new "
    "path reproduced historical v1 fold probabilities within 1e-6 before the run).",
    "During inference, the launcher was restarted twice while earlier worker processes were still "
    "running. Duplicate workers computed some identical units; four workers stopped with Windows "
    "PermissionError when replacing a file already written by the other process. All 60,000 fold "
    "artifacts exist, no temporary file remained, and finalize re-verified every artifact.",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


# --------------------------------------------------------------------------- loading
def load_labels(pre: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reference labels from the hash-verified filtered_v2 inputs (identical across variants)."""
    labels = {}
    for record in pre["records"]:
        if record["variant"] != "filtered_v2":
            continue
        path = Path(record["output_path"])
        if sha256_file(path) != record["output_sha256"]:
            raise ValueError(f"{path.name}: input hash differs from preprocess manifest")
        with np.load(path, allow_pickle=False) as z:
            labels[record["record_key"]] = {
                "subject_id": str(z["subject_id"].item()),
                "y": z["y"].astype(np.int8),
                "original_epoch_index": z["original_epoch_index"].astype(np.int32),
            }
    return labels


def load_arm(output_root: Path, arm: str, arm_manifest: dict[str, Any], pre_index: dict[tuple[str, str], str],
             labels: dict[str, dict[str, Any]], variant: str, protocol_sha: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted subject ids, subject confusion matrices) for one arm."""
    subjects, confusions = [], []
    for entry in sorted(arm_manifest["ensemble_records"], key=lambda e: e["subject_id"]):
        path = output_root / "ensemble" / arm / f"{entry['record_key']}.npz"
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{arm}/{entry['record_key']}: ensemble hash differs from arm manifest")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"].item()))
            prediction = z["prediction"].astype(np.int8)
            index = z["original_epoch_index"].astype(np.int32)
        if (meta["arm"] != arm or meta["protocol_sha256"] != protocol_sha
                or meta["input_npz_sha256"] != pre_index[(variant, entry["record_key"])]):
            raise ValueError(f"{arm}/{entry['record_key']}: ensemble provenance differs")
        ref = labels[entry["record_key"]]
        if not np.array_equal(index, ref["original_epoch_index"]) or ref["subject_id"] != entry["subject_id"]:
            raise ValueError(f"{arm}/{entry['record_key']}: prediction not aligned with reference labels")
        valid = ref["y"] >= 0
        subjects.append(entry["subject_id"])
        confusions.append(confusion_matrix_5(ref["y"][valid], prediction[valid]))
    return np.asarray(subjects), np.stack(confusions)


# --------------------------------------------------------------------------- vectorised pooled metrics
def pooled_stats(conf: np.ndarray) -> dict[str, np.ndarray]:
    """Pooled metrics for confusion arrays of shape (..., 5, 5); zero when a denominator is zero."""
    conf = conf.astype(np.float64)
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    support = conf.sum(-1)
    predicted = conf.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        total = conf.sum((-1, -2))
        accuracy = tp.sum(-1) / total
        expected = (support * predicted).sum(-1) / (total * total)
        kappa = np.where(expected < 1, (accuracy - expected) / (1 - expected), 0.0)
        n3_to_n2 = np.where(support[..., 3] > 0, conf[..., 3, 2] / support[..., 3], 0.0)
    out = {"macro_f1": f1.mean(-1), "accuracy": accuracy, "cohen_kappa": kappa, "n3_to_n2_rate": n3_to_n2}
    for i, stage in enumerate(STAGES):
        out[f"{stage.lower()}_precision"] = precision[..., i]
        out[f"{stage.lower()}_recall"] = recall[..., i]
        out[f"{stage.lower()}_f1"] = f1[..., i]
    return out


def draw_weights(samples: np.ndarray, n: int) -> np.ndarray:
    weights = np.zeros((len(samples), n), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(len(samples)), samples.shape[1]), samples.ravel()), 1.0)
    return weights


def pooled_distribution(conf: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    resampled = (weights @ conf.reshape(len(conf), 25).astype(np.float64)).reshape(-1, 5, 5)
    return pooled_stats(resampled)


def interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def present_class_macro_f1(confusions: np.ndarray) -> np.ndarray:
    stats = pooled_stats(confusions)
    f1 = np.stack([stats[f"{s.lower()}_f1"] for s in STAGES], -1)
    present = confusions.sum(-1) > 0
    return (f1 * present).sum(-1) / present.sum(-1)


# --------------------------------------------------------------------------- analyses
class Analysis:
    def __init__(self, subjects: np.ndarray, confusions: dict[str, np.ndarray], resamples: int, seed: int) -> None:
        self.subjects = subjects
        self.conf = confusions
        self.values = {arm: _subject_macro_f1(c) for arm, c in confusions.items()}
        rng = np.random.default_rng(seed)
        self.samples = rng.integers(0, len(subjects), size=(resamples, len(subjects)), dtype=np.int32)
        self.weights = draw_weights(self.samples, len(subjects))
        self._dist: dict[str, dict[str, np.ndarray]] = {}

    def dist(self, arm: str) -> dict[str, np.ndarray]:
        if arm not in self._dist:
            self._dist[arm] = pooled_distribution(self.conf[arm], self.weights)
        return self._dist[arm]

    def pooled_difference(self, left: str, right: str, stat: str) -> dict[str, Any]:
        observed = float(pooled_stats(self.conf[left].sum(0))[stat] - pooled_stats(self.conf[right].sum(0))[stat])
        return {"observed_difference": observed, "ci95": interval(self.dist(left)[stat] - self.dist(right)[stat])}

    def pooled_single(self, arm: str, stat: str) -> dict[str, Any]:
        return {"estimate": float(pooled_stats(self.conf[arm].sum(0))[stat]), "ci95": interval(self.dist(arm)[stat])}

    def contrast(self, left: str, right: str, *, with_test: bool, values: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        values = values or self.values
        primary = _mean_paired_bootstrap(values[left], values[right], self.samples)
        out = {"contrast": f"{left} - {right}",
               "subject_mean_macro_f1_difference": primary["observed_difference"],
               "ci95": [primary["ci95_low"], primary["ci95_high"]]}
        if with_test:
            out["wilcoxon"] = _wilcoxon_summary(values[left], values[right])
        out["pooled_macro_f1"] = self.pooled_difference(left, right, "macro_f1")
        out["pooled_accuracy"] = self.pooled_difference(left, right, "accuracy")
        out["pooled_cohen_kappa"] = self.pooled_difference(left, right, "cohen_kappa")
        return out

    def family(self, pairs: list[tuple[str, str]], values: dict[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
        rows = [self.contrast(l, r, with_test=True, values=values) for l, r in pairs]
        for row, p in zip(rows, holm_adjust([row["wilcoxon"]["p_value"] for row in rows]), strict=True):
            row["wilcoxon"]["holm_adjusted_p_value"] = p
            row["wilcoxon"]["holm_family_size"] = len(rows)
            row["confirmed"] = bool(p < 0.05 and row["ci95"][0] > 0)
        return rows

    def describe(self, arm: str) -> dict[str, Any]:
        pooled = metrics_from_confusion(self.conf[arm].sum(0))
        vec = pooled_stats(self.conf[arm].sum(0))
        if not np.isclose(vec["macro_f1"], pooled["macro_f1"]) or not np.isclose(vec["cohen_kappa"], pooled["cohen_kappa"]):
            raise AssertionError("vectorised metrics disagree with sleeptcn.metrics")
        absolute = _absolute_subject_bootstrap(self.values[arm], self.samples)
        return {"subject_mean_macro_f1": absolute["estimate"],
                "subject_mean_macro_f1_ci95": [absolute["ci95_low"], absolute["ci95_high"]],
                "pooled": pooled, "n3_to_n2_rate": float(vec["n3_to_n2_rate"])}


def pair(text: str) -> tuple[str, str]:
    left, right = (part.strip() for part in text.split(" - "))
    return left, right


def run_analysis(protocol: dict[str, Any], confusions: dict[str, np.ndarray], subjects: np.ndarray,
                 extra: dict[str, Any]) -> dict[str, Any]:
    stats = protocol["statistics"]
    a = Analysis(subjects, confusions, int(stats["bootstrap"]["resamples"]), int(stats["bootstrap"]["seed"]))
    hyp = protocol["hypotheses"]
    conf_pairs = [pair(hyp["confirmatory_family"][h]["contrast"]) for h in ("H1", "H2", "H3")]
    sec = hyp["secondary"]
    report: dict[str, Any] = {"descriptive": {arm: a.describe(arm) for arm in confusions}}
    family = a.family(conf_pairs)
    for name, row in zip(("H1", "H2", "H3"), family, strict=True):
        row["hypothesis"] = name
        row["v1_estimate"] = hyp["confirmatory_family"][name]["v1_estimate"]
    report["confirmatory_family"] = family
    report["S1_seed123_replication"] = a.family([pair(c) for c in sec["S1_seed123_replication"]["contrasts"]])
    report["S2_null_pair_and_run_to_run"] = [a.contrast(*pair(c), with_test=False)
                                              for c in sec["S2_null_pair_and_run_to_run_reference"]["contrasts"]]
    s3 = {"contrast": a.contrast("ADAST_full_e30", "SourceOnly_full_e30", with_test=False)}
    for stat in ("n3_recall", "n1_f1", "n2_f1", "rem_f1"):
        s3[stat] = a.pooled_difference("ADAST_full_e30", "SourceOnly_full_e30", stat)
    s3["absolute"] = {f"ADAST_full_e30 - {ref}": a.contrast("ADAST_full_e30", ref, with_test=False)
                      for ref in ("E3_s42", "E4_s42")}
    report["S3_adast"] = s3
    report["S4_n3_under_detection"] = {
        arm: {stat: a.pooled_single(arm, stat) for stat in ("n3_recall", "n3_precision", "n3_to_n2_rate")}
        for arm in sec["S4_n3_under_detection"]["arms"]}
    report["S4_expectation_met"] = all(report["S4_n3_under_detection"][arm]["n3_to_n2_rate"]["estimate"] > 0.5
                                       for arm in ("E0_s42", "E3_s42"))
    present = {arm: present_class_macro_f1(c) for arm, c in confusions.items()}
    report["S5_present_class_macro_f1"] = {
        "subjects_with_absent_reference_class": int((confusions["E3_s42"].sum(-1) == 0).any(-1).sum()),
        "absent_class_counts": {s: int((confusions["E3_s42"].sum(-1)[:, i] == 0).sum()) for i, s in enumerate(STAGES)},
        "family": a.family(conf_pairs, values=present)}
    report.update(extra)
    return report


def pooled_with_v1(protocol: dict[str, Any], holdout: dict[str, np.ndarray], holdout_subjects: np.ndarray,
                   v1_zero_shot: Path, v1_components: Path) -> dict[str, Any]:
    """S7: descriptive H1-H3 on v1 test (180) + holdout (500)."""
    v1 = {}
    for exp, manifest_path in (("E0", v1_zero_shot), ("E3", v1_zero_shot), ("E6", v1_zero_shot), ("E2", v1_components)):
        predictions = load_ensemble_predictions(read_json(manifest_path), exp).sorted()
        subjects = np.unique(predictions.subject_id)
        v1[exp] = np.stack([confusion_matrix_5(predictions.true_label[predictions.subject_id == s],
                                               predictions.predicted_label[predictions.subject_id == s]) for s in subjects])
        v1_subjects = subjects
    if set(v1_subjects) & set(holdout_subjects):
        raise ValueError("v1 test and holdout subjects overlap")
    combined = {f"{exp}_s42": np.concatenate([v1[exp], holdout[f"{exp}_s42"]]) for exp in ("E0", "E2", "E3", "E6")}
    stats = protocol["statistics"]["bootstrap"]
    a = Analysis(np.concatenate([v1_subjects, holdout_subjects]), combined, int(stats["resamples"]), int(stats["seed"]))
    pairs = [pair(protocol["hypotheses"]["confirmatory_family"][h]["contrast"]) for h in ("H1", "H2", "H3")]
    return {"subjects": int(len(a.subjects)), "note": "descriptive only; never used for confirmation",
            "source_sha256": {"v1_zero_shot_run_manifest": sha256_file(v1_zero_shot),
                              "v1_components_run_manifest": sha256_file(v1_components)},
            "contrasts": [a.contrast(l, r, with_test=False) for l, r in pairs]}


# --------------------------------------------------------------------------- report
def fmt(x: float) -> str:
    return f"{x:+.4f}"


def markdown(report: dict[str, Any]) -> str:
    lines = ["# SHHS1 holdout: pre-specified analysis", "",
             f"Subjects: {report['subjects']}; valid epochs: {report['valid_epochs']}.", "",
             "## Confirmatory family (Holm over 3, seed 42)", "",
             "| | Contrast | Δ subject-mean MF1 | 95% CI | Holm p | W/T/L | Confirmed | v1 |",
             "|---|---|---:|---|---:|---|---|---:|"]
    for row in report["confirmatory_family"]:
        w = row["wilcoxon"]
        lines.append(f"| {row['hypothesis']} | {row['contrast']} | {fmt(row['subject_mean_macro_f1_difference'])} | "
                     f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] | {w['holm_adjusted_p_value']:.3g} | "
                     f"{w['wins']}/{w['ties']}/{w['losses']} | {'yes' if row['confirmed'] else 'no'} | {row['v1_estimate']:+.4f} |")
    lines += ["", "## Descriptive", "", "| Arm | Subject-mean MF1 (95% CI) | Pooled MF1 | Accuracy | κ | N3 recall | N3→N2 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for arm, d in report["descriptive"].items():
        p = d["pooled"]
        lines.append(f"| {arm} | {d['subject_mean_macro_f1']:.4f} [{d['subject_mean_macro_f1_ci95'][0]:.4f}, "
                     f"{d['subject_mean_macro_f1_ci95'][1]:.4f}] | {p['macro_f1']:.4f} | {p['accuracy']:.4f} | "
                     f"{p['cohen_kappa']:.4f} | {p['per_class']['N3']['recall']:.4f} | {d['n3_to_n2_rate']:.3f} |")
    lines += ["", "## S1 seed-123 replication (separate Holm)", ""]
    for row in report["S1_seed123_replication"]:
        lines.append(f"- {row['contrast']}: {fmt(row['subject_mean_macro_f1_difference'])} "
                     f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}], Holm p={row['wilcoxon']['holm_adjusted_p_value']:.3g}")
    lines += ["", "## S2 null pair and run-to-run reference (CI only)", ""]
    for row in report["S2_null_pair_and_run_to_run"]:
        lines.append(f"- {row['contrast']}: {fmt(row['subject_mean_macro_f1_difference'])} [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]")
    s3 = report["S3_adast"]
    lines += ["", "## S3 ADAST", "",
              f"- ADAST − source-only: {fmt(s3['contrast']['subject_mean_macro_f1_difference'])} "
              f"[{s3['contrast']['ci95'][0]:.4f}, {s3['contrast']['ci95'][1]:.4f}]"]
    for stat in ("n3_recall", "n1_f1", "n2_f1", "rem_f1"):
        lines.append(f"- Δ {stat}: {fmt(s3[stat]['observed_difference'])} [{s3[stat]['ci95'][0]:.4f}, {s3[stat]['ci95'][1]:.4f}]")
    for name, row in s3["absolute"].items():
        lines.append(f"- {name}: {fmt(row['subject_mean_macro_f1_difference'])} [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]")
    lines += ["", "## S4 N3 under-detection", ""]
    for arm, d in report["S4_n3_under_detection"].items():
        lines.append(f"- {arm}: recall {d['n3_recall']['estimate']:.3f}, precision {d['n3_precision']['estimate']:.3f}, "
                     f"N3→N2 {d['n3_to_n2_rate']['estimate']:.3f} [{d['n3_to_n2_rate']['ci95'][0]:.3f}, {d['n3_to_n2_rate']['ci95'][1]:.3f}]")
    lines.append(f"- Expectation (N3→N2 > 0.5 for E0 and E3): {'met' if report['S4_expectation_met'] else 'not met'}")
    s5 = report["S5_present_class_macro_f1"]
    lines += ["", "## S5 present-class macro-F1", "",
              f"Subjects with at least one absent reference class: {s5['subjects_with_absent_reference_class']} "
              f"({s5['absent_class_counts']})."]
    for row in s5["family"]:
        lines.append(f"- {row['contrast']}: {fmt(row['subject_mean_macro_f1_difference'])} "
                     f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}], Holm p={row['wilcoxon']['holm_adjusted_p_value']:.3g}")
    s6 = report["S6_e6_input_scale"]
    lines += ["", "## S6 E6 input scale (label-free)", "",
              f"In-window SD of filtered_zscore_v2 inputs: median {s6['median']:.3f} (IQR {s6['q1']:.3f}–{s6['q3']:.3f}); Sleep-EDF median 0.77."]
    if "S7_pooled_with_v1" in report:
        lines += ["", "## S7 pooled with v1 (descriptive only)", ""]
        for row in report["S7_pooled_with_v1"]["contrasts"]:
            lines.append(f"- {row['contrast']}: {fmt(row['subject_mean_macro_f1_difference'])} [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]")
    lines += ["", "## Deviations", ""] + [f"- {d}" for d in report["deviations"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "shhs_holdout_confirmatory_v1.json")
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, default=ROOT / "runs" / "holdout_inference_v1")
    parser.add_argument("--v1-zero-shot-manifest", type=Path)
    parser.add_argument("--v1-components-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "holdout_analysis_v1" / "analysis.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists: the pre-specified analysis is run exactly once")

    protocol_raw = args.protocol.read_bytes()
    protocol, protocol_sha = json.loads(protocol_raw), sha256_bytes(protocol_raw)
    run = read_json(args.inference_root / "run_manifest.json")
    if run.get("status") != "complete_all_arms_no_metrics_computed" or run.get("protocol_sha256") != protocol_sha:
        raise ValueError("Inference is not complete for all arms under this protocol")
    pre_raw = args.preprocess_manifest.read_bytes()
    pre = json.loads(pre_raw)
    if sha256_bytes(pre_raw) != run["preprocess_manifest_sha256"]:
        raise ValueError("Preprocess manifest differs from the one used for inference")
    labels = load_labels(pre)
    pre_index = {(r["variant"], r["record_key"]): r["output_sha256"] for r in pre["records"]}
    variants = {arm: ("filtered_v2" if arm in ("ADAST_full_e30", "SourceOnly_full_e30")
                      else protocol["arms"][arm]["data_variant"]) for arm in ARMS}

    confusions, subjects = {}, None
    for arm in ARMS:
        manifest_path = args.inference_root / "arm_manifests" / f"{arm}.json"
        if sha256_file(manifest_path) != run["arm_manifest_sha256"][arm]:
            raise ValueError(f"{arm}: arm manifest differs from run manifest")
        subj, conf = load_arm(args.inference_root, arm, read_json(manifest_path), pre_index, labels,
                              variants[arm], protocol_sha)
        if subjects is None:
            subjects = subj
        elif not np.array_equal(subjects, subj):
            raise ValueError(f"{arm}: subject set differs")
        confusions[arm] = conf
    assert subjects is not None
    if len(subjects) != protocol["cohort"]["n_target"]:
        raise ValueError("Analysed subjects differ from protocol cohort size")

    extra: dict[str, Any] = {
        "S6_e6_input_scale": {**pre["summary"]["s6_zscore_input_std_in_window"], "sleep_edf_median": 0.77},
        "deviations": DEVIATIONS,
    }
    report = run_analysis(protocol, confusions, subjects, extra)
    if args.v1_zero_shot_manifest and args.v1_components_manifest:
        report["S7_pooled_with_v1"] = pooled_with_v1(protocol, confusions, subjects,
                                                     args.v1_zero_shot_manifest, args.v1_components_manifest)
    else:
        report["deviations"].append("S7 not run: v1 run manifests were not supplied.")
    report = {"schema_version": 1, "status": "complete", "protocol_sha256": protocol_sha,
              "inference_run_manifest_sha256": sha256_file(args.inference_root / "run_manifest.json"),
              "preprocess_manifest_sha256": run["preprocess_manifest_sha256"],
              "subjects": int(len(subjects)),
              "valid_epochs": int(sum(c.sum() for c in [confusions["E3_s42"]])),
              "bootstrap": protocol["statistics"]["bootstrap"], **report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    md = args.output.with_suffix(".md")
    md.write_text(markdown(report), encoding="utf-8")
    for path in (args.output, md):
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
