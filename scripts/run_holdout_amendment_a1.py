"""Amendment A1 to the SHHS1 holdout protocol: AdaBN and a test-time input-amplitude sweep for E4 seed 42.

Locked configuration: ``configs/shhs_holdout_amendment_a1.json`` (document
``docs/SHHS_HOLDOUT_AMENDMENT_A1.md``). The analyses are descriptive and post-confirmatory; they do
not change the H1-H3 decisions of the parent protocol.

Modes (run in this order; every stage is resumable and refuses to overwrite a different result):
  prepare   label-free AdaBN sample (40 epochs per record) and per-record amplitude statistics
  gate      checkpoint hash gate for E4 seed 42 and reproduction of stored parent fold probabilities
  worker    one of N parallel processes over the 5 conditions x 10 folds (streams records from disk)
  finalize  ten-fold ensembles and manifests once every fold artifact exists (no metrics)
  analyze   the pre-specified descriptive analysis, run once (write-once report)
  progress  print done/total fold artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import run_holdout_inference as rhi  # noqa: E402
from sleeptcn.evaluation.shhs_zero_shot import ensemble_probabilities  # noqa: E402
from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.io.serialization import atomic_savez, atomic_write_json  # noqa: E402

ARM = "E4_s42"
EXPERIMENT, SEED, VARIANT = "E4", 42, "bandpass_v2"
FOLDS = tuple(range(10))
SCALE_CURVE = ("scale_0.50", "scale_0.75", ARM, "scale_1.50", "scale_2.00")
STAGES = ("W", "N1", "N2", "N3", "REM")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


# --------------------------------------------------------------------------- configuration
def load_context(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.amendment.read_bytes()
    amendment = json.loads(raw)
    if amendment.get("status") != "locked_before_amendment_inference":
        raise ValueError("Amendment A1 is not locked")
    parent_sha = sha256_file(args.protocol)
    if parent_sha != amendment["parent_protocol"]["config_sha256"]:
        raise ValueError("Parent protocol differs from the one named in the amendment")
    ctx = rhi.load_inputs(args)  # verifies the parent lock and the preprocess manifest
    if ctx["preprocess_manifest_sha256"] != amendment["cohort"]["preprocess_manifest_sha256"]:
        raise ValueError("Preprocess manifest differs from the one named in the amendment")
    model = amendment["model"]
    if (model["arm"], model["experiment"], model["seed"], model["data_variant"]) != (ARM, EXPERIMENT, SEED, VARIANT):
        raise ValueError("Amendment model differs from this script")
    conditions = amendment["conditions"]
    ctx.update({"amendment": amendment, "amendment_sha256": sha256_bytes(raw), "conditions": conditions,
                "condition_names": list(conditions), "entries_main": ctx["entries"][VARIANT]})
    if len(ctx["entries_main"]) != amendment["cohort"]["participants"]:
        raise ValueError("Cohort size differs from the amendment")
    return ctx


def base_metadata(ctx: dict[str, Any], condition: str) -> dict[str, Any]:
    spec = ctx["conditions"][condition]
    return {"schema_version": 1, "amendment": "A1", "amendment_sha256": ctx["amendment_sha256"],
            "parent_protocol_sha256": ctx["protocol_sha256"],
            "preprocess_manifest_sha256": ctx["preprocess_manifest_sha256"], "arm": ARM,
            "condition": condition, "condition_type": spec["type"], "input_factor": float(spec["factor"]),
            "data_variant": VARIANT, "seed": SEED, "role": rhi.ROLE, "device": "cpu",
            "torch_version": torch.__version__}


# --------------------------------------------------------------------------- models
def load_models(args: argparse.Namespace, fold: int):
    from sleeptcn.experiment import build_context
    from sleeptcn.test_gate import _load_extractor, _load_sequence_model

    context = build_context(args.restore_workspace.resolve(), EXPERIMENT, fold, SEED, "cpu", smoke=False,
                            allow_test_evaluation=True, num_workers=0, resume=True)
    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    sequence_kind, sequence, sequence_path = _load_sequence_model(context)
    if extractor_kind != "resnet1d" or sequence_kind != "tcn":
        raise ValueError("E4 must use the ResNet-1D encoder and the TCN")
    hashes = {"extractor_sha256": extractor_hash, "sequence_checkpoint_sha256": sha256_file(sequence_path)}
    return extractor.eval(), sequence.eval(), hashes


@torch.no_grad()
def predict(extractor: nn.Module, sequence: nn.Module, x: np.ndarray, factor: float, batch_size: int) -> np.ndarray:
    """Same computation as shhs_zero_shot.infer_probabilities (ResNet + TCN), with an input factor."""
    extractor.eval()
    sequence.eval()
    parts = []
    for start in range(0, len(x), batch_size):
        signals = torch.from_numpy(x[start:start + batch_size]).unsqueeze(1)
        if factor != 1.0:
            signals = signals * factor
        parts.append(extractor.extract_features(signals).cpu().numpy())
    features = np.concatenate(parts).astype(np.float32, copy=False)
    logits = sequence(torch.from_numpy(features).unsqueeze(0), padding_mask=None)
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
    if probabilities.shape != (len(x), 5) or not np.isfinite(probabilities).all():
        raise ValueError("invalid probability matrix")
    if not np.allclose(probabilities.sum(1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("probabilities do not sum to one")
    return probabilities


class _StopForward(Exception):
    pass


def batchnorm_order(model: nn.Module) -> list[str]:
    """Names of BatchNorm1d layers in forward-execution order of extract_features."""
    order: list[str] = []
    handles = [m.register_forward_pre_hook(lambda _m, _i, name=name: order.append(name))
               for name, m in model.named_modules() if isinstance(m, nn.BatchNorm1d)]
    try:
        with torch.no_grad():
            model.eval().extract_features(torch.zeros(2, 1, 3000))
    finally:
        for handle in handles:
            handle.remove()
    seen: list[str] = []
    for name in order:
        if name not in seen:
            seen.append(name)
    return seen


@torch.no_grad()
def adapt_batchnorm(model: nn.Module, sample: np.ndarray, batch_size: int) -> dict[str, dict[str, np.ndarray]]:
    """AdaBN: replace running statistics layer by layer with target statistics (affine unchanged)."""
    model.eval()
    modules = dict(model.named_modules())
    state: dict[str, dict[str, np.ndarray]] = {}
    for name in batchnorm_order(model):
        bn = modules[name]
        acc: dict[str, Any] = {"n": 0, "s": None, "ss": None}

        def hook(_module, inputs, acc=acc):
            values = inputs[0].double()
            total = values.sum((0, 2))
            squares = (values * values).sum((0, 2))
            acc["n"] += values.shape[0] * values.shape[2]
            acc["s"] = total if acc["s"] is None else acc["s"] + total
            acc["ss"] = squares if acc["ss"] is None else acc["ss"] + squares
            raise _StopForward

        handle = bn.register_forward_pre_hook(hook)
        try:
            for start in range(0, len(sample), batch_size):
                try:
                    model.extract_features(torch.from_numpy(sample[start:start + batch_size]).unsqueeze(1))
                except _StopForward:
                    pass
        finally:
            handle.remove()
        mean = acc["s"] / acc["n"]
        var = (acc["ss"] / acc["n"] - mean * mean).clamp_min(0.0)
        bn.running_mean.copy_(mean.to(bn.running_mean.dtype))
        bn.running_var.copy_(var.to(bn.running_var.dtype))
        state[name] = {"running_mean": bn.running_mean.numpy().copy(), "running_var": bn.running_var.numpy().copy()}
    return state


def load_batchnorm_state(model: nn.Module, path: Path) -> None:
    modules = dict(model.named_modules())
    with np.load(path, allow_pickle=False) as z:
        names = [str(n) for n in z["layer_names"]]
        if names != batchnorm_order(model):
            raise ValueError(f"{path}: layer order differs")
        for name in names:
            modules[name].running_mean.copy_(torch.from_numpy(z[f"{name}.running_mean"]))
            modules[name].running_var.copy_(torch.from_numpy(z[f"{name}.running_var"]))


# --------------------------------------------------------------------------- prepare
def sample_indices(subject_id: str, n_epochs: int, per_record: int) -> np.ndarray:
    rng = np.random.default_rng([2041, int(subject_id)])
    return np.sort(rng.choice(n_epochs, size=per_record, replace=False)).astype(np.int32)


def prepare(args: argparse.Namespace, ctx: dict[str, Any]) -> dict[str, Any]:
    out = args.output_root
    manifest_path = out / "prepare_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["amendment_sha256"] != ctx["amendment_sha256"] or \
                sha256_file(out / "adabn_sample.npz") != manifest["adabn_sample_sha256"]:
            raise ValueError("Existing prepare output belongs to another amendment or was modified")
        return manifest
    per_record = int(ctx["amendment"]["adabn"]["sample"]["epochs_per_record"])
    xs, keys, indices, shhs_sd = [], [], [], []
    for entry in ctx["entries_main"]:
        record = rhi.load_record(entry, VARIANT, args.processed_root)  # hash-verified; labels not loaded
        idx = sample_indices(record["subject_id"], len(record["x"]), per_record)
        xs.append(record["x"][idx])
        keys.extend([record["record_key"]] * per_record)
        indices.append(record["original_epoch_index"][idx])
        shhs_sd.append(float(record["x"].astype(np.float64).std()))
    sample = np.concatenate(xs).astype(np.float32)
    if len(sample) != int(ctx["amendment"]["adabn"]["sample"]["total_epochs"]):
        raise ValueError("AdaBN sample size differs from the amendment")
    edf_sd, edf_files = [], []
    for path in sorted(args.sleep_edf_root.glob("SC*.npz")):
        with np.load(path, allow_pickle=False) as z:
            edf_sd.append(float(z["x"].astype(np.float64).std()))
        edf_files.append({"file": path.name, "sha256": sha256_file(path)})
    if len(edf_sd) != 153:
        raise ValueError(f"Expected 153 Sleep-EDF SC recordings, found {len(edf_sd)}")
    out.mkdir(parents=True, exist_ok=True)
    atomic_savez(out / "adabn_sample.npz", {"x": sample, "record_key": np.asarray(keys),
                                            "original_epoch_index": np.concatenate(indices)})

    def summary(values: list[float]) -> dict[str, float]:
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        return {"n": len(values), "median": float(med), "q1": float(q1), "q3": float(q3)}

    manifest = {"schema_version": 1, "amendment_sha256": ctx["amendment_sha256"],
                "preprocess_manifest_sha256": ctx["preprocess_manifest_sha256"], "labels_read": False,
                "adabn_sample_sha256": sha256_file(out / "adabn_sample.npz"), "adabn_sample_epochs": len(sample),
                "amplitude_sd_uv": {"sleep_edf": summary(edf_sd), "shhs1_holdout": summary(shhs_sd)},
                "amplitude_ratio_shhs_over_edf_median": float(np.median(shhs_sd) / np.median(edf_sd)),
                "sleep_edf_files": edf_files}
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, sort_keys=False)
    return manifest


# --------------------------------------------------------------------------- gate
def gate(args: argparse.Namespace, ctx: dict[str, Any]) -> dict[str, Any]:
    gate_args = argparse.Namespace(**{**vars(args), "output_root": args.output_root})
    checkpoint = rhi.checkpoint_gate(gate_args, ctx, [ARM])  # writes/validates checkpoint_gate.json
    checks = []
    for fold in (0, 9):
        extractor, sequence, _ = load_models(args, fold)
        for entry in ctx["entries_main"][:2]:
            record = rhi.load_record(entry, VARIANT, args.processed_root)
            ours = predict(extractor, sequence, record["x"], 1.0, args.batch_size)
            with np.load(rhi.fold_path(args.parent_inference_root, ARM, fold, entry["record_key"]), allow_pickle=False) as z:
                stored = z["probabilities"].copy()
            checks.append({"fold": fold, "record_key": entry["record_key"],
                           "max_abs_difference": float(np.abs(ours - stored).max()),
                           "argmax_agreement": float((ours.argmax(1) == stored.argmax(1)).mean())})
    passed = checkpoint["status"] == "passed" and all(
        c["max_abs_difference"] <= 1e-5 and c["argmax_agreement"] == 1.0 for c in checks)
    report = {"schema_version": 1, "amendment_sha256": ctx["amendment_sha256"], "status": "passed" if passed else "failed",
              "checkpoint_gate_sha256": sha256_file(args.output_root / "checkpoint_gate.json"),
              "reproduction_of_parent_E4_s42": checks}
    atomic_write_json(args.output_root / "amendment_gate.json", report, ensure_ascii=False, sort_keys=False)
    if not passed:
        raise ValueError(f"Amendment gate failed: {checks}")
    return report


def require_gate(args: argparse.Namespace, ctx: dict[str, Any]) -> None:
    path = args.output_root / "amendment_gate.json"
    if not path.exists() or read_json(path).get("status") != "passed" or \
            read_json(path)["amendment_sha256"] != ctx["amendment_sha256"]:
        raise ValueError("Run --mode prepare and --mode gate first")
    if not (args.output_root / "prepare_manifest.json").exists():
        raise ValueError("Run --mode prepare first")


# --------------------------------------------------------------------------- work units
def all_units(ctx: dict[str, Any]) -> list[tuple[str, int]]:
    return [(condition, fold) for fold in FOLDS for condition in ctx["condition_names"]]


def fold_file(args: argparse.Namespace, condition: str, fold: int, key: str) -> Path:
    return args.output_root / "fold_predictions" / condition / f"fold_{fold:02d}" / f"{key}.npz"


def adabn_state_for_fold(args: argparse.Namespace, ctx: dict[str, Any], extractor: nn.Module, fold: int) -> str:
    path = args.output_root / "adabn_state" / f"fold_{fold:02d}.npz"
    prep = read_json(args.output_root / "prepare_manifest.json")
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["adabn_sample_sha256"].item()) != prep["adabn_sample_sha256"]:
                raise ValueError(f"{path}: built from another AdaBN sample")
        load_batchnorm_state(extractor, path)
        return sha256_file(path)
    sample_path = args.output_root / "adabn_sample.npz"
    if sha256_file(sample_path) != prep["adabn_sample_sha256"]:
        raise ValueError("AdaBN sample changed after prepare")
    with np.load(sample_path, allow_pickle=False) as z:
        sample = z["x"].copy()
    state = adapt_batchnorm(extractor, sample, args.batch_size)
    arrays: dict[str, Any] = {"layer_names": np.asarray(list(state)),
                              "adabn_sample_sha256": np.array(prep["adabn_sample_sha256"])}
    for name, values in state.items():
        arrays[f"{name}.running_mean"] = values["running_mean"]
        arrays[f"{name}.running_var"] = values["running_var"]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(path, arrays)
    return sha256_file(path)


def run_unit(args: argparse.Namespace, ctx: dict[str, Any], condition: str, fold: int) -> int:
    pending = [e for e in ctx["entries_main"] if not fold_file(args, condition, fold, e["record_key"]).exists()]
    if not pending:
        return 0
    spec = ctx["conditions"][condition]
    extractor, sequence, hashes = load_models(args, fold)
    meta = {**base_metadata(ctx, condition), **hashes, "outer_fold": fold}
    if spec["type"] == "adabn":
        meta["adabn_state_sha256"] = adabn_state_for_fold(args, ctx, extractor, fold)
    elif spec["type"] != "input_scale":
        raise ValueError(f"Unknown condition type {spec['type']}")
    factor = float(spec["factor"])
    started = time.perf_counter()
    for index, entry in enumerate(pending, start=1):
        path = fold_file(args, condition, fold, entry["record_key"])
        if path.exists():  # written meanwhile by another process
            continue
        record = rhi.load_record(entry, VARIANT, args.processed_root)
        probabilities = predict(extractor, sequence, record["x"], factor, args.batch_size)
        metadata = {**meta, "artifact_type": "amendment_a1_fold_prediction", "record_key": record["record_key"],
                    "subject_id": record["subject_id"], "input_npz_sha256": record["input_sha256"]}
        try:
            rhi.save_probabilities(path, probabilities, record, metadata)
        except PermissionError:
            if not path.exists():
                raise
        if index % 50 == 0 or index == len(pending):
            rate = (time.perf_counter() - started) / index
            print(f"[{condition}] fold {fold:02d} {index}/{len(pending)} records ({rate:.2f}s/record)", flush=True)
    return len(pending)


def progress(args: argparse.Namespace, ctx: dict[str, Any]) -> tuple[int, int]:
    root = args.output_root / "fold_predictions"
    done = sum(1 for _ in root.glob("*/fold_*/*.npz")) if root.exists() else 0
    return done, len(ctx["entries_main"]) * len(FOLDS) * len(ctx["condition_names"])


# --------------------------------------------------------------------------- finalize
def finalize(args: argparse.Namespace, ctx: dict[str, Any]) -> dict[str, Any]:
    done, total = progress(args, ctx)
    if done != total:
        raise ValueError(f"Not finalized: {done}/{total} fold artifacts present")
    arm_hashes = {}
    for condition in ctx["condition_names"]:
        records = []
        for entry in ctx["entries_main"]:
            parts, reference = [], None
            for fold in FOLDS:
                path = fold_file(args, condition, fold, entry["record_key"])
                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(str(z["metadata_json"].item()))
                    if (meta["amendment_sha256"] != ctx["amendment_sha256"] or meta["condition"] != condition
                            or meta["outer_fold"] != fold or meta["input_npz_sha256"] != entry["output_sha256"]):
                        raise ValueError(f"{path}: provenance differs")
                    index = z["original_epoch_index"].copy()
                    reference = index if reference is None else reference
                    np.testing.assert_array_equal(index, reference)
                    parts.append(z["probabilities"].copy())
            mean = ensemble_probabilities(parts)
            out = args.output_root / "ensemble" / condition / f"{entry['record_key']}.npz"
            metadata = {**base_metadata(ctx, condition), "artifact_type": "amendment_a1_ensemble", "folds": list(FOLDS),
                        "record_key": entry["record_key"], "subject_id": entry["subject_id"],
                        "input_npz_sha256": entry["output_sha256"]}
            atomic_savez(out, {"metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
                               "probabilities": mean, "prediction": np.argmax(mean, axis=1).astype(np.int8),
                               "original_epoch_index": reference})
            records.append({"record_key": entry["record_key"], "subject_id": entry["subject_id"], "sha256": sha256_file(out)})
        manifest_path = args.output_root / "arm_manifests" / f"{condition}.json"
        atomic_write_json(manifest_path, {"condition": condition, "status": "complete", "ensemble_records": records},
                          ensure_ascii=False, sort_keys=False)
        arm_hashes[condition] = sha256_file(manifest_path)
        print(f"[{condition}] ensembles written", flush=True)
    run_manifest = {"schema_version": 1, "status": "complete_no_metrics_computed",
                    "amendment_sha256": ctx["amendment_sha256"],
                    "prepare_manifest_sha256": sha256_file(args.output_root / "prepare_manifest.json"),
                    "amendment_gate_sha256": sha256_file(args.output_root / "amendment_gate.json"),
                    "arm_manifest_sha256": arm_hashes,
                    "execution": {"device": "cpu", "torch_version": torch.__version__, "deterministic_algorithms": True}}
    atomic_write_json(args.output_root / "run_manifest.json", run_manifest, ensure_ascii=False, sort_keys=False)
    return run_manifest


# --------------------------------------------------------------------------- analysis
def load_condition(args: argparse.Namespace, ctx: dict[str, Any], condition: str,
                   labels: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    from sleeptcn.metrics import confusion_matrix_5

    manifest = read_json(args.output_root / "arm_manifests" / f"{condition}.json")
    subjects, confusions = [], []
    for entry in sorted(manifest["ensemble_records"], key=lambda e: e["subject_id"]):
        path = args.output_root / "ensemble" / condition / f"{entry['record_key']}.npz"
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path}: hash differs from manifest")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"].item()))
            prediction = z["prediction"].astype(np.int8)
            index = z["original_epoch_index"].astype(np.int32)
        ref = labels[entry["record_key"]]
        if meta["condition"] != condition or meta["amendment_sha256"] != ctx["amendment_sha256"] or \
                not np.array_equal(index, ref["original_epoch_index"]):
            raise ValueError(f"{path}: provenance or alignment differs")
        valid = ref["y"] >= 0
        subjects.append(entry["subject_id"])
        confusions.append(confusion_matrix_5(ref["y"][valid], prediction[valid]))
    return np.asarray(subjects), np.stack(confusions)


def analyze(args: argparse.Namespace, ctx: dict[str, Any]) -> dict[str, Any]:
    import analyze_holdout_confirmatory as aho

    out_dir = args.output_root / "analysis"
    if (out_dir / "analysis.json").exists():
        raise FileExistsError("Amendment analysis already exists (write-once)")
    run_manifest = read_json(args.output_root / "run_manifest.json")
    if run_manifest["status"] != "complete_no_metrics_computed" or run_manifest["amendment_sha256"] != ctx["amendment_sha256"]:
        raise ValueError("Amendment inference is not complete")
    deviations = read_json(args.deviations_file) if args.deviations_file else []
    pre = read_json(args.preprocess_manifest)
    labels = aho.load_labels(pre)
    pre_index = {(r["variant"], r["record_key"]): r["output_sha256"] for r in pre["records"]}
    confusions, subjects_ref = {}, None
    for arm, variant in ((ARM, VARIANT), ("ADAST_full_e30", "filtered_v2"), ("SourceOnly_full_e30", "filtered_v2")):
        arm_manifest = read_json(args.parent_inference_root / "arm_manifests" / f"{arm}.json")
        subjects, conf = aho.load_arm(args.parent_inference_root, arm, arm_manifest, pre_index, labels, variant,
                                      ctx["protocol_sha256"])
        subjects_ref = subjects if subjects_ref is None else subjects_ref
        if not np.array_equal(subjects, subjects_ref):
            raise ValueError("subject order differs between arms")
        confusions[arm] = conf
    for condition in ctx["condition_names"]:
        subjects, conf = load_condition(args, ctx, condition, labels)
        if not np.array_equal(subjects, subjects_ref):
            raise ValueError("subject order differs between conditions")
        confusions[condition] = conf
    stats = ctx["amendment"]["statistics"]["bootstrap"]
    a = aho.Analysis(subjects_ref, confusions, int(stats["resamples"]), int(stats["seed"]))
    report: dict[str, Any] = {
        "schema_version": 1, "status": "complete", "amendment": "A1", "amendment_sha256": ctx["amendment_sha256"],
        "run_manifest_sha256": sha256_file(args.output_root / "run_manifest.json"),
        "subjects": int(len(subjects_ref)), "bootstrap": stats, "hypothesis_tests": "none",
        "amplitude": read_json(args.output_root / "prepare_manifest.json")["amplitude_sd_uv"],
        "amplitude_ratio_shhs_over_edf_median": read_json(args.output_root / "prepare_manifest.json")[
            "amplitude_ratio_shhs_over_edf_median"],
        "descriptive": {}, "contrasts_vs_E4_s42": {}, "adabn_vs_adast": {}, "deviations": deviations}
    for arm in confusions:
        row = a.describe(arm)
        row["pooled_n3"] = {s: a.pooled_single(arm, s) for s in ("n3_recall", "n3_precision", "n3_to_n2_rate")}
        report["descriptive"][arm] = row
    for condition in ctx["condition_names"]:
        row = a.contrast(condition, ARM, with_test=False)
        for stat in ("n3_recall", "n3_to_n2_rate", "n1_f1", "n2_f1", "n3_f1", "rem_f1"):
            row[stat] = a.pooled_difference(condition, ARM, stat)
        report["contrasts_vs_E4_s42"][condition] = row
    adast = a.contrast("adabn", "ADAST_full_e30", with_test=False)
    adast["n3_recall"] = a.pooled_difference("adabn", "ADAST_full_e30", "n3_recall")
    report["adabn_vs_adast"] = {"adabn - ADAST_full_e30": adast,
                                "adabn - SourceOnly_full_e30": a.contrast("adabn", "SourceOnly_full_e30", with_test=False)}
    curve = [float(aho.pooled_stats(confusions[c].sum(0))["n3_recall"]) for c in SCALE_CURVE]
    n3n2 = [float(aho.pooled_stats(confusions[c].sum(0))["n3_to_n2_rate"]) for c in SCALE_CURVE]
    report["scale_curve"] = {"factors": [0.5, 0.75, 1.0, 1.5, 2.0], "n3_recall": curve, "n3_to_n2_rate": n3n2,
                             "n3_recall_monotonic_increasing": bool(np.all(np.diff(curve) > 0)),
                             "n3_to_n2_monotonic_decreasing": bool(np.all(np.diff(n3n2) < 0))}
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "analysis.json", report, ensure_ascii=False, sort_keys=False)
    (out_dir / "analysis.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report: dict[str, Any]) -> str:
    f = lambda x: f"{x:.4f}"  # noqa: E731
    lines = ["# Amendment A1: AdaBN and input-amplitude sweep (E4 seed 42, SHHS1 holdout)", "",
             f"Participants: {report['subjects']}. Descriptive only; no hypothesis tests.", "",
             "| Condition | Subject-mean macro-F1 (95% CI) | Pooled macro-F1 | N3 recall | N3 precision | N3->N2 |",
             "|---|---|---|---|---|---|"]
    for arm, row in report["descriptive"].items():
        ci = row["subject_mean_macro_f1_ci95"]
        n3 = row["pooled_n3"]
        lines.append(f"| {arm} | {f(row['subject_mean_macro_f1'])} [{f(ci[0])}, {f(ci[1])}] | "
                     f"{f(row['pooled']['macro_f1'])} | {f(n3['n3_recall']['estimate'])} | "
                     f"{f(n3['n3_precision']['estimate'])} | {f(n3['n3_to_n2_rate']['estimate'])} |")
    lines += ["", "| Condition - E4_s42 | d subject-mean macro-F1 (95% CI) | d N3 recall (95% CI) | d N3->N2 (95% CI) |",
              "|---|---|---|---|"]
    for condition, row in report["contrasts_vs_E4_s42"].items():
        lines.append(f"| {condition} | {f(row['subject_mean_macro_f1_difference'])} [{f(row['ci95'][0])}, {f(row['ci95'][1])}] | "
                     f"{f(row['n3_recall']['observed_difference'])} [{f(row['n3_recall']['ci95'][0])}, {f(row['n3_recall']['ci95'][1])}] | "
                     f"{f(row['n3_to_n2_rate']['observed_difference'])} [{f(row['n3_to_n2_rate']['ci95'][0])}, {f(row['n3_to_n2_rate']['ci95'][1])}] |")
    lines += ["", "AdaBN versus ADAST:"]
    for name, row in report["adabn_vs_adast"].items():
        lines.append(f"- {name}: {f(row['subject_mean_macro_f1_difference'])} [{f(row['ci95'][0])}, {f(row['ci95'][1])}]")
    amp = report["amplitude"]
    lines += ["", f"Amplitude SD (uV), median [IQR]: Sleep-EDF {amp['sleep_edf']['median']:.2f} "
              f"[{amp['sleep_edf']['q1']:.2f}, {amp['sleep_edf']['q3']:.2f}]; SHHS1 {amp['shhs1_holdout']['median']:.2f} "
              f"[{amp['shhs1_holdout']['q1']:.2f}, {amp['shhs1_holdout']['q3']:.2f}]; ratio "
              f"{report['amplitude_ratio_shhs_over_edf_median']:.3f}",
              f"N3 recall over factors 0.5-2.0: {[round(v, 4) for v in report['scale_curve']['n3_recall']]}",
              f"Deviations: {report['deviations'] or 'none'}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amendment", type=Path, default=ROOT / "configs" / "shhs_holdout_amendment_a1.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "shhs_holdout_confirmatory_v1.json")
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed" / "shhs_holdout_v1")
    parser.add_argument("--sleep-edf-root", type=Path, default=ROOT / "data" / "processed" / "bandpass_v2")
    parser.add_argument("--restore-workspace", type=Path, default=ROOT / "runs" / "holdout_ckpt_restore_20261010")
    parser.add_argument("--adast-backup", type=Path, default=ROOT / "runs" / "adast_fullsource_checkpoints_backup_20261010")
    parser.add_argument("--parent-inference-root", type=Path, default=ROOT / "runs" / "holdout_inference_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "holdout_amendment_a1")
    parser.add_argument("--mode", required=True,
                        choices=("prepare", "gate", "worker", "finalize", "analyze", "progress"))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--deviations-file", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_flush_denormal(True)
    ctx = load_context(args)

    if args.mode == "progress":
        done, total = progress(args, ctx)
        print(f"{done}/{total}")
    elif args.mode == "prepare":
        manifest = prepare(args, ctx)
        print(f"AdaBN sample: {manifest['adabn_sample_epochs']} epochs; amplitude ratio SHHS1/Sleep-EDF "
              f"{manifest['amplitude_ratio_shhs_over_edf_median']:.3f}")
    elif args.mode == "gate":
        report = gate(args, ctx)
        print(f"Amendment gate: {report['status']}")
    elif args.mode == "worker":
        require_gate(args, ctx)
        units = all_units(ctx)[args.worker_index::args.num_workers]
        started = time.perf_counter()
        for condition, fold in units:
            run_unit(args, ctx, condition, fold)
        print(f"Worker {args.worker_index}/{args.num_workers}: {len(units)} units in {time.perf_counter() - started:.0f}s")
    elif args.mode == "finalize":
        require_gate(args, ctx)
        finalize(args, ctx)
        print("STATUS: AMENDMENT A1 ENSEMBLES COMPLETE (no metrics computed)")
    elif args.mode == "analyze":
        report = analyze(args, ctx)
        print((args.output_root / "analysis" / "analysis.md").read_text(encoding="utf-8"))
        print(f"Monotonic N3 recall: {report['scale_curve']['n3_recall_monotonic_increasing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
