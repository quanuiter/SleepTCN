"""Run the recovered full-training E3 fold-0 calibration pilot on CPU.

Checkpoint hashes must match the historical inventory. Target fitting consumes
only the five label-independent adaptation recordings. Test labels are opened
only after all test inference and parameter fitting have completed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyedflib
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.calibration import (
    STAGES, adjust_prior, apply_temperature, estimate_target_prior, fit_temperature, probabilities,
)
from sleeptcn.cpu_followups import confusion_diagnostics, label_free_signal_variants, paired_diagnostic_bootstrap
from sleeptcn.demo import (
    DemoRecord, load_demo_models, load_locked_prediction, load_processed_demo_record,
    predict_record, validate_asset_manifest,
)
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.preprocessing import sha256_file
from sleeptcn.shhs_preprocessing import SHHSPreprocessConfig, normalize_uv_unit

PROTOCOL = ROOT / "configs/teacher_revision_recovered_fold0_pilot_v1.json"
ARMS = ("raw", "calibrated", "calibrated_em")


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_npz(path: Path, **arrays) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return sha256_file(path)


def checked_roles(selection: dict, audit: dict, selection_sha: str) -> dict:
    if (selection.get("dataset") != "SHHS Visit 1" or selection.get("selection_seed") != 42
            or audit.get("status") != "passed" or audit.get("manifest_sha256") != selection_sha):
        raise ValueError("selection/audit identity mismatch")
    roles = {role: sorted([s for s in selection["subjects"] if s["role"] == role],
                          key=lambda s: s["subject_id"]) for role in ["adaptation", "test"]}
    ids = [s["subject_id"] for subjects in roles.values() for s in subjects]
    if len(roles["adaptation"]) != 5 or len(roles["test"]) != 180 or len(set(ids)) != len(ids):
        raise ValueError("expected five adaptation and 180 disjoint test subjects")
    for role, subjects in roles.items():
        for subject in subjects:
            name = subject["edf_filename"]
            item = audit["subjects"][subject["subject_id"]]
            if (Path(name).name != name or not item["passed"] or item["role"] != role
                    or item["edf_filename"] != name):
                raise ValueError("subject role/EDF differs from technical audit")
    return roles


def read_full_record(edf: Path, expected_hash: str) -> tuple[np.ndarray, dict]:
    if sha256_file(edf) != expected_hash:
        raise ValueError("raw EEG hash mismatch")
    cfg = replace(SHHSPreprocessConfig(), wake_edge_minutes=0,
                  trim_anchor_policy="full_record_signal_length_no_labels")
    reader = pyedflib.EdfReader(str(edf))
    try:
        names = [name.strip() for name in reader.getSignalLabels()]
        if names.count(cfg.source_channel) != 1:
            raise ValueError("expected one primary EEG channel")
        index = names.index(cfg.source_channel)
        if (reader.getSampleFrequency(index) != cfg.source_sampling_rate_hz
                or normalize_uv_unit(reader.getPhysicalDimension(index)) != "uv"):
            raise ValueError("unexpected EEG rate or physical unit")
        signal, duration = reader.readSignal(index), reader.getFileDuration()
    finally:
        reader.close()
    if not np.isclose(signal.size / cfg.source_sampling_rate_hz, duration, atol=1e-6, rtol=0):
        raise ValueError("EEG duration mismatch")
    x, clip_fraction, metadata = label_free_signal_variants(signal, cfg)["filtered_v2"]
    return x, {"source_edf_sha256": expected_hash, "clip_fraction": clip_fraction,
               "preprocessing": metadata, "uses_annotations": False,
               "window": cfg.trim_anchor_policy}


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(probabilities(p)[np.arange(len(y)), y]).mean())


def benchmark_positions(indices: np.ndarray, full_length: int) -> np.ndarray:
    values = np.asarray(indices)
    if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer)
            or values[0] < 0 or values[-1] >= full_length or np.any(np.diff(values) != 1)):
        raise ValueError("benchmark mask is outside or misaligned with full recording")
    return values.astype(np.int64)


def summarize_confusions(confusions: dict[str, np.ndarray]) -> dict:
    if tuple(confusions) != ARMS:
        raise ValueError("expected the three specified arms in canonical order")
    arrays = {arm: np.asarray(cm) for arm, cm in confusions.items()}
    reference = arrays["raw"]
    if reference.ndim != 3 or reference.shape[1:] != (5, 5) or not len(reference):
        raise ValueError("expected subject-wise 5x5 confusion matrices")
    for cm in arrays.values():
        if cm.shape != reference.shape or not np.array_equal(cm.sum(-1), reference.sum(-1)):
            raise ValueError("arms do not share subject-level reference support")
    arms = {arm: {"subject_mean_macro_f1": float(confusion_diagnostics(cm)["macro_f1"].mean()),
                  "pooled": metrics_from_confusion(cm.sum(0))} for arm, cm in arrays.items()}
    comparisons = {}
    for left, right in [("calibrated_em", "calibrated"), ("calibrated", "raw")]:
        result = paired_diagnostic_bootstrap(arrays[left], arrays[right])
        differences = (confusion_diagnostics(arrays[left])["macro_f1"]
                       - confusion_diagnostics(arrays[right])["macro_f1"])
        rng = np.random.default_rng(2031)
        draws = rng.integers(0, len(reference), (10000, len(reference)), dtype=np.int32)
        low, high = np.quantile(differences[draws].mean(1), [.025, .975])
        result["subject_mean_macro_f1"] = {"difference": float(differences.mean()),
                                           "ci95": [float(low), float(high)]}
        comparisons[f"{left}_minus_{right}"] = result
    return {"arms": arms, "comparisons": comparisons,
            "subjects": len(reference), "valid_epochs": int(reference.sum()),
            "subjects_without_reference_n3": int((reference[:, 3].sum(1) == 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shhs-root", type=Path, required=True)
    parser.add_argument("--adaptation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("private participant outputs must stay in ignored runs storage")
    if output.exists():
        raise FileExistsError("refusing to overwrite a previous pilot")
    shhs = args.shhs_root
    protocol = json.loads(PROTOCOL.read_bytes())
    if (protocol["experiment"], protocol["outer_fold"], protocol["seed"], protocol["arms"]) != (
            "E3", 0, 123, list(ARMS)):
        raise ValueError("unexpected pilot specification")
    assets = ROOT / "demo/assets"
    asset_manifest = validate_asset_manifest(assets)
    inventory_path = shhs / "manifests" / protocol["checkpoint_inventory"]
    inventory = json.loads(inventory_path.read_bytes())
    historical_run = json.loads((shhs / "zero_shot_e4_seed123_v1/test/run_manifest.json").read_bytes())
    historical_protocol = shhs / "manifests/shhs_e4_seed123_extension_protocol_v1.json"
    if (inventory["status"] != "passed" or inventory["seed"] != 123
            or sha256_file(inventory_path) != historical_run["checkpoint_inventory_sha256"]
            or sha256_file(historical_protocol) != historical_run["protocol_sha256"]
            or inventory["protocol_sha256"] != historical_run["protocol_sha256"]):
        raise ValueError("historical inventory/protocol linkage mismatch")
    fold = next(f for f in inventory["folds"] if f["experiment"] == "E3" and f["outer_fold"] == 0)
    entry = asset_manifest["experiments"]["E3"]
    source_split = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    source_config = ROOT / "configs/experiments_v2.json"
    checkpoint_hashes = {}
    for component, stage in [("extractor", "resnet1d"), ("sequence", "sequence/tcn")]:
        path = assets / entry[component]["path"]
        digest = sha256_file(path)
        if digest not in {item["sha256"] for item in fold["checkpoints"]}:
            raise ValueError("recovered checkpoint does not match historical fold inventory")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {"experiment_id": "E3", "outer_fold": 0, "seed": 123,
                    "data_variant": "filtered_v2", "stage": stage,
                    "split_sha256": sha256_file(source_split),
                    "config_sha256": sha256_file(source_config)}
        if any(payload["metadata"].get(k) != value for k, value in expected.items()):
            raise ValueError("recovered checkpoint provenance mismatch")
        checkpoint_hashes[component] = digest
    split = json.loads(source_split.read_bytes())["outer_runs"][0]
    source_roles = [{key[:5] for key in split[role]["record_keys"]}
                    for role in ["train", "validation", "test"]]
    if any(source_roles[i] & source_roles[j] for i in range(3) for j in range(i)):
        raise ValueError("source subject roles overlap")
    selection_path = shhs / "manifests/shhs1_subject_manifest_seed42.json"
    selection = json.loads(selection_path.read_bytes())
    audit_path = shhs / "manifests/selected_audit_seed42.json"
    audit = json.loads(audit_path.read_bytes())
    roles = checked_roles(selection, audit, sha256_file(selection_path))
    adaptation_manifest_path = args.adaptation_cache / "private_manifest.json"
    adaptation = json.loads(adaptation_manifest_path.read_bytes())
    if (adaptation["summary"]["selection_manifest_sha256"] != sha256_file(selection_path)
            or {r["subject_id"] for r in adaptation["records"]}
            != {s["subject_id"] for s in roles["adaptation"]}):
        raise ValueError("adaptation cache uses a different selection")
    edf_dir = shhs / "shhs/polysomnography/edfs/shhs1"
    if any(not (edf_dir / s["edf_filename"]).is_file() for s in roles["test"]):
        raise FileNotFoundError("some locked test EDFs are missing")
    torch.set_num_threads(4)
    models = load_demo_models(assets, "E3", validated_manifest=asset_manifest)
    output.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__).resolve(), ROOT / "src/sleeptcn/calibration.py",
                  ROOT / "src/sleeptcn/cpu_followups.py", ROOT / "src/sleeptcn/demo.py",
                  ROOT / "src/sleeptcn/models.py", ROOT / "src/sleeptcn/preprocessing.py",
                  ROOT / "src/sleeptcn/shhs_preprocessing.py"]
    provenance = {"pilot_protocol_sha256": sha256_file(PROTOCOL), "protocol": protocol,
                  "checkpoint_sha256": checkpoint_hashes,
                  "historical_inventory_sha256": sha256_file(inventory_path),
                  "source_split_sha256": sha256_file(source_split),
                  "selection_sha256": sha256_file(selection_path),
                  "technical_audit_sha256": sha256_file(audit_path),
                  "adaptation_manifest_sha256": sha256_file(adaptation_manifest_path),
                  "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
                  "torch": torch.__version__, "numpy": np.__version__}
    write_json(output / "execution_specification.json", provenance)

    # Verify numerical replay on the predetermined first source test record.
    key = sorted(split["test"]["record_keys"])[0]
    record = load_processed_demo_record(ROOT / "data/processed/filtered_v2" / (key + ".npz"))
    live, locked = predict_record(record, models, batch_size=64), load_locked_prediction(
        assets, "E3", record, validated_manifest=asset_manifest)
    valid = record.labels >= 0
    replay = {"epochs": int(valid.sum()),
              "maximum_probability_difference": float(np.abs(live.probabilities[valid] - locked.probabilities[valid]).max()),
              "prediction_disagreements": int((live.predicted[valid] != locked.predicted[valid]).sum())}
    if replay["maximum_probability_difference"] > 1e-4 or replay["prediction_disagreements"] != 0:
        raise ValueError("checkpoint replay differs from archived source prediction")
    write_json(output / "replay_check.json", replay)

    source_p, source_y, source_ids, source_inputs = [], [], [], []
    for i, key in enumerate(sorted(split["validation"]["record_keys"]), 1):
        path = ROOT / "data/processed/filtered_v2" / (key + ".npz")
        record = load_processed_demo_record(path)
        prediction = predict_record(record, models, batch_size=64)
        valid = record.labels >= 0
        source_p.append(prediction.probabilities[valid])
        source_y.append(record.labels[valid])
        source_ids.extend(f"{key}:{int(index)}" for index in record.original_epoch_index[valid])
        source_inputs.append({"record_key": key, "sha256": sha256_file(path)})
        print(f"source validation inference {i}/{len(split['validation']['record_keys'])}", flush=True)
    source_p, source_y = np.concatenate(source_p), np.concatenate(source_y)
    if (len(source_y) != split["validation"]["valid_epochs"] or not np.array_equal(
            np.bincount(source_y, minlength=5), [split["validation"]["label_counts"][str(i)] for i in range(5)])):
        raise ValueError("source validation counts differ from locked split")
    source_hash = save_npz(output / "source_validation.npz", probabilities=source_p,
                           labels=source_y, epoch_id=np.array(source_ids))
    write_json(output / "source_input_manifest.json", {"records": source_inputs})
    temperature = fit_temperature(source_p, source_y)
    calibrated_source = apply_temperature(source_p, temperature)
    source_prior = calibrated_source.mean(0)

    adaptation_p, adaptation_ids = [], []
    for i, item in enumerate(sorted(adaptation["records"], key=lambda r: r["subject_id"]), 1):
        saved = item["variants"]["filtered_v2"]
        path = Path(saved["path"])
        if sha256_file(path) != saved["sha256"]:
            raise ValueError("adaptation signal hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"x", "original_epoch_index", "metadata_json"}:
                raise ValueError("adaptation signal contains unexpected label/schema fields")
            meta = json.loads(str(z["metadata_json"]))
            if (meta["role"] != "adaptation" or meta["uses_sleep_annotations"]
                    or meta["subject_id"] != item["subject_id"]
                    or meta["source_edf_sha256"] != audit["subjects"][item["subject_id"]]["edf_sha256"]
                    or meta["epoch_selection"] != "full_record_signal_length_no_labels"
                    or not np.array_equal(z["original_epoch_index"], np.arange(len(z["x"])))):
                raise ValueError("adaptation window or identity mismatch")
            record = DemoRecord(path.stem, z["x"], None, z["original_epoch_index"],
                                "SHHS adaptation", "full record, no annotation", "filtered_v2")
        prediction = predict_record(record, models, batch_size=64)
        adaptation_p.append(prediction.probabilities)
        adaptation_ids.extend(f"{item['subject_id']}:{int(index)}" for index in record.original_epoch_index)
        print(f"adaptation inference {i}/5", flush=True)
    adaptation_p = np.concatenate(adaptation_p)
    adaptation_hash = save_npz(output / "target_adaptation.npz", probabilities=adaptation_p,
                               epoch_id=np.array(adaptation_ids))
    em = estimate_target_prior(apply_temperature(adaptation_p, temperature), source_prior)
    fit = {"temperature": temperature, "source_prior": source_prior.tolist(),
           "target_prior": em.target_prior.tolist(), "em_converged": em.converged,
           "em_iterations": em.iterations, "em_log_likelihood": list(em.log_likelihood),
           "source_validation_epochs": len(source_p), "adaptation_epochs": len(adaptation_p),
           "source_validation_nll_raw": nll(source_p, source_y),
           "source_validation_nll_calibrated": nll(calibrated_source, source_y),
           "source_validation_sha256": source_hash, "adaptation_probabilities_sha256": adaptation_hash,
           "fitted_before_test_inference_and_evaluation": True, "target_labels_used": False}
    write_json(output / "fitted_parameters.json", fit)
    if not em.converged:
        raise RuntimeError("EM did not converge; no target evaluation performed")
    fit_hash = sha256_file(output / "fitted_parameters.json")
    print("temperature and EM fitted; parameters frozen before test inference", flush=True)

    target_folder = output / "target_probabilities"
    target_folder.mkdir()
    target_entries = []
    for i, subject in enumerate(roles["test"], 1):
        subject_id = subject["subject_id"]
        source_hash = audit["subjects"][subject_id]["edf_sha256"]
        edf = edf_dir / subject["edf_filename"]
        x, preprocessing = read_full_record(edf, source_hash)
        record = DemoRecord(edf.stem, x, None, np.arange(len(x), dtype=np.int64),
                            "SHHS test", "full record, no annotation", "filtered_v2")
        raw = predict_record(record, models, batch_size=64).probabilities
        calibrated = apply_temperature(raw, temperature)
        adjusted = adjust_prior(calibrated, source_prior, em.target_prior)
        path = target_folder / (edf.stem + ".npz")
        metadata = {"subject_id": subject_id, "role": "test", "preprocessing": preprocessing,
                    "fitted_parameters_sha256": fit_hash, "checkpoint_sha256": checkpoint_hashes}
        digest = save_npz(path, raw=raw, calibrated=calibrated, calibrated_em=adjusted,
                          original_epoch_index=record.original_epoch_index,
                          metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
        target_entries.append({"subject_id": subject_id, "record_key": edf.stem,
                               "path": str(path), "sha256": digest, "epochs": len(x),
                               "source_edf_sha256": source_hash})
        if i % 10 == 0 or i == 180:
            print(f"full-record test inference {i}/180; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    write_json(output / "private_inference_manifest.json", {"status": "complete",
               "fitted_parameters_sha256": fit_hash, "records": target_entries})

    # All outputs exist before historical labels/masks are opened for scoring.
    if sha256_file(output / "fitted_parameters.json") != fit_hash:
        raise ValueError("fitted parameters changed during test inference")
    confusions = {arm: [] for arm in ARMS}
    evaluation_inputs = []
    for item in target_entries:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("test prediction hash changed")
        label_path = shhs / "processed_v1/filtered_v2" / (item["record_key"] + ".npz")
        with np.load(label_path, allow_pickle=False) as labels, np.load(item["path"], allow_pickle=False) as pred:
            if (str(labels["subject_id"]) != item["subject_id"] or str(labels["role"]) != "test"
                    or str(labels["source_edf_sha256"]) != item["source_edf_sha256"]):
                raise ValueError("evaluation labels are from another recording/role")
            positions = benchmark_positions(labels["original_epoch_index"], len(pred["raw"]))
            y = labels["y"]
            if not np.array_equal(labels["valid_mask"], y >= 0):
                raise ValueError("evaluation valid mask mismatch")
            for arm in ARMS:
                p = probabilities(pred[arm])
                confusions[arm].append(confusion_matrix_5(y, p[positions].argmax(1)))
        evaluation_inputs.append({"subject_id": item["subject_id"], "sha256": sha256_file(label_path)})
    arrays = {arm: np.stack(cm) for arm, cm in confusions.items()}
    results = summarize_confusions(arrays)
    if results["subjects"] != 180 or results["valid_epochs"] != 169012:
        raise ValueError("evaluation cohort support differs from historical benchmark")
    results.update({"status": "complete_exploratory_single_fold_pilot", "experiment": "E3",
                    "seed": 123, "outer_fold": 0, "fitted_parameters": fit,
                    "fitted_parameters_sha256": fit_hash, "provenance": provenance,
                    "checkpoint_replay": replay, "full_test_epochs": sum(e["epochs"] for e in target_entries),
                    "wall_seconds": time.perf_counter() - started,
                    "inference_manifest_sha256": sha256_file(output / "private_inference_manifest.json")})
    write_json(output / "private_evaluation_manifest.json", {"records": evaluation_inputs})
    save_npz(output / "private_subject_confusions.npz", **arrays)
    write_json(output / "aggregate_results.json", results)
    print(json.dumps({"status": results["status"], "temperature": temperature,
                       "em_iterations": em.iterations, "wall_seconds": results["wall_seconds"],
                       "arms": {arm: {"subject_mean_macro_f1": value["subject_mean_macro_f1"],
                                      "n3": value["pooled"]["per_class"]["N3"]}
                                for arm, value in results["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
