"""Verify ADAST pilot provenance, prediction replay, scoring and source holdout."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.cpu_followups import confusion_diagnostics, paired_diagnostic_bootstrap
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.preprocessing import sha256_file


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--shhs-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.pilot
    result_path = output / "aggregate_results.json"
    result = json.loads(result_path.read_bytes())
    if result["status"] != "complete_exploratory_single_fold_harmonized_ADAST_pilot":
        raise ValueError("incomplete ADAST pilot")
    provenance = result["provenance"]
    for relative, digest in provenance["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest or sha256_file(output / "code_snapshot" / relative) != digest:
            raise ValueError("code snapshot mismatch")
    for relative, digest in provenance["upstream_files_sha256"].items():
        if sha256_file(args.upstream / relative) != digest:
            raise ValueError("upstream model changed")
    inputs_path = output / "private_inputs.json"
    if sha256_file(inputs_path) != result["input_manifest_sha256"]:
        raise ValueError("input manifest changed")
    inputs = json.loads(inputs_path.read_bytes())
    for entry in inputs["source_records"] + inputs["adaptation_records"] + list(inputs["cached_arrays"].values()):
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("input artifact changed")
    locked_path = args.shhs_root / "manifests/shhs1_subject_manifest_seed42.json"
    locked = json.loads(locked_path.read_bytes())
    locked_adaptation = {e["subject_id"] for e in locked["subjects"] if e["role"] == "adaptation"}
    if {e["subject_id"] for e in inputs["adaptation_records"]} != locked_adaptation:
        raise ValueError("adaptation subjects differ from locked selection")
    cfg = provenance["protocol"]
    helper = load("verify_adast_helpers", ROOT / "scripts/run_adast_cpu_pilot.py")
    signal_helper = load("verify_adast_signal", ROOT / "scripts/run_recovered_e3_cpu_pilot.py")
    module = load("verify_adast_models", args.upstream / "models/models.py")
    upstream_cfg = load("verify_adast_cfg", args.upstream / "config_files/configs.py").Config()
    torch.set_num_threads(4)
    models = {}
    for arm, selected in result["selections"].items():
        path = output / arm / "final.pt"
        if sha256_file(path) != selected["checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        network = helper.build_models(module, upstream_cfg, cfg["seed"])
        if helper.model_digest(network) != selected["initial_state_sha256"]:
            raise ValueError("initialization mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["protocol_sha256"] != provenance["protocol_sha256"] or payload["epochs_completed"] != 30
                or selected["updates"] != 1140 or payload["history"] != selected["history"]):
            raise ValueError("training budget/protocol mismatch")
        for name, model in network.items():
            model.load_state_dict(payload["models"][name], strict=True)
        models[arm] = network
    left = result["selections"]["adast"]
    right = result["selections"]["source_only"]
    if (left["initial_state_sha256"] != right["initial_state_sha256"]
            or [r["source_order_sha256"] for r in left["history"]] != [r["source_order_sha256"] for r in right["history"]]):
        raise ValueError("controls not matched")
    for index, row in enumerate(left["history"]):
        expected_lr = .001 if index < 9 else .0001
        if row["encoder_lr_after_epoch"] != expected_lr or row["updates"] != 38:
            raise ValueError("scheduler/update budget differs")
    manifest_path = output / "private_inference_manifest.json"
    if sha256_file(manifest_path) != result["inference_manifest_sha256"]:
        raise ValueError("target manifest changed")
    entries = json.loads(manifest_path.read_bytes())["records"]
    expected_test = {e["subject_id"] for e in locked["subjects"] if e["role"] == "test"}
    if len(entries) != 180 or {e["subject_id"] for e in entries} != expected_test:
        raise ValueError("target subjects differ from locked selection")
    frozen_hash = sha256_file(output / "both_arms_selected.json")
    cm = {arm: [] for arm in models}
    replay = {}
    for index, entry in enumerate(entries):
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("prediction changed")
        reference = args.shhs_root / "processed_v1/filtered_v2" / (entry["record_key"] + ".npz")
        with np.load(entry["path"], allow_pickle=False) as pred, np.load(reference, allow_pickle=False) as z:
            meta = json.loads(str(pred["metadata_json"]))
            if (meta["selected_sha256"] != frozen_hash or meta["subject_id"] != entry["subject_id"]
                    or meta["source_edf_sha256"] != entry["source_edf_sha256"]
                    or str(z["role"]) != "test" or str(z["subject_id"]) != entry["subject_id"]
                    or not np.array_equal(pred["original_epoch_index"], np.arange(entry["epochs"]))):
                raise ValueError("prediction/reference alignment mismatch")
            positions = signal_helper.benchmark_positions(z["original_epoch_index"], entry["epochs"])
            for arm in models:
                cm[arm].append(confusion_matrix_5(z["y"], pred[arm][positions].argmax(1)))
            if index == 0:
                edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
                signal, _ = signal_helper.read_full_record(edf, entry["source_edf_sha256"])
                for arm, network in models.items():
                    actual = helper.predict(network, signal, "source" if arm == "source_only" else "target")
                    replay[arm] = {"max_logit_difference": float(np.max(np.abs(actual - pred[arm]))),
                                   "argmax_disagreements": int(np.count_nonzero(actual.argmax(1) != pred[arm].argmax(1)))}
                    np.testing.assert_allclose(actual, pred[arm], atol=1e-6, rtol=1e-6)
    matrices = {arm: np.stack(values) for arm, values in cm.items()}
    with np.load(output / "private_subject_confusions.npz", allow_pickle=False) as stored:
        for arm, values in matrices.items():
            if not np.array_equal(values, stored[arm]) or values.sum() != 169012:
                raise ValueError("subject confusions mismatch")
            if metrics_from_confusion(values.sum(0)) != result["arms"][arm]["pooled"]:
                raise ValueError("pooled metrics mismatch")
            if float(confusion_diagnostics(values)["macro_f1"].mean()) != result["arms"][arm]["subject_mean_macro_f1"]:
                raise ValueError("subject mean metric mismatch")
    recomputed = paired_diagnostic_bootstrap(matrices["adast"], matrices["source_only"])
    if recomputed["metrics"] != result["adast_minus_source_only"]["metrics"]:
        raise ValueError("paired bootstrap does not replay")
    split_path = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    if sha256_file(split_path) != provenance["source_split_sha256"]:
        raise ValueError("source split changed")
    split = json.loads(split_path.read_bytes())["outer_runs"][0]
    if {e["record_key"] for e in inputs["source_records"]} != set(split["train"]["record_keys"]):
        raise ValueError("source train inputs differ from fold")
    source_cm = {arm: {} for arm in models}
    source_hashes = []
    for key in sorted(split["test"]["record_keys"]):
        path = ROOT / "data/processed/filtered_v2" / (key + ".npz")
        with np.load(path, allow_pickle=False) as z:
            subject = str(z["subject_id"])
            for arm, network in models.items():
                # Source-domain diagnostic uses source attention for both arms.
                logits = helper.predict(network, z["x"], "source")
                matrix = confusion_matrix_5(z["y"], logits.argmax(1))
                source_cm[arm].setdefault(subject, np.zeros((5, 5), dtype=np.int64))
                source_cm[arm][subject] += matrix
        source_hashes.append(sha256_file(path))
    source_results = {}
    for arm, values in source_cm.items():
        stacked = np.stack(list(values.values()))
        if len(stacked) != 8 or stacked.sum() != 19506:
            raise ValueError("source outer-test support differs")
        source_results[arm] = {"subjects": 8, "pooled": metrics_from_confusion(stacked.sum(0)),
            "subject_mean_macro_f1": float(confusion_diagnostics(stacked)["macro_f1"].mean())}
    verification = {"status": "passed", "input_hashes_verified": True, "code_hashes_verified": True,
        "target_prediction_hashes_verified": 180, "target_confusions_and_bootstrap_recomputed": True,
        "locked_adaptation_and_test_subjects_verified": True,
        "matched_initialization_source_sample_order_and_budget_verified": True,
        "full_record_prediction_replay": replay, "source_test_used_for_selection": False,
        "source_test_attention_policy": "source_attention_for_both_arms", "source_test": source_results,
        "source_test_input_hashes": source_hashes,
        "aggregate_results_sha256": sha256_file(result_path),
        "verification_script_sha256": sha256_file(Path(__file__))}
    helper.write_json(output / "verification.json", verification)
    print(json.dumps({"status": "passed", "prediction_replay": replay,
                      "source_test": source_results}, indent=2))


if __name__ == "__main__":
    main()
