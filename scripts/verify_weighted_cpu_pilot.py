"""Verify matched TCN checkpoints/target artifacts and score held-out source data."""
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
from sleeptcn.cpu_followups import confusion_diagnostics
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.models import SleepTCN
from sleeptcn.preprocessing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--shhs-root", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.pilot / "aggregate_results.json").read_bytes())
    if result["status"] != "complete_exploratory_single_fold_weighted_loss_pilot":
        raise ValueError("weighted pilot incomplete")
    cache_path = args.source_cache / "private_manifest.json"
    if sha256_file(cache_path) != result["provenance"]["source_cache_manifest_sha256"]:
        raise ValueError("source cache manifest changed")
    cache = json.loads(cache_path.read_bytes())
    for relative, digest in result["provenance"]["code_sha256"].items():
        if sha256_file(args.pilot / "code_snapshot" / relative) != digest:
            raise ValueError("execution code snapshot hash mismatch")
    if (result["selections"]["unweighted"]["initial_state_sha256"]
            != result["selections"]["weighted"]["initial_state_sha256"]):
        raise ValueError("initial weights differ")
    script = args.pilot / "code_snapshot/scripts/run_weighted_tcn_cpu_pilot.py"
    # Read model verification helper from the unchanged workspace file; compare it first.
    live_script = ROOT / "scripts/run_weighted_tcn_cpu_pilot.py"
    if sha256_file(live_script) != sha256_file(script):
        raise ValueError("verification helper differs from executed snapshot")
    spec = importlib.util.spec_from_file_location("weighted_verify_helpers", live_script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(4)
    models, source_results = {}, {}
    for arm in ["unweighted", "weighted"]:
        path = args.pilot / arm / "best.pt"
        selected = result["selections"][arm]
        if sha256_file(path) != selected["checkpoint_sha256"]:
            raise ValueError("selected checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["epoch"] != selected["selected_epoch"] or payload["protocol_sha256"] != result["provenance"]["protocol_sha256"]:
            raise ValueError("selected checkpoint metadata mismatch")
        model = SleepTCN(input_dim=128).eval()
        model.load_state_dict(payload["model_state"], strict=True)
        models[arm] = model
        source_results[arm] = {}
        for role in ["validation", "test"]:
            entries = [e for e in cache["records"] if e["role"] == role]
            records, subject_cm = [], {}
            for entry in entries:
                if sha256_file(Path(entry["path"])) != entry["sha256"]:
                    raise ValueError("source feature hash mismatch")
                with np.load(entry["path"], allow_pickle=False) as z:
                    records.append((torch.from_numpy(z["features"]), torch.from_numpy(z["labels"].astype(np.int64))))
                if role == "test":
                    with torch.inference_mode():
                        pred = model(records[-1][0].unsqueeze(0), padding_mask=None).squeeze(0).argmax(-1).numpy()
                    cm = confusion_matrix_5(records[-1][1].numpy(), pred)
                    subject_cm.setdefault(entry["subject_id"], np.zeros((5, 5), dtype=np.int64))
                    subject_cm[entry["subject_id"]] += cm
            metrics = helper.evaluate_source(model, records, 8)
            if role == "validation" and metrics["macro_f1"] != selected["selected_validation_macro_f1"]:
                raise ValueError("selected validation macro-F1 does not replay")
            source_results[arm][role] = {"pooled": metrics}
            if role == "test":
                matrices = np.stack(list(subject_cm.values()))
                source_results[arm][role]["subject_mean_macro_f1"] = float(confusion_diagnostics(matrices)["macro_f1"].mean())
                source_results[arm][role]["subjects"] = len(matrices)
    private = json.loads((args.pilot / "private_inference_manifest.json").read_bytes())
    if private["status"] != "complete" or len(private["records"]) != 180:
        raise ValueError("target inference incomplete")
    matrices = {arm: np.zeros((5, 5), dtype=np.int64) for arm in models}
    for entry in private["records"]:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("target probability hash mismatch")
        labels = args.shhs_root / "processed_v1/filtered_v2" / (entry["record_key"] + ".npz")
        with np.load(entry["path"], allow_pickle=False) as pred, np.load(labels, allow_pickle=False) as reference:
            meta = json.loads(str(pred["metadata_json"]))
            if meta["selected_checkpoints"] != result["selections"] or meta["subject_id"] != entry["subject_id"]:
                raise ValueError("target checkpoint/subject metadata mismatch")
            if not np.array_equal(pred["original_epoch_index"], np.arange(entry["epochs"])):
                raise ValueError("target full-record alignment changed")
            positions = reference["original_epoch_index"]
            for arm in models:
                matrices[arm] += confusion_matrix_5(reference["y"], pred[arm][positions].argmax(1))
    for arm in models:
        if not np.array_equal(matrices[arm], result["arms"][arm]["pooled"]["confusion_matrix"]):
            raise ValueError("target confusion does not reproduce")
    verification = {"status": "passed", "checkpoints_verified": 2, "target_prediction_hashes_verified": 180,
                    "target_confusions_recomputed": True, "matched_initial_weights_verified": True,
                    "selected_source_validation_scores_reproduced": True,
                    "source_results": source_results, "source_test_used_for_selection": False,
                    "aggregate_results_sha256": sha256_file(args.pilot / "aggregate_results.json"),
                    "verification_script_sha256": sha256_file(Path(__file__).resolve())}
    with (args.pilot / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(verification, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "source_test": {
        arm: {"macro_f1": data["test"]["pooled"]["macro_f1"],
              "n3": data["test"]["pooled"]["per_class"]["N3"]} for arm, data in source_results.items()}}, indent=2))


if __name__ == "__main__":
    main()
