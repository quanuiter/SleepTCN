"""Recover interrupted target inference without retraining or changing selection."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.cpu_followups import confusion_diagnostics, paired_diagnostic_bootstrap
from sleeptcn.demo import load_demo_models, validate_asset_manifest
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.models import SleepTCN
from sleeptcn.preprocessing import sha256_file


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_prediction(path, entry, selections):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"unweighted", "weighted", "original_epoch_index", "metadata_json"}:
            raise ValueError("unexpected prediction schema")
        meta = json.loads(str(z["metadata_json"]))
        if meta["subject_id"] != entry["subject_id"] or meta["selected_checkpoints"] != selections:
            raise ValueError("prediction identity/selection mismatch")
        if not np.array_equal(z["original_epoch_index"], np.arange(entry["epochs"])):
            raise ValueError("prediction full-record indices mismatch")
        for arm in selections:
            p = z[arm]
            if (p.shape != (entry["epochs"], 5) or not np.isfinite(p).all()
                    or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6)):
                raise ValueError("invalid probabilities")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--verified-pilot", required=True, type=Path)
    parser.add_argument("--shhs-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.pilot.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("require ignored runs output")
    if (output / "aggregate_results.json").exists():
        raise FileExistsError("completed results must not be overwritten")
    provenance = json.loads((output / "execution_specification.json").read_bytes())
    for relative, digest in provenance["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest or sha256_file(output / "code_snapshot" / relative) != digest:
            raise ValueError("original executed source has changed")
    original_result = args.verified_pilot / "aggregate_results.json"
    if sha256_file(original_result) != provenance["recovered_pilot_sha256"]:
        raise ValueError("original pilot changed")
    pilot = json.loads(original_result.read_bytes())
    original_manifest = args.verified_pilot / "private_inference_manifest.json"
    if sha256_file(original_manifest) != pilot["inference_manifest_sha256"]:
        raise ValueError("original target manifest changed")
    original = json.loads(original_manifest.read_bytes())["records"]
    if len(original) != 180 or len({e["subject_id"] for e in original}) != 180:
        raise ValueError("require 180 unique locked test subjects")
    selections = json.loads((output / "both_arms_selected.json").read_bytes())
    if selections["weighted"]["initial_state_sha256"] != selections["unweighted"]["initial_state_sha256"]:
        raise ValueError("initial states differ")
    torch.set_num_threads(4)
    models = {}
    for arm, selected in selections.items():
        checkpoint = output / arm / "best.pt"
        if sha256_file(checkpoint) != selected["checkpoint_sha256"]:
            raise ValueError("selected checkpoint changed")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["epoch"] != selected["selected_epoch"] or state["protocol_sha256"] != provenance["protocol_sha256"]:
            raise ValueError("checkpoint selection metadata mismatch")
        model = SleepTCN(input_dim=128).eval()
        model.load_state_dict(state["model_state"], strict=True)
        models[arm] = model
    helper = load_helper("recovery_signal_helpers", ROOT / "scripts/run_recovered_e3_cpu_pilot.py")
    writer = load_helper("recovery_training_helpers", ROOT / "scripts/run_weighted_tcn_cpu_pilot.py")
    assets = ROOT / "demo/assets"
    asset_manifest = validate_asset_manifest(assets)
    if asset_manifest["experiments"]["E3"]["extractor"]["sha256"] != pilot["provenance"]["checkpoint_sha256"]["extractor"]:
        raise ValueError("encoder changed")
    encoder = load_demo_models(assets, "E3", validated_manifest=asset_manifest).extractor.eval()
    pred_folder = output / "target_probabilities"
    existing = []
    for entry in original:
        path = pred_folder / (entry["record_key"] + ".npz")
        if path.exists():
            validate_prediction(path, entry, selections)
            existing.append({"path": str(path), "sha256": sha256_file(path)})
    recovery = {"status": "inference_only_no_retraining", "existing_files": existing,
                "selected_arms_sha256": sha256_file(output / "both_arms_selected.json"),
                "recovery_script_sha256": sha256_file(Path(__file__))}
    recovery_path = output / "private_recovery_specification.json"
    if not recovery_path.exists():
        writer.write_json(recovery_path, recovery)
        shutil.copyfile(__file__, output / "code_snapshot/scripts/resume_weighted_cpu_pilot_evaluation.py")
    else:
        saved = json.loads(recovery_path.read_bytes())
        if saved["recovery_script_sha256"] != recovery["recovery_script_sha256"]:
            raise ValueError("recovery script changed between attempts")
        for item in saved["existing_files"]:
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("previously recovered artifact changed")
    print(f"Validated {len(existing)}/180 existing files; continuing missing inference only", flush=True)
    targets = []
    for i, entry in enumerate(original, 1):
        path = pred_folder / (entry["record_key"] + ".npz")
        if not path.exists():
            edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
            x, meta = helper.read_full_record(edf, entry["source_edf_sha256"])
            with torch.inference_mode():
                parts = [encoder.extract_features(torch.from_numpy(x[start:start+64]).unsqueeze(1))
                         for start in range(0, len(x), 64)]
                features = torch.cat(parts).unsqueeze(0)
                predictions = {arm: torch.softmax(model(features, padding_mask=None), -1).squeeze(0).numpy()
                               for arm, model in models.items()}
            with path.open("xb") as stream:
                np.savez_compressed(stream, **predictions, original_epoch_index=np.arange(len(x)),
                                    metadata_json=np.array(json.dumps({"subject_id": entry["subject_id"],
                                        "preprocessing": meta, "selected_checkpoints": selections})))
            validate_prediction(path, entry, selections)
        targets.append({**entry, "path": str(path), "sha256": sha256_file(path)})
        if i % 20 == 0 or i == len(original):
            print(f"Recovered target inference {i}/180", flush=True)
    manifest_path = output / "private_inference_manifest.json"
    manifest = {"status": "complete", "records": targets}
    if manifest_path.exists():
        if json.loads(manifest_path.read_bytes()) != manifest:
            raise ValueError("existing completed manifest differs")
    else:
        writer.write_json(manifest_path, manifest)
    # Only now read target reference labels. Selection remains fixed throughout.
    confusions = {arm: [] for arm in models}
    for entry in targets:
        reference = args.shhs_root / "processed_v1/filtered_v2" / (entry["record_key"] + ".npz")
        with np.load(reference, allow_pickle=False) as z, np.load(entry["path"], allow_pickle=False) as pred:
            if (str(z["subject_id"]) != entry["subject_id"] or str(z["role"]) != "test"
                    or str(z["source_edf_sha256"]) != entry["source_edf_sha256"]):
                raise ValueError("reference identity mismatch")
            positions = helper.benchmark_positions(z["original_epoch_index"], entry["epochs"])
            for arm in models:
                confusions[arm].append(confusion_matrix_5(z["y"], pred[arm][positions].argmax(1)))
    cm = {arm: np.stack(values) for arm, values in confusions.items()}
    contrast = paired_diagnostic_bootstrap(cm["weighted"], cm["unweighted"])
    difference = confusion_diagnostics(cm["weighted"])["macro_f1"] - confusion_diagnostics(cm["unweighted"])["macro_f1"]
    draws = np.random.default_rng(2031).integers(0, len(difference), (10000, len(difference)), dtype=np.int32)
    contrast["subject_mean_macro_f1"] = {"difference": float(difference.mean()),
        "ci95": np.quantile(difference[draws].mean(1), [.025, .975]).tolist()}
    if any(matrix.sum() != 169012 for matrix in cm.values()):
        raise ValueError("benchmark support differs")
    results = {"status": "complete_exploratory_single_fold_weighted_loss_pilot", "provenance": provenance,
               "selections": selections, "subjects": 180, "valid_epochs": 169012,
               "arms": {arm: {"subject_mean_macro_f1": float(confusion_diagnostics(matrix)["macro_f1"].mean()),
                               "pooled": metrics_from_confusion(matrix.sum(0))} for arm, matrix in cm.items()},
               "weighted_minus_unweighted": contrast,
               "recovery": {"specification_sha256": sha256_file(recovery_path),
                            "script_sha256": sha256_file(Path(__file__)),
                            "inference_manifest_sha256": sha256_file(manifest_path)}}
    confusion_path = output / "private_subject_confusions.npz"
    if confusion_path.exists():
        with np.load(confusion_path, allow_pickle=False) as z:
            if any(not np.array_equal(z[arm], matrix) for arm, matrix in cm.items()):
                raise ValueError("existing confusion artifact differs")
    else:
        with confusion_path.open("xb") as stream:
            np.savez_compressed(stream, **cm)
    writer.write_json(output / "aggregate_results.json", results)
    print(json.dumps({"status": results["status"], "arms": results["arms"],
                      "subject_mean_contrast": contrast["subject_mean_macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
