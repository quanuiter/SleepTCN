"""Read-only artifact audit; write aggregate-only results to a new local output.

Does not expose participant identifiers, alter historical manifests or run models.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion


def signal_pairs(e3_root: Path, e4_root: Path, expected: int) -> dict:
    names3 = {p.name for p in e3_root.glob("*.npz")}
    names4 = {p.name for p in e4_root.glob("*.npz")}
    if names3 != names4 or len(names3) != expected:
        raise ValueError("paired signal file sets do not match expected record count")
    maximum = 0.0
    samples = 0
    for i, name in enumerate(sorted(names3), 1):
        with np.load(e3_root / name, allow_pickle=False) as a, \
                np.load(e4_root / name, allow_pickle=False) as b:
            for key in ["y", "valid_mask", "original_epoch_index", "source_edf_sha256"
                        if "source_edf_sha256" in a.files else "source_psg_sha256"]:
                if not np.array_equal(a[key], b[key]):
                    raise ValueError("paired source identity, labels or epoch alignment differs")
            x3, x4 = a["x"].astype(np.float64), b["x"].astype(np.float64)
            if x3.shape != x4.shape or not np.isfinite(x3).all() or not np.isfinite(x4).all():
                raise ValueError("signal shape/finite-value check failed")
            if "clip_fraction" in a.files and float(a["clip_fraction"]) != 0:
                raise ValueError("clipping is active; this is not a scale-only contrast")
            residual = np.abs(100 * x3 - x4)
            if not np.all(residual <= 2e-5 + 3e-7 * np.abs(x4)):
                raise ValueError("E3/E4 scale relationship failed declared float32 tolerance")
            maximum = max(maximum, float(residual.max()))
            samples += x3.size
        if i % 25 == 0 or i == expected:
            print(f"signal audit: {i}/{expected} records", flush=True)
    return {"records": expected, "samples_in_saved_windows": samples,
            "relation": "100*x_E3 ~= x_E4", "rtol": 3e-7, "atol_uv": 2e-5,
            "maximum_absolute_residual_uv": maximum,
            "scope": "saved benchmark windows; not all continuous pre-trim samples"}


def predictions(manifest_path: Path) -> dict:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    manifest_root = manifest_path.parents[2] / "manifests"
    inventory_raw = (manifest_root / "zero_shot_checkpoint_inventory_e4_seed123_v1.json").read_bytes()
    protocol_raw = (manifest_root / "shhs_e4_seed123_extension_protocol_v1.json").read_bytes()
    inventory, protocol = json.loads(inventory_raw), json.loads(protocol_raw)
    if (hashlib.sha256(inventory_raw).hexdigest() != manifest["checkpoint_inventory_sha256"]
            or hashlib.sha256(protocol_raw).hexdigest() != manifest["protocol_sha256"]
            or inventory["protocol_sha256"] != manifest["protocol_sha256"]
            or inventory["seed"] != 123 or protocol["checkpoint_seed"] != 123):
        raise ValueError("seed-123 inventory/protocol linkage does not match run manifest")
    lookup, result = {}, {}
    for experiment in ["E3", "E4"]:
        entries = [e for e in manifest["ensemble_records"] if e["experiment"] == experiment]
        if len(entries) != 180 or len({e["subject_id"] for e in entries}) != 180:
            raise ValueError("expected 180 unique subjects per configuration")
        matrices, subject_metrics = [], {}
        for entry in entries:
            raw_npz = Path(entry["path"]).read_bytes()
            if hashlib.sha256(raw_npz).hexdigest() != entry["sha256"]:
                raise ValueError("prediction artifact hash mismatch")
            with np.load(io.BytesIO(raw_npz), allow_pickle=False) as z:
                meta = json.loads(str(z["metadata_json"]))
                if any(meta[k] != entry[k] for k in ["experiment", "subject_id", "record_key", "role"]):
                    raise ValueError("prediction metadata mismatch")
                if (meta["protocol_sha256"] != manifest["protocol_sha256"]
                        or meta["checkpoint_inventory_sha256"] != manifest["checkpoint_inventory_sha256"]
                        or meta["folds"] != list(range(10))
                        or meta["data_variant"] != protocol["experiments"][experiment]["data_variant"]):
                    raise ValueError("prediction protocol/checkpoint/variant linkage mismatch")
                valid = z["valid_mask"]
                y, pred, indices = z["y"][valid], z["prediction"][valid], z["original_epoch_index"][valid]
                if not np.array_equal(z["probabilities"].argmax(1), z["prediction"]):
                    raise ValueError("probability argmax differs from stored prediction")
                cm = confusion_matrix_5(y, pred)
                if not np.array_equal(cm, entry["metrics"]["confusion_matrix"]):
                    raise ValueError("recomputed confusion differs from manifest")
                if experiment == "E3":
                    lookup[entry["subject_id"]] = (y.copy(), indices.copy())
                else:
                    paired = lookup[entry["subject_id"]]
                    if not np.array_equal(y, paired[0]) or not np.array_equal(indices, paired[1]):
                        raise ValueError("E3/E4 epoch alignment differs")
            matrices.append(cm)
            subject_metrics[entry["subject_id"]] = metrics_from_confusion(cm)["macro_f1"]
        pooled = metrics_from_confusion(np.sum(matrices, axis=0))
        estimate = float(np.mean(list(subject_metrics.values())))
        if not np.isclose(estimate, manifest["metrics"][experiment]["subject_macro_f1_mean"], atol=1e-12):
            raise ValueError("subject mean differs from manifest")
        result[experiment] = {"subjects": 180, "hash_matches": 180,
                              "subject_mean_macro_f1": estimate, "pooled": pooled}
    result["E4_minus_E3_subject_mean_macro_f1"] = (
        result["E4"]["subject_mean_macro_f1"] - result["E3"]["subject_mean_macro_f1"])
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "seed": 123,
            "inventory_protocol_seed_linkage_verified": True,
            "results": result, "scope": "recomputed stored predictions; no inference or CI rerun"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shhs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shhs = args.shhs_root
    inventories = {}
    for name in ["zero_shot_checkpoint_inventory_v1.json", "component_checkpoint_inventory_v1.json",
                 "zero_shot_checkpoint_inventory_e4_seed123_v1.json"]:
        path = shhs / "manifests" / name
        data = json.loads(path.read_bytes())
        unique = {item["path"]: item for fold in data["folds"] for item in fold["checkpoints"]}
        available = [item for p, item in unique.items() if Path(p).is_file()]
        inventories[name] = {"referenced_checkpoints": len(unique), "available_at_recorded_paths": len(available)}
    resource = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "full_edf_checkpoints": len(list((ROOT / "runs/v2/full").rglob("*.pt"))),
                "smoke_checkpoints": len(list((ROOT / "runs/smoke").rglob("*.pt"))),
                "full_edf_prediction_npz": len(list((ROOT / "runs/v2/full").rglob("*.npz"))),
                "checkpoint_inventory_availability": inventories}
    result = {"date": "2026-10-01", "resources": resource,
              "edf_saved_signal_pairs": signal_pairs(ROOT / "data/processed/filtered_v2",
                                                      ROOT / "data/processed/bandpass_v2", 153),
              "shhs_saved_signal_pairs": signal_pairs(shhs / "processed_v1/filtered_v2",
                                                       shhs / "processed_extension_e4_v1/bandpass_v2", 200),
              "shhs_seed123_e3_e4": predictions(shhs / "zero_shot_e4_seed123_v1/test/run_manifest.json")}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("aggregate-only input audit complete", flush=True)


if __name__ == "__main__":
    main()
