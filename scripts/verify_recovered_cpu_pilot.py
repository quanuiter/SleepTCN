"""Replay the standalone calibration CLI and verify the completed real-data pilot."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.calibration import apply_temperature, probabilities
from sleeptcn.demo import validate_asset_manifest
from sleeptcn.preprocessing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    root = args.pilot.resolve()
    if not root.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("participant outputs must remain inside ignored runs storage")
    result = json.loads((root / "aggregate_results.json").read_bytes())
    fit = result["fitted_parameters"]
    inventory = json.loads((root / "private_inference_manifest.json").read_bytes())
    if (result["status"] != "complete_exploratory_single_fold_pilot" or inventory["status"] != "complete"
            or len(inventory["records"]) != 180):
        raise ValueError("real-data pilot is incomplete")
    for filename, expected in [("fitted_parameters.json", result["fitted_parameters_sha256"]),
                               ("source_validation.npz", fit["source_validation_sha256"]),
                               ("target_adaptation.npz", fit["adaptation_probabilities_sha256"]),
                               ("private_inference_manifest.json", result["inference_manifest_sha256"])]:
        if sha256_file(root / filename) != expected:
            raise ValueError("pilot input/fitted-parameter hash changed")
    for path, expected in result["provenance"]["code_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError("executed pilot code has changed")
    arrays, ids, expected_arms = [], [], {arm: [] for arm in ["calibrated", "calibrated_em"]}
    for item in inventory["records"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError("test probability artifact hash changed")
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"raw", "calibrated", "calibrated_em", "original_epoch_index", "metadata_json"}:
                raise ValueError("unexpected target schema")
            if not np.array_equal(z["original_epoch_index"], np.arange(item["epochs"])):
                raise ValueError("target inference is not full-record aligned")
            arrays.append(z["raw"])
            for arm in expected_arms:
                expected_arms[arm].append(z[arm])
            ids.extend(f"{item['subject_id']}:{int(i)}" for i in z["original_epoch_index"])
    raw = np.concatenate(arrays)
    with (root / "target_inference.npz").open("xb") as stream:
        np.savez_compressed(stream, probabilities=raw, epoch_id=np.array(ids))
    manifest = {"schema_version": 1, "class_order": ["W", "N1", "N2", "N3", "REM"],
                "outer_fold": 0, "checkpoint_sha256": result["provenance"]["checkpoint_sha256"]["sequence"],
                "source_split_sha256": result["provenance"]["source_split_sha256"],
                "source_role": "validation", "adaptation_window_policy": "label_independent",
                "source_validation": "source_validation.npz", "target_adaptation": "target_adaptation.npz",
                "target_inference": "target_inference.npz"}
    manifest_path = root / "standalone_cli_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    spec = importlib.util.spec_from_file_location("standalone_calibration", ROOT / "scripts/run_calibration_pilot.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli_result = cli.run(manifest_path, root / "standalone_cli_output.npz")
    residuals = {}
    with np.load(root / "standalone_cli_output.npz", allow_pickle=False) as z:
        if not np.array_equal(z["epoch_id"], np.array(ids)):
            raise ValueError("standalone CLI epoch alignment differs")
        for arm, parts in expected_arms.items():
            expected = np.concatenate(parts)
            residuals[arm] = float(np.abs(z[arm] - expected).max())
            np.testing.assert_allclose(z[arm], expected, atol=1e-12, rtol=1e-12)
    assets = ROOT / "demo/assets"
    asset_manifest = validate_asset_manifest(assets)
    prediction = assets / asset_manifest["experiments"]["E3"]["prediction"]["path"]
    with np.load(prediction, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata_json"]))
        if (metadata["outer_fold"] != 0 or metadata["seed"] != 123 or metadata["role"] != "test"
                or metadata["checkpoint_sha256"] != result["provenance"]["checkpoint_sha256"]["sequence"]):
            raise ValueError("held-out source prediction provenance mismatch")
        p, y = probabilities(softmax(z["logits"].astype(np.float64), axis=1)), z["true_label"]
    calibrated = apply_temperature(p, fit["temperature"])
    nll = lambda values: float(-np.log(probabilities(values)[np.arange(len(y)), y]).mean())
    verification = {"status": "passed", "target_files_hash_verified": 180,
                    "full_target_epochs": len(raw), "standalone_cli": cli_result,
                    "standalone_cli_maximum_probability_residual": residuals,
                    "source_outer_test": {"epochs": len(y), "raw_nll": nll(p),
                                          "calibrated_nll": nll(calibrated),
                                          "argmax_changes": int((p.argmax(1) != calibrated.argmax(1)).sum()),
                                          "used_to_fit_temperature": False},
                    "verification_code_sha256": sha256_file(Path(__file__).resolve()),
                    "aggregate_results_sha256": sha256_file(root / "aggregate_results.json")}
    with (root / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(verification, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
