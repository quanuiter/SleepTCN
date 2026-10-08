"""Single-fold, CPU calibration/EM pilot with deliberately label-free target files.

See docs/TEACHER_REVISION_EXECUTION_V1.md for the input contract and limitations.
No target metrics, fold selection or ensemble fitting is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.calibration import (STAGES, apply_temperature, estimate_target_prior,
                                  adjust_prior, fit_temperature, probabilities)


def load_input(path: Path, *, labelled: bool) -> tuple[dict, str]:
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        expected = {"probabilities", "epoch_id"} | ({"labels"} if labelled else set())
        if set(z.files) != expected:
            raise ValueError(f"{path.name}: expected keys {sorted(expected)}; target labels are forbidden")
        arrays = {key: z[key] for key in z.files}
    p = probabilities(arrays["probabilities"])
    ids = arrays["epoch_id"]
    if ids.shape != (len(p),) or ids.dtype.kind not in "US" or len(np.unique(ids)) != len(ids):
        raise ValueError("epoch_id must be unique strings aligned to probability rows")
    arrays["probabilities"] = p
    return arrays, hashlib.sha256(data).hexdigest()


def run(manifest_path: Path, output: Path) -> dict:
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get("schema_version") != 1 or manifest.get("class_order") != list(STAGES):
        raise ValueError("expected schema 1 and class order W,N1,N2,N3,REM")
    if manifest.get("source_role") != "validation" or not manifest.get("source_split_sha256"):
        raise ValueError("source role must be validation with split provenance")
    if manifest.get("adaptation_window_policy") != "label_independent":
        raise ValueError("adaptation window must not be selected using target sleep labels")
    if not isinstance(manifest.get("outer_fold"), int) or not 0 <= manifest["outer_fold"] < 10:
        raise ValueError("outer_fold must be 0..9")
    checkpoint = manifest.get("checkpoint_sha256", "")
    split = manifest["source_split_sha256"]
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
           for value in [checkpoint, split]):
        raise ValueError("checkpoint and split identities must be lowercase SHA-256 values")
    arrays, hashes = {}, {}
    for role in ["source_validation", "target_adaptation", "target_inference"]:
        arrays[role], hashes[role] = load_input(manifest_path.parent / manifest[role],
                                              labelled=(role == "source_validation"))
    source = arrays["source_validation"]
    temperature = fit_temperature(source["probabilities"], source["labels"])
    calibrated_source = apply_temperature(source["probabilities"], temperature)
    # Source-validation mean posterior, fixed before consulting target labels.
    source_prior = calibrated_source.mean(axis=0)
    adaptation = apply_temperature(arrays["target_adaptation"]["probabilities"], temperature)
    result = estimate_target_prior(adaptation, source_prior)
    if not result.converged:
        raise RuntimeError("EM did not converge in 1000 iterations; no pilot output written")
    inference = arrays["target_inference"]
    calibrated = apply_temperature(inference["probabilities"], temperature)
    adjusted = adjust_prior(calibrated, source_prior, result.target_prior)
    meta = {
        "status": "research_pilot_not_confirmatory", "schema_version": 1,
        "class_order": list(STAGES), "outer_fold": manifest["outer_fold"],
        "checkpoint_sha256": checkpoint, "source_split_sha256": split,
        "input_sha256": hashes, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__).resolve(), ROOT / "src/sleeptcn/calibration.py"]},
        "calibration": "source_validation_epoch_weighted_temperature_NLL_bounds_0.05_20",
        "temperature": temperature, "source_prior_rule": "calibrated_validation_mean_posterior",
        "source_prior": source_prior.tolist(), "target_prior": result.target_prior.tolist(),
        "adaptation_window_policy": "label_independent", "em_iterations": result.iterations,
        "em_tolerance": 1e-8, "em_max_iterations": 1000, "probability_floor": 1e-12,
        "em_log_likelihood": list(result.log_likelihood),
        "target_labels_used": False,
        "role_provenance_note": "Manifest declarations require independent split/window audit.",
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, raw=inference["probabilities"], calibrated=calibrated,
                        calibrated_em=adjusted, epoch_id=inference["epoch_id"],
                        metadata_json=np.array(json.dumps(meta, sort_keys=True)))
    # Exclusive creation: historical predictions and prior outputs cannot be overwritten.
    with output.open("xb") as stream:
        stream.write(buffer.getvalue())
    return {"status": meta["status"], "epochs": len(calibrated),
            "temperature": temperature, "em_iterations": result.iterations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), indent=2))
