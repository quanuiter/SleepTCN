"""Audit resumable calibration outputs; independently replay a completed campaign."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.calibration import apply_temperature, adjust_prior, fit_temperature, estimate_target_prior
from sleeptcn.metrics import confusion_matrix_5
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import load_restored_pair, infer_pair, write_once_json


def audit_prediction(path, entry, fits, fit_hash, spec_hash):
    with np.load(path, allow_pickle=False) as z:
        expected = {"subject_id": entry["subject_id"], "source_edf_sha256": entry["source_edf_sha256"],
                    "fitted_sha256": fit_hash, "specification_sha256": spec_hash}
        if json.loads(str(z["metadata_json"])) != expected:
            raise ValueError("prediction identity differs")
        np.testing.assert_array_equal(z["original_epoch_index"], np.arange(entry["epochs"]))
        raw = z["fold_raw"]
        if raw.shape != (10, entry["epochs"], 5) or not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
            raise ValueError("invalid fold probabilities")
        np.testing.assert_allclose(raw.sum(-1), 1., rtol=0, atol=1e-6)
        calibrated = [apply_temperature(raw[f], fits[f]["temperature"]) for f in range(10)]
        adjusted = [adjust_prior(calibrated[f], fits[f]["source_prior"], fits[f]["target_prior"]) for f in range(10)]
        for arm, parts in [("raw", raw), ("calibrated", calibrated), ("calibrated_em", adjusted)]:
            expected_p = np.mean(np.asarray(parts, dtype=np.float64), axis=0).astype(np.float32)
            np.testing.assert_array_equal(z[arm], expected_p)
        return {k: z[k].copy() for k in ("fold_raw", "raw", "calibrated", "calibrated_em")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--restored", required=True, type=Path)
    parser.add_argument("--shhs-root", required=True, type=Path)
    parser.add_argument("--adaptation-cache", required=True, type=Path)
    parser.add_argument("--verified-pilot", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign
    specification = campaign / "execution_specification.json"
    spec = json.loads(specification.read_bytes())
    for relative, digest in spec["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest or sha256_file(campaign / "code_snapshot" / relative) != digest:
            raise ValueError("executed code changed")
    for path, key in [(args.restored / "restoration_manifest.json", "restoration_sha256"),
                      (args.verified_pilot / "private_inference_manifest.json", "target_manifest_sha256"),
                      (args.adaptation_cache / "private_manifest.json", "adaptation_manifest_sha256"),
                      (args.restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json", "split_sha256")]:
        if sha256_file(path) != spec[key]:
            raise ValueError(f"input manifest changed: {key}")
    fitted_path = campaign / "all_folds_fitted.json"
    fit_document = json.loads(fitted_path.read_bytes())
    fits = fit_document["folds"]
    if fit_document["status"] != "complete_before_target_inference" or [f["fold"] for f in fits] != list(range(10)):
        raise ValueError("incomplete fits")
    for f in fits:
        if f != json.loads((campaign / f"fits/fold_{f['fold']:02d}.json").read_bytes()):
            raise ValueError("fit manifests disagree")
    targets = json.loads((args.verified_pilot / "private_inference_manifest.json").read_bytes())["records"]
    if len(targets) != 180 or len({e["subject_id"] for e in targets}) != 180:
        raise ValueError("target membership differs")
    expected_files = {e["record_key"] + ".npz" for e in targets}
    paths = list((campaign / "predictions").glob("*.npz"))
    if not {p.name for p in paths} <= expected_files:
        raise ValueError("unexpected prediction files")
    fit_hash, spec_hash = sha256_file(fitted_path), sha256_file(specification)
    for e in targets:
        path = campaign / "predictions" / (e["record_key"] + ".npz")
        if path.exists():
            audit_prediction(path, e, fits, fit_hash, spec_hash)
    print(json.dumps({"prediction_files_verified": len(paths), "expected": 180, "preflight": args.preflight}), flush=True)
    if args.preflight:
        return
    if len(paths) != 180:
        raise ValueError("campaign is not complete")
    result_path = campaign / "aggregate_results.json"
    result = json.loads(result_path.read_bytes())
    manifest = campaign / "private_inference_manifest.json"
    if (result["provenance"] != spec or result["fitted_sha256"] != fit_hash
            or result["inference_manifest_sha256"] != sha256_file(manifest)):
        raise ValueError("result provenance differs")
    records = json.loads(manifest.read_bytes())["records"]
    if [e["record_key"] for e in records] != [e["record_key"] for e in targets]:
        raise ValueError("completed target order differs")
    loader = importlib.util.spec_from_file_location("calibration_verify_helpers", ROOT / "scripts/run_recovered_e3_cpu_pilot.py")
    helper = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helper)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    models = {f: load_restored_pair(args.restored, "E3", f, spec["protocol"]["seed"]) for f in range(10)}
    adapt = json.loads((args.adaptation_cache / "private_manifest.json").read_bytes())["records"]
    if len(adapt) != 5 or {e["subject_id"] for e in adapt} & {e["subject_id"] for e in targets}:
        raise ValueError("adaptation subjects not disjoint")
    for f, (encoder, sequence, hashes) in models.items():
        fit = fits[f]
        if hashes != fit["checkpoint_sha256"] or hashes != spec["checkpoint_sha256"][str(f)] or fit["target_labels_used"]:
            raise ValueError("fit provenance differs")
        val = args.restored / f"runs/v2/full/E3/fold_{f:02d}/seed_{spec['protocol']['seed']}/predictions/validation.npz"
        if sha256_file(val) != fit["source_validation_sha256"]:
            raise ValueError("source validation changed")
        with np.load(val, allow_pickle=False) as z:
            valid = z["true_label"] >= 0
            source_p = softmax(z["logits"][valid].astype(np.float64), axis=1)
            temperature = fit_temperature(source_p, z["true_label"][valid])
        np.testing.assert_allclose(temperature, fit["temperature"], rtol=0, atol=1e-10)
        source_prior = apply_temperature(source_p, temperature).mean(0)
        np.testing.assert_allclose(source_prior, fit["source_prior"], rtol=0, atol=1e-12)
        adapt_p = []
        for item in sorted(adapt, key=lambda e: e["subject_id"]):
            variant = item["variants"]["filtered_v2"]
            if sha256_file(Path(variant["path"])) != variant["sha256"]:
                raise ValueError("adaptation input changed")
            with np.load(variant["path"], allow_pickle=False) as z:
                if set(z.files) != {"x", "original_epoch_index", "metadata_json"}:
                    raise ValueError("adaptation contains unexpected fields")
                adapt_p.append(infer_pair(encoder, sequence, z["x"], spec["protocol"]["batch_size"]))
        adapt_p = np.concatenate(adapt_p)
        if len(adapt_p) != fit["adaptation_epochs"] or len(adapt_p) != 4989:
            raise ValueError("adaptation support differs")
        em = estimate_target_prior(apply_temperature(adapt_p, temperature), source_prior)
        if not em.converged or em.iterations != fit["em_iterations"]:
            raise ValueError("EM convergence differs")
        np.testing.assert_allclose(em.target_prior, fit["target_prior"], rtol=0, atol=1e-10)
    cm = {arm: [] for arm in spec["protocol"]["arms"]}
    replays, reference_hashes = [], []
    # Includes both sides of the interrupted/resumed boundary, not just endpoints.
    for ordinal, entry in enumerate(records):
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("prediction hash changed")
        predictions = audit_prediction(entry["path"], entry, fits, fit_hash, spec_hash)
        ref = args.shhs_root / "processed_v1/filtered_v2" / (entry["record_key"] + ".npz")
        with np.load(ref, allow_pickle=False) as z:
            if str(z["subject_id"]) != entry["subject_id"] or str(z["source_edf_sha256"]) != entry["source_edf_sha256"] or str(z["role"]) != "test":
                raise ValueError("reference identity differs")
            positions = helper.benchmark_positions(z["original_epoch_index"], entry["epochs"])
            for arm in cm:
                cm[arm].append(confusion_matrix_5(z["y"], predictions[arm][positions].argmax(1)))
        reference_hashes.append({"record_key": entry["record_key"], "sha256": sha256_file(ref)})
        if ordinal in [0, 74, 75, 179]:
            edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
            x, _ = helper.read_full_record(edf, entry["source_edf_sha256"])
            for f, (encoder, sequence, _) in models.items():
                actual = infer_pair(encoder, sequence, x, spec["protocol"]["batch_size"])
                saved = predictions["fold_raw"][f]
                residual = float(np.max(np.abs(actual - saved)))
                if residual > 3e-6 or np.any(actual.argmax(1) != saved.argmax(1)):
                    raise ValueError("checkpoint replay differs")
                replays.append({"record_ordinal": ordinal, "fold": f, "max_probability_residual": residual})
    cm = {arm: np.stack(values) for arm, values in cm.items()}
    recalculated = helper.summarize_confusions(cm)
    if any(result[k] != value for k, value in recalculated.items()):
        raise ValueError("aggregate metrics/bootstrap differ")
    with np.load(campaign / "private_subject_confusions.npz", allow_pickle=False) as z:
        for arm in cm:
            np.testing.assert_array_equal(cm[arm], z[arm])
    document = {"status": "passed", "subjects_verified": 180, "source_calibration_and_em_refitted": 10,
                "fold_prediction_matrices_verified": 1800, "checkpoint_replays": replays,
                "reference_hashes": reference_hashes, "metrics_and_bootstrap_recomputed": True,
                "aggregate_results_sha256": sha256_file(result_path), "verifier_sha256": sha256_file(Path(__file__))}
    write_once_json(campaign / "verification.json", document)
    print(json.dumps({"status": "passed", "subjects": 180, "replayed_fold_records": len(replays)}))


if __name__ == "__main__":
    main()
