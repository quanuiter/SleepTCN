"""Ten-fold E3 calibration/EM campaign with fold-specific source-only calibration."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.special import softmax
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.calibration import fit_temperature, apply_temperature, estimate_target_prior, adjust_prior
from sleeptcn.preprocessing import sha256_file
from sleeptcn.metrics import confusion_matrix_5
from sleeptcn.revision_campaign import load_restored_pair, infer_pair, mean_ten, write_once_json

PROTOCOL = ROOT / "configs/teacher_revision_calibration_10fold_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--shhs-root", type=Path, required=True)
    parser.add_argument("--adaptation-cache", type=Path, required=True)
    parser.add_argument("--verified-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("require private runs output")
    if (output / "aggregate_results.json").exists():
        raise FileExistsError("completed campaign must not be overwritten")
    cfg = json.loads(PROTOCOL.read_bytes())
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    helper_path = ROOT / "scripts/run_recovered_e3_cpu_pilot.py"
    loader = importlib.util.spec_from_file_location("tenfold_calibration_helpers", helper_path)
    helper = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helper)
    old = json.loads((args.verified_pilot / "aggregate_results.json").read_bytes())
    target_path = args.verified_pilot / "private_inference_manifest.json"
    if sha256_file(target_path) != old["inference_manifest_sha256"]:
        raise ValueError("full-record subject manifest changed")
    targets = json.loads(target_path.read_bytes())["records"]
    adapt_path = args.adaptation_cache / "private_manifest.json"
    if sha256_file(adapt_path) != old["provenance"]["adaptation_manifest_sha256"]:
        raise ValueError("adaptation selection changed")
    adapt = json.loads(adapt_path.read_bytes())
    if (len(targets) != 180 or len(adapt["records"]) != 5
            or {e["subject_id"] for e in targets} & {e["subject_id"] for e in adapt["records"]}):
        raise ValueError("target/adaptation subject mismatch")
    restored = json.loads((args.restored / "restoration_manifest.json").read_bytes())
    lookup = {e["git_path"]: e for e in restored["records"]}
    split_path = args.restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    splits = json.loads(split_path.read_bytes())["outer_runs"]
    models = {f: load_restored_pair(args.restored, "E3", f, cfg["seed"]) for f in cfg["folds"]}
    code_paths = [Path(__file__).resolve(), PROTOCOL, helper_path, ROOT / "src/sleeptcn/revision_campaign.py",
                  ROOT / "src/sleeptcn/calibration.py", ROOT / "src/sleeptcn/models.py",
                  ROOT / "src/sleeptcn/cpu_followups.py", ROOT / "src/sleeptcn/preprocessing.py",
                  ROOT / "src/sleeptcn/shhs_preprocessing.py", ROOT / "src/sleeptcn/metrics.py"]
    spec = {"protocol": cfg, "protocol_sha256": sha256_file(PROTOCOL),
            "restoration_sha256": sha256_file(args.restored / "restoration_manifest.json"),
            "target_manifest_sha256": sha256_file(target_path), "adaptation_manifest_sha256": sha256_file(adapt_path),
            "split_sha256": sha256_file(split_path), "checkpoint_sha256": {str(f): m[2] for f,m in models.items()},
            "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths}}
    write_once_json(output / "execution_specification.json", spec)
    for path in code_paths:
        dest = output / "code_snapshot" / path.relative_to(ROOT)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    fit_records = []
    for fold, (encoder, sequence, hashes) in models.items():
        destination = output / f"fits/fold_{fold:02d}.json"
        if destination.exists():
            fitted = json.loads(destination.read_bytes())
            if fitted["checkpoint_sha256"] != hashes or fitted["protocol_sha256"] != sha256_file(PROTOCOL):
                raise ValueError("resume fitted parameters differ")
        else:
            relative = f"runs/v2/full/E3/fold_{fold:02d}/seed_{cfg['seed']}/predictions/validation.npz"
            entry = lookup[relative]
            source_path = args.restored / relative
            if sha256_file(source_path) != entry["sha256"]:
                raise ValueError("source validation artifact changed")
            with np.load(source_path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata_json"]))
                expected = {"checkpoint_sha256": hashes[1], "experiment_id": "E3", "outer_fold": fold,
                            "role": "validation", "seed": cfg["seed"], "smoke": False,
                            "split_sha256": sha256_file(split_path), "data_variant": "filtered_v2"}
                if any(meta.get(k) != v for k,v in expected.items()):
                    raise ValueError("source calibration role/checkpoint mismatch")
                if set(z["record_key"].tolist()) != set(splits[fold]["validation"]["record_keys"]):
                    raise ValueError("source validation membership mismatch")
                valid = z["true_label"] >= 0
                labels = z["true_label"][valid]
                p = softmax(z["logits"][valid].astype(np.float64), axis=1)
                if len(labels) != splits[fold]["validation"]["valid_epochs"]:
                    raise ValueError("source validation support mismatch")
            temperature = fit_temperature(p, labels)
            source_prior = apply_temperature(p, temperature).mean(0)
            adaptation_p = []
            for record in sorted(adapt["records"], key=lambda e: e["subject_id"]):
                item = record["variants"]["filtered_v2"]
                if sha256_file(Path(item["path"])) != item["sha256"]:
                    raise ValueError("adaptation signal changed")
                with np.load(item["path"], allow_pickle=False) as z:
                    if set(z.files) != {"x", "original_epoch_index", "metadata_json"}:
                        raise ValueError("adaptation input contains unexpected fields/labels")
                    adaptation_p.append(infer_pair(encoder, sequence, z["x"], cfg["batch_size"]))
            adaptation_p = np.concatenate(adaptation_p)
            if len(adaptation_p) != 4989:
                raise ValueError("adaptation support mismatch")
            em = estimate_target_prior(apply_temperature(adaptation_p, temperature), source_prior)
            if not em.converged:
                raise ValueError(f"EM did not converge at fold {fold}; no result will be selected or substituted")
            fitted = {"fold": fold, "checkpoint_sha256": hashes, "protocol_sha256": sha256_file(PROTOCOL),
                      "source_validation_sha256": entry["sha256"], "source_validation_epochs": len(labels),
                      "temperature": temperature, "source_prior": source_prior.tolist(),
                      "target_prior": em.target_prior.tolist(), "em_iterations": em.iterations,
                      "adaptation_epochs": len(adaptation_p), "target_labels_used": False}
            write_once_json(destination, fitted)
        fit_records.append(fitted)
        print(f"Calibration fold {fold}/9 fitted: T={fitted['temperature']:.4f}, EM iterations={fitted['em_iterations']}", flush=True)
    write_once_json(output / "all_folds_fitted.json", {"status": "complete_before_target_inference", "folds": fit_records})
    fit_hash = sha256_file(output / "all_folds_fitted.json")
    entries = []
    began = time.perf_counter()
    for ordinal, entry in enumerate(targets, 1):
        path = output / "predictions" / (entry["record_key"] + ".npz")
        meta = {"subject_id": entry["subject_id"], "source_edf_sha256": entry["source_edf_sha256"],
                "fitted_sha256": fit_hash, "specification_sha256": sha256_file(output / "execution_specification.json")}
        if not path.exists():
            edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
            x, _ = helper.read_full_record(edf, entry["source_edf_sha256"])
            raw, calibrated, adjusted = {}, {}, {}
            for fold, (encoder, sequence, _) in models.items():
                fitted = fit_records[fold]
                raw[fold] = infer_pair(encoder, sequence, x, cfg["batch_size"])
                calibrated[fold] = apply_temperature(raw[fold], fitted["temperature"])
                adjusted[fold] = adjust_prior(calibrated[fold], fitted["source_prior"], fitted["target_prior"])
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                np.savez_compressed(stream, fold_raw=np.stack([raw[f] for f in range(10)]),
                    raw=mean_ten(raw), calibrated=mean_ten(calibrated), calibrated_em=mean_ten(adjusted),
                    original_epoch_index=np.arange(len(x)), metadata_json=np.array(json.dumps(meta)))
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z["metadata_json"])) != meta or not np.array_equal(z["original_epoch_index"], np.arange(entry["epochs"])):
                raise ValueError("resumed prediction identity mismatch")
            if z["fold_raw"].shape != (10, entry["epochs"], 5) or not np.isfinite(z["fold_raw"]).all():
                raise ValueError("invalid resumed predictions")
        entries.append({**entry, "path": str(path), "sha256": sha256_file(path)})
        if ordinal % 10 == 0 or ordinal == 1:
            print(f"Ten-fold calibration inference {ordinal}/180, {time.perf_counter()-began:.1f}s", flush=True)
    write_once_json(output / "private_inference_manifest.json", {"status": "complete", "records": entries})
    cm = {arm: [] for arm in cfg["arms"]}
    for entry in entries:
        reference = args.shhs_root / "processed_v1/filtered_v2" / (entry["record_key"] + ".npz")
        with np.load(reference, allow_pickle=False) as z, np.load(entry["path"], allow_pickle=False) as pred:
            if str(z["subject_id"]) != entry["subject_id"] or str(z["source_edf_sha256"]) != entry["source_edf_sha256"] or str(z["role"]) != "test":
                raise ValueError("benchmark reference identity mismatch")
            positions = helper.benchmark_positions(z["original_epoch_index"], entry["epochs"])
            for arm in cm:
                cm[arm].append(confusion_matrix_5(z["y"], pred[arm][positions].argmax(1)))
    cm = {arm: np.stack(values) for arm, values in cm.items()}
    result = helper.summarize_confusions(cm)
    if result["valid_epochs"] != 169012:
        raise ValueError("benchmark support mismatch")
    result.update(status="complete_additional_ten_fold_calibration", provenance=spec, fitted_sha256=fit_hash,
                  inference_manifest_sha256=sha256_file(output / "private_inference_manifest.json"))
    with (output / "private_subject_confusions.npz").open("xb") as stream:
        np.savez_compressed(stream, **cm)
    write_once_json(output / "aggregate_results.json", result)
    print(json.dumps({"status": result["status"], "means": {k: v["subject_mean_macro_f1"] for k,v in result["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
