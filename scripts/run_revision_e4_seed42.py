"""Additional E4 seed42 ten-fold inference, paired with verified historical E3."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.preprocessing import sha256_file
from sleeptcn.metrics import confusion_matrix_5
from sleeptcn.revision_campaign import load_restored_pair, infer_pair, mean_ten, summarize_pair, write_once_json

PROTOCOL = ROOT / "configs/teacher_revision_e4_seed42_extension_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--shhs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("require private runs output")
    if (output / "aggregate_results.json").exists():
        raise FileExistsError("completed run must not be overwritten")
    cfg = json.loads(PROTOCOL.read_bytes())
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    old_path = args.shhs_root / "zero_shot_v1/test/run_manifest.json"
    old = json.loads(old_path.read_bytes())
    inventory = args.shhs_root / "manifests/zero_shot_checkpoint_inventory_v1.json"
    if sha256_file(inventory) != old["checkpoint_inventory_sha256"]:
        raise ValueError("historical E3 inventory changed")
    pre_path = args.shhs_root / "manifests/preprocess_combined_e4_seed123_v1.json"
    pre = json.loads(pre_path.read_bytes())
    lookup = {(e["record_key"], e["variant"]): e for e in pre["records"]}
    subjects = sorted([e for e in old["ensemble_records"] if e["experiment"] == "E3"], key=lambda e: e["record_key"])
    if len(subjects) != 180 or len({e["subject_id"] for e in subjects}) != 180:
        raise ValueError("expected 180 unique subjects")
    models = {fold: load_restored_pair(args.restored, "E4", fold, 42) for fold in range(10)}
    code_paths = [Path(__file__).resolve(), PROTOCOL, ROOT / "src/sleeptcn/revision_campaign.py",
                  ROOT / "src/sleeptcn/models.py", ROOT / "src/sleeptcn/cpu_followups.py",
                  ROOT / "src/sleeptcn/metrics.py"]
    spec = {"protocol": cfg, "protocol_sha256": sha256_file(PROTOCOL),
            "restoration_manifest_sha256": sha256_file(args.restored / "restoration_manifest.json"),
            "historical_E3_manifest_sha256": sha256_file(old_path),
            "processed_manifest_sha256": sha256_file(pre_path),
            "checkpoint_sha256": {str(f): v[2] for f, v in models.items()},
            "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
            "torch": str(torch.__version__)}
    write_once_json(output / "execution_specification.json", spec)
    for path in code_paths:
        dest = output / "code_snapshot" / path.relative_to(ROOT)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    # Validate runtime/checkpoint reconstruction against existing E3 before
    # generating any new E4 target predictions.
    first = subjects[0]
    first_input = lookup[first["record_key"], "filtered_v2"]
    if sha256_file(Path(first_input["output_path"])) != first_input["output_sha256"]:
        raise ValueError("replay input changed")
    with np.load(first_input["output_path"], allow_pickle=False) as z:
        x = z["x"]
    replay_parts, replay_details = {}, []
    for fold in range(10):
        encoder, sequence, hashes = load_restored_pair(args.restored, "E3", fold, 42)
        entry = next(e for e in old["fold_records"] if e["experiment"] == "E3"
                     and e["record_key"] == first["record_key"] and e["outer_fold"] == fold)
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("E3 fold prediction hash mismatch")
        actual = infer_pair(encoder, sequence, x, cfg["batch_size"])
        with np.load(entry["path"], allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"]))
            if meta["extractor_sha256"] != hashes[0] or meta["sequence_checkpoint_sha256"] != hashes[1]:
                raise ValueError("E3 replay checkpoint mismatch")
            difference = float(np.max(np.abs(actual - z["probabilities"])))
            disagreements = int(np.count_nonzero(actual.argmax(1) != z["probabilities"].argmax(1)))
        if difference > 3e-6 or disagreements:
            raise ValueError("E3 numerical replay failed")
        replay_parts[fold] = actual
        replay_details.append({"fold": fold, "maximum_probability_residual": difference, "argmax_disagreements": disagreements})
    if sha256_file(Path(first["path"])) != first["sha256"]:
        raise ValueError("E3 ensemble artifact changed")
    with np.load(first["path"], allow_pickle=False) as z:
        np.testing.assert_allclose(mean_ten(replay_parts), z["probabilities"], atol=3e-6, rtol=0)
        if np.any(mean_ten(replay_parts).argmax(1) != z["probabilities"].argmax(1)):
            raise ValueError("E3 ensemble decisions do not replay")
    write_once_json(output / "E3_replay_verification.json", {"status": "passed", "folds": replay_details})
    print("Historical E3 first-subject replay passed for all ten folds and ensemble", flush=True)
    records = []
    begun = time.perf_counter()
    for i, subject in enumerate(subjects, 1):
        entry = lookup[subject["record_key"], "bandpass_v2"]
        path = Path(entry["output_path"])
        if sha256_file(path) != entry["output_sha256"] or entry["subject_id"] != subject["subject_id"] or entry["role"] != "test":
            raise ValueError("E4 input identity changed")
        if sha256_file(Path(subject["path"])) != subject["sha256"]:
            raise ValueError("E3 ensemble changed")
        with np.load(path, allow_pickle=False) as z, np.load(subject["path"], allow_pickle=False) as ref:
            for field in ["y", "valid_mask", "original_epoch_index"]:
                if not np.array_equal(z[field], ref[field]):
                    raise ValueError("E3/E4 benchmark alignment differs")
            x = z["x"]
            indices = z["original_epoch_index"]
        dest = output / "predictions" / (subject["record_key"] + ".npz")
        meta = {"subject_id": subject["subject_id"], "record_key": subject["record_key"],
                "input_sha256": entry["output_sha256"], "specification_sha256": sha256_file(output / "execution_specification.json")}
        if not dest.exists():
            parts = {fold: infer_pair(enc, seq, x, cfg["batch_size"]) for fold, (enc, seq, _) in models.items()}
            dest.parent.mkdir(exist_ok=True)
            with dest.open("xb") as stream:
                np.savez_compressed(stream, fold_probabilities=np.stack([parts[f] for f in range(10)]),
                    probabilities=mean_ten(parts), original_epoch_index=indices, metadata_json=np.array(json.dumps(meta)))
        with np.load(dest, allow_pickle=False) as z:
            if json.loads(str(z["metadata_json"])) != meta or not np.array_equal(indices, z["original_epoch_index"]):
                raise ValueError("resume identity/alignment mismatch")
            np.testing.assert_array_equal(z["probabilities"], mean_ten(dict(enumerate(z["fold_probabilities"]))))
        records.append({"subject_id": subject["subject_id"], "record_key": subject["record_key"],
                        "path": str(dest), "sha256": sha256_file(dest), "E3_reference": subject,
                        "input_path": str(path), "input_sha256": entry["output_sha256"]})
        if i % 10 == 0 or i == 1:
            elapsed = time.perf_counter() - begun
            print(f"E4 seed42 ten-fold ensemble: {i}/180 subjects, {elapsed:.1f}s elapsed", flush=True)
    write_once_json(output / "private_inference_manifest.json", {"status": "complete", "records": records})
    left, right = [], []
    for item in records:
        with np.load(item["path"], allow_pickle=False) as pred, np.load(item["E3_reference"]["path"], allow_pickle=False) as ref:
            left.append(confusion_matrix_5(ref["y"], pred["probabilities"].argmax(1)))
            right.append(confusion_matrix_5(ref["y"], ref["probabilities"].argmax(1)))
    result = summarize_pair(left, right, "E4", "E3")
    if result["valid_epochs"] != 169012 or abs(result["arms"]["E3"]["subject_mean_macro_f1"] - old["metrics"]["E3"]["subject_macro_f1_mean"]) > 1e-12:
        raise ValueError("historical benchmark metrics do not replay")
    result.update(status="complete_additional_seed42_ten_fold_E4_E3_comparison", provenance=spec,
                  inference_manifest_sha256=sha256_file(output / "private_inference_manifest.json"))
    with (output / "private_subject_confusions.npz").open("xb") as stream:
        np.savez_compressed(stream, E4=np.stack(left), E3=np.stack(right))
    write_once_json(output / "aggregate_results.json", result)
    print(json.dumps({"status": result["status"], "means": {k: v["subject_mean_macro_f1"] for k,v in result["arms"].items()},
                      "contrast": result["paired_difference"]["subject_mean_macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
