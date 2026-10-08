"""Recompute the E4 extension results and replay all folds on boundary records."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.preprocessing import sha256_file
from sleeptcn.metrics import confusion_matrix_5
from sleeptcn.revision_campaign import load_restored_pair, infer_pair, mean_ten, summarize_pair, write_once_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--restored", required=True, type=Path)
    args = parser.parse_args()
    result_path = args.campaign / "aggregate_results.json"
    result = json.loads(result_path.read_bytes())
    if result["status"] != "complete_additional_seed42_ten_fold_E4_E3_comparison":
        raise ValueError("campaign incomplete")
    spec = result["provenance"]
    for relative, digest in spec["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest or sha256_file(args.campaign / "code_snapshot" / relative) != digest:
            raise ValueError("executed code changed")
    if sha256_file(args.restored / "restoration_manifest.json") != spec["restoration_manifest_sha256"]:
        raise ValueError("restoration manifest changed")
    manifest = args.campaign / "private_inference_manifest.json"
    if sha256_file(manifest) != result["inference_manifest_sha256"]:
        raise ValueError("inference manifest changed")
    records = json.loads(manifest.read_bytes())["records"]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    models = {f: load_restored_pair(args.restored, "E4", f, 42) for f in range(10)}
    left, right, replay = [], [], []
    for ordinal, entry in enumerate(records):
        for path, digest in [(entry["path"], entry["sha256"]), (entry["input_path"], entry["input_sha256"]),
                             (entry["E3_reference"]["path"], entry["E3_reference"]["sha256"])]:
            if sha256_file(Path(path)) != digest:
                raise ValueError("input/reference/prediction hash mismatch")
        with np.load(entry["path"], allow_pickle=False) as z, np.load(entry["E3_reference"]["path"], allow_pickle=False) as ref:
            meta = json.loads(str(z["metadata_json"]))
            if meta["subject_id"] != entry["subject_id"] or not np.array_equal(z["original_epoch_index"], ref["original_epoch_index"]):
                raise ValueError("subject or epoch mismatch")
            p = mean_ten(dict(enumerate(z["fold_probabilities"])))
            np.testing.assert_array_equal(p, z["probabilities"])
            left.append(confusion_matrix_5(ref["y"], p.argmax(1)))
            right.append(confusion_matrix_5(ref["y"], ref["probabilities"].argmax(1)))
            if ordinal in [0, len(records)-1]:
                with np.load(entry["input_path"], allow_pickle=False) as inputs:
                    x = inputs["x"]
                actual = {}
                for f, (enc, seq, hashes) in models.items():
                    if hashes != spec["checkpoint_sha256"][str(f)]:
                        raise ValueError("inference checkpoint differs")
                    actual[f] = infer_pair(enc, seq, x, spec["protocol"]["batch_size"])
                    error = float(np.max(np.abs(actual[f] - z["fold_probabilities"][f])))
                    if error > 3e-6 or np.any(actual[f].argmax(1) != z["fold_probabilities"][f].argmax(1)):
                        raise ValueError("fold replay changed decisions or exceeded tolerance")
                    replay.append({"record_ordinal": ordinal, "fold": f, "max_probability_residual": error})
                if np.any(mean_ten(actual).argmax(1) != p.argmax(1)):
                    raise ValueError("ensemble replay decisions differ")
    recomputed = summarize_pair(left, right, "E4", "E3")
    if any(recomputed[k] != result[k] for k in recomputed):
        raise ValueError("aggregate metrics/bootstrap do not replay")
    with np.load(args.campaign / "private_subject_confusions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(left, z["E4"])
        np.testing.assert_array_equal(right, z["E3"])
    document = {"status": "passed", "subjects_verified": len(records), "fold_prediction_matrices_verified": len(records)*10,
                "metrics_and_bootstrap_recomputed": True, "checkpoint_prediction_replay": replay,
                "aggregate_results_sha256": sha256_file(result_path),
                "verifier_sha256": sha256_file(Path(__file__))}
    write_once_json(args.campaign / "verification.json", document)
    print(json.dumps({"status": "passed", "subjects": len(records), "replayed_fold_records": len(replay)}, indent=2))


if __name__ == "__main__":
    main()
