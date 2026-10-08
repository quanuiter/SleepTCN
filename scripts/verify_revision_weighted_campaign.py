"""Independently recompute target confusions and bootstrap for the matched ensemble."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.metrics import confusion_matrix_5
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import summarize_pair, write_once_json
from sleeptcn.revision_weighted_campaign import audit_prediction, require_ten_folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--shhs-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    spec = json.loads((root/"execution_specification.json").read_bytes())
    result = json.loads((root/"aggregate_results.json").read_bytes())
    frozen = json.loads((root/"all_checkpoints_frozen.json").read_bytes())
    require_ten_folds(frozen["records"])
    if result["status"] != "complete_additional_ten_fold_weighted_loss" or result["provenance"] != spec:
        raise ValueError("require completed matching ensemble result")
    for relative, digest in spec["code_sha256"].items():
        if sha256_file(ROOT/relative) != digest or sha256_file(root/"code_snapshot"/relative) != digest:
            raise ValueError("executed code differs")
    frozen_hash = sha256_file(root/"all_checkpoints_frozen.json")
    spec_hash = sha256_file(root/"execution_specification.json")
    if frozen_hash != result["frozen_checkpoints_sha256"]:
        raise ValueError("frozen checkpoints manifest differs")
    for fold in frozen["records"]:
        for selected in fold["arms"].values():
            if sha256_file(Path(selected["checkpoint_path"])) != selected["checkpoint_sha256"]:
                raise ValueError("frozen checkpoint changed")
    manifest_path = root/"private_inference_manifest.json"
    if sha256_file(manifest_path) != result["inference_manifest_sha256"]:
        raise ValueError("target manifest changed")
    entries = json.loads(manifest_path.read_bytes())["records"]
    if len(entries) != 180 or len({e["subject_id"] for e in entries}) != 180:
        raise ValueError("require complete unique target coverage")
    loader = importlib.util.spec_from_file_location("weighted_independent_mask", ROOT/"scripts/run_recovered_e3_cpu_pilot.py")
    helper = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helper)
    matrices = {a: [] for a in ["unweighted", "weighted"]}
    for entry in entries:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("prediction changed")
        prediction = audit_prediction(entry["path"], entry, frozen_hash, spec_hash)
        reference = args.shhs_root/"processed_v1/filtered_v2"/(entry["record_key"]+".npz")
        with np.load(reference, allow_pickle=False) as z:
            if str(z["subject_id"]) != entry["subject_id"] or str(z["source_edf_sha256"]) != entry["source_edf_sha256"] or str(z["role"]) != "test":
                raise ValueError("reference identity differs")
            positions = helper.benchmark_positions(z["original_epoch_index"], entry["epochs"])
            for arm in matrices:
                matrices[arm].append(confusion_matrix_5(z["y"], prediction[arm][positions].argmax(1)))
    matrices = {a: np.stack(cm) for a,cm in matrices.items()}
    with np.load(root/"private_subject_confusions.npz", allow_pickle=False) as z:
        for arm, cm in matrices.items():
            np.testing.assert_array_equal(z[arm], cm)
    expected = summarize_pair(matrices["weighted"], matrices["unweighted"], "weighted", "unweighted")
    for key, value in expected.items():
        if result[key] != value:
            raise ValueError("target metrics or bootstrap did not reproduce")
    document = {"status": "passed", "source_selected_checkpoint_hashes_verified": 20,
        "target_prediction_files_verified": 180, "fold_arm_probabilities_audited": 3600,
        "all_subject_confusions_recomputed": True, "paired_bootstrap_recomputed": True,
        "aggregate_results_sha256": sha256_file(root/"aggregate_results.json"),
        "verifier_sha256": sha256_file(Path(__file__))}
    write_once_json(root/"independent_verification.json", document)
    print(json.dumps(document, indent=2), flush=True)


if __name__ == "__main__":
    main()
