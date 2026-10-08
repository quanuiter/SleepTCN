"""Independently verify a completed next-fold CPU pair using only source data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.calibration import source_class_weights
from sleeptcn.demo import load_processed_demo_record
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.models import SleepTCN
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import load_restored_pair, extract_features, write_once_json
from sleeptcn.revision_weighted import state_digest
from sleeptcn.training import collate_feature_sequences


@torch.inference_mode()
def score_source(model, records, batch_size):
    matrix = np.zeros((5, 5), dtype=np.int64)
    for start in range(0, len(records), batch_size):
        batch = collate_feature_sequences(records[start:start+batch_size])
        predictions = model(batch.features, padding_mask=batch.padding_mask).argmax(-1).numpy()
        matrix += confusion_matrix_5(batch.targets.numpy().ravel(), predictions.ravel())
    return metrics_from_confusion(matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--restored", required=True, type=Path)
    args = parser.parse_args()
    result_path = args.trial / "aggregate_results.json"
    result = json.loads(result_path.read_bytes())
    if result["status"] != "complete_source_only_next_fold_cpu_trial" or result["target_inference_performed"]:
        raise ValueError("require completed source-only trial")
    spec = result["provenance"]
    if spec != json.loads((args.trial / "execution_specification.json").read_bytes()):
        raise ValueError("execution specification differs")
    for relative, digest in spec["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest or sha256_file(args.trial / "code_snapshot" / relative) != digest:
            raise ValueError("executed code changed")
    split_path = args.restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    if sha256_file(split_path) != spec["split_sha256"]:
        raise ValueError("split changed")
    split = json.loads(split_path.read_bytes())["outer_runs"][result["fold"]]
    cache = json.loads((args.cache / "private_manifest.json").read_bytes())
    roles = {}
    for role in ["train", "validation", "test"]:
        entries = [e for e in cache["records"] if e["role"] == role]
        if {e["record_key"] for e in entries} != set(split[role]["record_keys"]):
            raise ValueError("source role membership differs")
        records = []
        for entry in entries:
            source = ROOT / "data/processed/filtered_v2" / (entry["record_key"] + ".npz")
            if sha256_file(source) != entry["input_sha256"] or sha256_file(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("source/cache hash differs")
            record = load_processed_demo_record(source)
            with np.load(entry["path"], allow_pickle=False) as z:
                np.testing.assert_array_equal(z["labels"], record.labels)
                np.testing.assert_array_equal(z["original_epoch_index"], record.original_epoch_index)
                metadata = json.loads(str(z["metadata_json"]))
                if metadata["encoder_sha256"] != spec["encoder_sha256"] or metadata["fold"] != result["fold"] or metadata["role"] != role:
                    raise ValueError("source/cache provenance differs")
                records.append((torch.from_numpy(z["features"].copy()), torch.from_numpy(z["labels"].astype(np.int64))))
        if sum(int((y >= 0).sum()) for _,y in records) != split[role]["valid_epochs"]:
            raise ValueError("source role support differs")
        roles[role] = records
    weights = source_class_weights(np.concatenate([y.numpy() for _,y in roles["train"]]))
    np.testing.assert_allclose(weights, result["train_class_weights"], rtol=0, atol=1e-12)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(result["seed"])
    initial = state_digest(SleepTCN(input_dim=128))
    orders = {}
    replay_metrics = {}
    for arm in ["unweighted", "weighted"]:
        selected = result["selections"][arm]
        if selected != json.loads((args.trial / arm / "selection.json").read_bytes()):
            raise ValueError("selection manifests disagree")
        path = args.trial / arm / "best.pt"
        if sha256_file(path) != selected["checkpoint_sha256"] or selected["initial_state_sha256"] != initial:
            raise ValueError("checkpoint/initialization identity differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["epoch"] != selected["selected_epoch"] or payload["identity"]["fold"] != result["fold"] or payload["identity"]["arm"] != arm:
            raise ValueError("selected payload provenance differs")
        expected_weights = weights if arm == "weighted" else np.ones(5)
        np.testing.assert_allclose(selected["loss_weights"], expected_weights, rtol=1e-6)
        history = selected["history"]
        best = max(history, key=lambda e: e["validation_macro_f1"])
        if best["epoch"] != selected["selected_epoch"] or len(history) != selected["epochs_completed"]:
            raise ValueError("source-only selection/epoch accounting differs")
        orders[arm] = [row["batch_order_sha256"] for row in history]
        model = SleepTCN(input_dim=128).eval()
        model.load_state_dict(payload["model_state"], strict=True)
        replay_metrics[arm] = {name: score_source(model, roles[role], spec["protocol"]["batch_size_records"])
            for name, role in [("validation", "validation"), ("outer_test", "test")]}
        if replay_metrics[arm] != result["source_results"][arm]:
            raise ValueError("source confusion/metrics did not replay")
        if replay_metrics[arm]["validation"]["macro_f1"] != selected["selected_validation_macro_f1"]:
            raise ValueError("selection metric differs")
    common = min(map(len, orders.values()))
    if orders["weighted"][:common] != orders["unweighted"][:common]:
        raise ValueError("source batch orders differ")
    encoder, _, hashes = load_restored_pair(args.restored, "E3", result["fold"], result["seed"])
    if hashes[0] != spec["encoder_sha256"]:
        raise ValueError("encoder differs")
    replays = []
    for role in ["train", "validation", "test"]:
        entry = next(e for e in reversed(cache["records"]) if e["role"] == role)
        record = load_processed_demo_record(ROOT / "data/processed/filtered_v2" / (entry["record_key"] + ".npz"))
        actual = extract_features(encoder, record.x, 64)
        with np.load(entry["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(actual, z["features"])
        replays.append({"role": role, "max_feature_residual": 0.})
    document = {"status": "passed", "source_cache_records_verified": len(cache["records"]),
        "source_role_membership_and_support_verified": True, "class_weights_from_training_only": True,
        "initialization_and_matched_batch_order_verified": True,
        "strict_best_source_validation_selection_verified": True,
        "validation_and_outer_test_confusions_recomputed": True,
        "feature_replays_at_opposite_cache_boundaries": replays,
        "target_inference_performed": False, "aggregate_results_sha256": sha256_file(result_path),
        "verifier_sha256": sha256_file(Path(__file__))}
    write_once_json(args.trial / "independent_verification.json", document)
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
