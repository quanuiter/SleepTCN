"""Reconstruct source exposure and unlabelled pseudo-label counts from a verified ADAST run."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import write_once_json


def main():
    base = ROOT / "runs/teacher_revision_cpu_20261001/adast_fold0"
    result_path = base / "aggregate_results.json"
    result = json.loads(result_path.read_bytes())
    verified = json.loads((base / "verification.json").read_bytes())
    if verified["status"] != "passed" or verified["aggregate_results_sha256"] != sha256_file(result_path):
        raise ValueError("ADAST run must be verified")
    cfg = result["provenance"]["protocol"]
    y = np.load(base / "source_train_y.npy", mmap_mode="r", allow_pickle=False)
    selected = []
    for epoch, row in enumerate(result["selections"]["adast"]["history"]):
        order = np.random.default_rng(cfg["seed"] + epoch).permutation(len(y))[:cfg["steps_per_epoch"]*cfg["batch_size"]]
        import hashlib
        if hashlib.sha256(order.tobytes()).hexdigest() != row["source_order_sha256"]:
            raise ValueError("source sample order did not replay")
        selected.append(order)
    if [r["source_order_sha256"] for r in result["selections"]["adast"]["history"]] != [
        r["source_order_sha256"] for r in result["selections"]["source_only"]["history"]]:
        raise ValueError("matched source sample orders differ")
    seen = np.unique(np.concatenate(selected))
    pseudo = {}
    for round_index in range(cfg["rounds"]):
        path = base / "adast" / f"pseudo_round_{round_index}.npy"
        labels = np.load(path, allow_pickle=False)
        if labels.shape != (4989,) or not np.isin(labels, range(5)).all():
            raise ValueError("unexpected pseudo-label shape/classes")
        pseudo[str(round_index)] = {"counts": np.bincount(labels, minlength=5).tolist(),
                                   "sha256": sha256_file(path),
                                   "target_supervision_weight": cfg["target_loss_weights_by_round"][round_index]}
    document = {"status": "source_order_replayed_no_retraining", "class_order": ["W", "N1", "N2", "N3", "REM"],
        "source_train_epochs": len(y), "source_presentations_total": sum(map(len, selected)),
        "unique_source_epochs_seen": len(seen), "unique_source_fraction": len(seen)/len(y),
        "full_source_equivalent_passes": sum(map(len, selected))/len(y),
        "updates_per_arm": result["selections"]["adast"]["updates"], "epoch_source_batches": cfg["steps_per_epoch"],
        "labelled_source_class_counts": np.bincount(y, minlength=5).tolist(),
        "seen_source_class_counts": np.bincount(y[seen], minlength=5).tolist(),
        "pseudo_labels": pseudo, "true_target_labels_used": False,
        "source_loss_weights_by_round": cfg["source_loss_weights_by_round"],
        "adversarial_weight": cfg["adversarial_weight"],
        "aggregate_results_sha256": sha256_file(result_path), "analyzer_sha256": sha256_file(Path(__file__)),
        "interpretation": "Budget and pseudo-label diagnostics do not isolate causes of performance loss; round-0 pseudo labels have zero supervision weight."}
    write_once_json(ROOT / "runs/teacher_revision_cpu_20261003/adast_budget_diagnostic.json", document)
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
