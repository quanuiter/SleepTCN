"""CPU-only E4/E3 class-error bootstrap and source-only train-weight preparation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.cpu_followups import paired_diagnostic_bootstrap
from sleeptcn.shhs_analysis import load_ensemble_predictions, _subject_confusions
from sleeptcn.statistics import assert_paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run_manifest.read_bytes())
    if run.get("status") != "complete" or run.get("role") != "test":
        raise ValueError("expected a completed test manifest")
    manifests = args.run_manifest.parents[2] / "manifests"
    inventory_raw = (manifests / "zero_shot_checkpoint_inventory_e4_seed123_v1.json").read_bytes()
    protocol_raw = (manifests / "shhs_e4_seed123_extension_protocol_v1.json").read_bytes()
    inventory, protocol = json.loads(inventory_raw), json.loads(protocol_raw)
    if (hashlib.sha256(inventory_raw).hexdigest() != run["checkpoint_inventory_sha256"]
            or hashlib.sha256(protocol_raw).hexdigest() != run["protocol_sha256"]
            or inventory["protocol_sha256"] != run["protocol_sha256"]
            or inventory["seed"] != 123 or protocol["checkpoint_seed"] != 123):
        raise ValueError("seed-123 inventory/protocol linkage mismatch")
    a, b = [load_ensemble_predictions(run, exp).sorted() for exp in ["E4", "E3"]]
    assert_paired(a, b)
    subjects_a, cm_a = _subject_confusions(a)
    subjects_b, cm_b = _subject_confusions(b)
    if not np.array_equal(subjects_a, subjects_b):
        raise ValueError("paired subject order differs")
    diagnostic = paired_diagnostic_bootstrap(cm_a, cm_b)
    diagnostic["comparison"] = "E4-E3_seed123"
    diagnostic["subjects_with_reference_n3"] = int((cm_a[:, 3].sum(1) > 0).sum())
    diagnostic["subjects_without_reference_n3"] = int((cm_a[:, 3].sum(1) == 0).sum())
    diagnostic["p_values"] = "not_computed_unregistered_diagnostic_family"

    split_path = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    split = json.loads(split_path.read_bytes())
    counts_by_record = {}
    for record in (ROOT / "data/processed/filtered_v2").glob("*.npz"):
        with np.load(record, allow_pickle=False) as z:
            if str(z["preprocess_version"]) != "filtered_v2":
                raise ValueError("source preprocessing metadata mismatch")
            y = z["y"]
            if not np.isin(y, [-1, 0, 1, 2, 3, 4]).all():
                raise ValueError("unsupported source labels")
            key = str(z["record_key"])
            if key in counts_by_record:
                raise ValueError("duplicate source record")
            counts_by_record[key] = np.bincount(y[y >= 0], minlength=5)
    if len(counts_by_record) != split["summary"]["records"] or len(split["outer_runs"]) != 10:
        raise ValueError("source record/fold count mismatch")
    weights = []
    for fold in split["outer_runs"]:
        roles = [set(fold[role]["record_keys"]) for role in ["train", "validation", "test"]]
        if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
            raise ValueError("source roles overlap")
        subjects = [{key[:5] for key in keys} for keys in roles]
        if subjects[0] & subjects[1] or subjects[0] & subjects[2] or subjects[1] & subjects[2]:
            raise ValueError("source subjects overlap")
        for role, keys in zip(["train", "validation", "test"], roles):
            measured = np.sum([counts_by_record[key] for key in sorted(keys)], axis=0)
            expected = [fold[role]["label_counts"][str(k)] for k in range(5)]
            if not np.array_equal(measured, expected):
                raise ValueError("source class counts differ from locked split")
        counts = np.sum([counts_by_record[key] for key in sorted(roles[0])], axis=0)
        if np.any(counts == 0):
            raise ValueError("source training class absent")
        # Inverse-frequency source-only weights, without expanding epoch labels.
        values = counts.sum() / (5.0 * counts)
        weights.append({"outer_fold": fold["outer_fold"], "training_records": len(roles[0]),
                        "valid_train_epochs": int(counts.sum()), "class_counts": counts.tolist(),
                        "class_weights": values.tolist()})
    result = {"date": "2026-10-01", "status": "complete_cpu_analysis_not_new_inference",
              "class_order": ["W", "N1", "N2", "N3", "REM"], "diagnostic": diagnostic,
              "source_training_loss_preparation": {"policy": "N/(5*n_c)", "folds": weights,
                                                    "training_executed": False},
              "input_sha256": {"shhs_run_manifest": hashlib.sha256(args.run_manifest.read_bytes()).hexdigest(),
                               "source_split": hashlib.sha256(split_path.read_bytes()).hexdigest()},
              "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__).resolve(), ROOT / "src/sleeptcn/cpu_followups.py"]}}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(diagnostic, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
