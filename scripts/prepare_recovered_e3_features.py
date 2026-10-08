"""Cache all source-fold features and benchmark CPU TCN training cost.

Uses the hash-verified recovered full E3 encoder; writes no trained checkpoint.
The timing pass uses training labels only and is not an intervention experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.calibration import source_class_weights
from sleeptcn.demo import load_demo_models, load_processed_demo_record, validate_asset_manifest
from sleeptcn.models import SleepTCN
from sleeptcn.preprocessing import sha256_file
from sleeptcn.training import collate_feature_sequences, masked_cross_entropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "data/cache").resolve()) or output.exists():
        raise ValueError("require a new private data/cache subdirectory")
    pilot = json.loads((args.verified_pilot / "execution_specification.json").read_bytes())
    split_path = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    if sha256_file(split_path) != pilot["source_split_sha256"]:
        raise ValueError("split differs from recovered checkpoint audit")
    assets = ROOT / "demo/assets"
    manifest = validate_asset_manifest(assets)
    encoder_entry = manifest["experiments"]["E3"]["extractor"]
    if sha256_file(assets / encoder_entry["path"]) != pilot["checkpoint_sha256"]["extractor"]:
        raise ValueError("encoder differs from recovered checkpoint audit")
    models = load_demo_models(assets, "E3", validated_manifest=manifest)
    torch.set_num_threads(4)
    split = json.loads(split_path.read_bytes())["outer_runs"][0]
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    records, counts, role_epochs = [], np.zeros(5, dtype=np.int64), {}
    for role in ["train", "validation", "test"]:
        folder = output / role
        folder.mkdir()
        role_epochs[role] = 0
        for i, key in enumerate(sorted(split[role]["record_keys"]), 1):
            path = ROOT / "data/processed/filtered_v2" / (key + ".npz")
            record = load_processed_demo_record(path)
            parts = []
            with torch.inference_mode():
                for start in range(0, len(record.x), 64):
                    parts.append(models.extractor.extract_features(
                        torch.from_numpy(record.x[start:start+64]).unsqueeze(1)).numpy())
            features = np.concatenate(parts).astype(np.float32, copy=False)
            if features.shape != (len(record.x), 128) or not np.isfinite(features).all():
                raise ValueError("invalid recovered-encoder features")
            valid = record.labels >= 0
            if role == "train":
                counts += np.bincount(record.labels[valid], minlength=5)
            metadata = {"record_key": key, "subject_id": key[:5], "role": role,
                        "outer_fold": 0, "encoder_seed": 123, "data_variant": "filtered_v2",
                        "encoder_sha256": encoder_entry["sha256"],
                        "split_sha256": sha256_file(split_path), "input_sha256": sha256_file(path)}
            dest = folder / (key + ".npz")
            with dest.open("xb") as stream:
                np.savez_compressed(stream, features=features, labels=record.labels,
                                    original_epoch_index=record.original_epoch_index,
                                    metadata_json=np.array(json.dumps(metadata)))
            with np.load(dest, allow_pickle=False) as z:
                if not np.array_equal(z["features"], features) or not np.array_equal(z["labels"], record.labels):
                    raise ValueError("feature round-trip mismatch")
            records.append({**metadata, "path": str(dest), "sha256": sha256_file(dest),
                            "epochs": len(features), "valid_epochs": int(valid.sum())})
            role_epochs[role] += int(valid.sum())
            if i % 20 == 0 or i == len(split[role]["record_keys"]):
                print(f"verified source feature cache {role}: {i}/{len(split[role]['record_keys'])}", flush=True)
        if role_epochs[role] != split[role]["valid_epochs"]:
            raise ValueError("cached role support differs from split")
    if not np.array_equal(counts, [split["train"]["label_counts"][str(i)] for i in range(5)]):
        raise ValueError("train-only class counts differ from split")
    weights = counts.sum() / (5. * counts)
    extraction_seconds = time.perf_counter() - started

    # A single training-only epoch measures throughput without selecting a model.
    train = []
    for item in records:
        if item["role"] == "train":
            with np.load(item["path"], allow_pickle=False) as z:
                train.append((torch.from_numpy(z["features"]),
                              torch.from_numpy(z["labels"].astype(np.int64))))
    train_weights = source_class_weights(np.concatenate([pair[1].numpy() for pair in train]))
    np.testing.assert_allclose(train_weights, weights)
    torch.manual_seed(123)
    model = SleepTCN(input_dim=128).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    generator = torch.Generator().manual_seed(123)
    order = torch.randperm(len(train), generator=generator).tolist()
    began, batch_seconds = time.perf_counter(), []
    for batch_index, start in enumerate(range(0, len(order), 8), 1):
        batch = collate_feature_sequences([train[k] for k in order[start:start+8]])
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.features, padding_mask=batch.padding_mask)
        loss = masked_cross_entropy(logits, batch.targets, torch.tensor(weights, dtype=torch.float32))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        batch_seconds.append(time.perf_counter() - tick)
        if batch_index % 4 == 0:
            print(f"CPU training timing {batch_index}/{(len(order)+7)//8} batches", flush=True)
    training_seconds = time.perf_counter() - began
    summary = {"status": "complete_features_and_timing_only_no_trained_model_selected",
               "experiment": "E3", "outer_fold": 0, "encoder_seed": 123,
               "source_records": len(records), "valid_epochs_by_role": role_epochs,
               "train_class_counts": counts.tolist(), "train_class_weights": weights.tolist(),
               "feature_dimension": 128, "encoder_sha256": encoder_entry["sha256"],
               "source_split_sha256": sha256_file(split_path),
               "feature_extraction_seconds": extraction_seconds,
               "cpu_training_epoch_seconds": training_seconds, "timed_batches": len(batch_seconds),
               "cpu_threads": 4, "batch_size_records": 8,
               "max_300_epochs_two_arms_training_only_hours_estimate": training_seconds * 600 / 3600,
               "timing_limitations": "One train epoch only; estimate excludes validation, checkpoint I/O, and early stopping.",
               "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__).resolve(),
                   ROOT / "src/sleeptcn/models.py", ROOT / "src/sleeptcn/training.py", ROOT / "src/sleeptcn/demo.py"]}}
    for name, document in [("private_manifest.json", {"summary": summary, "records": records}),
                            ("aggregate_summary.json", summary)]:
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
