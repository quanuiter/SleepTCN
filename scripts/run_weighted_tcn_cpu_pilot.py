"""Matched unweighted/weighted TCN retraining with a recovered frozen encoder.

All selection uses source validation. Both arms finish selection before SHHS
scoring. Outputs are an explicitly post-hoc single-fold pilot, not a new 10-fold
campaign. The historical runner and manifests are not modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.cpu_followups import confusion_diagnostics, paired_diagnostic_bootstrap
from sleeptcn.demo import load_demo_models, validate_asset_manifest
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.models import SleepTCN
from sleeptcn.preprocessing import sha256_file
from sleeptcn.training import collate_feature_sequences, masked_cross_entropy

PROTOCOL = ROOT / "configs/teacher_revision_weighted_fold0_pilot_v1.json"


def write_json(path, document):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_source(model, records, batch_size):
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    for start in range(0, len(records), batch_size):
        batch = collate_feature_sequences(records[start:start+batch_size])
        logits = model(batch.features, padding_mask=batch.padding_mask)
        cm += confusion_matrix_5(batch.targets.numpy().ravel(), logits.argmax(-1).numpy().ravel())
    return metrics_from_confusion(cm)


def train_arm(train, validation, weights, cfg, folder):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = SleepTCN(input_dim=128)
    initial_hash = state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    weights = torch.tensor(weights, dtype=torch.float32)
    best, stale, best_epoch, history = -float("inf"), 0, 0, []
    folder.mkdir(exist_ok=False)
    started = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        tick = time.perf_counter()
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), cfg["batch_size_records"]):
            batch = collate_feature_sequences([train[i] for i in order[start:start+cfg["batch_size_records"]]])
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.features, padding_mask=batch.padding_mask)
            loss = masked_cross_entropy(logits, batch.targets, weights)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite source training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_source(model, validation, cfg["batch_size_records"])
        score = metrics["macro_f1"]
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_state, "epoch": epoch, "validation_metrics": metrics,
                        "loss_weights": weights.tolist(), "initial_state_sha256": initial_hash,
                        "selection": "source_validation_macro_f1", "protocol_sha256": sha256_file(PROTOCOL)},
                       folder / "best.pt")
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss_batch_mean": float(np.mean(losses)),
               "validation_macro_f1": score, "best_epoch": best_epoch, "stale_epochs": stale,
               "seconds": time.perf_counter() - tick}
        history.append(row)
        with (folder / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        # Save a recoverable epoch boundary in this newly created run directory.
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch, "rng_state": torch.get_rng_state(),
                    "loader_generator_state": generator.get_state(), "best_epoch": best_epoch,
                    "best_score": best, "stale_epochs": stale}, folder / "latest.tmp")
        (folder / "latest.tmp").replace(folder / "latest.pt")
        print(f"{folder.name} epoch {epoch}: val_macro_f1={score:.5f}, best={best:.5f}, stale={stale}/{cfg['patience']}, {row['seconds']:.1f}s", flush=True)
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    result = {"epochs_completed": len(history), "selected_epoch": best_epoch,
              "selected_validation_macro_f1": best, "initial_state_sha256": initial_hash,
              "checkpoint_sha256": sha256_file(folder / "best.pt"),
              "training_and_validation_seconds": time.perf_counter() - started,
              "loss_weights": weights.tolist()}
    write_json(folder / "selection.json", result)
    return model, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--verified-pilot", required=True, type=Path)
    parser.add_argument("--shhs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("require new ignored runs subdirectory")
    cfg = json.loads(PROTOCOL.read_bytes())
    cache_manifest = args.source_cache / "private_manifest.json"
    cache = json.loads(cache_manifest.read_bytes())
    pilot_result_path = args.verified_pilot / "aggregate_results.json"
    pilot = json.loads(pilot_result_path.read_bytes())
    if (cache["summary"]["encoder_sha256"] != pilot["provenance"]["checkpoint_sha256"]["extractor"]
            or cache["summary"]["source_split_sha256"] != pilot["provenance"]["source_split_sha256"]):
        raise ValueError("source cache and recovered encoder/split differ")
    split_path = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    if sha256_file(split_path) != cache["summary"]["source_split_sha256"]:
        raise ValueError("source split changed")
    split = json.loads(split_path.read_bytes())["outer_runs"][0]
    roles, role_entries = {}, {}
    for role in ["train", "validation"]:
        entries = [entry for entry in cache["records"] if entry["role"] == role]
        if {e["record_key"] for e in entries} != set(split[role]["record_keys"]):
            raise ValueError("source cache role differs from locked split")
        records = []
        for entry in entries:
            if sha256_file(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("source cache hash mismatch")
            with np.load(entry["path"], allow_pickle=False) as z:
                records.append((torch.from_numpy(z["features"]), torch.from_numpy(z["labels"].astype(np.int64))))
        roles[role], role_entries[role] = records, entries
    train_subjects = {e["subject_id"] for e in role_entries["train"]}
    if train_subjects & {e["subject_id"] for e in role_entries["validation"]}:
        raise ValueError("source role leakage")
    labels = np.concatenate([y.numpy() for _, y in roles["train"]])
    counts = np.bincount(labels[labels >= 0], minlength=5)
    if not np.array_equal(counts, cache["summary"]["train_class_counts"]):
        raise ValueError("source training class counts differ")
    weights = counts.sum() / (5. * counts)
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    # Preserve exact code/config used for this new pair without changing old git snapshots.
    code_paths = [Path(__file__).resolve(), PROTOCOL, ROOT / "src/sleeptcn/models.py",
                  ROOT / "src/sleeptcn/training.py", ROOT / "src/sleeptcn/metrics.py",
                  ROOT / "src/sleeptcn/cpu_followups.py", ROOT / "scripts/run_recovered_e3_cpu_pilot.py"]
    for path in code_paths:
        dest = output / "code_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    provenance = {"protocol": cfg, "protocol_sha256": sha256_file(PROTOCOL),
                  "source_cache_manifest_sha256": sha256_file(cache_manifest),
                  "recovered_pilot_sha256": sha256_file(pilot_result_path),
                  "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
                  "source_train_class_counts": counts.tolist(), "cpu_threads": 4,
                  "torch_version": torch.__version__, "numpy_version": np.__version__}
    write_json(output / "execution_specification.json", provenance)
    models, selections = {}, {}
    for arm, loss_weights in [("unweighted", np.ones(5)), ("weighted", weights)]:
        models[arm], selections[arm] = train_arm(roles["train"], roles["validation"], loss_weights, cfg, output / arm)
    if selections["unweighted"]["initial_state_sha256"] != selections["weighted"]["initial_state_sha256"]:
        raise ValueError("paired initial weights differ")
    write_json(output / "both_arms_selected.json", selections)
    del roles

    # Extract each target full record once and apply both selected TCNs.
    helper_spec = importlib.util.spec_from_file_location("recovered_helpers", ROOT / "scripts/run_recovered_e3_cpu_pilot.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    assets = ROOT / "demo/assets"
    asset_manifest = validate_asset_manifest(assets)
    encoder = load_demo_models(assets, "E3", validated_manifest=asset_manifest).extractor
    if asset_manifest["experiments"]["E3"]["extractor"]["sha256"] != cache["summary"]["encoder_sha256"]:
        raise ValueError("target encoder differs from training cache")
    original = json.loads((args.verified_pilot / "private_inference_manifest.json").read_bytes())
    if sha256_file(args.verified_pilot / "private_inference_manifest.json") != pilot["inference_manifest_sha256"]:
        raise ValueError("target selection manifest changed")
    pred_folder = output / "target_probabilities"
    pred_folder.mkdir()
    targets = []
    for i, entry in enumerate(original["records"], 1):
        edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
        x, meta = helper.read_full_record(edf, entry["source_edf_sha256"])
        parts = []
        with torch.inference_mode():
            for start in range(0, len(x), 64):
                parts.append(encoder.extract_features(torch.from_numpy(x[start:start+64]).unsqueeze(1)))
            features = torch.cat(parts).unsqueeze(0)
            predictions = {arm: torch.softmax(model(features, padding_mask=None), -1).squeeze(0).numpy()
                           for arm, model in models.items()}
        path = pred_folder / (entry["record_key"] + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **predictions, original_epoch_index=np.arange(len(x)),
                                metadata_json=np.array(json.dumps({"subject_id": entry["subject_id"],
                                    "preprocessing": meta, "selected_checkpoints": selections})))
        targets.append({**entry, "path": str(path), "sha256": sha256_file(path), "epochs": len(x)})
        if i % 20 == 0 or i == len(original["records"]):
            print(f"selected weighted pair: full-record target inference {i}/180", flush=True)
    write_json(output / "private_inference_manifest.json", {"status": "complete", "records": targets})
    confusions = {arm: [] for arm in models}
    for entry in targets:
        path = args.shhs_root / "processed_v1/filtered_v2" / (entry["record_key"] + ".npz")
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("weighted-pair probability hash mismatch")
        with np.load(path, allow_pickle=False) as z, np.load(entry["path"], allow_pickle=False) as pred:
            if (str(z["subject_id"]) != entry["subject_id"] or str(z["role"]) != "test"
                    or str(z["source_edf_sha256"]) != entry["source_edf_sha256"]):
                raise ValueError("target reference identity mismatch")
            positions = helper.benchmark_positions(z["original_epoch_index"], entry["epochs"])
            for arm in models:
                confusions[arm].append(confusion_matrix_5(z["y"], pred[arm][positions].argmax(1)))
    cm = {arm: np.stack(values) for arm, values in confusions.items()}
    contrast = paired_diagnostic_bootstrap(cm["weighted"], cm["unweighted"])
    difference = confusion_diagnostics(cm["weighted"])["macro_f1"] - confusion_diagnostics(cm["unweighted"])["macro_f1"]
    rng = np.random.default_rng(2031)
    draws = rng.integers(0, len(difference), (10000, len(difference)), dtype=np.int32)
    contrast["subject_mean_macro_f1"] = {"difference": float(difference.mean()),
                                       "ci95": np.quantile(difference[draws].mean(1), [.025, .975]).tolist()}
    if cm["unweighted"].sum() != 169012 or len(cm["unweighted"]) != 180:
        raise ValueError("weighted evaluation benchmark support mismatch")
    results = {"status": "complete_exploratory_single_fold_weighted_loss_pilot", "provenance": provenance,
               "selections": selections, "subjects": 180, "valid_epochs": 169012,
               "arms": {arm: {"subject_mean_macro_f1": float(confusion_diagnostics(matrix)["macro_f1"].mean()),
                               "pooled": metrics_from_confusion(matrix.sum(0))} for arm, matrix in cm.items()},
               "weighted_minus_unweighted": contrast}
    with (output / "private_subject_confusions.npz").open("xb") as stream:
        np.savez_compressed(stream, **cm)
    write_json(output / "aggregate_results.json", results)
    print(json.dumps({"status": results["status"], "arms": results["arms"],
                      "subject_mean_contrast": contrast["subject_mean_macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
