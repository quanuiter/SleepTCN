"""Resume the remaining matched CPU folds, then freeze/evaluate the ten-fold pair.

Historical artifacts and the completed next-fold trial are reused without changes.
Target scoring starts only after all twenty source-selected checkpoints are frozen.
Each invocation is capped at five hours; no scientific settings are shortened.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.models import SleepTCN
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import extract_features, load_restored_pair, mean_ten, summarize_pair, write_once_json
from sleeptcn.revision_weighted import train_arm
from sleeptcn.revision_weighted_campaign import CampaignBudget, CampaignStop, audit_prediction, require_ten_folds
from run_revision_weighted_cpu_trial import prepare_cache, verify_pair

PROTOCOL = ROOT / "configs/teacher_revision_weighted_10fold_v1.json"
RECIPE_KEYS = ["seed", "learning_rate", "batch_size_records", "max_epochs", "patience", "gradient_clip_norm"]


def checked_result(folder, verification_name):
    result_path = folder / "aggregate_results.json"
    result = json.loads(result_path.read_bytes())
    verification = json.loads((folder / verification_name).read_bytes())
    if verification["status"] != "passed" or verification["aggregate_results_sha256"] != sha256_file(result_path):
        raise ValueError("reused result lacks matching successful verification")
    return result


def snapshot(output, paths):
    for path in paths:
        destination = output / "code_snapshot" / path.relative_to(ROOT)
        if destination.exists():
            if sha256_file(destination) != sha256_file(path):
                raise ValueError("executed snapshot changed")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def verify_source(folder, cache, restored, budget):
    budget()
    command = [sys.executable, str(ROOT / "scripts/verify_revision_weighted_cpu_trial.py"),
               "--trial", str(folder), "--cache", str(cache), "--restored", str(restored)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=budget.remaining())
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    checked_result(folder, "independent_verification.json")
    budget()


def verify_target(args, budget):
    budget.publish(phase="independent_target_verification")
    budget()
    command = [sys.executable, str(ROOT/"scripts/verify_revision_weighted_campaign.py"),
        "--campaign", str(args.output), "--shhs-root", str(args.shhs_root)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=budget.remaining())
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    checked_result(args.output, "independent_verification.json")
    budget()


def run_fold(fold, args, cfg, code_paths, budget):
    folder = args.output / "folds" / f"fold_{fold:02d}"
    cache = args.cache_root / f"fold_{fold:02d}"
    if (folder / "aggregate_results.json").exists():
        verify_source(folder, cache, args.restored, budget)
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    budget.publish(phase="source_cache", fold=fold, arm=None, epoch=0)
    tick = time.perf_counter()
    encoder, _, hashes = load_restored_pair(args.restored, "E3", fold, cfg["seed"])
    split = args.restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    spec = {"fold": fold, "seed": cfg["seed"], "protocol": cfg,
        "protocol_sha256": sha256_file(PROTOCOL), "split_sha256": sha256_file(split),
        "encoder_sha256": hashes[0], "restoration_sha256": sha256_file(args.restored / "restoration_manifest.json"),
        "campaign_specification_sha256": sha256_file(args.output / "execution_specification.json"),
        "resource_limits": {"cpu_threads": 4, "shared_attempt_max_seconds": args.max_seconds},
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
        "torch_version": str(torch.__version__), "target_inference_performed": False}
    write_once_json(folder / "execution_specification.json", spec)
    snapshot(folder, code_paths)
    roles, weights, manifest = prepare_cache(encoder, hashes[0], split, fold, cache, budget)
    selections = {}
    for arm, values in [("unweighted", np.ones(5)), ("weighted", weights)]:
        budget()
        budget.publish(phase="training", fold=fold, arm=arm, epoch=0)
        identity = {"fold": fold, "seed": cfg["seed"], "arm": arm,
            "protocol_sha256": spec["protocol_sha256"], "encoder_sha256": hashes[0],
            "cache_manifest_sha256": sha256_file(cache / "private_manifest.json"),
            "execution_specification_sha256": sha256_file(folder / "execution_specification.json")}
        _, selections[arm] = train_arm(roles["train"], roles["validation"], values, cfg,
            folder / arm, identity, runtime_guard=budget)
    if selections["unweighted"]["initial_state_sha256"] != selections["weighted"]["initial_state_sha256"]:
        raise ValueError("paired initialization differs")
    write_once_json(folder / "both_arms_selected.json", selections)
    budget.publish(phase="source_verification", fold=fold, arm=None)
    source, verification = verify_pair(folder, cache, roles, selections, cfg, encoder, manifest, budget)
    result = {"status": "complete_source_only_next_fold_cpu_trial", "fold": fold, "seed": cfg["seed"],
        "provenance": spec, "selections": selections, "source_results": source,
        "train_class_weights": weights.tolist(), "source_cache_seconds": budget.feature_seconds,
        "trial_elapsed_seconds": time.perf_counter() - tick,
        "timing_scope": "current_attempt_fold_work; training_selection_seconds_are_cumulative_if_resumed",
        "throughput_projection": None, "target_inference_performed": False}
    write_once_json(folder / "aggregate_results.json", result)
    verification["aggregate_results_sha256"] = sha256_file(folder / "aggregate_results.json")
    write_once_json(folder / "verification.json", verification)
    verify_source(folder, cache, args.restored, budget)
    print(f"Fold {fold} complete and independently verified", flush=True)
    return folder


def freeze_checkpoints(folders, args, cfg, budget):
    records, pooled = [], {arm: np.zeros((5, 5), dtype=np.int64) for arm in cfg["arms"]}
    splits = json.loads((args.restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json").read_bytes())["outer_runs"]
    test_subjects = []
    for fold, folder in enumerate(folders):
        budget()
        result = checked_result(folder, "verification.json" if fold == 0 else "independent_verification.json")
        encoder, _, hashes = load_restored_pair(args.restored, "E3", fold, cfg["seed"])
        del encoder
        source_cfg = result["provenance"]["protocol"]
        if any(source_cfg[k] != cfg[k] for k in RECIPE_KEYS):
            raise ValueError("reused training recipe differs")
        if fold == 0:
            original = json.loads((args.verified_pilot / "aggregate_results.json").read_bytes())
            if hashes[0] != original["provenance"]["checkpoint_sha256"]["extractor"]:
                raise ValueError("fold-zero encoder differs")
            source = json.loads((folder / "verification.json").read_bytes())["source_results"]
        else:
            if result["fold"] != fold or result["provenance"]["encoder_sha256"] != hashes[0]:
                raise ValueError("fold or encoder identity differs")
            source = result["source_results"]
        if result["selections"]["weighted"]["initial_state_sha256"] != result["selections"]["unweighted"]["initial_state_sha256"]:
            raise ValueError("paired initial states differ")
        arms = {}
        for arm in cfg["arms"]:
            selection = result["selections"][arm]
            path = folder / arm / "best.pt"
            if sha256_file(path) != selection["checkpoint_sha256"]:
                raise ValueError("selected checkpoint changed")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["epoch"] != selection["selected_epoch"]:
                raise ValueError("selected checkpoint epoch differs")
            if fold == 0:
                metric = source[arm]["test"]["pooled"]
                counts = np.array(result["provenance"]["source_train_class_counts"])
            else:
                metric = source[arm]["outer_test"]
                counts = np.array([splits[fold]["train"]["label_counts"][str(i)] for i in range(5)])
            expected = np.ones(5) if arm == "unweighted" else counts.sum() / (5 * counts)
            np.testing.assert_allclose(selection["loss_weights"], expected, rtol=1e-6)
            if metric["n_valid_epochs"] != splits[fold]["test"]["valid_epochs"]:
                raise ValueError("source test support differs")
            pooled[arm] += np.array(metric["confusion_matrix"])
            arms[arm] = {"checkpoint_path": str(path.resolve()), "checkpoint_sha256": sha256_file(path),
                "selected_epoch": selection["selected_epoch"],
                "selected_validation_macro_f1": selection["selected_validation_macro_f1"],
                "source_result_sha256": sha256_file(folder / "aggregate_results.json")}
        records.append({"fold": fold, "encoder_sha256": hashes[0], "arms": arms})
        test_subjects.extend(splits[fold]["test"]["subject_ids"])
    require_ten_folds(records)
    if len(test_subjects) != len(set(test_subjects)) or len(test_subjects) != 78 or any(cm.sum() != 195469 for cm in pooled.values()):
        raise ValueError("source outer test roles do not form complete disjoint coverage")
    frozen = {"status": "all_twenty_source_selected_checkpoints_frozen_before_new_target_inference",
              "records": records, "source_outer_test_pooled": {a: metrics_from_confusion(cm) for a,cm in pooled.items()}}
    write_once_json(args.output / "all_checkpoints_frozen.json", frozen)
    return frozen


def target_campaign(args, frozen, spec, budget):
    helper_spec = importlib.util.spec_from_file_location("weighted_campaign_signal_helpers", ROOT / "scripts/run_recovered_e3_cpu_pilot.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    manifest_path = args.verified_pilot / "private_inference_manifest.json"
    targets = json.loads(manifest_path.read_bytes())["records"]
    if len(targets) != 180 or len({e["subject_id"] for e in targets}) != 180:
        raise ValueError("require all 180 locked target subjects")
    frozen_hash = sha256_file(args.output / "all_checkpoints_frozen.json")
    spec_hash = sha256_file(args.output / "execution_specification.json")
    models = {}
    for record in frozen["records"]:
        budget()
        fold = record["fold"]
        encoder, _, hashes = load_restored_pair(args.restored, "E3", fold, 123)
        if hashes[0] != record["encoder_sha256"]:
            raise ValueError("frozen encoder changed")
        arms = {}
        for arm, selected in record["arms"].items():
            if sha256_file(Path(selected["checkpoint_path"])) != selected["checkpoint_sha256"]:
                raise ValueError("frozen sequence checkpoint changed")
            payload = torch.load(selected["checkpoint_path"], map_location="cpu", weights_only=True)
            model = SleepTCN(input_dim=128).eval()
            model.load_state_dict(payload["model_state"], strict=True)
            arms[arm] = model
        models[fold] = encoder, arms
    entries = []
    for ordinal, entry in enumerate(targets, 1):
        budget()
        budget.publish(phase="target_inference", fold=None, arm=None, target_records_completed=ordinal-1)
        path = args.output / "predictions" / (entry["record_key"] + ".npz")
        if not path.exists():
            edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
            x, _ = helper.read_full_record(edf, entry["source_edf_sha256"])
            if len(x) != entry["epochs"]:
                raise ValueError("full record support differs")
            parts = {a: {} for a in ["unweighted", "weighted"]}
            with torch.inference_mode():
                for fold, (encoder, arms) in models.items():
                    budget()
                    features = torch.from_numpy(extract_features(encoder, x, 64)).unsqueeze(0)
                    for arm, model in arms.items():
                        parts[arm][fold] = torch.softmax(model(features, padding_mask=None), -1).squeeze(0).numpy()
            metadata = {"subject_id": entry["subject_id"], "source_edf_sha256": entry["source_edf_sha256"],
                        "frozen_checkpoints_sha256": frozen_hash, "specification_sha256": spec_hash}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **{a: mean_ten(p) for a,p in parts.items()},
                    **{"fold_"+a: np.stack([p[f] for f in range(10)]) for a,p in parts.items()},
                    original_epoch_index=np.arange(len(x)), metadata_json=np.array(json.dumps(metadata)))
            temporary.replace(path)
        audit_prediction(path, entry, frozen_hash, spec_hash)
        entries.append({**entry, "path": str(path.resolve()), "sha256": sha256_file(path)})
        if ordinal % 10 == 0 or ordinal == 1:
            print(f"Weighted ten-fold target inference {ordinal}/180", flush=True)
    write_once_json(args.output / "private_inference_manifest.json", {"status": "complete", "records": entries})
    budget.publish(phase="target_evaluation", target_records_completed=180)
    matrices = {a: [] for a in ["unweighted", "weighted"]}
    for entry in entries:
        budget()
        prediction = audit_prediction(entry["path"], entry, frozen_hash, spec_hash)
        reference = args.shhs_root / "processed_v1/filtered_v2" / (entry["record_key"] + ".npz")
        with np.load(reference, allow_pickle=False) as z:
            if str(z["subject_id"]) != entry["subject_id"] or str(z["source_edf_sha256"]) != entry["source_edf_sha256"] or str(z["role"]) != "test":
                raise ValueError("target reference identity differs")
            positions = helper.benchmark_positions(z["original_epoch_index"], entry["epochs"])
            for arm in matrices:
                matrices[arm].append(confusion_matrix_5(z["y"], prediction[arm][positions].argmax(1)))
    matrices = {a: np.stack(cm) for a,cm in matrices.items()}
    result = summarize_pair(matrices["weighted"], matrices["unweighted"], "weighted", "unweighted")
    if result["valid_epochs"] != 169012:
        raise ValueError("target benchmark support differs")
    budget.publish(phase="target_replay_verification")
    replay_count = 0
    for entry in [entries[0], entries[-1]]:
        edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
        x, _ = helper.read_full_record(edf, entry["source_edf_sha256"])
        with np.load(entry["path"], allow_pickle=False) as saved, torch.inference_mode():
            for fold, (encoder, arms) in models.items():
                budget()
                features = torch.from_numpy(extract_features(encoder, x, 64)).unsqueeze(0)
                for arm, model in arms.items():
                    actual = torch.softmax(model(features, padding_mask=None), -1).squeeze(0).numpy()
                    np.testing.assert_array_equal(actual, saved["fold_"+arm][fold])
                    replay_count += 1
    for entry in entries:
        budget()
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("target prediction changed")
    cm_path = args.output / "private_subject_confusions.npz"
    if cm_path.exists():
        with np.load(cm_path, allow_pickle=False) as z:
            for arm, cm in matrices.items():
                np.testing.assert_array_equal(z[arm], cm)
    else:
        temporary = cm_path.with_suffix(".tmp.npz")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **matrices)
        temporary.replace(cm_path)
    result.update(status="complete_additional_ten_fold_weighted_loss", provenance=spec,
        frozen_checkpoints_sha256=frozen_hash,
        inference_manifest_sha256=sha256_file(args.output / "private_inference_manifest.json"),
        source_outer_test_pooled=frozen["source_outer_test_pooled"])
    write_once_json(args.output / "aggregate_results.json", result)
    write_once_json(args.output / "verification.json", {"status": "passed",
        "source_selected_checkpoint_count": 20, "independent_source_verification_complete": True,
        "target_prediction_files_audited": 180, "target_fold_arm_probabilities": 3600,
        "target_confusions_and_bootstrap_computed": True, "target_fold_arm_replays": replay_count,
        "target_labels_used_for_selection": False,
        "aggregate_results_sha256": sha256_file(args.output / "aggregate_results.json")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["restored", "fold0", "fold1", "verified-pilot", "shhs-root", "output", "cache-root"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=18000)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if not args.output.is_relative_to((ROOT / "runs").resolve()) or not args.cache_root.is_relative_to((ROOT / "data/cache").resolve()):
        raise ValueError("require private output and cache")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(PROTOCOL.read_bytes())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    budget = CampaignBudget(args.max_seconds, args.output / "progress.json")
    if (args.output / "aggregate_results.json").exists():
        checked_result(args.output, "verification.json")
        verify_target(args, budget)
        budget.publish(phase="complete", status="complete_additional_ten_fold_weighted_loss")
        print("Campaign already completed and independently verified; no retraining", flush=True)
        return 0
    paths = [Path(__file__).resolve(), PROTOCOL, ROOT / "scripts/run_revision_weighted_cpu_trial.py",
        ROOT / "scripts/verify_revision_weighted_cpu_trial.py", ROOT / "scripts/run_recovered_e3_cpu_pilot.py",
        ROOT / "scripts/verify_revision_weighted_campaign.py",
        *[ROOT / "src/sleeptcn" / (name+".py") for name in ["revision_weighted_campaign", "revision_weighted",
        "revision_campaign", "models", "training", "metrics", "demo", "calibration", "cpu_followups",
        "preprocessing", "shhs_preprocessing"]]]
    original = checked_result(args.verified_pilot, "verification.json")
    target_manifest = args.verified_pilot / "private_inference_manifest.json"
    if sha256_file(target_manifest) != original["inference_manifest_sha256"]:
        raise ValueError("locked target manifest changed")
    checked_result(args.fold0, "verification.json")
    checked_result(args.fold1, "independent_verification.json")
    split_path = args.restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    if original["provenance"]["source_split_sha256"] != sha256_file(split_path):
        raise ValueError("reused pilot split differs")
    for fold, folder in enumerate([args.fold0, args.fold1]):
        reused = json.loads((folder/"aggregate_results.json").read_bytes())
        if any(reused["provenance"]["protocol"][k] != cfg[k] for k in RECIPE_KEYS):
            raise ValueError("reused optimization recipe differs")
        encoder, _, hashes = load_restored_pair(args.restored, "E3", fold, cfg["seed"])
        del encoder
        expected_encoder = original["provenance"]["checkpoint_sha256"]["extractor"] if fold == 0 else reused["provenance"]["encoder_sha256"]
        if hashes[0] != expected_encoder:
            raise ValueError("reused encoder differs")
        for arm, selected in reused["selections"].items():
            if sha256_file(folder/arm/"best.pt") != selected["checkpoint_sha256"]:
                raise ValueError("reused sequence checkpoint differs")
    spec = {"protocol": cfg, "protocol_sha256": sha256_file(PROTOCOL),
        "restoration_sha256": sha256_file(args.restored / "restoration_manifest.json"),
        "split_sha256": sha256_file(args.restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json"),
        "target_manifest_sha256": sha256_file(target_manifest),
        "reused_results_sha256": {str(i): sha256_file(p/"aggregate_results.json") for i,p in enumerate([args.fold0,args.fold1])},
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
        "resource_limits": {"cpu_threads": 4, "attempt_max_seconds": args.max_seconds},
        "torch_version": str(torch.__version__)}
    write_once_json(args.output / "execution_specification.json", spec)
    snapshot(args.output, paths)
    try:
        budget.publish(status="running", completed_source_folds=2)
        folders = [args.fold0, args.fold1]
        for fold in range(2, 10):
            folders.append(run_fold(fold, args, cfg, paths, budget))
            budget.publish(completed_source_folds=fold+1)
        budget.publish(phase="freezing_all_checkpoints", completed_source_folds=10)
        frozen = freeze_checkpoints(folders, args, cfg, budget)
        result = target_campaign(args, frozen, spec, budget)
        verify_target(args, budget)
        budget.publish(phase="complete", status=result["status"])
        print(json.dumps({"status": result["status"], "subject_mean_macro_f1":
            {a:v["subject_mean_macro_f1"] for a,v in result["arms"].items()}}, indent=2), flush=True)
        return 0
    except (CampaignStop, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        budget.publish(status="stopped_resource_budget_or_interruption", reason=str(error),
            completed_epoch_checkpoints_retained=True)
        ordinal = len(list(args.output.glob("stop_attempt_*.json"))) + 1
        write_once_json(args.output / f"stop_attempt_{ordinal:02d}.json", json.loads((args.output/"progress.json").read_bytes()))
        print(f"Campaign stopped: {error}", flush=True)
        return 2
    except Exception as error:
        budget.publish(status="failed", reason=f"{type(error).__name__}: {error}", completed_epoch_checkpoints_retained=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
