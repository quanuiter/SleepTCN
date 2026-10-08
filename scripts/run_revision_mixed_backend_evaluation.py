"""Freeze verified CPU/CUDA source pairs, then evaluate locally without training."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import load_restored_pair, write_once_json
from sleeptcn.revision_weighted_campaign import CampaignBudget, CampaignStop, require_ten_folds
from sleeptcn.metrics import metrics_from_confusion
from run_revision_weighted_campaign import checked_result, snapshot, target_campaign, verify_target, RECIPE_KEYS


def main(args):
    restored = ROOT / "runs/teacher_revision_cpu_20261002/restored_checkpoints"
    cpu_campaign = ROOT / "runs/teacher_revision_cpu_20261003/weighted_10fold_campaign"
    gpu_campaign = ROOT / "runs/colab_training_20261004"
    folders = [ROOT / "runs/teacher_revision_cpu_20261001/weighted_e3_fold0",
               ROOT / "runs/teacher_revision_cpu_20261003/weighted_fold01_trial"]
    folders += [cpu_campaign / f"folds/fold_{fold:02d}" for fold in range(2, 6)]
    folders += [gpu_campaign / f"fold_{fold:02d}/gpu_results" for fold in range(6, 10)]
    protocol = ROOT / "configs/teacher_revision_weighted_10fold_v1.json"
    cfg = json.loads(protocol.read_bytes())
    splits_path = restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    splits = json.loads(splits_path.read_bytes())["outer_runs"]
    records, evidence = [], []
    pooled = {arm: np.zeros((5, 5), dtype=np.int64) for arm in cfg["arms"]}
    all_subjects = []
    for fold, folder in enumerate(folders):
        result = checked_result(folder, "verification.json" if fold == 0 else "independent_verification.json")
        if fold < 6:
            recipe = result["provenance"]["protocol"]
            if any(recipe[k] != cfg[k] for k in RECIPE_KEYS):
                raise ValueError("Reused CPU recipe differs")
        else:
            spec = json.loads((folder / "execution_specification.json").read_bytes())
            if spec["protocol"] != cfg or spec["target_access"]:
                raise ValueError("CUDA recipe/scope differs")
        encoder, _, hashes = load_restored_pair(restored, "E3", fold, cfg["seed"])
        del encoder
        if fold > 0 and result["provenance"]["encoder_sha256"] != hashes[0]:
            raise ValueError("Encoder provenance differs")
        if fold == 0:
            recovered = json.loads((ROOT / "runs/teacher_revision_cpu_20261001/recovered_e3_fold0/aggregate_results.json").read_bytes())
            if recovered["provenance"]["checkpoint_sha256"]["extractor"] != hashes[0]:
                raise ValueError("Fold zero encoder differs")
            source = json.loads((folder / "verification.json").read_bytes())["source_results"]
        else:
            source = result["source_results"]
        selections = result["selections"]
        if selections["weighted"]["initial_state_sha256"] != selections["unweighted"]["initial_state_sha256"]:
            raise ValueError("Paired initialization differs")
        arms = {}
        counts = np.array([splits[fold]["train"]["label_counts"][str(i)] for i in range(5)])
        for arm, selection in selections.items():
            path = folder / arm / "best.pt"
            if sha256_file(path) != selection["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint hash differs")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["epoch"] != selection["selected_epoch"]:
                raise ValueError("Selected epoch differs")
            np.testing.assert_allclose(selection["loss_weights"], np.ones(5) if arm == "unweighted" else counts.sum() / (5 * counts), rtol=1e-6)
            metric = source[arm]["test"]["pooled"] if fold == 0 else source[arm]["outer_test"]
            if metric["n_valid_epochs"] != splits[fold]["test"]["valid_epochs"]:
                raise ValueError("Source test support differs")
            pooled[arm] += np.asarray(metric["confusion_matrix"])
            arms[arm] = {"checkpoint_path": str(path.resolve()), "checkpoint_sha256": sha256_file(path),
                         "selected_epoch": selection["selected_epoch"],
                         "selected_validation_macro_f1": selection["selected_validation_macro_f1"],
                         "source_result_sha256": sha256_file(folder / "aggregate_results.json"),
                         "training_backend": "cpu" if fold < 6 else "cuda"}
        records.append({"fold": fold, "encoder_sha256": hashes[0], "arms": arms})
        evidence.append({"fold": fold, "backend": "cpu" if fold < 6 else "cuda",
                         "result_sha256": sha256_file(folder / "aggregate_results.json"),
                         "verification_sha256": sha256_file(folder / ("verification.json" if fold == 0 else "independent_verification.json"))})
        all_subjects.extend(splits[fold]["test"]["subject_ids"])
    require_ten_folds(records)
    if len(all_subjects) != len(set(all_subjects)) or len(all_subjects) != 78 or any(cm.sum() != 195469 for cm in pooled.values()):
        raise ValueError("Source outer test coverage differs")
    output = ROOT / "runs/teacher_revision_gpu_20261004/weighted_10fold_mixed_backend"
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__).resolve(), protocol, ROOT / "scripts/run_revision_weighted_campaign.py",
             ROOT / "scripts/verify_revision_weighted_campaign.py", ROOT / "scripts/run_recovered_e3_cpu_pilot.py",
             ROOT / "scripts/verify_colab_source_training.py"] + sorted((ROOT / "src/sleeptcn").rglob("*.py"))
    pilot = ROOT / "runs/teacher_revision_cpu_20261001/recovered_e3_fold0"
    pilot_result = checked_result(pilot, "verification.json")
    if sha256_file(pilot / "private_inference_manifest.json") != pilot_result["inference_manifest_sha256"]:
        raise ValueError("Locked target manifest differs from verified pilot")
    if sha256_file(splits_path) != pilot_result["provenance"]["source_split_sha256"]:
        raise ValueError("Source splits differ from verified pilot")
    spec = {"protocol": cfg, "protocol_sha256": sha256_file(protocol), "source_evidence": evidence,
            "restoration_sha256": sha256_file(restored / "restoration_manifest.json"),
            "split_sha256": sha256_file(splits_path),
            "target_manifest_sha256": sha256_file(pilot / "private_inference_manifest.json"),
            "backend_policy": "folds0-5 CPU2.5.1; folds6-9 fresh paired CUDA2.11; CPU partial fold6 not imported",
            "inference_backend": "cpu", "resource_limits": {"attempt_max_seconds": args.max_seconds, "cpu_threads": 4},
            "code_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}}
    write_once_json(output / "execution_specification.json", spec)
    snapshot(output, paths)
    frozen = {"status": "all_twenty_source_selected_checkpoints_frozen_before_new_target_inference",
              "records": records, "source_outer_test_pooled": {a:metrics_from_confusion(cm) for a,cm in pooled.items()},
              "backend_policy": spec["backend_policy"]}
    write_once_json(output / "all_checkpoints_frozen.json", frozen)
    if args.freeze_only:
        print("FROZEN 20 verified source-selected checkpoints; no target inference started", flush=True)
        return 0
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    budget = CampaignBudget(args.max_seconds, output / "progress.json")
    inputs = SimpleNamespace(output=output, restored=restored, verified_pilot=pilot, shhs_root=args.shhs_root.resolve())
    try:
        budget.publish(status="running", phase="target_inference", completed_source_folds=10)
        if not (output / "aggregate_results.json").exists():
            target_campaign(inputs, frozen, spec, budget)
        verify_target(inputs, budget)
        budget.publish(status="complete_additional_ten_fold_weighted_loss", phase="complete")
        return 0
    except (CampaignStop, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        budget.publish(status="stopped_resource_budget_or_interruption", reason=str(error), predictions_retained=True)
        return 2
    except Exception as error:
        budget.publish(status="failed", reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shhs-root", type=Path, default=Path("E:/research/Dataset/SHHS_v1"))
    parser.add_argument("--max-seconds", type=float, default=18000)
    parser.add_argument("--freeze-only", action="store_true")
    sys.exit(main(parser.parse_args()))
