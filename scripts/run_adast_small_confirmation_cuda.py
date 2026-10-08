"""Three fresh full-source models on fold 1; reuse immutable measured training."""
import argparse
import copy
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
import run_adast_development_cuda as base

EXPECTED_ARMS = {
    "source_only_full_source": {},
    "adast_reference_full_source": {},
    "adast_keep_source_ce_full_source": {"source_loss_weights_by_round": [1., 1.]},
}
STATUS = "complete_three_arm_new_fold_confirmation"

def effective_config(cfg, key):
    if cfg["arms"] != list(EXPECTED_ARMS) or cfg["arm_overrides"] != EXPECTED_ARMS:
        raise ValueError("Require the locked three confirmation arms")
    if key not in EXPECTED_ARMS:
        raise ValueError("Unknown confirmation arm")
    result = copy.deepcopy(cfg)
    result.update(EXPECTED_ARMS[key])
    return result

def train_arm(key, models, utils, arrays, cfg, folder, identity, guard, device):
    return base.train_development_arm(key, models, utils, arrays, effective_config(cfg, key),
                                      folder, identity, guard, device)

def check_matched(rows):
    if len(rows) != 3 or [r["arm"] for r in rows] != list(EXPECTED_ARMS):
        raise ValueError("Require all three completed fresh models")
    if len({r["initial_state_sha256"] for r in rows}) != 1:
        raise ValueError("Matched initialization differs")
    orders = [[h["source_order_sha256"] for h in r["history"]] for r in rows]
    if not all(order == orders[0] for order in orders):
        raise ValueError("Matched source sampling differs")
    if [h["target_order_sha256"] for h in rows[1]["history"]] != [h["target_order_sha256"] for h in rows[2]["history"]]:
        raise ValueError("Matched adaptation sampling differs")

def export(output):
    paths = sorted(p for p in output.rglob("*") if p.is_file() and p.suffix in {".pt", ".json", ".npz", ".npy", ".log"})
    archive = output.parent / "SleepTCN_ADAST_Small_Confirmation_Results_20261004.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in paths:
            z.write(path, path.relative_to(output).as_posix())
        z.writestr("export_manifest.json", json.dumps({p.relative_to(output).as_posix(): base.reference.digest(p) for p in paths}, indent=2))
    print("RESULT_ARCHIVE", archive, "SHA256", base.reference.digest(archive), flush=True)

def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU training fallback")
    work = Path(__file__).resolve().parent
    manifest = json.loads((work / "manifest.json").read_bytes())
    if any(manifest[k] for k in ("source_outer_test_included", "target_test_data_included", "target_labels_included", "participant_ids_included")):
        raise ValueError("Forbidden data role")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe payload path")
        if base.reference.digest(work / name) != digest:
            raise ValueError("Frozen payload changed")
    cfg = json.loads((work / "protocol.json").read_bytes())
    effective_config(cfg, cfg["arms"][0])
    if (cfg["campaign_kind"] != "three_arm_new_fold_confirmation" or cfg["fold"] != 1 or cfg["seed"] != 123
        or cfg["rounds"] != 2 or cfg["epochs_per_round"] != 15 or cfg["batch_size"] != 128
        or manifest["files"]["run_adast_development_cuda.py"] != cfg["previous_development_runner_sha256"]
        or not 0 < args.max_seconds <= cfg["max_seconds"] <= 18000):
        raise ValueError("Confirmation scope/budget or immutable runner differs")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arrays = [np.load(work / "data" / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("train_x", "train_y", "validation_x", "validation_y", "adaptation_x")]
    sizes = [cfg["source_train_epochs"]]*2 + [cfg["source_validation_epochs"]]*2 + [cfg["adaptation_epochs"]]
    for array, size, label in zip(arrays, sizes, (False, True, False, True, False)):
        if array.shape != ((size,) if label else (size, 3000)) or array.dtype != (np.int64 if label else np.float32):
            raise ValueError("Input schema/support differs")
        if not (np.isin(array, range(5)).all() if label else np.isfinite(array).all()):
            raise ValueError("Invalid signal/labels")
    for name, labels in (("train", arrays[1]), ("validation", arrays[3])):
        if np.bincount(labels, minlength=5).tolist() != manifest[name + "_class_counts"]:
            raise ValueError("Source class support differs")
    if int(np.ceil(len(arrays[0])/cfg["batch_size"])) != cfg["full_source_steps_per_epoch"]:
        raise ValueError("Full-source step count differs")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve old/partial attempt; no automatic restart")
    output.mkdir(parents=True)
    module = base.reference.load_module("confirmation_models", work / "upstream/models.py")
    model_cfg = base.reference.load_module("confirmation_config", work / "upstream/configs.py").Config()
    utils = base.reference.load_module("confirmation_utils", work / "upstream/utils.py")
    spec = {"protocol": cfg, "manifest_sha256": base.reference.digest(work / "manifest.json"),
            "runner_sha256": base.reference.digest(Path(__file__)),
            "base_runner_sha256": base.reference.digest(work / "run_adast_development_cuda.py"),
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "backend": "CUDA_float32_deterministic_no_tf32", "max_seconds": args.max_seconds,
            "source_outer_test_access": False, "target_test_access": False, "target_true_label_access": False}
    base.reference.write_once(output / "execution_specification.json", spec)
    guard = base.reference.Guard(output, args.max_seconds)
    rows = []
    try:
        for key in cfg["arms"]:
            models = base.reference.build_models(module, model_cfg, cfg["seed"], "cuda")
            identity = {"arm": key, "fold": cfg["fold"],
                        "execution_specification_sha256": base.reference.digest(output / "execution_specification.json")}
            rows.append(train_arm(key, models, utils, arrays, cfg, output / key, identity, guard, "cuda"))
            del models
            torch.cuda.empty_cache()
            guard.publish(completed_arms=len(rows))
        check_matched(rows)
        result = {"status": STATUS, "arms": rows, "elapsed_seconds": time.monotonic()-guard.started,
                  "source_outer_test_access": False, "target_test_access": False}
        base.reference.write_once(output / "aggregate_results.json", result)
        base.reference.write_once(output / "verification.json", {"status": "passed",
            "matched_initial_states_and_all_three_orders": True,
            "aggregate_results_sha256": base.reference.digest(output / "aggregate_results.json")})
        guard.publish(status=STATUS, phase="complete")
    except Exception as error:
        guard.publish(status="stopped_resource_budget" if isinstance(error, base.reference.BudgetStop) else "failed",
                      reason=str(error), checkpoints_retained=True)
        raise
    finally:
        export(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=18000)
    main(parser.parse_args())
