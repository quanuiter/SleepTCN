"""Complete eight matched full-source pairs on CUDA using the existing training core."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch
import run_adast_development_cuda as base

ARMS = ["source_only_full_source", "adast_full_source"]
FOLDS = list(range(2, 10))
STATUS = "complete_eight_new_fullsource_pairs"
BASE_SHA = "fd6071fa77fc463e25890f34350ad1a22fc946bd9e37a95fe2d801a984a3b204"
TORCH_VERSION = "2.11.0+cu130"
EXPECTED = {
    "seed": 123, "completed_folds_reused": [0, 1], "folds_to_train": FOLDS,
    "arms": ARMS, "rounds": 2, "epochs_per_round": 15, "batch_size": 128,
    "limited_steps_per_epoch": 38, "source_loss_weights_by_round": [1, .1],
    "target_loss_weights_by_round": [0, .01], "adversarial_weight": 1,
    "similarity_weight": .001,
    "optimizer": {"name": "Adam", "lr": .001, "betas": [.5, .99], "weight_decay": .0003},
    "base_runner_sha256": BASE_SHA, "required_torch": TORCH_VERSION,
    "backend": "CUDA_float32_deterministic_no_tf32", "adaptation_subjects": 5,
    "adaptation_epochs": 4989, "max_seconds": 18000, "cpu_training_fallback": False,
    "automatic_restart": False, "automatic_continuation": False,
    "target_test_data_upload": False, "target_labels_upload": False,
    "checkpoint_primary": "final_epoch30_for_both_budgets",
    "checkpoint_secondary": "strict_maximum_source_attention_source_validation_macro_f1_first_tie",
    "no_target_score_based_checkpoint_or_configuration_choice": True,
}


def validate_protocol(cfg):
    for key, expected in EXPECTED.items():
        if cfg.get(key) != expected:
            raise ValueError("Completion protocol differs: " + key)
    sizes = cfg.get("remaining_fold_sizes", [])
    if [row["fold"] for row in sizes] != FOLDS:
        raise ValueError("Require exactly the eight remaining folds")
    for row in sizes:
        if min(row["train_epochs"], row["validation_epochs"]) <= 0:
            raise ValueError("Empty source role")
        if row["updates_per_model"] != 30 * math.ceil(row["train_epochs"] / 128):
            raise ValueError("Update budget differs")


def validate_roles(roles, size):
    if set(roles) != {"train", "validation", "test"}:
        raise ValueError("Missing or extra source role")
    for indices in roles.values():
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or not len(indices) or indices.min() < 0 or indices.max() >= size
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Invalid or duplicate source role index")
    joined = np.concatenate(list(roles.values()))
    if len(joined) != size or len(np.unique(joined)) != size:
        raise ValueError("Source roles overlap or fail coverage")


class RoleArray:
    """Read exactly the ordered role subset without allocating a full signal copy."""
    def __init__(self, full, indices):
        self.full = full
        self.indices = indices
        self.shape = (len(indices), *full.shape[1:])
        self.dtype = full.dtype

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        return self.full[self.indices[key]]


def fold_arrays(x, y, target, roles):
    validate_roles(roles, len(x))
    return [RoleArray(x, roles["train"]), y[roles["train"]].copy(),
            RoleArray(x, roles["validation"]), y[roles["validation"]].copy(), target]


def check_pair(rows, train_size):
    if len(rows) != 2 or [row["arm"] for row in rows] != ARMS:
        raise ValueError("Require exactly one complete source-only/ADAST pair")
    if rows[0]["initial_state_sha256"] != rows[1]["initial_state_sha256"]:
        raise ValueError("Paired initialization differs")
    orders = [[h["source_order_sha256"] for h in row["history"]] for row in rows]
    if orders[0] != orders[1]:
        raise ValueError("Paired source order differs")
    steps = math.ceil(train_size / 128)
    for position, row in enumerate(rows):
        history = row["history"]
        if row["epochs_completed"] != 30 or len(history) != 30 or row["total_updates"] != 30 * steps:
            raise ValueError("Incomplete full-source training")
        expected_best = max(range(30), key=lambda i: history[i]["validation"]["source"]["macro_f1"]) + 1
        if row["best_epoch"] != expected_best:
            raise ValueError("Source-validation selection differs")
        for i, h in enumerate(history):
            if (h["global_epoch"] != i + 1 or h["updates"] != steps
                    or h["source_presentations"] != train_size or h["source_unique_seen"] != train_size):
                raise ValueError("Source epoch/update coverage differs")
            coefficients = {"source_ce": 1 if i < 15 else .1, "similarity": .001,
                            "adversarial": 1 if position else 0,
                            "target_pseudo_ce": (.01 if i >= 15 else 0) if position else 0}
            if h["loss_coefficients"] != coefficients:
                raise ValueError("Loss schedule differs")
            if not math.isclose(h["optimizer_lr_before_epoch"], .001 if i < 10 else .0001):
                raise ValueError("Learning-rate schedule differs")
            if h["target_training_presentations"] != (train_size if position else 0):
                raise ValueError("Target training role differs")
            if not position and (h["target_order_sha256"] is not None or h["target_unique_seen"] != 0):
                raise ValueError("Source-only consumed adaptation")


class CampaignGuard(base.reference.Guard):
    """Add a timing estimate to progress while retaining the existing shared limit."""
    def publish(self, **values):
        if "epoch_seconds" in values:
            key = (values.get("fold", self.state.get("fold")),
                   values.get("arm", self.state.get("arm")))
            self.timings.setdefault(key, []).append(values["epoch_seconds"])
            self.state["measured_epoch_count"] = sum(len(v) for v in self.timings.values())
            self.state["eta_seconds_from_observed_epochs"] = self.estimate_remaining()
        super().publish(**values)

    def __init__(self, output, seconds, sizes):
        super().__init__(output, seconds)
        self.sizes = {r["fold"]: r["train_epochs"] for r in sizes}
        self.timings = {}

    def estimate_remaining(self):
        rates = {arm: [] for arm in ARMS}
        for (fold, arm), seconds in self.timings.items():
            rates[arm].append(float(np.mean(seconds)) / math.ceil(self.sizes[fold] / 128))
        if not all(rates.values()):
            return None
        remaining = 0.
        for fold in FOLDS:
            for arm in ARMS:
                epochs_left = 30 - len(self.timings.get((fold, arm), []))
                remaining += epochs_left * math.ceil(self.sizes[fold] / 128) * max(rates[arm])
        return remaining


def export(output):
    paths = sorted(p for p in output.rglob("*")
                   if p.is_file() and p.suffix in {".pt", ".json", ".npz", ".npy", ".log"})
    archive = output.parent / "SleepTCN_ADAST_Fullsource_Completion_Results_20261006.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in paths:
            zipped.write(path, path.relative_to(output).as_posix())
        zipped.writestr("export_manifest.json", json.dumps(
            {p.relative_to(output).as_posix(): base.reference.digest(p) for p in paths}, indent=2))
    print("RESULT_ARCHIVE", archive, "SHA256", base.reference.digest(archive), flush=True)
    return archive


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU training fallback")
    if str(torch.__version__) != TORCH_VERSION:
        raise RuntimeError("Require matching PyTorch " + TORCH_VERSION)
    work = Path(__file__).resolve().parent
    cfg = json.loads((work / "protocol.json").read_bytes())
    validate_protocol(cfg)
    if not 0 < args.max_seconds <= cfg["max_seconds"]:
        raise ValueError("Require at most one five-hour attempt")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve existing attempt; no automatic restart")
    manifest = json.loads((work / "manifest.json").read_bytes())
    forbidden = ("target_test_data_included", "target_labels_included", "participant_ids_included", "raw_edf_included")
    if any(manifest[k] for k in forbidden):
        raise ValueError("Forbidden payload data")
    if (manifest["folds_to_train"] != FOLDS or not manifest["source_all_records_included"]
            or not manifest["source_role_training_restricted"] or manifest["adaptation_subjects"] != 5):
        raise ValueError("Payload role scope differs")
    expected_files = {"protocol.json", "run_adast_fullsource_completion_cuda.py",
                      "run_adast_development_cuda.py", "adast_reference_adapter.py", "development_metrics.py",
                      "upstream/models.py", "upstream/configs.py", "upstream/utils.py", "upstream/LICENSE",
                      "data/source_x.npy", "data/source_y.npy", "data/adaptation_x.npy"}
    expected_files.update(f"data/fold_{fold:02d}_roles.npz" for fold in FOLDS)
    if set(manifest["files"]) != expected_files:
        raise ValueError("Payload file allowlist differs")
    for name, digest in manifest["files"].items():
        if base.reference.digest(work / name) != digest:
            raise ValueError("Payload changed: " + name)
    if manifest["files"]["run_adast_development_cuda.py"] != BASE_SHA:
        raise ValueError("Historical training core differs")
    if shutil.disk_usage(output.parent if output.parent.exists() else work).free < 5_000_000_000:
        raise RuntimeError("Need 5 GB free for checkpoint and result export")
    x = np.load(work / "data/source_x.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(work / "data/source_y.npy", mmap_mode="r", allow_pickle=False)
    target = np.load(work / "data/adaptation_x.npy", mmap_mode="r", allow_pickle=False)
    if (x.shape != (195469, 3000) or x.dtype != np.float32 or y.shape != (195469,)
            or y.dtype != np.int64 or target.shape != (4989, 3000) or target.dtype != np.float32):
        raise ValueError("Input array schema differs")
    if not np.isin(y, range(5)).all() or not np.isfinite(target).all():
        raise ValueError("Invalid labels or adaptation signals")
    for start in range(0, len(x), 2048):
        if not np.isfinite(x[start:start + 2048]).all():
            raise ValueError("Nonfinite source signal")
    roles_by_fold = {}
    for size in cfg["remaining_fold_sizes"]:
        fold = size["fold"]
        with np.load(work / f"data/fold_{fold:02d}_roles.npz", allow_pickle=False) as zipped:
            roles = {name: zipped[name].copy() for name in zipped.files}
        validate_roles(roles, len(x))
        if len(roles["train"]) != size["train_epochs"] or len(roles["validation"]) != size["validation_epochs"]:
            raise ValueError("Fold role size differs")
        roles_by_fold[fold] = roles
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True)
    module = base.reference.load_module("completion_models", work / "upstream/models.py")
    model_cfg = base.reference.load_module("completion_config", work / "upstream/configs.py").Config()
    utils = base.reference.load_module("completion_utils", work / "upstream/utils.py")
    spec = {"protocol": cfg, "manifest_sha256": base.reference.digest(work / "manifest.json"),
            "runner_sha256": base.reference.digest(Path(__file__)), "base_runner_sha256": BASE_SHA,
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "backend": cfg["backend"], "max_seconds": args.max_seconds,
            "source_outer_test_access_during_training": False, "target_test_access": False,
            "target_true_label_access": False, "reused_models_present_in_this_archive": False}
    base.reference.write_once(output / "execution_specification.json", spec)
    guard = CampaignGuard(output, args.max_seconds, cfg["remaining_fold_sizes"])
    rows = []
    try:
        for fold in FOLDS:
            guard()
            arrays = fold_arrays(x, y, target, roles_by_fold[fold])
            fold_folder = output / f"fold_{fold:02d}"
            fold_folder.mkdir()
            pair = []
            for key in ARMS:
                guard()
                guard.publish(fold=fold, arm=key, phase="initializing", epochs_completed=0)
                models = base.reference.build_models(module, model_cfg, cfg["seed"], "cuda")
                identity = {"arm": key, "fold": fold,
                            "execution_specification_sha256": base.reference.digest(output / "execution_specification.json")}
                pair.append(base.train_development_arm(key, models, utils, arrays, copy.deepcopy(cfg),
                                                       fold_folder / key, identity, guard, "cuda"))
                del models
                gc.collect()
                torch.cuda.empty_cache()
                guard.publish(completed_models=len(rows) * 2 + len(pair))
            check_pair(pair, len(arrays[0]))
            rows.append({"fold": fold, "arms": pair})
            base.reference.write_once(fold_folder / "pair_verification.json",
                                      {"status": "passed", "matched_initialization_and_source_orders": True})
            guard.publish(completed_pairs=len(rows))
        result = {"status": STATUS, "folds": rows, "new_models_completed": 16,
                  "reused_folds_separate": [0, 1], "elapsed_seconds": time.monotonic() - guard.started,
                  "source_outer_test_access_during_training": False, "target_test_access": False}
        base.reference.write_once(output / "aggregate_results.json", result)
        base.reference.write_once(output / "verification.json", {
            "status": "passed", "complete_matched_pairs": 8,
            "aggregate_results_sha256": base.reference.digest(output / "aggregate_results.json")})
        guard.publish(status=STATUS, phase="complete", completed_models=16)
    except BaseException as error:
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
