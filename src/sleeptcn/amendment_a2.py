"""Amendment A2: post-confirmatory Sleep-EDF training and locked-test evaluation.

A2 reuses the unchanged runner (``experiment.run_experiment``) and experiment config
(``configs/experiments_v2.json``). It only builds the run context itself, so that an
arm can pair an existing experiment (E3, E4 or E6) with an A2 data variant and keep
its runs under ``runs/a2``. Training and test evaluation are separated as in the
original campaigns: the Sleep-EDF test folds are read only after every A2 run has
finished validation-only training and passed independent run validation.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io.hashing import sha256_file
from .io.serialization import atomic_write_json, read_json

A2_CONFIG = Path("configs/shhs_holdout_amendment_a2.json")
LOCKED_STATUS = "locked_before_a2_training"
EXPERIMENT_CONFIG = Path("configs/experiments_v2.json")
SPLIT = Path("data/splits/sleepedf_sc_10fold_seed42_v2.json")
REPRO_ARM = "REPRO_E4"
TEST_CONFIRMATION = "OPEN-A2-LOCKED-TEST-ONCE"
A2_ROOT = Path("runs/a2")


@dataclass(frozen=True, order=True)
class A2Target:
    arm: str
    seed: int
    fold: int

    @property
    def key(self) -> str:
        return f"{self.arm}/seed_{self.seed}/fold_{self.fold:02d}"


def load_a2_config(workspace: Path, *, require_locked: bool = True) -> tuple[dict[str, Any], str]:
    path = workspace / A2_CONFIG
    config = read_json(path)
    if config.get("amendment") != "A2":
        raise ValueError("not the A2 amendment config")
    if require_locked and config.get("status") != LOCKED_STATUS:
        raise ValueError(
            f"A2 config status is {config.get('status')!r}; lock it ({LOCKED_STATUS!r}) and commit first"
        )
    return config, sha256_file(path)


def arm_spec(config: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm == REPRO_ARM:
        return {"experiment": "E4", "data_variant": "bandpass_v2", "seeds": [123]}
    return config["training"]["arms"][arm]


def training_targets(config: dict[str, Any]) -> list[A2Target]:
    """All A2 runs in the locked ``run_order``: each (arm, seed) pair with all its folds.

    ``run_order`` must list every (arm, seed) of ``arms`` exactly once.
    """
    arms = config["training"]["arms"]
    folds = [int(f) for f in config["training"]["folds"]]
    order = [(str(arm), int(seed)) for arm, seed in config["training"]["run_order"]]
    declared = {(arm, int(seed)) for arm, spec in arms.items() for seed in spec["seeds"]}
    if len(order) != len(set(order)) or set(order) != declared:
        raise ValueError("run_order must list every (arm, seed) of the A2 arms exactly once")
    if REPRO_ARM in arms:
        raise ValueError(f"{REPRO_ARM} is reserved for the reproduction check")
    return [A2Target(arm, seed, fold) for arm, seed in order for fold in folds]


def repro_target() -> A2Target:
    return A2Target(REPRO_ARM, 123, 0)


def run_root(workspace: Path, target: A2Target) -> Path:
    return workspace / A2_ROOT / "full" / target.arm / f"fold_{target.fold:02d}" / f"seed_{target.seed}"


def cache_root(workspace: Path, target: A2Target) -> Path:
    return (workspace / "data" / "cache" / "features" / "a2" / "full"
            / f"fold_{target.fold:02d}" / f"seed_{target.seed}")


def build_a2_context(workspace: Path, config: dict[str, Any], target: A2Target, device: str, *,
                     num_workers: int, allow_test_evaluation: bool, resume: bool):
    """Same fields as ``experiment.build_context``, with the A2 variant and run root."""
    import torch

    from .experiment import RunContext

    spec = arm_spec(config, target.arm)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    experiment_config = read_json(workspace / EXPERIMENT_CONFIG)
    if spec["experiment"] not in experiment_config["experiments"]:
        raise ValueError(f"unknown experiment {spec['experiment']}")
    return RunContext(
        workspace=workspace,
        experiment_id=spec["experiment"],
        outer_fold=target.fold,
        seed=target.seed,
        device=torch_device,
        smoke=False,
        allow_test_evaluation=allow_test_evaluation,
        resume=resume,
        num_workers=num_workers,
        config=experiment_config,
        config_sha256=sha256_file(workspace / EXPERIMENT_CONFIG),
        split_sha256=sha256_file(workspace / SPLIT),
        data_variant=spec["data_variant"],
        run_root=run_root(workspace, target),
        cache_root=cache_root(workspace, target),
    )


def _manifest(workspace: Path, target: A2Target) -> dict[str, Any] | None:
    path = run_root(workspace, target) / "run_manifest.json"
    return read_json(path) if path.is_file() else None


def training_state(workspace: Path, target: A2Target) -> str:
    """'missing', 'partial', 'trained' (validation only) or 'tested'."""
    manifest = _manifest(workspace, target)
    if manifest is None:
        return "missing" if not run_root(workspace, target).exists() else "partial"
    if manifest.get("status") != "complete":
        return "partial"
    if not (run_root(workspace, target) / "validation_report.json").is_file():
        return "partial"
    return "tested" if "test" in manifest.get("metrics_roles", []) else "trained"


def train_target(workspace: Path, config: dict[str, Any], config_sha256: str, target: A2Target, *,
                 device: str, num_workers: int) -> dict[str, Any]:
    """Validation-only training of one A2 run with the unchanged runner, then run validation."""
    from .experiment import run_experiment
    from .run_validation import validate_run

    workspace = workspace.resolve()
    state = training_state(workspace, target)
    if state in ("trained", "tested"):
        return {"target": target.key, "state": state, "skipped": True}
    root = run_root(workspace, target)
    context = build_a2_context(workspace, config, target, device, num_workers=num_workers,
                               allow_test_evaluation=False, resume=state == "partial")
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "a2_target.json", {
        "schema_version": 1, "amendment": "A2", "a2_config_sha256": config_sha256,
        "arm": target.arm, "experiment_id": context.experiment_id, "data_variant": context.data_variant,
        "seed": target.seed, "outer_fold": target.fold, "device": device,
    })
    started = time.time()
    run_experiment(context)
    elapsed = time.time() - started
    report = validate_run(workspace, root)
    atomic_write_json(root / "validation_report.json", report)
    record = {"target": target.key, "state": "trained", "resumed": state == "partial",
              "wall_seconds": round(elapsed, 1), "finished_unix": round(time.time(), 1)}
    log = workspace / A2_ROOT / "monitoring" / "training_times.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


# --------------------------------------------------------------------------- locked test
def _git_clean(workspace: Path) -> str:
    commit = subprocess.run(["git", "-C", str(workspace), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(workspace), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if status:
        raise RuntimeError("A2 test evaluation requires a clean Git worktree:\n" + status)
    return commit


def _baseline(workspace: Path, target: A2Target) -> dict[str, Any]:
    from .run_validation import validate_run

    root = run_root(workspace, target)
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("smoke"):
        raise ValueError(f"{target.key}: training run is not complete")
    if manifest.get("allow_test_evaluation") is not False or manifest.get("metrics_roles") != ["validation"]:
        raise ValueError(f"{target.key}: test role is not locked")
    if (root / "predictions" / "test.npz").exists() or (root / "metrics" / "test.json").exists():
        raise ValueError(f"{target.key}: test artifacts already exist")
    report = validate_run(workspace, root)
    if not report.get("passed") or set(report.get("roles", {})) != {"validation"}:
        raise ValueError(f"{target.key}: independent validation did not pass")
    return {
        "state": "pending",
        "validation_prediction_sha256": sha256_file(root / "predictions" / "validation.npz"),
        "validation_metrics_sha256": sha256_file(root / "metrics" / "validation.json"),
        "extractor_sha256": manifest["extractor_sha256"],
        "sequence_checkpoint_sha256": manifest["sequence_checkpoint_sha256"],
    }


def test_journal_path(workspace: Path) -> Path:
    return workspace / A2_ROOT / "test_campaign_a2.json"


def test_preflight(workspace: Path, targets: Iterable[A2Target]) -> dict[str, Any]:
    """Check every A2 run is complete, validated and still test-locked; reads no test data."""
    workspace = workspace.resolve()
    targets = list(targets)
    missing = [t.key for t in targets if training_state(workspace, t) != "trained"]
    if missing:
        raise ValueError(f"{len(missing)} A2 runs are not trained and test-locked, e.g. {missing[:3]}")
    return {
        "schema_version": 1, "amendment": "A2", "status": "prepared",
        "source_git_commit": _git_clean(workspace),
        "target_order": [t.key for t in targets],
        "targets": {t.key: _baseline(workspace, t) for t in targets},
    }


def _evaluate_test(workspace: Path, config: dict[str, Any], target: A2Target, entry: dict[str, Any], *,
                   device: str, num_workers: int, source_commit: str) -> dict[str, Any]:
    from .experiment import _save_role_predictions, _selected_records, feature_sequences
    from .run_validation import validate_run
    from .test_gate import _load_extractor, _load_sequence_model
    from .training_data import resolve_fold_partitions

    root = run_root(workspace, target)
    for name, key in (("predictions/validation.npz", "validation_prediction_sha256"),
                      ("metrics/validation.json", "validation_metrics_sha256")):
        if sha256_file(root / name) != entry[key]:
            raise ValueError(f"{target.key}: validation artifact changed: {name}")
    context = build_a2_context(workspace, config, target, device, num_workers=num_workers,
                               allow_test_evaluation=True, resume=True)
    manifest = read_json(root / "run_manifest.json")
    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    if extractor_hash != entry["extractor_sha256"]:
        raise ValueError(f"{target.key}: extractor hash changed")
    sequence_kind, sequence_model, sequence_path = _load_sequence_model(context)
    if sha256_file(sequence_path) != entry["sequence_checkpoint_sha256"]:
        raise ValueError(f"{target.key}: sequence checkpoint hash changed")
    partitions = resolve_fold_partitions(workspace / "data" / "processed", workspace / SPLIT,
                                         target.fold, context.data_variant)
    test_records = _selected_records(partitions.test, context.data_variant, None)
    sequences = feature_sequences(context, test_records, "test", extractor_kind=extractor_kind,
                                  extractor=extractor, extractor_sha256=extractor_hash)
    _save_role_predictions(context, sequence_model, sequences, sequence_kind, "test", sequence_path)
    updated = dict(manifest)
    updated["role_records"] = {**manifest["role_records"],
                               "test": [r.info.record_key for r in test_records]}
    updated["allow_test_evaluation"] = True
    updated["metrics_roles"] = ["validation", "test"]
    updated["test_evaluation"] = {"amendment": "A2", "source_git_commit": source_commit,
                                  "selection": "prelocked_best_checkpoint"}
    atomic_write_json(root / "run_manifest.json", updated)
    report = validate_run(workspace, root)
    atomic_write_json(root / "validation_report.json", report)
    return {"state": "complete",
            "test_prediction_sha256": report["roles"]["test"]["prediction_sha256"],
            "test_valid_epochs": report["roles"]["test"]["valid_epochs"]}


def execute_test(workspace: Path, config: dict[str, Any], targets: Iterable[A2Target], *,
                 device: str, num_workers: int, confirmation: str) -> dict[str, Any]:
    if confirmation != TEST_CONFIRMATION:
        raise ValueError(f"confirmation must equal {TEST_CONFIRMATION!r}")
    workspace = workspace.resolve()
    targets = list(targets)
    path = test_journal_path(workspace)
    if path.is_file():
        journal = read_json(path)
        if journal.get("source_git_commit") != _git_clean(workspace):
            raise ValueError("Git HEAD differs from the prepared A2 test campaign")
    else:
        journal = test_preflight(workspace, targets)
        journal["status"] = "running"
        atomic_write_json(path, journal)
    for index, target in enumerate(targets, start=1):
        entry = journal["targets"][target.key]
        if entry.get("state") == "complete":
            continue
        print(f"[{index}/{len(targets)}] {target.key}: test", flush=True)
        entry.update(_evaluate_test(workspace, config, target, entry, device=device,
                                    num_workers=num_workers, source_commit=journal["source_git_commit"]))
        atomic_write_json(path, journal)
    journal["status"] = "complete"
    atomic_write_json(path, journal)
    return journal


# --------------------------------------------------------------------------- reproduction check
def reproduction_summary(workspace: Path, *, reference_commit: str = "8af1d12") -> dict[str, Any]:
    """Compare the A2 re-run of E4 fold 0 seed 123 with the archived seed-123 run."""
    import io

    import torch

    def git_bytes(path: str) -> bytes:
        return subprocess.run(["git", "-C", str(workspace), "show", f"{reference_commit}:{path}"],
                              capture_output=True, check=True).stdout

    reference = "runs/v2/full/E4/fold_00/seed_123"
    new_root = run_root(workspace, repro_target())
    out: dict[str, Any] = {"reference_commit": reference_commit, "stages": {}}
    for stage in ("resnet1d", "sequence/tcn"):
        old = torch.load(io.BytesIO(git_bytes(f"{reference}/checkpoints/{stage}/latest.pt")),
                         map_location="cpu", weights_only=False)["progress"]
        new = torch.load(new_root / "checkpoints" / stage / "latest.pt", map_location="cpu",
                         weights_only=False)["progress"]
        out["stages"][stage] = {k: {"archived": old[k], "rerun": new[k]}
                                for k in ("completed_epochs", "best_epoch", "best_metric")}
    import numpy as np

    old_metrics = json.loads(git_bytes(f"{reference}/metrics/validation.json"))["metrics"]
    new_metrics = read_json(new_root / "metrics" / "validation.json")["metrics"]
    out["validation_macro_f1"] = {"archived": old_metrics["macro_f1"], "rerun": new_metrics["macro_f1"]}
    with np.load(io.BytesIO(git_bytes(f"{reference}/predictions/validation.npz"))) as old, \
            np.load(new_root / "predictions" / "validation.npz") as new:
        same_epochs = np.array_equal(old["original_epoch_index"], new["original_epoch_index"]) and \
            np.array_equal(old["record_key"], new["record_key"])
        out["same_validation_epochs"] = bool(same_epochs)
        if same_epochs:
            out["predicted_label_agreement"] = float((old["predicted_label"] == new["predicted_label"]).mean())
            out["max_abs_logit_difference"] = float(np.abs(old["logits"] - new["logits"]).max())
    out["identical"] = bool(out.get("max_abs_logit_difference") == 0.0)
    return out
