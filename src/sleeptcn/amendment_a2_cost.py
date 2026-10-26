"""Amendment A2, analysis E: decomposition of the seed-123 training wall-clock time.

* Optimiser steps, completed epochs and validation events of every stage of every fold come
  from ``progress`` in the archived ``latest.pt`` checkpoints at commit 8af1d12.
* Observed wall-clock time per configuration and fold comes from the archived campaign
  records ``runs/v2/monitoring/fold_XX_seed123_time.txt`` at the same commit.
* Time per optimiser step is measured through ``engine.run_loader`` (the training loop) with
  each stage's optimiser, loss, batch size and gradient clipping, on pre-built synthetic
  batches. Sequence batches use the real fold-0 training record lengths.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_COMMIT = "8af1d12"
SEED = 123
FOLDS = tuple(range(10))
CONFIGS = ("E0", "E2", "E3", "E4", "E6")
BENCHMARK_FILE = Path("runs/a2_step_time_benchmark.json")
WARMUP_STEPS = 20
TIMED_STEPS = 200
ROUNDS = 3
LENGTH_SEED = 2045


def stage_names(config: str) -> list[str]:
    from .features import expected_15cnn_keys

    if config == "E0":
        return [f"cnn15/{key}" for key in expected_15cnn_keys()] + ["sequence/bilstm"]
    return ["resnet1d", "sequence/tcn"]


def stage_kind(stage: str) -> str:
    return {"resnet1d": "resnet1d", "sequence/bilstm": "bilstm", "sequence/tcn": "tcn"}.get(stage, "cnn15")


def _git_bytes(workspace: Path, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(workspace), "show", f"{REFERENCE_COMMIT}:{path}"],
                          capture_output=True, check=True).stdout


def archived_steps(workspace: Path) -> dict[str, Any]:
    """Per configuration, fold and stage: completed epochs, validation events and optimiser steps."""
    import torch

    rows = []
    for config in CONFIGS:
        for fold in FOLDS:
            for stage in stage_names(config):
                path = f"runs/v2/full/{config}/fold_{fold:02d}/seed_{SEED}/checkpoints/{stage}/latest.pt"
                progress = torch.load(io.BytesIO(_git_bytes(workspace, path)), map_location="cpu",
                                      weights_only=False)["progress"]
                rows.append({"config": config, "fold": fold, "stage": stage, "kind": stage_kind(stage),
                             "completed_epochs": int(progress["completed_epochs"]),
                             "validation_events": int(progress["validation_events"]),
                             "optimizer_steps": int(progress["global_steps"]),
                             "best_epoch": int(progress["best_epoch"])})
    return {"reference_commit": REFERENCE_COMMIT, "seed": SEED, "rows": rows}


def archived_wall_clock(workspace: Path) -> dict[str, Any]:
    """Seconds per configuration and fold from the archived START/DONE campaign records."""
    pattern = re.compile(r"^(START|DONE) (E\d) (\S+)$")
    seconds: dict[str, dict[int, float]] = {}
    for fold in FOLDS:
        text = _git_bytes(workspace, f"runs/v2/monitoring/fold_{fold:02d}_seed{SEED}_time.txt").decode("utf-8")
        starts: dict[str, datetime] = {}
        for line in text.splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            event, config, stamp = match.groups()
            moment = datetime.fromisoformat(stamp)
            if event == "START":
                starts[config] = moment
            else:
                seconds.setdefault(config, {})[fold] = (moment - starts.pop(config)).total_seconds()
    for config in CONFIGS:
        if sorted(seconds.get(config, {})) != list(FOLDS):
            raise ValueError(f"{config}: wall-clock records missing for some folds")
    return {"reference_commit": REFERENCE_COMMIT, "seed": SEED,
            "seconds": {c: {f"fold_{f:02d}": seconds[c][f] for f in FOLDS} for c in CONFIGS},
            "total_seconds": {c: float(sum(seconds[c].values())) for c in CONFIGS}}


# --------------------------------------------------------------------------- step-time benchmark
def fold0_training_lengths(workspace: Path) -> list[int]:
    split = json.loads((workspace / "data/splits/sleepedf_sc_10fold_seed42_v2.json").read_text(encoding="utf-8"))
    run = next(r for r in split["outer_runs"] if int(r["outer_fold"]) == 0)
    lengths = []
    for key in run["train"]["record_keys"]:
        with np.load(workspace / "data/processed/bandpass_v2" / f"{key}.npz", allow_pickle=False) as z:
            lengths.append(int(len(z["y"])))
    return lengths


def _epoch_batches(count: int, batch: int, generator: np.random.Generator):
    import torch

    return [(torch.from_numpy(generator.standard_normal((batch, 1, 3000), dtype=np.float32)),
             torch.from_numpy(generator.integers(0, 5, batch)).long()) for _ in range(count)]


def _sequence_batches(count: int, per_batch: int, feature_dim: int, lengths: list[int],
                      generator: np.random.Generator):
    import torch

    from .training import collate_feature_sequences

    batches, pool = [], []
    for _ in range(count):
        if len(pool) < per_batch:
            pool += list(generator.permutation(lengths))
        chosen, pool = pool[:per_batch], pool[per_batch:]
        items = [(torch.from_numpy(generator.standard_normal((n, feature_dim), dtype=np.float32)),
                  torch.from_numpy(generator.integers(0, 5, n)).long()) for n in chosen]
        batches.append(collate_feature_sequences(items))
    return batches


def benchmark_step_times(workspace: Path, device: str) -> dict[str, Any]:
    """Mean time per optimiser step of each stage kind through ``engine.run_loader``."""
    import torch

    from .engine import epoch_forward, run_loader, sequence_forward
    from .io.serialization import read_json
    from .models import EEGResNet1D, SleepCNN
    from .training import masked_cross_entropy
    from .workflows.model_factory import build_sequence_model

    config = read_json(workspace / "configs/experiments_v2.json")
    components = config["components"]
    lengths = fold0_training_lengths(workspace)
    torch_device = torch.device(device)
    weights = torch.tensor([0.6, 1.8, 0.6, 3.2, 1.5])

    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return masked_cross_entropy(logits, targets, weights)

    specs = {
        "cnn15": (SleepCNN(), lambda m: torch.optim.Adam(m.parameters(), lr=components["cnn15"]["learning_rate"]),
                  None, epoch_forward,
                  lambda n, g: _epoch_batches(n, components["cnn15"]["batch_size"], g)),
        "resnet1d": (EEGResNet1D(),
                     lambda m: torch.optim.AdamW(m.parameters(), lr=components["resnet1d"]["learning_rate"],
                                                 weight_decay=components["resnet1d"]["weight_decay"]),
                     None, epoch_forward,
                     lambda n, g: _epoch_batches(n, components["resnet1d"]["batch_size_epochs"], g)),
        "bilstm": (build_sequence_model(config, "bilstm"),
                   lambda m: torch.optim.Adam(m.parameters(), lr=components["bilstm"]["learning_rate"]),
                   1.0, sequence_forward("bilstm"),
                   lambda n, g: _sequence_batches(n, components["bilstm"]["batch_size_records"], 75, lengths, g)),
        "tcn": (build_sequence_model(config, "common_tcn", input_dim=128),
                lambda m: torch.optim.Adam(m.parameters(), lr=components["common_tcn"]["learning_rate"]),
                1.0, sequence_forward("tcn"),
                lambda n, g: _sequence_batches(n, components["common_tcn"]["batch_size_records"], 128, lengths, g)),
    }

    def synchronize() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    results = {}
    for kind, (model, make_optimizer, clip, forward, make_batches) in specs.items():
        generator = np.random.default_rng(LENGTH_SEED)
        model = model.to(torch_device)
        optimizer = make_optimizer(model)
        run_loader(model, make_batches(WARMUP_STEPS, generator), torch_device, forward, loss,
                   optimizer=optimizer, gradient_clip_norm=clip)
        rounds = []
        for _ in range(ROUNDS):
            batches = make_batches(TIMED_STEPS, generator)
            synchronize()
            started = time.perf_counter()
            run_loader(model, batches, torch_device, forward, loss, optimizer=optimizer, gradient_clip_norm=clip)
            synchronize()
            rounds.append((time.perf_counter() - started) / TIMED_STEPS)
        results[kind] = {"mean_seconds_per_step": float(np.mean(rounds)), "round_means": rounds,
                         "range_over_rounds": [float(min(rounds)), float(max(rounds))]}
        del model, optimizer
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    runtime = {"device": device, "torch_version": torch.__version__}
    if torch_device.type == "cuda":
        runtime["gpu"] = torch.cuda.get_device_name(0)
    return {"schema_version": 1, "amendment": "A2", **runtime, "warmup_steps": WARMUP_STEPS,
            "timed_steps": TIMED_STEPS, "rounds": ROUNDS, "length_seed": LENGTH_SEED,
            "fold0_training_records": len(lengths), "fold0_mean_record_epochs": float(np.mean(lengths)),
            "kinds": results}


def decompose(steps: dict[str, Any], wall: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    """Per configuration: steps, steps x mean step time, remainder, and E0/config ratios."""
    times = {kind: v["mean_seconds_per_step"] for kind, v in benchmark["kinds"].items()}
    out: dict[str, Any] = {}
    for config in CONFIGS:
        rows = [r for r in steps["rows"] if r["config"] == config]
        by_kind: dict[str, int] = {}
        for row in rows:
            by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + row["optimizer_steps"]
        compute = sum(n * times[kind] for kind, n in by_kind.items())
        observed = wall["total_seconds"][config]
        out[config] = {"optimizer_steps_by_kind": by_kind, "optimizer_steps": sum(by_kind.values()),
                       "completed_epochs": sum(r["completed_epochs"] for r in rows),
                       "networks_trained_per_fold": len(stage_names(config)),
                       "step_compute_seconds_on_benchmark_device": compute,
                       "observed_wall_clock_seconds_v100": observed}
        if "V100" in benchmark.get("gpu", ""):  # same GPU model: absolute remainder is meaningful
            out[config]["remainder_seconds"] = observed - compute
            out[config]["remainder_fraction"] = (observed - compute) / observed
    for config in CONFIGS:
        if config == "E0":
            continue
        e0, other = out["E0"], out[config]
        steps_ratio = e0["optimizer_steps"] / other["optimizer_steps"]
        out[config]["ratios_vs_E0"] = {
            "observed_wall_clock": e0["observed_wall_clock_seconds_v100"] / other["observed_wall_clock_seconds_v100"],
            "optimizer_steps": steps_ratio,
            "step_compute": e0["step_compute_seconds_on_benchmark_device"] / other["step_compute_seconds_on_benchmark_device"],
            "mean_step_time": (e0["step_compute_seconds_on_benchmark_device"] / e0["optimizer_steps"])
            / (other["step_compute_seconds_on_benchmark_device"] / other["optimizer_steps"]),
        }
    return out
