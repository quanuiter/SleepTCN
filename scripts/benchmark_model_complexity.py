"""Gate-6 benchmark using the locked, trained pipelines on one device."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.hashing import combined_sha256, sha256_file
from sleeptcn.experiment import build_context
from sleeptcn.features import MANIPULATIONS, MANIPULATION_PREFIX, STAGE_NAMES
from sleeptcn.run_validation import validate_run
from sleeptcn.test_gate import (
    ACTIVE_EXPERIMENTS,
    _load_extractor,
    _load_sequence_model,
)
from sleeptcn.workflows.provenance import clean_git_commit


BENCHMARK_SCHEMA_VERSION = 2


class CNN15Pipeline(nn.Module):
    """Exact current/previous/next feature ordering followed by the sequence model."""

    def __init__(self, extractors: dict[str, nn.Module], sequence: nn.Module, kind: str):
        super().__init__()
        expected = tuple(
            f"{MANIPULATION_PREFIX[m]}_{stage}"
            for m in MANIPULATIONS
            for stage in STAGE_NAMES
        )
        if tuple(extractors) != expected:
            raise ValueError("15CNN extractors are not in canonical order")
        if kind not in {"bilstm", "tcn"}:
            raise ValueError("invalid sequence kind")
        self.extractors = nn.ModuleDict(extractors)
        self.sequence = sequence
        self.kind = kind

    @staticmethod
    def _shift(x: torch.Tensor, manipulation: str) -> torch.Tensor:
        if manipulation == "current":
            return x
        if manipulation == "previous":
            return torch.cat((x[:, :1], x[:, :-1]), dim=1)
        if manipulation == "next":
            return torch.cat((x[:, 1:], x[:, -1:]), dim=1)
        raise ValueError(manipulation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length = x.shape[:2]
        columns = []
        for manipulation in MANIPULATIONS:
            flat = self._shift(x, manipulation).reshape(batch * length, 1, 3000)
            prefix = MANIPULATION_PREFIX[manipulation]
            for stage in STAGE_NAMES:
                columns.append(
                    torch.softmax(self.extractors[f"{prefix}_{stage}"](flat), dim=-1)
                )
        features = torch.cat(columns, dim=-1).reshape(batch, length, 75)
        if self.kind == "bilstm":
            lengths = torch.full((batch,), length, dtype=torch.long)
            return self.sequence(features, lengths)
        return self.sequence(features, padding_mask=None)


class ResNetTCNPipeline(nn.Module):
    def __init__(self, extractor: nn.Module, sequence: nn.Module):
        super().__init__()
        self.extractor = extractor
        self.sequence = sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length = x.shape[:2]
        features = self.extractor.extract_features(
            x.reshape(batch * length, 1, 3000)
        ).reshape(batch, length, 128)
        return self.sequence(features, padding_mask=None)


def _git_commit(workspace: Path) -> str:
    return clean_git_commit(
        workspace,
        dirty_message="official benchmark requires a clean Git worktree",
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_report(model: nn.Module) -> dict[str, Any]:
    counts = Counter(type(module).__name__ for module in model.modules())
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_bytes_fp32": parameters * 4,
        "component_models": 16 if isinstance(model, CNN15Pipeline) else 2,
        "module_class_counts": dict(sorted(counts.items())),
    }


def benchmark_forward(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    model = model.to(device).eval()
    x = x.to(device)
    expected_shape = (x.shape[0], x.shape[1], 5)
    with torch.inference_mode():
        output = model(x)
        if tuple(output.shape) != expected_shape or not torch.isfinite(output).all():
            raise ValueError("pipeline output contract failed")
        for _ in range(warmup):
            model(x)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
        elapsed = []
        for _ in range(repeats):
            _synchronize(device)
            started = time.perf_counter_ns()
            model(x)
            _synchronize(device)
            elapsed.append((time.perf_counter_ns() - started) / 1_000_000_000)
    milliseconds = np.asarray(elapsed, dtype=np.float64) * 1000.0
    median_seconds = float(statistics.median(elapsed))
    epochs = int(x.shape[0] * x.shape[1])
    report: dict[str, Any] = {
        "latency_ms_median": float(np.median(milliseconds)),
        "latency_ms_mean": float(np.mean(milliseconds)),
        "latency_ms_std": float(np.std(milliseconds, ddof=1)) if repeats > 1 else 0.0,
        "latency_ms_p95": float(np.quantile(milliseconds, 0.95)),
        "latency_ms_min": float(np.min(milliseconds)),
        "latency_ms_max": float(np.max(milliseconds)),
        "throughput_epochs_per_second": float(epochs / median_seconds),
        "timed_forward_passes": repeats,
        "latency_samples_ms": milliseconds.tolist(),
    }
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        report.update(
            {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": peak_allocated
                - baseline_allocated,
            }
        )
    return report


def _load_pipeline(
    workspace: Path,
    experiment: str,
    fold: int,
    seed: int,
    device: str,
) -> tuple[nn.Module, dict[str, Any]]:
    context = build_context(
        workspace,
        experiment,
        fold,
        seed,
        device,
        smoke=False,
        allow_test_evaluation=True,
        num_workers=0,
        resume=True,
    )
    report = validate_run(workspace, context.run_root)
    if not report.get("passed") or set(report.get("roles", {})) != {
        "validation",
        "test",
    }:
        raise ValueError(f"{experiment}: run artifacts are not Gate-4 complete")
    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    sequence_kind, sequence, sequence_path = _load_sequence_model(context)
    if extractor_kind == "cnn15":
        pipeline: nn.Module = CNN15Pipeline(extractor, sequence, sequence_kind)
    elif extractor_kind == "resnet1d" and sequence_kind == "tcn":
        pipeline = ResNetTCNPipeline(extractor, sequence)
    else:
        raise ValueError(f"unsupported locked pipeline: {extractor_kind}/{sequence_kind}")
    manifest = json.loads(
        (context.run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["extractor_sha256"] != extractor_hash:
        raise ValueError(f"{experiment}: extractor hash mismatch")
    if manifest["sequence_checkpoint_sha256"] != sha256_file(sequence_path):
        raise ValueError(f"{experiment}: sequence checkpoint hash mismatch")
    return pipeline, {
        "experiment_id": experiment,
        "data_variant": context.data_variant,
        "extractor_kind": extractor_kind,
        "sequence_kind": sequence_kind,
        "extractor_sha256": extractor_hash,
        "sequence_checkpoint_sha256": sha256_file(sequence_path),
        "run_manifest_sha256": report["manifest_sha256"],
    }


def _environment(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_compute_capability": list(properties.major_minor)
                if hasattr(properties, "major_minor")
                else [properties.major, properties.minor],
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            }
        )
    return payload


def build_report(
    workspace: Path,
    *,
    mode: str,
    device_name: str,
    fold: int,
    seed: int,
    batch_records: int,
    sequence_length: int,
    warmup: int,
    repeats: int,
    rounds: int,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if mode not in {"parameters", "latency"}:
        raise ValueError("mode must be parameters or latency")
    if fold != 0 or seed != 42:
        raise ValueError("Gate 6 is frozen to fold 00 and seed 42")
    if min(batch_records, sequence_length, repeats, rounds) <= 0 or warmup < 0:
        raise ValueError("invalid benchmark dimensions")
    if mode == "latency" and (
        batch_records,
        sequence_length,
        warmup,
        repeats,
        rounds,
    ) != (1, 100, 20, 100, 3):
        raise ValueError(
            "official latency protocol is fixed to batch=1, length=100, "
            "warmup=20, repeats=100, rounds=3"
        )
    device = torch.device("cpu" if mode == "parameters" else device_name)
    if mode == "latency" and device.type != "cuda":
        raise ValueError("official latency mode requires CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    commit = _git_commit(workspace)
    torch.manual_seed(seed)
    input_cpu = torch.randn(
        batch_records, sequence_length, 1, 3000, dtype=torch.float32
    )
    rng = np.random.default_rng(seed)
    execution_orders = []
    if mode == "parameters":
        execution_orders.append(list(ACTIVE_EXPERIMENTS))
    else:
        for _ in range(rounds):
            execution_orders.append(
                [
                    ACTIVE_EXPERIMENTS[index]
                    for index in rng.permutation(len(ACTIVE_EXPERIMENTS))
                ]
            )
    model_reports: dict[str, Any] = {}
    round_reports: dict[str, list[dict[str, Any]]] = {
        experiment: [] for experiment in ACTIVE_EXPERIMENTS
    }
    for round_index, order in enumerate(execution_orders):
        for experiment in order:
            pipeline, provenance = _load_pipeline(
                workspace, experiment, fold, seed, str(device)
            )
            if experiment not in model_reports:
                model_reports[experiment] = {
                    **provenance,
                    **parameter_report(pipeline),
                }
            if mode == "latency":
                timing = benchmark_forward(
                    pipeline,
                    input_cpu,
                    device,
                    warmup=warmup,
                    repeats=repeats,
                )
                timing["round"] = round_index + 1
                round_reports[experiment].append(timing)
            pipeline.to("cpu")
            del pipeline
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if mode == "latency":
        for experiment in ACTIVE_EXPERIMENTS:
            reports = round_reports[experiment]
            samples = np.asarray(
                [value for report in reports for value in report["latency_samples_ms"]],
                dtype=np.float64,
            )
            model_reports[experiment]["latency"] = {
                "rounds": reports,
                "all_samples_ms_median": float(np.median(samples)),
                "all_samples_ms_p95": float(np.quantile(samples, 0.95)),
                "all_samples_ms_mean": float(np.mean(samples)),
                "all_samples_ms_std": float(np.std(samples, ddof=1)),
                "throughput_epochs_per_second_from_all_sample_median": float(
                    batch_records * sequence_length / (np.median(samples) / 1000.0)
                ),
                "maximum_peak_allocated_bytes": max(
                    report["peak_allocated_bytes"] for report in reports
                ),
                "maximum_peak_reserved_bytes": max(
                    report["peak_reserved_bytes"] for report in reports
                ),
                "maximum_incremental_peak_allocated_bytes": max(
                    report["incremental_peak_allocated_bytes"] for report in reports
                ),
                "total_timed_forward_passes": len(samples),
            }
    benchmark_files = {
        "scripts/benchmark_model_complexity.py": sha256_file(
            workspace / "scripts" / "benchmark_model_complexity.py"
        ),
        "src/sleeptcn/models.py": sha256_file(
            workspace / "src" / "sleeptcn" / "models.py"
        ),
        "src/sleeptcn/test_gate.py": sha256_file(
            workspace / "src" / "sleeptcn" / "test_gate.py"
        ),
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "complete",
        "mode": mode,
        "scope": "locked_checkpoint_end_to_end_forward_excludes_disk_io_preprocessing_and_training",
        "git_commit": commit,
        "benchmark_code_sha256": combined_sha256(benchmark_files),
        "fold": fold,
        "seed": seed,
        "batch_records": batch_records,
        "sequence_length": sequence_length,
        "warmup": warmup,
        "repeats": repeats,
        "rounds": rounds if mode == "latency" else 1,
        "experiment_execution_orders": execution_orders,
        "environment": _environment(device),
        "models": model_reports,
        "interpretation_limits": [
            "Latency compares inference on the same synthetic tensor shape.",
            "Disk I/O, preprocessing, feature-cache serialization and training are excluded.",
            "E2, E3, E4 and E6 share one architecture; timing differences are measurement noise.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mode", choices=("parameters", "latency"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-records", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.workspace,
        mode=args.mode,
        device_name=args.device,
        fold=args.fold,
        seed=args.seed,
        batch_records=args.batch_records,
        sequence_length=args.sequence_length,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
