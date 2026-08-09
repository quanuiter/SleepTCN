"""Report parameters and reproducible forward-pass latency on one device."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.models import BiLSTMSleepNet, EEGResNet1D, SleepCNN, SleepTCN


class CNN15Pipeline(nn.Module):
    def __init__(self, sequence_kind: str) -> None:
        super().__init__()
        self.extractors = nn.ModuleList([SleepCNN() for _ in range(15)])
        self.sequence = (
            BiLSTMSleepNet()
            if sequence_kind == "bilstm"
            else SleepTCN(input_dim=75)
        )
        self.sequence_kind = sequence_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length = x.shape[:2]
        flat = x.reshape(batch * length, 1, 3000)
        features = torch.cat(
            [torch.softmax(model(flat), dim=-1) for model in self.extractors], dim=-1
        ).reshape(batch, length, 75)
        if self.sequence_kind == "bilstm":
            lengths = torch.full((batch,), length, dtype=torch.long, device=x.device)
            return self.sequence(features, lengths)
        return self.sequence(features)


class ResNetTCNPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = EEGResNet1D()
        self.sequence = SleepTCN(input_dim=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length = x.shape[:2]
        features = self.extractor.extract_features(
            x.reshape(batch * length, 1, 3000)
        ).reshape(batch, length, 128)
        return self.sequence(features)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    model = model.to(device).eval()
    x = x.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        synchronize(device)
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(x)
            synchronize(device)
            elapsed.append(time.perf_counter() - started)
    milliseconds = np.asarray(elapsed) * 1000.0
    epochs = int(x.shape[0] * x.shape[1])
    report: dict[str, float | int] = {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "latency_ms_median": float(statistics.median(milliseconds.tolist())),
        "latency_ms_p95": float(np.quantile(milliseconds, 0.95)),
        "throughput_epochs_per_second": float(
            epochs / statistics.median(elapsed)
        ),
    }
    if device.type == "cuda":
        report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-records", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_records, args.sequence_length, args.repeats) <= 0 or args.warmup < 0:
        raise ValueError("invalid benchmark size")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    x = torch.randn(args.batch_records, args.sequence_length, 1, 3000)
    model_factories = {
        "E0_15cnn_bilstm": lambda: CNN15Pipeline("bilstm"),
        "E1_15cnn_tcn": lambda: CNN15Pipeline("tcn"),
        "E2_to_E6_resnet_tcn": ResNetTCNPipeline,
    }
    model_reports = {}
    for name, factory in model_factories.items():
        model = factory()
        model_reports[name] = benchmark(
            model, x, device, args.warmup, args.repeats
        )
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema_version": 1,
        "scope": "end_to_end_forward_only_excludes_preprocessing_and_training",
        "device": str(device),
        "batch_records": args.batch_records,
        "sequence_length": args.sequence_length,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "models": model_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
