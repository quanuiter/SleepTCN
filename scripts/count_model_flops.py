"""Count forward FLOPs of the E0 and ResNet-1D-TCN pipelines (post hoc, architecture only).

Builds untrained pipelines from the locked configuration, checks the parameter counts against
Table 6 of the manuscript and counts floating-point operations for one 100-epoch sequence with
torch.utils.flop_counter. Multiply-accumulates count as two FLOPs; recurrent cells are counted
analytically because the counter does not see fused LSTM kernels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.features import MANIPULATIONS, MANIPULATION_PREFIX, STAGE_NAMES  # noqa: E402
from sleeptcn.models import EEGResNet1D, SleepCNN  # noqa: E402
from sleeptcn.workflows.model_factory import build_sequence_model  # noqa: E402

spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "benchmark_model_complexity.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def lstm_flops(module: torch.nn.LSTM, steps: int, batch: int) -> int:
    total = 0
    directions = 2 if module.bidirectional else 1
    for layer in range(module.num_layers):
        inp = module.input_size if layer == 0 else module.hidden_size * directions
        per_step = 2 * 4 * module.hidden_size * (inp + module.hidden_size)
        total += per_step * steps * batch * directions
    return total


def count(model: torch.nn.Module, x: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    flops = int(counter.get_total_flops())
    lstm = [m for m in model.modules() if isinstance(m, torch.nn.LSTM)]
    extra = sum(lstm_flops(m, x.shape[1], x.shape[0]) for m in lstm)
    params = int(sum(p.numel() for p in model.parameters()))
    return {"parameters": params, "counted_flops": flops, "lstm_flops_analytic": extra,
            "total_flops": flops + extra, "flops_per_epoch": (flops + extra) / (x.shape[0] * x.shape[1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiments_v2.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "revision" / "model_flops.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(0)
    x = torch.randn(1, args.epochs, 3000)
    keys = [f"{MANIPULATION_PREFIX[m]}_{s}" for m in MANIPULATIONS for s in STAGE_NAMES]
    e0 = bench.CNN15Pipeline({k: SleepCNN() for k in keys}, build_sequence_model(config, "bilstm"), "bilstm")
    e1 = bench.CNN15Pipeline({k: SleepCNN() for k in keys},
                             build_sequence_model(config, "common_tcn", input_dim=75), "tcn")
    e3 = bench.ResNetTCNPipeline(EEGResNet1D(), build_sequence_model(config, "common_tcn", input_dim=128))
    report = {"epochs": args.epochs, "torch": torch.__version__,
              "E0": count(e0, x), "E1": count(e1, x), "E2_E3_E4_E6": count(e3, x)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
