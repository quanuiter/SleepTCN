"""Kiem thu hop dong toan tuyen tren CPU bang du lieu da tien xu ly that.

Day la kiem thu ky thuat, khong phai mot thi nghiem danh gia do chinh xac.
Moi nhanh chi chay mot buoc toi uu nho de phat hien loi hinh dang, nhan,
mat na, gradient va gia tri khong huu han truoc khi thue GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--record-key", default="SC4002E")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cpu_contract_smoke.json"),
    )
    return parser.parse_args()


def assert_finite_gradients(model: nn.Module, name: str) -> None:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise AssertionError(f"{name}: khong co gradient")
    if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise AssertionError(f"{name}: gradient co NaN/Inf")


def one_epoch_model_step(
    model: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    name: str,
) -> tuple[float, list[int]]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_function(logits, targets)
    if not math.isfinite(float(loss.detach())):
        raise AssertionError(f"{name}: loss khong huu han")
    loss.backward()
    assert_finite_gradients(model, name)
    optimizer.step()
    return float(loss.detach()), list(logits.shape)


def one_sequence_step(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    masked_cross_entropy: Callable[..., torch.Tensor],
    name: str,
    *,
    use_lengths: bool,
) -> tuple[float, list[int]]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    if use_lengths:
        lengths = torch.tensor([features.shape[1]], dtype=torch.long)
        logits = model(features, lengths)
    else:
        padding_mask = torch.zeros(features.shape[:2], dtype=torch.bool)
        logits = model(features, padding_mask=padding_mask)
    loss = masked_cross_entropy(logits, targets)
    if not math.isfinite(float(loss.detach())):
        raise AssertionError(f"{name}: loss khong huu han")
    loss.backward()
    assert_finite_gradients(model, name)
    optimizer.step()
    return float(loss.detach()), list(logits.shape)


def ignored_window(y: np.ndarray) -> np.ndarray:
    ignored = np.flatnonzero(y == -1)
    if len(ignored) == 0:
        raise AssertionError("Ban ghi smoke khong co epoch Movement/Unknown")
    center = int(ignored[0])
    start = max(0, center - 4)
    stop = min(len(y), center + 5)
    indices = np.arange(start, stop)
    if len(indices) < 2 or not np.any(y[indices] >= 0):
        raise AssertionError("Cua so smoke khong co du nhan hop le")
    return indices


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    sys.path.insert(0, str(workspace / "src"))

    from sleeptcn.dataset import load_record
    from sleeptcn.features import expected_15cnn_keys, extract_15cnn_features
    from sleeptcn.metrics import compute_metrics
    from sleeptcn.models import BiLSTMSleepNet, EEGResNet1D, SleepCNN, SleepTCN
    from sleeptcn.training import masked_cross_entropy

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    started = time.perf_counter()

    paper = load_record(
        workspace / "data" / "processed" / "paper_raw_v1" / f"{args.record_key}.npz",
        "paper_raw_v1",
    )
    filtered = load_record(
        workspace / "data" / "processed" / "filtered_v2" / f"{args.record_key}.npz",
        "filtered_v2",
    )
    if not np.array_equal(paper.y, filtered.y):
        raise AssertionError("Hai bien the khong con cung truc thoi gian/nhan")
    if not np.array_equal(paper.original_epoch_index, filtered.original_epoch_index):
        raise AssertionError("Hai bien the khong con cung chi so epoch goc")

    valid_indices = np.flatnonzero(paper.valid_mask)[:4]
    if len(valid_indices) != 4:
        raise AssertionError("Ban ghi smoke co it hon 4 epoch hop le")
    raw_x = torch.from_numpy(paper.x[valid_indices, None, :])
    filtered_x = torch.from_numpy(filtered.x[valid_indices, None, :])
    epoch_targets = torch.from_numpy(paper.y[valid_indices].astype(np.int64))

    results: dict[str, object] = {}
    sleep_cnn = SleepCNN()
    loss, shape = one_epoch_model_step(
        sleep_cnn, raw_x, epoch_targets, masked_cross_entropy, "SleepCNN"
    )
    results["e0_sleepcnn_raw"] = {"loss": loss, "logits_shape": shape}

    for variant, x in (("paper_raw_v1", raw_x), ("filtered_v2", filtered_x)):
        resnet = EEGResNet1D()
        loss, shape = one_epoch_model_step(
            resnet, x, epoch_targets, masked_cross_entropy, f"ResNet-{variant}"
        )
        results[f"resnet_{variant}"] = {"loss": loss, "logits_shape": shape}

    sequence_indices = ignored_window(paper.y)
    sequence_x = torch.from_numpy(paper.x[sequence_indices, None, :])
    sequence_targets = torch.from_numpy(
        paper.y[sequence_indices].astype(np.int64)
    ).unsqueeze(0)

    feature_cnns = {key: SleepCNN().eval() for key in expected_15cnn_keys()}
    cnn_features = torch.from_numpy(
        extract_15cnn_features(
            paper.x[sequence_indices], feature_cnns, device="cpu", batch_size=4
        )
    ).unsqueeze(0)
    if cnn_features.shape[-1] != 75:
        raise AssertionError("Dac trung 15CNN khong co 75 chieu")

    bilstm = BiLSTMSleepNet(input_dim=75)
    loss, shape = one_sequence_step(
        bilstm,
        cnn_features,
        sequence_targets,
        masked_cross_entropy,
        "15CNN-BiLSTM",
        use_lengths=True,
    )
    results["e0_15cnn_bilstm"] = {"loss": loss, "logits_shape": shape}

    cnn_tcn = SleepTCN(input_dim=75)
    loss, shape = one_sequence_step(
        cnn_tcn,
        cnn_features,
        sequence_targets,
        masked_cross_entropy,
        "15CNN-TCN",
        use_lengths=False,
    )
    results["e1_15cnn_tcn"] = {
        "loss": loss,
        "logits_shape": shape,
        "receptive_field_epochs": cnn_tcn.receptive_field,
    }

    for experiment, record in (("e2_raw", paper), ("e3_filtered", filtered)):
        extractor = EEGResNet1D().eval()
        sequence_signal = torch.from_numpy(record.x[sequence_indices, None, :])
        with torch.no_grad():
            resnet_features = extractor.extract_features(sequence_signal).unsqueeze(0)
        if resnet_features.shape[-1] != 128:
            raise AssertionError("Dac trung ResNet khong co 128 chieu")
        tcn = SleepTCN(input_dim=128)
        loss, shape = one_sequence_step(
            tcn,
            resnet_features,
            sequence_targets,
            masked_cross_entropy,
            f"ResNet-TCN-{experiment}",
            use_lengths=False,
        )
        results[f"{experiment}_resnet_tcn"] = {
            "loss": loss,
            "logits_shape": shape,
            "receptive_field_epochs": tcn.receptive_field,
        }

    metric_contract = compute_metrics(
        paper.y[sequence_indices],
        np.zeros(len(sequence_indices), dtype=np.int64),
    )

    payload = {
        "status": "pass",
        "purpose": "technical_contract_only_not_model_quality",
        "device": "cpu",
        "seed": 42,
        "torch_version": torch.__version__,
        "record_key": args.record_key,
        "record_epochs": paper.info.epochs,
        "sequence_original_epoch_indices": paper.original_epoch_index[
            sequence_indices
        ].tolist(),
        "sequence_labels": paper.y[sequence_indices].tolist(),
        "ignored_epochs_in_sequence": int(np.sum(paper.y[sequence_indices] == -1)),
        "metric_valid_epochs": metric_contract["n_valid_epochs"],
        "metric_classes_reported": list(metric_contract["per_class"]),
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = args.output
    if not output.is_absolute():
        output = workspace / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
