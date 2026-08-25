"""Vong huan luyen co checkpoint, khong truy cap tap test."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .metrics import compute_metrics
from .training import PaddedBatch


SCHEMA_VERSION = 1
ForwardBatch = Callable[
    [nn.Module, Any, torch.device], tuple[torch.Tensor, torch.Tensor]
]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainingProgress:
    completed_epochs: int = 0
    global_steps: int = 0
    validation_events: int = 0
    best_metric: float = float("-inf")
    best_epoch: int = -1
    bad_validations: int = 0


@dataclass(frozen=True)
class EpochSummary:
    loss: float
    valid_epochs: int
    batches: int
    metrics: dict[str, object]


@dataclass(frozen=True)
class FitResult:
    progress: TrainingProgress
    stopped_early: bool
    best_checkpoint: Path
    latest_checkpoint: Path
    history: tuple[dict[str, object], ...]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def epoch_forward(
    model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    return model(inputs), targets


def sequence_forward(kind: str) -> ForwardBatch:
    if kind not in {"bilstm", "tcn"}:
        raise ValueError(f"unknown sequence model kind: {kind}")

    def forward(
        model: nn.Module, batch: PaddedBatch, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = batch.features.to(device, non_blocking=True)
        targets = batch.targets.to(device, non_blocking=True)
        if kind == "bilstm":
            logits = model(features, batch.lengths)
        else:
            logits = model(
                features,
                padding_mask=batch.padding_mask.to(device, non_blocking=True),
            )
        return logits, targets

    return forward


def _valid_target_mask(targets: torch.Tensor) -> torch.Tensor:
    return (targets >= 0) & (targets < 5)


def run_loader(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device | str,
    forward_batch: ForwardBatch,
    loss_function: LossFunction,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> EpochSummary:
    """Chay mot luot train hoac evaluation va chi tinh tren nhan 0..4."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    training = optimizer is not None
    device = torch.device(device)
    model.train(training)
    total_loss = 0.0
    total_valid = 0
    batches = 0
    true_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, targets = forward_batch(model, batch, device)
            if logits.shape[:-1] != targets.shape or logits.shape[-1] != 5:
                raise ValueError("forward batch returned incompatible logits/targets")
            valid = _valid_target_mask(targets)
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                raise ValueError("batch has no valid target")
            loss = loss_function(logits, targets)
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise FloatingPointError("loss is not a finite scalar")
            if training:
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradients:
                    raise RuntimeError("training step produced no gradient")
                if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
                    raise FloatingPointError("gradient contains NaN or infinity")
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            total_loss += float(loss.detach()) * valid_count
            total_valid += valid_count
            batches += 1
            true_parts.append(targets[valid].detach().cpu().numpy())
            predicted_parts.append(logits.argmax(dim=-1)[valid].detach().cpu().numpy())
    if total_valid == 0:
        raise ValueError("loader yielded no valid target")
    true = np.concatenate(true_parts)
    predicted = np.concatenate(predicted_parts)
    return EpochSummary(
        loss=total_loss / total_valid,
        valid_epochs=total_valid,
        batches=batches,
        metrics=compute_metrics(true, predicted),
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _validate_hash(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: TrainingProgress,
    *,
    experiment_id: str,
    stage: str,
    outer_fold: int,
    seed: int,
    config_sha256: str,
    split_sha256: str,
    data_variant: str,
    selection_metric: str,
    loader_generator: torch.Generator | None,
    history: list[dict[str, object]],
) -> dict[str, Any]:
    _validate_hash(config_sha256, "config_sha256")
    _validate_hash(split_sha256, "split_sha256")
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "experiment_id": experiment_id,
            "stage": stage,
            "outer_fold": outer_fold,
            "seed": seed,
            "config_sha256": config_sha256,
            "split_sha256": split_sha256,
            "data_variant": data_variant,
            "model_class": type(model).__name__,
            "selection_metric": selection_metric,
        },
        "progress": asdict(progress),
        "history": history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "loader_generator_state": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
    }


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_and_validate_payload(
    path: Path, expected_metadata: dict[str, Any]
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    metadata = payload.get("metadata", {})
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint metadata mismatch: {mismatches}")
    return payload


def load_model_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    expected_metadata: dict[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """Nap trong so de suy luan, khong phuc hoi optimizer/RNG."""
    payload = _load_and_validate_payload(path, expected_metadata)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(torch.device(device))
    return payload["metadata"]


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_metadata: dict[str, Any],
    device: torch.device | str,
    loader_generator: torch.Generator | None = None,
    restore_rng: bool = True,
) -> TrainingProgress:
    payload = _load_and_validate_payload(path, expected_metadata)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    device = torch.device(device)
    model.to(device)
    _optimizer_to(optimizer, device)
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    generator_state = payload.get("loader_generator_state")
    if generator_state is not None:
        if loader_generator is None:
            raise ValueError("checkpoint needs a loader_generator to resume exactly")
        loader_generator.set_state(generator_state)
    return TrainingProgress(**payload["progress"])


def fit_model(
    model: nn.Module,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    loss_function: LossFunction,
    forward_batch: ForwardBatch,
    *,
    device: torch.device | str,
    max_epochs: int,
    patience: int,
    checkpoint_dir: Path,
    experiment_id: str,
    stage: str,
    outer_fold: int,
    seed: int,
    config_sha256: str,
    split_sha256: str,
    data_variant: str,
    selection_metric: str = "validation_macro_f1",
    gradient_clip_norm: float | None = None,
    loader_generator: torch.Generator | None = None,
    resume_from: Path | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> FitResult:
    """Fit chi nhan train va validation loader; tap test khong co trong API."""
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    if selection_metric not in {"validation_macro_f1", "validation_loss"}:
        raise ValueError("unsupported selection_metric")
    device = torch.device(device)
    model.to(device)
    metadata = {
        "experiment_id": experiment_id,
        "stage": stage,
        "outer_fold": outer_fold,
        "seed": seed,
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "data_variant": data_variant,
        "model_class": type(model).__name__,
        "selection_metric": selection_metric,
    }
    progress = TrainingProgress()
    history: list[dict[str, object]] = []
    if resume_from is not None:
        resume_payload = _load_and_validate_payload(resume_from, metadata)
        progress = load_checkpoint(
            resume_from,
            model,
            optimizer,
            expected_metadata=metadata,
            device=device,
            loader_generator=loader_generator,
        )
        stored_history = resume_payload.get("history", [])
        if not isinstance(stored_history, list):
            raise ValueError("checkpoint history must be a list")
        history = list(stored_history)
    if progress.completed_epochs > max_epochs:
        raise ValueError("checkpoint is beyond requested max_epochs")
    checkpoint_dir = checkpoint_dir.resolve()
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    stopped_early = False

    for epoch in range(progress.completed_epochs, max_epochs):
        train_summary = run_loader(
            model,
            train_loader,
            device,
            forward_batch,
            loss_function,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
            max_batches=max_train_batches,
        )
        progress.global_steps += train_summary.batches
        validation_summary = run_loader(
            model,
            validation_loader,
            device,
            forward_batch,
            loss_function,
            max_batches=max_validation_batches,
        )
        metric = (
            float(validation_summary.metrics["macro_f1"])
            if selection_metric == "validation_macro_f1"
            else validation_summary.loss
        )
        progress.completed_epochs = epoch + 1
        progress.validation_events += 1
        improved = progress.best_epoch < 0 or (
            metric > progress.best_metric
            if selection_metric == "validation_macro_f1"
            else metric < progress.best_metric
        )
        if improved:
            progress.best_metric = metric
            progress.best_epoch = epoch
            progress.bad_validations = 0
        else:
            progress.bad_validations += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_summary.loss,
                "train_metrics": train_summary.metrics,
                "validation_loss": validation_summary.loss,
                "train_macro_f1": float(train_summary.metrics["macro_f1"]),
                "validation_metrics": validation_summary.metrics,
                "validation_macro_f1": float(validation_summary.metrics["macro_f1"]),
                "selection_metric": selection_metric,
                "selection_value": metric,
                "improved": improved,
            }
        )
        payload = _checkpoint_payload(
            model,
            optimizer,
            progress,
            experiment_id=experiment_id,
            stage=stage,
            outer_fold=outer_fold,
            seed=seed,
            config_sha256=config_sha256,
            split_sha256=split_sha256,
            data_variant=data_variant,
            selection_metric=selection_metric,
            loader_generator=loader_generator,
            history=history,
        )
        if improved:
            save_checkpoint_atomic(best_path, payload)
        save_checkpoint_atomic(latest_path, payload)
        if progress.bad_validations >= patience:
            stopped_early = True
            break
    if not best_path.is_file() or not latest_path.is_file():
        raise AssertionError("fit completed without required checkpoints")
    return FitResult(
        progress=progress,
        stopped_early=stopped_early,
        best_checkpoint=best_path,
        latest_checkpoint=latest_path,
        history=tuple(history),
    )
