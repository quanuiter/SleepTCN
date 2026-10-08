"""Restart-safe matched TCN training; same optimization recipe as the first pilot."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .metrics import confusion_matrix_5, metrics_from_confusion
from .models import SleepTCN
from .preprocessing import sha256_file
from .revision_campaign import write_once_json
from .training import collate_feature_sequences, masked_cross_entropy


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_source(model, records, batch_size, device="cpu"):
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    for start in range(0, len(records), batch_size):
        batch = collate_feature_sequences(records[start:start+batch_size])
        logits = model(batch.features.to(device), padding_mask=batch.padding_mask.to(device))
        cm += confusion_matrix_5(batch.targets.numpy().ravel(), logits.argmax(-1).cpu().numpy().ravel())
    return metrics_from_confusion(cm)


def train_arm(train, validation, weights, cfg, folder, identity, device="cpu", runtime_guard=None):
    if not train or not validation or cfg["max_epochs"] < 1 or cfg["patience"] < 1:
        raise ValueError("nonempty training/validation data and positive epoch limits required")
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (5,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("require five finite positive loss weights")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = SleepTCN(input_dim=128).to(device)
    initial = state_digest(model)
    write_once_json(folder / "identity.json", {"identity": identity, "initial_state_sha256": initial,
                                              "loss_weights": list(map(float, weights)), "device": str(device),
                                              "training_config": cfg})
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    latest, best_path = folder / "latest.pt", folder / "best.pt"
    best, best_epoch, stale, done, elapsed, history = -float("inf"), 0, 0, 0, 0., []
    best_state = None
    selection_path = folder / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_bytes())
        if selection["checkpoint_sha256"] != sha256_file(best_path):
            raise ValueError("selected checkpoint changed")
        selected = torch.load(best_path, map_location=device, weights_only=True)
        if selected["identity"] != identity:
            raise ValueError("selected checkpoint provenance changed")
        model.load_state_dict(selected["model_state"])
        return model.eval(), selection
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("resume checkpoint identity changed")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        generator.set_state(saved["loader_generator_state"].cpu())
        torch.set_rng_state(saved["rng_state"].cpu())
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
        best, best_epoch, stale, done = saved["best_score"], saved["best_epoch"], saved["stale_epochs"], saved["epoch"]
        elapsed, history = saved["elapsed_seconds"], saved["history"]
        best_state = saved["best_model_state"]
    for epoch in range(done + 1, cfg["max_epochs"] + 1):
        if stale >= cfg["patience"]:
            break
        tick = time.perf_counter()
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), cfg["batch_size_records"]):
            if runtime_guard is not None:
                runtime_guard({"phase": "training_batch", "arm": folder.name, "epoch": epoch,
                               "saved_epochs": len(history), "history": history})
            batch = collate_feature_sequences([train[i] for i in order[start:start+cfg["batch_size_records"]]])
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.features.to(device), padding_mask=batch.padding_mask.to(device))
            loss = masked_cross_entropy(logits, batch.targets.to(device), weights)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        if runtime_guard is not None:
            runtime_guard({"phase": "validation", "arm": folder.name, "epoch": epoch,
                           "saved_epochs": len(history), "history": history})
        metrics = evaluate_source(model, validation, cfg["batch_size_records"], device)
        score = metrics["macro_f1"]
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        seconds = time.perf_counter() - tick
        elapsed += seconds
        row = {"epoch": epoch, "train_loss_batch_mean": float(np.mean(losses)), "validation_macro_f1": score,
               "best_epoch": best_epoch, "stale_epochs": stale, "seconds": seconds,
               "batch_order_sha256": hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()}
        history.append(row)
        saved = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "best_model_state": best_state, "identity": identity, "epoch": epoch,
                 "best_score": best, "best_epoch": best_epoch, "stale_epochs": stale,
                 "history": history, "elapsed_seconds": elapsed, "rng_state": torch.get_rng_state(),
                 "loader_generator_state": generator.get_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []}
        torch.save(saved, folder / "latest.tmp")
        (folder / "latest.tmp").replace(latest)
        print(f"{folder.parent.name}/{folder.name}: epoch {epoch}, val={score:.5f}, stale={stale}/{cfg['patience']}, {seconds:.1f}s", flush=True)
        if runtime_guard is not None:
            runtime_guard({"phase": "epoch_saved", "arm": folder.name, "epoch": epoch,
                           "saved_epochs": len(history), "history": history})
    if best_state is None:
        raise ValueError("no selected state")
    model.load_state_dict(best_state)
    selected = {"model_state": best_state, "identity": identity, "epoch": best_epoch,
                "validation_macro_f1": best, "initial_state_sha256": initial}
    if not best_path.exists():
        torch.save(selected, folder / "best.tmp")
        (folder / "best.tmp").replace(best_path)
    else:
        # After an interruption following final save, validate rather than overwrite.
        previous = torch.load(best_path, map_location=device, weights_only=True)
        if previous["identity"] != identity or previous["epoch"] != best_epoch:
            raise ValueError("existing best checkpoint differs")
        for key in best_state:
            if not torch.equal(previous["model_state"][key], best_state[key]):
                raise ValueError("existing best weights differ")
    selection = {"epochs_completed": len(history), "selected_epoch": best_epoch,
                 "selected_validation_macro_f1": best, "initial_state_sha256": initial,
                 "checkpoint_sha256": sha256_file(best_path), "checkpoint_path": str(best_path.resolve()),
                 "training_and_validation_seconds": elapsed, "loss_weights": weights.cpu().tolist(),
                 "history": history}
    write_once_json(selection_path, selection)
    return model.eval(), selection
