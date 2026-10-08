"""Provenance and inference helpers for the separate 2026-10 revision campaigns."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .cpu_followups import confusion_diagnostics, paired_diagnostic_bootstrap
from .metrics import metrics_from_confusion
from .models import EEGResNet1D, SleepTCN
from .preprocessing import sha256_file


def write_once_json(path, data):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_bytes()) != data:
            raise ValueError(f"existing manifest differs: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_restored_pair(root, experiment, fold, seed):
    root = Path(root)
    manifest = json.loads((root / "restoration_manifest.json").read_bytes())
    lookup = {e["git_path"]: e for e in manifest["records"]}
    models, digests = [], []
    variant = {"E3": "filtered_v2", "E4": "bandpass_v2"}[experiment]
    for stage, model in [("resnet1d", EEGResNet1D()), ("sequence/tcn", SleepTCN(input_dim=128))]:
        relative = f"runs/v2/full/{experiment}/fold_{fold:02d}/seed_{seed}/checkpoints/{stage}/best.pt"
        item = lookup[relative]
        path = root / relative
        if not item["audited_checkpoint"] or sha256_file(path) != item["sha256"]:
            raise ValueError("restored checkpoint hash mismatch")
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray,
                                                np.dtype, type(np.dtype(np.uint32))]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"experiment_id": experiment, "outer_fold": fold, "seed": seed,
                    "stage": stage, "data_variant": variant,
                    "config_sha256": sha256_file(root / "configs/experiments_v2.json"),
                    "split_sha256": sha256_file(root / "data/splits/sleepedf_sc_10fold_seed42_v2.json")}
        if any(payload["metadata"].get(k) != v for k, v in expected.items()):
            raise ValueError("restored checkpoint metadata mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        models.append(model.eval())
        digests.append(item["sha256"])
    return (*models, digests)


@torch.inference_mode()
def extract_features(encoder, x, batch_size=64):
    encoder.eval()
    return torch.cat([encoder.extract_features(torch.from_numpy(x[i:i+batch_size]).unsqueeze(1))
                      for i in range(0, len(x), batch_size)]).numpy()


@torch.inference_mode()
def infer_pair(encoder, sequence, x, batch_size=64):
    features = extract_features(encoder, x, batch_size)
    sequence.eval()
    p = torch.softmax(sequence(torch.from_numpy(features).unsqueeze(0), padding_mask=None), -1).squeeze(0).numpy()
    if p.shape != (len(x), 5) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1., atol=1e-6):
        raise ValueError("invalid probabilities")
    return p


def mean_ten(parts):
    if len(parts) != 10 or set(parts) != set(range(10)):
        raise ValueError("require exactly folds 0 through 9; no fold selection")
    shape = parts[0].shape
    if any(p.shape != shape or not np.isfinite(p).all() for p in parts.values()):
        raise ValueError("fold probabilities not aligned")
    total = np.zeros(shape, dtype=np.float64)
    for fold in range(10):
        total += parts[fold]
    return (total / 10).astype(np.float32)


def summarize_pair(left, right, left_name, right_name):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (5, 5):
        raise ValueError("invalid paired subject confusion shape")
    if not np.array_equal(left.sum(-1), right.sum(-1)):
        raise ValueError("paired reference supports differ")
    contrast = paired_diagnostic_bootstrap(left, right)
    difference = confusion_diagnostics(left)["macro_f1"] - confusion_diagnostics(right)["macro_f1"]
    draws = np.random.default_rng(2031).integers(0, len(left), (10000, len(left)), dtype=np.int32)
    contrast["subject_mean_macro_f1"] = {"difference": float(difference.mean()),
        "ci95": np.quantile(difference[draws].mean(1), [.025, .975]).tolist()}
    return {"subjects": len(left), "valid_epochs": int(left.sum()),
            "arms": {name: {"subject_mean_macro_f1": float(confusion_diagnostics(cm)["macro_f1"].mean()),
                             "pooled": metrics_from_confusion(cm.sum(0))}
                     for name, cm in [(left_name, left), (right_name, right)]},
            "contrast": f"{left_name}_minus_{right_name}", "paired_difference": contrast}
