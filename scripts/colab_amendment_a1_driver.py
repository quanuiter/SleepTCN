"""Colab GPU execution of amendment A1 conditions on the label-free, identifier-free package.

Same computation as scripts/run_holdout_amendment_a1.py (``predict`` and ``adapt_batchnorm``), moved to
CUDA. Units run in reverse protocol order so that they do not overlap with the local CPU workers, which
run in forward order. Outputs per unit: probabilities for every record index, plus the AdaBN state of
each fold. No labels exist in this package.

Usage in Colab (after extracting all parts into WORK):
    python colab_amendment_a1_driver.py --work /content/a1 --out /content/a1_results [--max-units N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

FOLDS = tuple(range(10))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_models(work: Path, fold: int, device: str):
    sys.path.insert(0, str(work / "src"))
    from sleeptcn.models import EEGResNet1D, SleepTCN

    tcn_cfg = json.loads((work / "configs" / "experiments_v2.json").read_bytes())["components"]["common_tcn"]
    resnet_path = work / "checkpoints" / f"fold_{fold:02d}" / "resnet1d.pt"
    tcn_path = work / "checkpoints" / f"fold_{fold:02d}" / "tcn.pt"
    extractor = EEGResNet1D()
    extractor.load_state_dict(torch.load(resnet_path, map_location="cpu", weights_only=False)["model_state"], strict=True)
    sequence = SleepTCN(input_dim=128, hidden_dim=int(tcn_cfg["hidden_dim"]), kernel_size=int(tcn_cfg["kernel_size"]),
                        n_blocks=int(tcn_cfg["residual_blocks"]), dropout=float(tcn_cfg["dropout"]))
    sequence.load_state_dict(torch.load(tcn_path, map_location="cpu", weights_only=False)["model_state"], strict=True)
    hashes = {"extractor_sha256": sha_file(resnet_path), "sequence_checkpoint_sha256": sha_file(tcn_path)}
    return extractor.eval(), sequence.eval(), hashes


@torch.no_grad()
def predict(extractor, sequence, x: np.ndarray, factor: float, batch_size: int, device: str) -> np.ndarray:
    parts = []
    for start in range(0, len(x), batch_size):
        signals = torch.from_numpy(x[start:start + batch_size]).unsqueeze(1).to(device)
        if factor != 1.0:
            signals = signals * factor
        parts.append(extractor.extract_features(signals).cpu().numpy())
    features = np.concatenate(parts).astype(np.float32, copy=False)
    logits = sequence(torch.from_numpy(features).unsqueeze(0).to(device), padding_mask=None)
    p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
    if p.shape != (len(x), 5) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("invalid probability matrix")
    return p


class _Stop(Exception):
    pass


def batchnorm_order_cpu(model: nn.Module) -> list[str]:
    order: list[str] = []
    handles = [m.register_forward_pre_hook(lambda _m, _i, name=name: order.append(name))
               for name, m in model.named_modules() if isinstance(m, nn.BatchNorm1d)]
    try:
        with torch.no_grad():
            model.eval().extract_features(torch.zeros(2, 1, 3000))
    finally:
        for h in handles:
            h.remove()
    return list(dict.fromkeys(order))


@torch.no_grad()
def adapt_batchnorm(model: nn.Module, order: list[str], sample: np.ndarray, batch_size: int, device: str):
    model.eval()
    modules = dict(model.named_modules())
    state = {}
    for name in order:
        bn = modules[name]
        acc = {"n": 0, "s": None, "ss": None}

        def hook(_m, inputs, acc=acc):
            v = inputs[0].double()
            total, squares = v.sum((0, 2)), (v * v).sum((0, 2))
            acc["n"] += v.shape[0] * v.shape[2]
            acc["s"] = total if acc["s"] is None else acc["s"] + total
            acc["ss"] = squares if acc["ss"] is None else acc["ss"] + squares
            raise _Stop

        handle = bn.register_forward_pre_hook(hook)
        try:
            for start in range(0, len(sample), batch_size):
                try:
                    model.extract_features(torch.from_numpy(sample[start:start + batch_size]).unsqueeze(1).to(device))
                except _Stop:
                    pass
        finally:
            handle.remove()
        mean = acc["s"] / acc["n"]
        var = (acc["ss"] / acc["n"] - mean * mean).clamp_min(0.0)
        bn.running_mean.copy_(mean.to(bn.running_mean.dtype))
        bn.running_var.copy_(var.to(bn.running_var.dtype))
        state[name] = {"running_mean": bn.running_mean.cpu().numpy().copy(),
                       "running_var": bn.running_var.cpu().numpy().copy()}
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-units", type=int, default=50)
    args = parser.parse_args()
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    work, out = args.work, args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((work / "package_manifest.json").read_bytes())
    amendment = json.loads((work / "configs" / "shhs_holdout_amendment_a1.json").read_bytes())
    assert sha_file(work / "configs" / "shhs_holdout_amendment_a1.json") == manifest["amendment_sha256"]
    assert not manifest["labels_included"] and not manifest["participant_ids_included"]
    conditions = amendment["conditions"]
    records = manifest["records"]
    runtime = {"device": device, "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
               "deterministic_algorithms": True}

    def load_x(idx: int) -> np.ndarray:
        return np.load(work / "records" / f"r{idx:03d}.npy", allow_pickle=False)

    # Reproduction check against stored parent CPU probabilities (scale 1.0).
    reference = np.load(work / "gate_reference.npz")
    gate = []
    for fold in (0, 9):
        e, s, _ = load_models(work, fold, device)
        e, s = e.to(device), s.to(device)
        for idx in (0, 1):
            p = predict(e, s, load_x(idx), 1.0, args.batch_size, device)
            ref = reference[f"p_f{fold:02d}_r{idx:03d}"]
            gate.append({"fold": fold, "idx": idx, "max_abs_difference": float(np.abs(p - ref).max()),
                         "argmax_agreement": float((p.argmax(1) == ref.argmax(1)).mean())})
    print("GPU reproduction check:", json.dumps(gate), flush=True)
    assert all(g["max_abs_difference"] <= 1e-4 and g["argmax_agreement"] == 1.0 for g in gate), "reproduction failed"
    (out / "colab_gate.json").write_text(json.dumps({"runtime": runtime, "checks": gate}, indent=2))

    units = [(c, f) for f in FOLDS for c in conditions][::-1][: args.max_units]
    sample = None
    started = time.time()
    for n, (condition, fold) in enumerate(units, start=1):
        path = out / f"unit_{condition}_f{fold:02d}.npz"
        if path.exists():
            continue
        spec = conditions[condition]
        extractor, sequence, hashes = load_models(work, fold, device)
        meta = {"condition": condition, "outer_fold": fold, "input_factor": float(spec["factor"]), **hashes, **runtime,
                "amendment_sha256": manifest["amendment_sha256"]}
        order = batchnorm_order_cpu(extractor)
        extractor, sequence = extractor.to(device), sequence.to(device)
        if spec["type"] == "adabn":
            if sample is None:
                sample = np.load(work / "adabn_sample_x.npy", allow_pickle=False)
            state = adapt_batchnorm(extractor, order, sample, args.batch_size, device)
            arrays = {"layer_names": np.asarray(list(state)), "adabn_sample_sha256": np.array(manifest["adabn_sample_sha256"])}
            for name, values in state.items():
                arrays[f"{name}.running_mean"] = values["running_mean"]
                arrays[f"{name}.running_var"] = values["running_var"]
            np.savez(out / f"adabn_state_f{fold:02d}.npz", **arrays)
            meta["adabn_state_file"] = f"adabn_state_f{fold:02d}.npz"
        probs = {}
        for r in records:
            x = load_x(r["idx"])
            assert len(x) == r["n_epochs"]
            probs[f"p{r['idx']:03d}"] = predict(extractor, sequence, x, float(spec["factor"]), args.batch_size, device)
        tmp = out / f"unit_{condition}_f{fold:02d}.tmp.npz"
        np.savez(tmp, meta_json=np.array(json.dumps(meta, sort_keys=True)), **probs)
        tmp.replace(path)
        print(f"[{n}/{len(units)}] {condition} fold {fold:02d} done ({time.time() - started:.0f}s)", flush=True)
    print("ALL REQUESTED UNITS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
