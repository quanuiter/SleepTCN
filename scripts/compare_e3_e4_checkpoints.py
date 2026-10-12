"""Compare E3 and E4 Sleep-EDF checkpoints (same seed, same fold) against between-seed baselines.

E3 (band-pass, clip, divide by 100) and E4 (band-pass only) share every training setting except the
constant input scale; clipping never activates. Because the ResNet-1D stem is a bias-free convolution
followed by batch normalisation, the factor of 100 is absorbed except through the BN stabilising
constant. This script measures how close the trained weights of the two configurations are.

The checkpoint reader uses only the Python standard library and NumPy, so it runs without PyTorch.
No participant data are read; only model checkpoints.

Usage (from the repository root):
    python scripts/compare_e3_e4_checkpoints.py --seed 42  --out runs/e3_e4_checkpoint_comparison/seed_42.json
    python scripts/compare_e3_e4_checkpoints.py --seed 123 --out runs/e3_e4_checkpoint_comparison/seed_123.json
    (--folds START STOP splits a seed into parts, e.g. seed_42_part1.json; all seed_*.json are summarised)
    python scripts/compare_e3_e4_checkpoints.py --summarise runs/e3_e4_checkpoint_comparison
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("runs/holdout_ckpt_restore_20261010/runs/v2/full")
DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
    "BFloat16Storage": np.uint16,
}


def load_checkpoint(path: Path) -> dict:
    """Read a torch zip checkpoint into nested dicts of NumPy arrays (no PyTorch needed)."""
    archive = zipfile.ZipFile(path)
    root = archive.namelist()[0].split("/")[0]

    class Storage:
        def __init__(self, kind: str, key: str):
            self.kind, self.key = kind, key

    def rebuild(storage, offset, size, stride, *_):
        buffer = np.frombuffer(archive.read(f"{root}/data/{storage.key}"), dtype=DTYPES[storage.kind])
        if len(size) == 0:
            return buffer[offset]
        strides = [s * buffer.itemsize for s in stride]
        return np.lib.stride_tricks.as_strided(buffer[offset:], shape=size, strides=strides).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name.endswith("Storage"):
                return name
            if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return rebuild
            if name == "_rebuild_parameter":
                return lambda data, *_: data
            if name == "OrderedDict":
                return collections.OrderedDict
            try:
                return super().find_class(module, name)
            except Exception:
                return lambda *a, **k: None

        def persistent_load(self, pid):
            kind = pid[1] if isinstance(pid[1], str) else getattr(pid[1], "__name__", str(pid[1]))
            return Storage(kind, pid[2])

    return Unpickler(archive.open(f"{root}/data.pkl")).load()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def weight_keys(state: dict) -> list[str]:
    return [k for k, v in state.items() if isinstance(v, np.ndarray) and v.ndim >= 2]


def layer_cosines(sa: dict, sb: dict) -> list[float]:
    return [cosine(sa[k], sb[k]) for k in weight_keys(sa)]


def compare_seed(root: Path, seed: int, other_seed: int, folds: range) -> dict:
    rows = []
    for fold in folds:
        def path(exp, s, component):
            sub = "resnet1d/best.pt" if component == "encoder" else "sequence/tcn/best.pt"
            return root / exp / f"fold_{fold:02d}" / f"seed_{s}" / "checkpoints" / sub

        enc = {(e, s): load_checkpoint(path(e, s, "encoder")) for e in ("E3", "E4") for s in (seed, other_seed)}
        seq = {e: load_checkpoint(path(e, seed, "sequence")) for e in ("E3", "E4")}
        e3, e4 = enc[("E3", seed)], enc[("E4", seed)]
        s3, s4 = e3["model_state"], e4["model_state"]
        same = layer_cosines(s3, s4)
        cross_e3 = layer_cosines(s3, enc[("E3", other_seed)]["model_state"])
        cross_e4 = layer_cosines(s4, enc[("E4", other_seed)]["model_state"])
        tcn = layer_cosines(seq["E3"]["model_state"], seq["E4"]["model_state"])
        rows.append({
            "fold": fold,
            "best_epoch": {"E3": e3["progress"]["best_epoch"], "E4": e4["progress"]["best_epoch"]},
            "global_steps": {"E3": e3["progress"]["global_steps"], "E4": e4["progress"]["global_steps"]},
            "encoder_stem_cosine": same[0],
            "encoder_median_layer_cosine": float(np.median(same)),
            "encoder_min_layer_cosine": float(np.min(same)),
            "between_seed_median_layer_cosine": {"E3": float(np.median(cross_e3)), "E4": float(np.median(cross_e4))},
            "between_seed_stem_cosine": {"E3": cross_e3[0], "E4": cross_e4[0]},
            "stem_bn_running_var_ratio_E4_over_E3": float(
                np.mean(s4["stem.1.running_var"]) / np.mean(s3["stem.1.running_var"])),
            "stem_bn_running_var_mean_E3": float(np.mean(s3["stem.1.running_var"])),
            "tcn_median_layer_cosine": float(np.median(tcn)),
            "sha256": {f"{e}_{c}": sha256(path(e, seed, c)) for e in ("E3", "E4") for c in ("encoder", "sequence")},
        })
    return {"seed": seed, "between_seed_reference": other_seed, "checkpoint_root": str(root), "folds": rows}


def summarise(directory: Path) -> dict:
    rows = []
    for file in sorted(directory.glob("seed_*.json")):
        data = json.loads(file.read_text())
        for r in data["folds"]:
            rows.append({**r, "seed": data["seed"]})

    def rng(values):
        return [float(np.min(values)), float(np.max(values))]

    same_epoch = {s: sum(r["best_epoch"]["E3"] == r["best_epoch"]["E4"] for r in rows if r["seed"] == s)
                  for s in (42, 123)}
    same_rows = [r for r in rows if r["best_epoch"]["E3"] == r["best_epoch"]["E4"]]
    summary = {
        "pairs": len(rows),
        "same_best_epoch": same_epoch,
        "encoder_stem_cosine_range": rng([r["encoder_stem_cosine"] for r in rows]),
        "encoder_median_layer_cosine_range": rng([r["encoder_median_layer_cosine"] for r in rows]),
        "between_seed_median_layer_cosine_range": rng(
            [v for r in rows for v in r["between_seed_median_layer_cosine"].values()]),
        "between_seed_stem_cosine_range": rng([v for r in rows for v in r["between_seed_stem_cosine"].values()]),
        "tcn_median_layer_cosine_range": rng([r["tcn_median_layer_cosine"] for r in rows]),
        "stem_bn_running_var_ratio_same_epoch_range": rng(
            [r["stem_bn_running_var_ratio_E4_over_E3"] for r in same_rows]),
        "stem_bn_running_var_ratio_all_range": rng([r["stem_bn_running_var_ratio_E4_over_E3"] for r in rows]),
        "stem_bn_running_var_mean_E3_range": rng([r["stem_bn_running_var_mean_E3"] for r in rows]),
        "relative_effect_of_bn_epsilon_E3_max": float(1e-5 / np.min([r["stem_bn_running_var_mean_E3"] for r in rows])),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, choices=(42, 123))
    parser.add_argument("--folds", type=int, nargs=2, default=(0, 10), metavar=("START", "STOP"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summarise", type=Path)
    args = parser.parse_args()
    if args.summarise:
        print(json.dumps(summarise(args.summarise), indent=2))
        return
    other = 123 if args.seed == 42 else 42
    result = compare_seed(args.root, args.seed, other, range(*args.folds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out} ({len(result['folds'])} folds)")


if __name__ == "__main__":
    main()
