"""Amendment A3, step 2: zero-shot inference with the locked checkpoints on CPU.

``--cohort telemetry`` applies the ten fold models of every arm to every telemetry recording and stores
the mean of their softmax outputs (float64 mean, then stored as float32) with the per-fold probabilities
discarded. ``--cohort sleep_edf_oof`` applies each fold model to the Sleep-EDF recordings of its own test
fold, which reproduces the out-of-fold predictions needed for the source cost of the amplitude rule.
Labels are not read. See docs/AMENDMENT_A3.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.experiment import build_context  # noqa: E402
from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.shhs_zero_shot import ExternalRecord, infer_probabilities  # noqa: E402
from sleeptcn.test_gate import _load_extractor, _load_sequence_model  # noqa: E402

CONFIG = json.loads((ROOT / "configs/amendment_a3.json").read_text(encoding="utf8"))
SPLIT = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
FOLDS = 10


def load_model(restore: Path, experiment: str, seed: int, fold: int):
    context = build_context(restore.resolve(), experiment, fold, seed, "cpu", smoke=False,
                            allow_test_evaluation=True, num_workers=0, resume=True)
    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    sequence_kind, sequence, sequence_path = _load_sequence_model(context)
    hashes = {"extractor_sha256": extractor_hash, "sequence_checkpoint_sha256": sha256_file(sequence_path)}
    return context.data_variant, (extractor_kind, extractor, sequence, sequence_kind), hashes


def predict(model, path: Path, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    extractor_kind, extractor, sequence, sequence_kind = model
    with np.load(path, allow_pickle=False) as z:
        x = np.ascontiguousarray(z["x"], dtype=np.float32)
        index = z["original_epoch_index"].copy()
    n = len(x)
    record = ExternalRecord(path=path, input_sha256="", record_key=path.stem, subject_id=path.stem[:5],
                            role="a3", variant=path.parent.name, x=x, y=np.zeros(n, dtype=np.int8),
                            valid_mask=np.ones(n, dtype=bool), original_epoch_index=index)
    with torch.inference_mode():
        probabilities = infer_probabilities(record, extractor_kind, extractor, sequence, sequence_kind, batch_size)
    return probabilities.astype(np.float64), index


def save(path: Path, probabilities: np.ndarray, index: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, probabilities=probabilities.astype(np.float32),
             prediction=probabilities.argmax(axis=1).astype(np.int8), original_epoch_index=index,
             metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    temporary.replace(path)


def run_telemetry(args, arm: dict) -> None:
    processed = ROOT / CONFIG["cohorts"]["sleep_edf_telemetry"]["processed_root"]
    out = args.output_root / "telemetry" / "ensemble" / arm["arm"]
    sums: dict[str, np.ndarray] = {}
    indices: dict[str, np.ndarray] = {}
    hashes = []
    paths = None
    for fold in range(FOLDS):
        variant, model, model_hashes = load_model(args.restore_workspace, arm["experiment"], arm["seed"], fold)
        hashes.append(model_hashes)
        paths = sorted((processed / variant).glob("*.npz"))
        if len(paths) != 44:
            raise ValueError(f"{variant}: expected 44 telemetry recordings, found {len(paths)}")
        if all((out / path.name).exists() for path in paths):
            return
        for path in paths:
            probabilities, index = predict(model, path, args.batch_size)
            sums[path.name] = sums.get(path.name, 0.0) + probabilities
            indices[path.name] = index
        print(arm["arm"], "fold", fold, flush=True)
    for path in paths:
        save(out / path.name, sums[path.name] / FOLDS, indices[path.name],
             {"amendment": "A3", "arm": arm["arm"], "cohort": "sleep_edf_telemetry", "folds": FOLDS,
              "input_sha256": sha256_file(path), "models": hashes})


def run_oof(args, arm: dict) -> None:
    split = json.loads(SPLIT.read_text(encoding="utf8"))
    out = args.output_root / "sleep_edf_oof" / arm["arm"]
    for fold in range(FOLDS):
        keys = split["folds"][fold]["record_keys"]
        if all((out / f"{key}.npz").exists() for key in keys):
            continue
        variant, model, model_hashes = load_model(args.restore_workspace, arm["experiment"], arm["seed"], fold)
        for key in keys:
            path = ROOT / "data/processed" / variant / f"{key}.npz"
            probabilities, index = predict(model, path, args.batch_size)
            save(out / f"{key}.npz", probabilities, index,
                 {"amendment": "A3", "arm": arm["arm"], "cohort": "sleep_edf_oof", "test_fold": fold,
                  "input_sha256": sha256_file(path), "model": model_hashes})
        print(arm["arm"], "fold", fold, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["telemetry", "sleep_edf_oof"], required=True)
    parser.add_argument("--restore-workspace", type=Path, default=ROOT / CONFIG["checkpoint_workspace"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/a3")
    parser.add_argument("--batch-size", type=int, default=CONFIG["encoder_batch_size"])
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    for arm in CONFIG["arms"]:
        if args.arms and arm["arm"] not in args.arms:
            continue
        (run_telemetry if args.cohort == "telemetry" else run_oof)(args, arm)
    print("DONE", args.cohort)


if __name__ == "__main__":
    main()
