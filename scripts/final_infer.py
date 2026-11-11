"""Final zero-shot inference on any device (CPU or CUDA) for Amendments A4 and A5.

Tasks
  device-check   reproduce stored telemetry ensemble probabilities of E0_s42 and E4_s42 for two recordings
                 (A5 requirement: within 1e-4, identical argmax); must pass before the other tasks
  a4             E4HP and E4LP (seeds 42, 123; runs/a4/full) on the SHHS1 holdout and telemetry A4 inputs
  a5             E0, E2, E3, E4 (seeds 42, 123; archived checkpoints) on ISRUC subgroup I, channels C4 and F4

Each recording is scored by the ten fold models; the mean of their softmax outputs (float64) is stored with
the argmax. Labels are never read. Finished recordings are skipped, so a task can be restarted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.experiment import build_context  # noqa: E402
from sleeptcn.features import extract_15cnn_features  # noqa: E402
from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.test_gate import _load_extractor, _load_sequence_model  # noqa: E402

RESTORE = ROOT / "runs/holdout_ckpt_restore_20261010"
OUT = ROOT / "runs/final"
FOLDS = 10
A5_ARMS = [("E0", 42), ("E2", 42), ("E3", 42), ("E4", 42), ("E0", 123), ("E2", 123), ("E3", 123), ("E4", 123)]
A4_INPUTS = {"telemetry": (ROOT / "data/processed/telemetry_a4", 44),
             "shhs1_holdout": (ROOT / "data/processed/shhs_holdout_a4", 500)}
ISRUC_ROOT = ROOT / "data/processed/isruc_a5"
CHECK_RECORDS = ("ST7011J", "ST7022J")


class Model:
    def __init__(self, context, device: torch.device, batch_size: int):
        self.extractor_kind, self.extractor, extractor_hash = _load_extractor(context)
        self.sequence_kind, self.sequence, sequence_path = _load_sequence_model(context)
        self.device, self.batch_size = device, batch_size
        if self.extractor_kind == "resnet1d":
            self.extractor = self.extractor.to(device).eval()
        self.sequence = self.sequence.to(device).eval()
        self.hashes = {"extractor_sha256": extractor_hash, "sequence_checkpoint_sha256": sha256_file(sequence_path)}
        self.data_variant = context.data_variant

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.extractor_kind == "cnn15":
            features = extract_15cnn_features(x, self.extractor, device=self.device, batch_size=self.batch_size)
        else:
            parts = [self.extractor.extract_features(
                torch.from_numpy(x[s:s + self.batch_size]).unsqueeze(1).to(self.device)).cpu()
                for s in range(0, len(x), self.batch_size)]
            features = torch.cat(parts).numpy()
        tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).unsqueeze(0).to(self.device)
        if self.sequence_kind == "bilstm":
            logits = self.sequence(tensor, torch.tensor([len(x)], dtype=torch.long))
        else:
            logits = self.sequence(tensor, padding_mask=None)
        p = torch.softmax(logits, dim=-1).squeeze(0).double().cpu().numpy()
        if p.shape != (len(x), 5) or not np.isfinite(p).all():
            raise ValueError("invalid probability matrix")
        return p


def load_x(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return np.ascontiguousarray(z["x"], dtype=np.float32), z["original_epoch_index"].copy()


def save(path: Path, probabilities: np.ndarray, index: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, probabilities=probabilities.astype(np.float32),
             prediction=probabilities.argmax(axis=1).astype(np.int8), original_epoch_index=index,
             metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    temporary.replace(path)


def archived_context(experiment: str, seed: int, fold: int, device: str):
    return build_context(RESTORE.resolve(), experiment, fold, seed, device, smoke=False,
                         allow_test_evaluation=True, num_workers=0, resume=True)


def ensemble(contexts, paths_for_variant, out: Path, metadata: dict, device, batch_size) -> int:
    """Score every recording with all fold models of one arm; returns the number of new files."""
    sums, indices, hashes, paths = {}, {}, [], None
    for fold, context in enumerate(contexts):
        model = Model(context, device, batch_size)
        paths = paths_for_variant(model.data_variant)
        if fold == 0 and all((out / p.name).exists() for p in paths):
            return 0
        hashes.append(model.hashes)
        for path in paths:
            x, index = load_x(path)
            sums[path.name] = sums.get(path.name, 0.0) + model(x)
            indices[path.name] = index
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for path in paths:
        save(out / path.name, sums[path.name] / len(contexts), indices[path.name],
             {**metadata, "folds": len(contexts), "input_sha256": sha256_file(path), "models": hashes})
    return len(paths)


def require_check() -> None:
    report = OUT / "device_check.json"
    if not report.exists() or not json.loads(report.read_text(encoding="utf-8"))["passed"]:
        raise SystemExit("run the device-check task first; it must pass")


def task_device_check(args, device) -> None:
    rows, passed = [], True
    for experiment in ("E0", "E4"):
        contexts = [archived_context(experiment, 42, fold, args.device) for fold in range(FOLDS)]
        stored_root = ROOT / "runs/a3/telemetry/ensemble" / f"{experiment}_s42"
        for key in CHECK_RECORDS:
            total = 0.0
            for context in contexts:
                model = Model(context, device, args.batch_size)
                x, _ = load_x(ROOT / "data/processed/telemetry_a3" / model.data_variant / f"{key}.npz")
                total = total + model(x)
            new = total / FOLDS
            with np.load(stored_root / f"{key}.npz") as z:
                old = z["probabilities"].astype(np.float64)
            difference = float(np.abs(new - old).max())
            same = bool(np.array_equal(new.argmax(1), old.argmax(1)))
            ok = difference < 1e-4 and same
            passed &= ok
            rows.append({"arm": f"{experiment}_s42", "record": key, "max_abs_difference": difference,
                         "identical_argmax": same, "passed": ok})
            print(rows[-1], flush=True)
    report = {"device": args.device, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "rows": rows, "passed": passed}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "device_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("DEVICE CHECK PASSED" if passed else "DEVICE CHECK FAILED")
    if not passed:
        raise SystemExit(1)


def task_a4(args, device) -> None:
    from sleeptcn.amendment_a4 import A4Target, build_a4_context, load_a4_config, training_state

    require_check()
    config, config_sha = load_a4_config(ROOT)
    for cohort, (root, expected) in A4_INPUTS.items():
        if not root.is_dir():
            print(f"A4 {cohort}: inputs not present on this machine ({root}); skipped", flush=True)
            continue
        for arm, spec in config["training"]["arms"].items():
            for seed in spec["seeds"]:
                targets = [A4Target(arm, seed, fold) for fold in range(FOLDS)]
                states = {training_state(ROOT, t) for t in targets}
                if states != {"tested"}:
                    raise SystemExit(f"{arm} seed {seed}: runs are {states}, expected all tested")
                contexts = [build_a4_context(ROOT, config, t, args.device, num_workers=0,
                                             allow_test_evaluation=True, resume=True) for t in targets]

                def paths_for_variant(variant, root=root, expected=expected):
                    paths = sorted((root / variant).glob("*.npz"))
                    if len(paths) != expected:
                        raise SystemExit(f"{root / variant}: expected {expected} inputs, found {len(paths)}")
                    return paths

                started = time.time()
                count = ensemble(contexts, paths_for_variant, OUT / "a4" / cohort / f"{arm}_s{seed}",
                                 {"amendment": "A4", "arm": f"{arm}_s{seed}", "cohort": cohort,
                                  "a4_config_sha256": config_sha, "device": args.device}, device, args.batch_size)
                print(f"A4 {cohort} {arm}_s{seed}: {count} new files in {time.time() - started:.0f} s", flush=True)


def task_a5(args, device) -> None:
    require_check()
    manifest = ISRUC_ROOT / "isruc_preprocess_manifest.json"
    if not manifest.exists():
        raise SystemExit("ISRUC inputs are missing; run scripts/a5_prepare_isruc.py first")
    expected = json.loads(manifest.read_text(encoding="utf-8"))["included_subjects"]
    config_sha = sha256_file(ROOT / "configs/amendment_a5.json")
    for channel in ("C4", "F4"):
        for experiment, seed in A5_ARMS:
            contexts = [archived_context(experiment, seed, fold, args.device) for fold in range(FOLDS)]

            def paths_for_variant(variant, channel=channel):
                paths = sorted((ISRUC_ROOT / channel / variant).glob("*.npz"))
                if len(paths) != len(expected):
                    raise SystemExit(f"{channel}/{variant}: expected {len(expected)} inputs, found {len(paths)}")
                return paths

            started = time.time()
            count = ensemble(contexts, paths_for_variant, OUT / "a5" / channel / f"{experiment}_s{seed}",
                             {"amendment": "A5", "arm": f"{experiment}_s{seed}", "channel": channel,
                              "a5_config_sha256": config_sha, "device": args.device}, device, args.batch_size)
            print(f"A5 {channel} {experiment}_s{seed}: {count} new files in {time.time() - started:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["device-check", "a4", "a5"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    {"device-check": task_device_check, "a4": task_a4, "a5": task_a5}[args.task](args, device)


if __name__ == "__main__":
    main()
