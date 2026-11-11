"""Post-hoc CPU timing of whole-night inference with the locked fold-0 seed-42 models.

Times the inference function used for the holdout evaluation (``infer_probabilities``: epoch encoder in
batches of 256 epochs, then the sequence model over the whole night, then softmax) for one fold model
of E0 (15-CNN + BiLSTM) and of E3 (ResNet-1D + TCN; E2, E4, E6 and E6R share this architecture) on the
holdout SHHS1 recording of median length. Disk I/O, preprocessing and model loading are excluded. The
timing is descriptive, depends on the machine and is not part of the confirmatory protocol or of
amendments A1 and A2.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.experiment import build_context  # noqa: E402
from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.shhs_zero_shot import ExternalRecord, infer_probabilities  # noqa: E402
from sleeptcn.test_gate import _load_extractor, _load_sequence_model  # noqa: E402

ARMS = ("E0", "E3")
FOLD, SEED, BATCH_SIZE = 0, 42, 256
WARMUP, ROUNDS = 1, 5


def median_length_record(folder: Path) -> Path:
    lengths = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            lengths.append((int(z["valid_mask"].shape[0]), path.name))
    lengths.sort()
    return folder / lengths[len(lengths) // 2][1]


def load_arm(restore: Path, processed: Path, arm: str, record_name: str) -> dict:
    context = build_context(restore.resolve(), arm, FOLD, SEED, "cpu", smoke=False,
                            allow_test_evaluation=True, num_workers=0, resume=True)
    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    sequence_kind, sequence, sequence_path = _load_sequence_model(context)
    path = processed / context.data_variant / record_name
    with np.load(path, allow_pickle=False) as z:
        x = np.ascontiguousarray(z["x"], dtype=np.float32)
        index = z["original_epoch_index"].copy()
    n = len(x)
    record = ExternalRecord(path=path, input_sha256=sha256_file(path), record_key=path.stem,
                            subject_id=path.stem, role="holdout", variant=context.data_variant, x=x,
                            y=np.zeros(n, dtype=np.int8), valid_mask=np.ones(n, dtype=bool),
                            original_epoch_index=index)
    return {"record": record, "extractor_kind": extractor_kind, "extractor": extractor,
            "sequence": sequence, "sequence_kind": sequence_kind,
            "info": {"data_variant": context.data_variant, "epochs": n,
                     "extractor_sha256": extractor_hash,
                     "sequence_checkpoint_sha256": sha256_file(sequence_path)}}


def run_once(arm: dict) -> float:
    started = time.perf_counter_ns()
    with torch.inference_mode():
        infer_probabilities(arm["record"], arm["extractor_kind"], arm["extractor"], arm["sequence"],
                            arm["sequence_kind"], BATCH_SIZE)
    return (time.perf_counter_ns() - started) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-workspace", type=Path,
                        default=ROOT / "runs/holdout_ckpt_restore_20261010")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/processed/shhs_holdout_v1")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/cpu_benchmark/whole_night_cpu_benchmark.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing benchmark; do not overwrite")
    record_name = median_length_record(args.processed_root / "bandpass_v2").name
    arms = {arm: load_arm(args.restore_workspace, args.processed_root, arm, record_name) for arm in ARMS}
    results = {}
    for threads in args.threads:
        torch.set_num_threads(threads)
        for arm in ARMS:
            for _ in range(WARMUP):
                run_once(arms[arm])
        samples = {arm: [] for arm in ARMS}
        for round_index in range(ROUNDS):
            order = ARMS if round_index % 2 == 0 else ARMS[::-1]
            for arm in order:
                samples[arm].append(run_once(arms[arm]))
        entry = {}
        for arm in ARMS:
            median = statistics.median(samples[arm])
            epochs = arms[arm]["info"]["epochs"]
            entry[arm] = {"seconds_samples": samples[arm], "seconds_median": median,
                          "seconds_min": min(samples[arm]), "seconds_max": max(samples[arm]),
                          "epochs_per_second": epochs / median,
                          "milliseconds_per_epoch": 1000.0 * median / epochs}
        entry["speedup_e3_over_e0"] = entry["E0"]["seconds_median"] / entry["E3"]["seconds_median"]
        results[str(threads)] = entry
        print(threads, {arm: round(entry[arm]["seconds_median"], 3) for arm in ARMS},
              round(entry["speedup_e3_over_e0"], 2), flush=True)
    report = {
        "schema_version": 1,
        "role": "post-hoc descriptive CPU timing; one fold model per arm; machine dependent",
        "scope": "infer_probabilities on one whole night; excludes disk I/O, preprocessing, model loading",
        "record": record_name, "fold": FOLD, "seed": SEED, "encoder_batch_size": BATCH_SIZE,
        "warmup_runs": WARMUP, "timed_runs_per_arm": ROUNDS, "order": "alternating",
        "environment": {"processor": platform.processor(), "platform": platform.platform(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__},
        "arms": {arm: arms[arm]["info"] for arm in ARMS},
        "threads": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
