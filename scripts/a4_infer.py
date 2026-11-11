"""Amendment A4: zero-shot CPU inference of the trained filter-edge arms on the holdout and telemetry.

Loads the ten fold models of every A4 arm from runs/a4/full (after the GPU results are unpacked), applies
them to the A4 input variant of each recording and stores the mean of their softmax outputs. Labels are
not read. See docs/AMENDMENT_A4.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.amendment_a4 import A4Target, build_a4_context, load_a4_config, training_state  # noqa: E402
from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.test_gate import _load_extractor, _load_sequence_model  # noqa: E402

spec = importlib.util.spec_from_file_location("a3_infer", ROOT / "scripts/a3_infer.py")
a3_infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3_infer)

INPUTS = {"shhs1_holdout": (ROOT / "data/processed/shhs_holdout_a4", 500),
          "telemetry": (ROOT / "data/processed/telemetry_a4", 44)}
FOLDS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=sorted(INPUTS), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    config, config_sha = load_a4_config(ROOT)
    root, expected = INPUTS[args.cohort]
    for arm, arm_spec in config["training"]["arms"].items():
        for seed in arm_spec["seeds"]:
            name = f"{arm}_s{seed}"
            out = ROOT / "runs/a4/zero_shot" / args.cohort / "ensemble" / name
            paths = sorted((root / arm_spec["data_variant"]).glob("*.npz"))
            if len(paths) != expected:
                raise ValueError(f"{name}: expected {expected} inputs, found {len(paths)}")
            if all((out / p.name).exists() for p in paths):
                continue
            sums, indices, hashes = {}, {}, []
            for fold in range(FOLDS):
                target = A4Target(arm, seed, fold)
                if training_state(ROOT, target) != "tested":
                    raise ValueError(f"{target.key}: run is not trained and tested")
                context = build_a4_context(ROOT, config, target, "cpu", num_workers=0,
                                           allow_test_evaluation=True, resume=True)
                extractor_kind, extractor, extractor_hash = _load_extractor(context)
                sequence_kind, sequence, sequence_path = _load_sequence_model(context)
                hashes.append({"extractor_sha256": extractor_hash,
                               "sequence_checkpoint_sha256": sha256_file(sequence_path)})
                model = (extractor_kind, extractor, sequence, sequence_kind)
                for path in paths:
                    probabilities, index = a3_infer.predict(model, path, args.batch_size)
                    sums[path.name] = sums.get(path.name, 0.0) + probabilities
                    indices[path.name] = index
                print(name, "fold", fold, flush=True)
            for path in paths:
                a3_infer.save(out / path.name, sums[path.name] / FOLDS, indices[path.name],
                              {"amendment": "A4", "arm": name, "cohort": args.cohort, "folds": FOLDS,
                               "a4_config_sha256": config_sha, "input_sha256": sha256_file(path),
                               "models": hashes})
    print("DONE", args.cohort)


if __name__ == "__main__":
    main()
