"""Prepare source-only feature bundles locally; performs no training."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.revision_campaign import load_restored_pair
from run_revision_weighted_cpu_trial import prepare_cache, RuntimeGuard
from build_colab_training_bundle import build


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    restored = ROOT / "runs/teacher_revision_cpu_20261002/restored_checkpoints"
    guard = RuntimeGuard(18000)
    for fold in [7, 8, 9]:
        encoder, _, hashes = load_restored_pair(restored, "E3", fold, 123)
        cache = ROOT / f"data/cache/revision_weighted_campaign_seed123_20261003/fold_{fold:02d}"
        prepare_cache(encoder, hashes[0], restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json", fold, cache, guard)
        build(fold)
        print(json.dumps({"phase": "source_bundle_ready", "fold": fold}), flush=True)
