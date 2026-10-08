"""Allowlisted Colab upload: exact source modules + eight ID-free source samples.

No raw EEG, target data, campaign checkpoints, Git files or credentials included.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/colab_t4_smoke_20261004"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data/cache/revision_weighted_campaign_seed123_20261003/fold_06"
    paths = sorted((cache / "train").glob("*.npz"))[:8]
    assert len(paths) == 8
    arrays = {}
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            arrays[f"features_{i:02}"] = z["features"].copy()
            arrays[f"labels_{i:02}"] = z["labels"].astype(np.int64)
    manifest = json.loads((cache / "private_manifest.json").read_bytes())
    counts = np.asarray(manifest["train_class_counts"], dtype=float)
    arrays["train_class_weights"] = (counts.sum() / (5 * counts)).astype(np.float32)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payloads = {name: (ROOT / name).read_bytes() for name in [
        "src/sleeptcn/__init__.py", "src/sleeptcn/models.py", "src/sleeptcn/training.py",
        "scripts/benchmark_colab_tcn.py"]}
    payloads["sample_records.npz"] = stream.getvalue()
    description = {"purpose": "bounded_source_only_TCN_CUDA_device_diagnostic",
        "not_a_fold_training_campaign": True, "sample_records": 8,
        "raw_eeg_included": False, "participant_ids_included": False,
        "target_data_included": False, "checkpoint_included": False,
        "files": {k: sha(v) for k, v in payloads.items()}}
    payloads["manifest.json"] = json.dumps(description, indent=2).encode()
    destination = OUT / "SleepTCN_T4_Smoke_20261004.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == set(payloads)
        assert archive.testzip() is None
        assert all(archive.read(name) == data for name, data in payloads.items())
    # The unpacked sample is for the local paired benchmark; never mutate cache.
    (OUT / "sample_records.npz").write_bytes(stream.getvalue())
    verification = {"status": "passed", "archive_sha256": sha(destination.read_bytes()),
        "archive_bytes": destination.stat().st_size, "allowlist_verified": True,
        "source_cache_sha256": [sha(path.read_bytes()) for path in paths], "manifest": description}
    (OUT / "bundle_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in verification.items() if k not in ["source_cache_sha256", "manifest"]}, indent=2))


if __name__ == "__main__":
    build()
