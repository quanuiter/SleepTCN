"""Package one source feature fold, without record identifiers or target data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(fold):
    cache = ROOT / f"data/cache/revision_weighted_campaign_seed123_20261003/fold_{fold:02d}"
    original = json.loads((cache / "private_manifest.json").read_bytes())
    destination = ROOT / f"runs/colab_training_20261004/fold_{fold:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"SleepTCN_Train_Fold{fold:02d}_20261004.zip"
    if archive_path.exists():
        print(json.dumps(json.loads((destination / "bundle_verification.json").read_bytes())), flush=True)
        return
    files, records = {}, []
    # Source code only, no repository configuration, notebook outputs, or datasets.
    paths = sorted((ROOT / "src/sleeptcn").rglob("*.py")) + [
        ROOT / "scripts/run_colab_source_training.py", ROOT / "configs/teacher_revision_weighted_10fold_v1.json"]
    for path in paths:
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for role in ["train", "validation", "test"]:
        entries = [e for e in original["records"] if e["role"] == role]
        for ordinal, entry in enumerate(entries):
            path = Path(entry["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Source cache hash differs")
            with np.load(path, allow_pickle=False) as z:
                features, labels = z["features"], z["labels"]
                if features.shape != (len(labels), 128) or not np.isfinite(features).all():
                    raise ValueError("Invalid feature cache")
                stream = io.BytesIO()
                np.savez_compressed(stream, features=features, labels=labels)
            name = f"data/{role}/{ordinal:03d}.npz"
            files[name] = stream.getvalue()
            records.append({"path": name, "role": role, "epochs": len(labels),
                            "valid_epochs": int((labels >= 0).sum())})
    manifest = {"fold": fold, "seed": 123, "source_specification": original["specification"],
                "original_cache_manifest_sha256": hashlib.sha256((cache / "private_manifest.json").read_bytes()).hexdigest(),
                "train_class_counts": original["train_class_counts"], "records": records,
                "target_data_included": False, "participant_ids_included": False,
                "raw_eeg_included": False, "checkpoint_included": False,
                "backend_policy": "fresh matched pair on CUDA; completed CPU folds remain unchanged",
                "files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}}
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(files) | {"manifest.json"}
    result = {"status": "passed", "fold": fold, "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
              "archive_bytes": archive_path.stat().st_size, "records": len(records),
              "target_data_included": False, "participant_ids_included": False}
    (destination / "bundle_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(6, 10), required=True)
    build(parser.parse_args().fold)
