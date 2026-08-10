"""Kiểm tra workspace trước CPU smoke hoặc GPU smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy
import pyedflib
import scipy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.dataset import inspect_record
from sleeptcn.splits import validate_split_structure


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    errors: list[str] = []
    if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
        errors.append("unsupported_python_version_expected_3.10_or_3.11")

    split_path = root / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    sidecar_path = split_path.with_suffix(split_path.suffix + ".sha256")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split_errors = validate_split_structure(split_manifest)
    errors.extend(f"split:{error}" for error in split_errors)
    expected_split_hash = sidecar_path.read_text(encoding="ascii").split()[0]
    actual_split_hash = sha256_file(split_path)
    if expected_split_hash != actual_split_hash:
        errors.append("split_manifest_sha256")

    processed_validation = json.loads(
        (root / "data/manifests/processed_validation_v2.json").read_text(
            encoding="utf-8"
        )
    )
    if processed_validation["summary"]["files_with_errors"] != 0:
        errors.append("processed_validation_file_errors")
    if processed_validation["summary"]["global_errors"]:
        errors.append("processed_validation_global_errors")

    experiment_config = json.loads(
        (root / "configs/experiments_v2.json").read_text(encoding="utf-8")
    )
    expected_experiments = {f"E{index}" for index in range(7)}
    if set(experiment_config["experiments"]) != expected_experiments:
        errors.append("experiment_registry")
    if experiment_config["dataset"]["ignored_label"] != -1:
        errors.append("ignored_label_config")
    if experiment_config["dataset"]["padding_label"] != -100:
        errors.append("padding_label_config")

    sample_results = {}
    for variant in (
        "paper_raw_v1",
        "bandpass_v2",
        "bandpass_clip_v2",
        "filtered_v2",
        "filtered_zscore_v2",
    ):
        path = root / "data/processed" / variant / "SC4002E.npz"
        info = inspect_record(path, variant)
        sample_results[variant] = {
            "record_key": info.record_key,
            "epochs": info.epochs,
            "ignored_epochs": info.ignored_epochs,
        }

    cuda_available = torch.cuda.is_available()
    if args.require_gpu and not cuda_available:
        errors.append("gpu_required_but_unavailable")
    report = {
        "schema_version": 1,
        "workspace": str(root),
        "passed": not errors,
        "errors": errors,
        "require_gpu": args.require_gpu,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pyedflib": pyedflib.__version__,
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        },
        "split_manifest_sha256": actual_split_hash,
        "split_summary": split_manifest["summary"],
        "sample_records": sample_results,
        "experiment_status": experiment_config["status"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
