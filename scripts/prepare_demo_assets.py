"""Materialize verified E0/E3/E6 demo assets from a Git ref."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


EXPERIMENTS = {
    "E0": {
        "data_variant": "paper_raw_v1",
        "extractor_kind": "cnn15",
        "sequence_kind": "bilstm",
    },
    "E3": {
        "data_variant": "filtered_v2",
        "extractor_kind": "resnet1d",
        "sequence_kind": "tcn",
    },
    "E6": {
        "data_variant": "filtered_zscore_v2",
        "extractor_kind": "resnet1d",
        "sequence_kind": "tcn",
    },
}
CNN_KEYS = tuple(
    f"{prefix}_{stage}"
    for prefix in ("C", "P", "N")
    for stage in ("W", "N1", "N2", "N3", "REM")
)


def git_bytes(workspace: Path, ref: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Cannot read {ref}:{path}: {message}")
    return result.stdout


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def file_entry(
    workspace: Path,
    ref: str,
    source_path: str,
    destination: Path,
    expected_sha256: str,
    output: Path,
) -> dict[str, str]:
    data = git_bytes(workspace, ref, source_path)
    observed = hashlib.sha256(data).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"SHA-256 mismatch in source branch: {source_path}")
    write_atomic(output / destination, data)
    print(f"{destination.as_posix()}: {len(data):,} bytes · {observed[:12]}…")
    return {"path": destination.as_posix(), "sha256": observed}


def completion_hash(workspace: Path, ref: str, path: str) -> str:
    document = json.loads(git_bytes(workspace, ref, path).decode("utf-8"))
    return str(document["best_checkpoint_sha256"])


def build_experiment(
    workspace: Path,
    ref: str,
    output: Path,
    experiment_id: str,
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    config = EXPERIMENTS[experiment_id]
    prefix = f"runs/v2/full/{experiment_id}/fold_{fold:02d}/seed_{seed}"
    run_manifest = json.loads(
        git_bytes(workspace, ref, f"{prefix}/run_manifest.json").decode("utf-8")
    )
    if (
        run_manifest.get("experiment_id") != experiment_id
        or run_manifest.get("outer_fold") != fold
        or run_manifest.get("seed") != seed
        or run_manifest.get("data_variant") != config["data_variant"]
        or run_manifest.get("status") != "complete"
    ):
        raise ValueError(f"source run manifest does not match requested {experiment_id}")

    validation = json.loads(
        git_bytes(workspace, ref, f"{prefix}/validation_report.json").decode("utf-8")
    )
    if not validation.get("passed"):
        raise ValueError(f"{experiment_id} validation report did not pass")
    result: dict[str, Any] = {
        "data_variant": config["data_variant"],
        "extractor_kind": config["extractor_kind"],
        "sequence_kind": config["sequence_kind"],
        "source_run_path": prefix,
    }

    prediction_source = f"{prefix}/predictions/test.npz"
    result["prediction"] = file_entry(
        workspace,
        ref,
        prediction_source,
        Path("predictions") / f"{experiment_id}_fold{fold:02d}_test.npz",
        str(validation["roles"]["test"]["prediction_sha256"]),
        output,
    )

    sequence_kind = config["sequence_kind"]
    sequence_source = f"{prefix}/checkpoints/sequence/{sequence_kind}/best.pt"
    sequence_complete = f"{prefix}/checkpoints/sequence/{sequence_kind}/complete.json"
    result["sequence"] = file_entry(
        workspace,
        ref,
        sequence_source,
        Path("checkpoints") / experiment_id / f"{sequence_kind}_best.pt",
        completion_hash(workspace, ref, sequence_complete),
        output,
    )

    if experiment_id == "E0":
        extractors: dict[str, dict[str, str]] = {}
        for key in CNN_KEYS:
            source = f"{prefix}/checkpoints/cnn15/{key}/best.pt"
            complete = f"{prefix}/checkpoints/cnn15/{key}/complete.json"
            extractors[key] = file_entry(
                workspace,
                ref,
                source,
                Path("checkpoints") / experiment_id / "cnn15" / f"{key}_best.pt",
                completion_hash(workspace, ref, complete),
                output,
            )
        result["extractors"] = extractors
    else:
        source = f"{prefix}/checkpoints/resnet1d/best.pt"
        complete = f"{prefix}/checkpoints/resnet1d/complete.json"
        result["extractor"] = file_entry(
            workspace,
            ref,
            source,
            Path("checkpoints") / experiment_id / "resnet1d_best.pt",
            completion_hash(workspace, ref, complete),
            output,
        )
    return result, list(run_manifest["role_records"]["test"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ref", default="run-in-docker")
    parser.add_argument("--fold", type=int, default=0, choices=range(10))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("demo/assets"))
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    experiments: dict[str, Any] = {}
    canonical_records: list[str] | None = None
    for experiment_id in EXPERIMENTS:
        print(f"\n[{experiment_id}]")
        entry, test_records = build_experiment(
            workspace, args.ref, output, experiment_id, args.fold, args.seed
        )
        if canonical_records is None:
            canonical_records = test_records
        elif test_records != canonical_records:
            raise ValueError(f"{experiment_id} test records are not aligned")
        experiments[experiment_id] = entry

    manifest = {
        "schema_version": 2,
        "source_ref": args.ref,
        "outer_fold": args.fold,
        "seed": args.seed,
        "experiments": experiments,
        "test_records": canonical_records,
    }
    rendered = (json.dumps(manifest, indent=2, sort_keys=False) + "\n").encode("utf-8")
    write_atomic(output / "demo_assets.json", rendered)
    print(f"\nManifest: {output / 'demo_assets.json'}")
    print("DEMO_ASSETS_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
