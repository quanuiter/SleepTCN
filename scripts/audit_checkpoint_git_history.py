"""Read historical checkpoint blobs without checkout or restoring run artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def blob(revision, path):
    return git("cat-file", "blob", f"{revision}:{path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="a005df3")
    parser.add_argument("--shhs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("audit output already exists")
    revision = git("rev-parse", args.revision).decode().strip()
    manifests = args.shhs_root / "manifests"
    requested, inventory_results = {}, {}
    for name in ["zero_shot_checkpoint_inventory_e4_seed123_v1.json",
                 "zero_shot_checkpoint_inventory_v1.json", "component_checkpoint_inventory_v1.json"]:
        raw = (manifests / name).read_bytes()
        inventory = json.loads(raw)
        digests = []
        for fold in inventory["folds"]:
            for checkpoint in fold["checkpoints"]:
                path = checkpoint["path"].replace("\\", "/")
                path = "runs/" + path.split("/runs/", 1)[1]
                expected = checkpoint["sha256"]
                if path in requested and requested[path]["expected_sha256"] != expected:
                    raise ValueError("inventories disagree")
                requested.setdefault(path, {"expected_sha256": expected, "inventories": []})["inventories"].append(name)
                digests.append(expected)
        inventory_results[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "referenced_checkpoints": len(digests)}
    # Seed-42 E4 was not in the historical SHHS inventory. Compare its blobs
    # with the same historical commit's completion markers and run manifests.
    for fold in range(10):
        base = f"runs/v2/full/E4/fold_{fold:02d}/seed_42"
        for stage in ["resnet1d", "sequence/tcn"]:
            path = f"{base}/checkpoints/{stage}/best.pt"
            marker = json.loads(blob(revision, f"{base}/checkpoints/{stage}/complete.json"))
            requested[path] = {"expected_sha256": marker["best_checkpoint_sha256"],
                               "inventories": [], "reference": "historical_completion_marker_and_run_manifest"}
    verified, groups = [], Counter()
    for index, (path, expected) in enumerate(sorted(requested.items()), 1):
        data = blob(revision, path)
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected["expected_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        pieces = path.split("/")
        experiment, fold, seed = pieces[3], int(pieces[4].split("_")[1]), int(pieces[5].split("_")[1])
        stage = "/".join(pieces[7:-1])
        result = {"git_path": path, "git_blob_oid": git("rev-parse", f"{revision}:{path}").decode().strip(),
                  "bytes": len(data), "sha256": digest, "reference_sources": expected["inventories"]}
        groups[f"{experiment}/seed_{seed}"] += 1
        if experiment in ["E3", "E4"]:
            base = "/".join(pieces[:6])
            marker = json.loads(blob(revision, "/".join(pieces[:-1]) + "/complete.json"))
            run = json.loads(blob(revision, base + "/run_manifest.json"))
            with torch.serialization.safe_globals([
                    np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]):
                payload = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
            metadata = payload["metadata"]
            variant = "filtered_v2" if experiment == "E3" else "bandpass_v2"
            expected_meta = {"experiment_id": experiment, "outer_fold": fold, "seed": seed,
                             "stage": stage, "data_variant": variant,
                             "config_sha256": run["config_sha256"], "split_sha256": run["split_sha256"]}
            if any(metadata.get(k) != v for k, v in expected_meta.items()):
                raise ValueError(f"checkpoint metadata mismatch: {path}")
            if (marker["best_checkpoint_sha256"] != digest or marker["smoke"] is not False
                    or marker["outer_fold"] != fold or marker["component_seed"] != seed
                    or marker["data_variant"] != variant or marker["stage"] != stage):
                raise ValueError(f"completion marker mismatch: {path}")
            if stage == "resnet1d" and run["extractor_sha256"] != digest:
                raise ValueError("run manifest encoder hash mismatch")
            if payload["progress"]["completed_epochs"] < 1 or not payload["model_state"]:
                raise ValueError("empty/untrained checkpoint")
            result.update(metadata_verified=True, completion_marker_verified=True,
                          metadata=metadata, training_progress=payload["progress"])
        verified.append(result)
        if index % 80 == 0 or index == len(requested):
            print(f"Verified Git blobs {index}/{len(requested)}", flush=True)
    branches = {}
    for ref in ["run-in-docker", "origin/run-in-docker", "8af1d12", "a005df3"]:
        names = git("ls-tree", "-r", "--name-only", ref).decode().splitlines()
        counts = Counter(f"{p.split('/')[3]}/{p.split('/')[5]}" for p in names
                         if p.startswith("runs/v2/full/") and "/checkpoints/" in p and p.endswith("/best.pt"))
        branches[ref] = {"commit": git("rev-parse", ref).decode().strip(), "best_checkpoint_counts": dict(counts)}
    report = {"status": "passed", "revision": revision, "read_only_git_audit": True,
              "checkout_changed": False, "checkpoints_restored_to_worktree": False,
              "verified_unique_paths": len(verified), "verified_total_bytes": sum(e["bytes"] for e in verified),
              "verified_groups": dict(groups), "metadata_and_marker_checks_E3_E4": sum(e.get("metadata_verified", False) for e in verified),
              "inventories": inventory_results, "branches": branches, "checkpoints": verified,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limitations": "E4 seed42 checked against historical completion markers/run manifests, not an existing SHHS E4 seed42 inference inventory. No new inference or accuracy result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ["status", "revision", "verified_unique_paths", "verified_total_bytes", "verified_groups", "metadata_and_marker_checks_E3_E4"]}, indent=2))


if __name__ == "__main__":
    main()
