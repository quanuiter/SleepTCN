"""Import a checked Colab result and replay source metrics locally (no training)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.models import SleepTCN
from sleeptcn.revision_weighted import evaluate_source
from sleeptcn.revision_campaign import write_once_json


def sha(data):
    return hashlib.sha256(data).hexdigest()


def preserve(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Refusing to overwrite different artifact")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)


def verify(args):
    raw = args.archive.read_bytes()
    if sha(raw) != args.expected_sha256:
        raise ValueError("Archive differs from hash printed by Colab")
    parent = ROOT / f"runs/colab_training_20261004/fold_{args.fold:02d}"
    bundle = parent / f"SleepTCN_Train_Fold{args.fold:02d}_20261004.zip"
    bundle_audit = json.loads((parent / "bundle_verification.json").read_bytes())
    if sha(bundle.read_bytes()) != bundle_audit["archive_sha256"]:
        raise ValueError("Source bundle differs")
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read("manifest.json"))
        manifest_hash = sha(z.read("manifest.json"))
        roles = {role: [] for role in ["validation", "test"]}
        for entry in manifest["records"]:
            if entry["role"] in roles:
                data = z.read(entry["path"])
                if sha(data) != manifest["files"][entry["path"]]:
                    raise ValueError("Source feature payload differs")
                import io
                with np.load(io.BytesIO(data), allow_pickle=False) as npz:
                    roles[entry["role"]].append((torch.from_numpy(npz["features"].copy()),
                                                  torch.from_numpy(npz["labels"].astype(np.int64))))
        expected_cfg = json.loads(z.read("configs/teacher_revision_weighted_10fold_v1.json"))
        protocol_hash = sha(z.read("configs/teacher_revision_weighted_10fold_v1.json"))
        runner_hash = sha(z.read("scripts/run_colab_source_training.py"))
        for filename in ["src/sleeptcn/models.py", "src/sleeptcn/training.py", "src/sleeptcn/revision_weighted.py"]:
            if sha((ROOT / filename).read_bytes()) != manifest["files"][filename]:
                raise ValueError("Frozen training code differs")
    expected_names = {"aggregate_results.json", "execution_specification.json", "process.json",
                      "progress.json", "run.log", "verification.json"}
    for arm in ["unweighted", "weighted"]:
        expected_names.update(f"{arm}/{name}" for name in ["identity.json", "selection.json", "latest.pt", "best.pt"])
    output = parent / "gpu_results"
    with zipfile.ZipFile(args.archive) as z:
        entries = z.infolist()
        if len(entries) != len(expected_names) or set(z.namelist()) != expected_names or z.testzip() is not None:
            raise ValueError("Unexpected result ZIP schema/CRC")
        for entry in entries:
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts or entry.file_size > 25_000_000:
                raise ValueError("Unsafe result ZIP entry")
            preserve(output / entry.filename, z.read(entry.filename))
    preserve(parent / f"SleepTCN_Fold{args.fold:02d}_CUDA_Checkpoints.zip", raw)
    result = json.loads((output / "aggregate_results.json").read_bytes())
    verification = json.loads((output / "verification.json").read_bytes())
    spec = json.loads((output / "execution_specification.json").read_bytes())
    if result["status"] != "complete_source_only_cuda_pair" or result["fold"] != args.fold or result["target_access"]:
        raise ValueError("Unexpected result status/scope")
    if verification["status"] != "passed" or verification["aggregate_results_sha256"] != sha((output / "aggregate_results.json").read_bytes()):
        raise ValueError("Internal verification differs")
    if spec["protocol"] != expected_cfg or spec["cpu_partial_checkpoint_imported"] or spec["target_access"]:
        raise ValueError("Training recipe or scope differs")
    provenance = result["provenance"]
    for key, expected in [("bundle_manifest_sha256", manifest_hash), ("protocol_sha256", protocol_hash),
                          ("runner_sha256", runner_hash), ("encoder_sha256", manifest["source_specification"]["encoder_sha256"])]:
        if provenance[key] != expected:
            raise ValueError("Provenance hash differs")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    counts = np.asarray(manifest["train_class_counts"])
    replayed = {}
    for arm, selected in result["selections"].items():
        checkpoint = output / arm / "best.pt"
        if sha(checkpoint.read_bytes()) != selected["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint hash differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        latest = torch.load(output / arm / "latest.pt", map_location="cpu", weights_only=True)
        if payload["identity"] != {**provenance, "arm": arm} or payload["epoch"] != selected["selected_epoch"]:
            raise ValueError("Selected checkpoint identity differs")
        history = selected["history"]
        best = max(history, key=lambda row: row["validation_macro_f1"])
        if best["epoch"] != selected["selected_epoch"] or best["validation_macro_f1"] != selected["selected_validation_macro_f1"]:
            raise ValueError("Selection is not earliest strict maximum validation score")
        if len(history) != selected["epochs_completed"] or latest["epoch"] != len(history) or latest["history"] != history:
            raise ValueError("Epoch history differs")
        if len(history) < expected_cfg["max_epochs"] and latest["stale_epochs"] < expected_cfg["patience"]:
            raise ValueError("Training terminated before scientific stopping rule")
        expected_weights = np.ones(5) if arm == "unweighted" else counts.sum() / (5 * counts)
        np.testing.assert_allclose(selected["loss_weights"], expected_weights, rtol=1e-6)
        model = SleepTCN(input_dim=128).eval()
        model.load_state_dict(payload["model_state"], strict=True)
        replayed[arm] = {}
        for role, name in [("validation", "validation"), ("test", "outer_test")]:
            metric = evaluate_source(model, roles[role], expected_cfg["batch_size_records"])
            np.testing.assert_array_equal(metric["confusion_matrix"], result["source_results"][arm][name]["confusion_matrix"])
            if metric != result["source_results"][arm][name]:
                raise ValueError("Source metric replay differs")
            replayed[arm][name] = metric
    selections = result["selections"]
    if selections["unweighted"]["initial_state_sha256"] != selections["weighted"]["initial_state_sha256"]:
        raise ValueError("Initial states differ")
    common = min(s["epochs_completed"] for s in selections.values())
    if [r["batch_order_sha256"] for r in selections["unweighted"]["history"][:common]] != [r["batch_order_sha256"] for r in selections["weighted"]["history"][:common]]:
        raise ValueError("Paired batch order differs")
    audit = {"status": "passed", "fold": args.fold, "archive_sha256": sha(raw),
             "aggregate_results_sha256": sha((output / "aggregate_results.json").read_bytes()),
             "selected_checkpoints": 2, "source_confusions_replayed_on_cpu": True,
             "recipe_and_stopping_rule_verified": True, "target_access": False,
             "source_results": replayed}
    write_once_json(output / "independent_verification.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k != "source_results"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(6, 10), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    verify(parser.parse_args())
