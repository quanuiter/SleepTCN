"""Verify exported diagnostics without executing archive contents or GPU training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/colab_t4_smoke_20261004"
EXPECTED = {"cpu.json", "cuda.json", "cpu.log", "cuda.log", "summary.json", "progress.json", "manifest.json"}
SOURCE_FIELDS = {"models_sha256": "src/sleeptcn/models.py",
                 "training_sha256": "src/sleeptcn/training.py",
                 "script_sha256": "scripts/benchmark_colab_tcn.py",
                 "sample_sha256": "sample_records.npz"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close(actual, expected):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            "Reported aggregate differs from recomputation")


def preserve(path, data):
    if path.exists():
        require(path.read_bytes() == data, "Existing artifact differs; refusing overwrite")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)


def verify(archive_path):
    archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(EXPECTED) and {i.filename for i in entries} == EXPECTED,
                "Archive allowlist or duplicate entry check failed")
        require(all(not i.is_dir() and i.file_size < 1_000_000 for i in entries), "Unexpected entry size/type")
        require(archive.testzip() is None, "ZIP CRC check failed")
        payloads = {name: archive.read(name) for name in EXPECTED}
    manifest = json.loads(payloads["manifest.json"])
    require(set(manifest) == EXPECTED - {"manifest.json"}, "Manifest allowlist differs")
    for name, expected_hash in manifest.items():
        require(digest(payloads[name]) == expected_hash, f"Hash mismatch: {name}")
    source = json.loads((RUN / "bundle_verification.json").read_bytes())
    require(source["status"] == "passed", "Source bundle was not verified")
    require(digest((RUN / "SleepTCN_T4_Smoke_20261004.zip").read_bytes()) == source["archive_sha256"],
            "Original source bundle hash differs")
    cpu, cuda, summary, progress = [json.loads(payloads[n + ".json"]) for n in ["cpu", "cuda", "summary", "progress"]]
    require(summary["status"] == "passed" and progress["status"] == "completed", "Non-terminal or failed result")
    require(summary["no_campaign_started"] and summary["bounded_device_diagnostic_only"] and progress["no_campaign_started"],
            "Unexpected campaign scope")
    require(cpu["torch"] == cuda["torch"] == summary["torch"], "Environment version differs")
    require(cuda["gpu"] == summary["gpu"] == "Tesla T4", "Unexpected GPU")
    cases = []
    for backend, result in [("cpu", cpu), ("cuda", cuda)]:
        require(result["status"] == "passed" and result["backend"] == backend, "Backend status differs")
        require(not result["target_data_access"] and not result["campaign_checkpoint_access"], "Unexpected data scope")
        require(result["float32_only"] and not result["tf32"] and result["deterministic_algorithms"], "Precision settings differ")
        for field, filename in SOURCE_FIELDS.items():
            expected_hash = source["manifest"]["files"][filename]
            require(result[field] == expected_hash, f"Source provenance differs: {field}")
            local = RUN / filename if filename == "sample_records.npz" else ROOT / filename
            require(digest(local.read_bytes()) == expected_hash, f"Current source differs: {filename}")
        require(len(result["cases"]) == 2, "Unexpected number of cases")
        require(payloads[backend + ".log"].decode().strip().endswith("FINAL_STATUS passed"), "Log status differs")
        for case in result["cases"]:
            check = case["numerical_check"]
            require(case["status"] == "passed" and check["passed"], "Numerical check failed")
            require(all(g["passed"] for g in check["gradients"]), "A gradient check failed")
            close(check["gradient_max_abs_residual"], max(g["max_abs_residual"] for g in check["gradients"]))
            require(math.isclose(check["loss_cpu"], check["loss_candidate"], rel_tol=1e-3, abs_tol=1e-4), "Loss check differs")
            require(0 <= check["first_adam_step_max_parameter_residual"] <= 1e-3, "Adam residual differs")
            for kind in ["resident", "with_transfer"]:
                timing = case[kind]
                require(timing["repetitions"] == len(timing["seconds"]) == 15 and timing["warmups"] == 3,
                        "Timing protocol differs")
                require(all(math.isfinite(v) and v > 0 for v in timing["seconds"]), "Invalid timing")
                close(timing["median_seconds"], statistics.median(timing["seconds"]))
    local_cpu = json.loads((RUN / "local_cpu251.json").read_bytes())
    require(local_cpu["status"] == "passed", "Local CPU diagnostic failed")
    for index, (c, g, row) in enumerate(zip(cpu["cases"], cuda["cases"], summary["cases"])):
        require(c["name"] == g["name"] == row["case"] and c["shape"] == g["shape"] == row["shape"], "Case mapping differs")
        ct, gt = [statistics.median(v["with_transfer"]["seconds"]) for v in [c, g]]
        close(row["cpu_seconds_per_step"], ct)
        close(row["gpu_seconds_per_step"], gt)
        close(row["cpu_over_gpu_speed_ratio"], ct / gt)
        close(row["gpu_gradient_max_abs_residual"], g["numerical_check"]["gradient_max_abs_residual"])
        close(row["gpu_peak_allocated_MiB"], g["peak_memory_allocated_bytes"] / 2**20)
        local = local_cpu["cases"][index]
        require(local["name"] == g["name"] and local_cpu["sample_sha256"] == cuda["sample_sha256"], "Local comparison differs")
        cases.append({**row, "local_cpu_over_gpu_ratio": statistics.median(local["with_transfer"]["seconds"]) / gt,
                      "gpu_numerical_check": {k: v for k, v in g["numerical_check"].items() if k != "gradients"}})
    require(len(summary["cases"]) == 2, "Unexpected summary size")
    checkpoint = ROOT / "runs/teacher_revision_cpu_20261003/weighted_10fold_campaign/folds/fold_06/unweighted/latest.pt"
    checkpoint_hash = digest(checkpoint.read_bytes())
    require(checkpoint_hash == "5fa37275ed3dbc0050951a2ed125ef721410adaeec79960dc31457938ee0e17c", "Frozen checkpoint differs")
    report = {"status": "passed", "archive_sha256": digest(archive_bytes), "archive_bytes": len(archive_bytes),
              "manifest_hashes_verified": True, "source_hashes_verified": True, "timings_recomputed": True,
              "cpu_checkpoint_preserved": True, "cases": cases,
              "scope": "export integrity and aggregate consistency; not a rerun of GPU tensor checks",
              "standalone_archive_hash_previously_observed_on_colab": False}
    preserve(RUN / "SleepTCN_T4_Smoke_Results_20261004.zip", archive_bytes)
    for name, data in payloads.items():
        preserve(RUN / "verified_results" / name, data)
    preserve(RUN / "verified_results/independent_verification.json", json.dumps(report, indent=2).encode())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().archive), indent=2))
