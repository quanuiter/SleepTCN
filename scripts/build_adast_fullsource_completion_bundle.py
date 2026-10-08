"""Prepare the remaining-fold CUDA input locally; this script does not upload or train."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "runs/adast_fullsource_completion_20261006"
OLD = ROOT / "runs/colab_adast_20261004/payload"
BASE = ROOT / "runs/adast_development_20261004/payload"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def storage_requirement(payload_bytes):
    # Payload + worst-case ZIP + duplicate upload parts + 5 GB for results/local verification.
    return 3 * payload_bytes + 5_000_000_000


def write_launcher(binding, parts, destination):
    template = (ROOT / "scripts/adast_fullsource_completion_colab_launch_template.py").read_text(encoding="utf8")
    replacements = {
        "MANIFEST_SHA_PLACEHOLDER": repr(binding["manifest_sha256"]),
        "PARTS_PLACEHOLDER": repr([{key: value for key, value in part.items() if key != "path"} for part in parts]),
        "SHA_PLACEHOLDER": repr(binding["archive_sha256"]),
        "BYTES_PLACEHOLDER": str(binding["archive_bytes"]),
    }
    for key in sorted(replacements, key=len, reverse=True):
        template = template.replace(key, replacements[key])
    if "PLACEHOLDER" in template:
        raise ValueError("Unresolved launcher token")
    compile(template, "fullsource_completion_colab_launch.py", "exec")
    with destination.open("x", encoding="utf8") as stream:
        stream.write(template)


def build():
    if RECORD.exists():
        raise FileExistsError("Preserve the existing prepared bundle or partial build")
    audit_path = ROOT / "runs/adast_fullsource_preflight_20261006/audit.json"
    audit = json.loads(audit_path.read_bytes())
    cfg_path = ROOT / "configs/adast_fullsource_completion_v1_20261006.json"
    cfg = json.loads(cfg_path.read_bytes())
    if (audit["status"] != "passed" or audit["reusable_models"] != 4 or audit["new_models_needed"] != 16
            or cfg["folds_to_train"] != list(range(2, 10))
            or cfg["remaining_fold_sizes"] != audit["remaining_folds"]
            or cfg["base_runner_sha256"] != audit["common_base_runner_sha256"]):
        raise ValueError("Completion scope differs from reuse audit")
    old_binding = json.loads((OLD.parent / "bundle_verification.json").read_bytes())
    if sha(OLD / "manifest.json") != old_binding["manifest_sha256"]:
        raise ValueError("Historical source manifest differs")
    old = json.loads((OLD / "manifest.json").read_bytes())
    frozen = json.loads((BASE / "manifest.json").read_bytes())
    data_names = ["data/source_x.npy", "data/source_y.npy", "data/adaptation_x.npy"]
    data_names += [f"data/fold_{fold:02d}_roles.npz" for fold in cfg["folds_to_train"]]
    code_names = ["adast_reference_adapter.py", "development_metrics.py", "run_adast_development_cuda.py",
                  "upstream/models.py", "upstream/configs.py", "upstream/utils.py", "upstream/LICENSE"]
    sources = [(OLD / name, name, old["files"][name]) for name in data_names]
    sources += [(BASE / name, name, frozen["files"][name]) for name in code_names]
    if any(old[name] for name in ("target_test_data_included", "target_labels_included", "participant_ids_included", "raw_edf_included")):
        raise ValueError("Forbidden historical payload role")
    if frozen["files"]["run_adast_development_cuda.py"] != cfg["base_runner_sha256"]:
        raise ValueError("Measured training core differs")
    if old["files"]["data/adaptation_x.npy"] != frozen["files"]["data/adaptation_x.npy"]:
        raise ValueError("Adaptation population/input differs")
    payload_bytes = sum(path.stat().st_size for path, _, _ in sources)
    temp_parent = Path(tempfile.gettempdir()).resolve()
    required = storage_requirement(payload_bytes)
    if shutil.disk_usage(temp_parent).free < required:
        raise RuntimeError(f"Need {required} free bytes in temporary storage")
    storage = Path(tempfile.mkdtemp(prefix="SleepTCN_ADAST_Fullsource_20261006_", dir=temp_parent))
    work = storage / "payload"
    work.mkdir()
    RECORD.mkdir(parents=True)
    pointer = {"status": "building_local_bundle_no_upload", "storage_root": storage.as_posix(),
               "payload_root": work.as_posix(), "required_local_free_bytes_at_build": required,
               "upload_permission_current_account": "pending", "source_records": 153,
               "adaptation_subjects": 5, "target_test_data_included": False}
    write_json(RECORD / "storage.json", pointer)
    for source, name, expected in sources:
        destination = work / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha(destination) != expected:
            raise ValueError("Copied input/code differs: " + name)
        print("PREPARED", name, destination.stat().st_size, flush=True)
    shutil.copyfile(ROOT / "scripts/run_adast_fullsource_completion_cuda.py",
                    work / "run_adast_fullsource_completion_cuda.py")
    shutil.copyfile(cfg_path, work / "protocol.json")
    files = sorted(path for path in work.rglob("*") if path.is_file())
    manifest = {
        "status": "prepared_local_input_for_eight_fullsource_pairs", "folds_to_train": cfg["folds_to_train"],
        "source_all_records_included": True, "source_role_training_restricted": True,
        "source_role_description": "All source records are shared across folds. Each model uses only its train and validation roles; its outer test role is excluded from training and selection.",
        "source_records": 153, "source_valid_epochs": old["source_valid_epochs"],
        "source_split_sha256": old["source_split_sha256"], "adaptation_subjects": 5, "adaptation_epochs": 4989,
        "target_test_data_included": False, "target_labels_included": False,
        "participant_ids_included": False, "raw_edf_included": False,
        "upstream_commit": frozen["upstream_commit"], "upstream_files_sha256": frozen["upstream_files_sha256"],
        "reuse_audit_sha256": sha(audit_path), "reused_models_in_this_payload": False,
        "files": {path.relative_to(work).as_posix(): sha(path) for path in files},
    }
    write_json(work / "manifest.json", manifest)
    archive = storage / "SleepTCN_ADAST_Fullsource_Completion_Input_20261006.zip"
    print("COMPRESSING LOCAL PAYLOAD; NO UPLOAD OR TRAINING", flush=True)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in files + [work / "manifest.json"]:
            zipped.write(path, path.relative_to(work).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if (len(zipped.namelist()) != len(set(zipped.namelist()))
                or set(zipped.namelist()) != set(manifest["files"]) | {"manifest.json"}
                or zipped.testzip() is not None):
            raise ValueError("ZIP allowlist/CRC differs")
    binding = {"status": "passed", "archive_filename": archive.name, "archive_path": archive.as_posix(),
               "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
               "manifest_sha256": sha(work / "manifest.json"), "payload_root": work.as_posix(),
               "new_models_planned": 16, "reused_models_separate": 4,
               "target_test_data_included": False, "target_labels_included": False,
               "participant_ids_included": False, "training_started": False, "uploaded": False}
    write_json(RECORD / "bundle_verification.json", binding)
    parts = []
    with archive.open("rb") as stream:
        while block := stream.read(256 * 1024 * 1024):
            part = storage / f"SleepTCN_ADAST_Fullsource_Completion_20261006.part{len(parts):02d}"
            with part.open("xb") as sink:
                sink.write(block)
            parts.append({"name": part.name, "path": part.as_posix(), "bytes": len(block),
                          "sha256": hashlib.sha256(block).hexdigest()})
    write_json(RECORD / "upload_parts.json", {**binding, "parts": parts})
    write_launcher(binding, parts, RECORD / "colab_launch.py")
    write_json(RECORD / "preparation_complete.json", {
        "status": "bundle_and_launcher_ready_local", "archive_sha256": binding["archive_sha256"],
        "archive_bytes": binding["archive_bytes"], "parts": len(parts),
        "training_started": False, "uploaded": False,
        "pending": ["notebook_access_and_GPU", "upload_permission_current_account"]})
    print(json.dumps({key: value for key, value in binding.items() if not key.endswith("path") and key != "payload_root"}, indent=2), flush=True)


if __name__ == "__main__":
    build()
