"""Amendment A2: preprocessing of the new input variant for Sleep-EDF and the SHHS1 holdout.

Both cohorts reuse the unchanged record pipelines (``preprocessing.process_record`` and
``shhs_preprocessing.process_subject``) with only ``filtered_zscore_rolling90_v3`` requested.
Every new record is checked against an existing variant of the same record: record key,
subject, labels, valid mask and original epoch indices must be identical, so the new variant
covers exactly the epochs that the frozen split, the evaluation windows and the parent
holdout analysis already use. Manifests are write-once and record every output hash.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .amendment_a2 import A2_ROOT, load_a2_config
from .io.hashing import sha256_file
from .io.serialization import atomic_write_json, read_json

VARIANT = "filtered_zscore_rolling90_v3"
SLEEPEDF_REFERENCE_VARIANT = "filtered_v2"
SHHS_REFERENCE_VARIANT = "bandpass_v2"
# Existing variant recomputed for a few records to show that the filter/clip pipeline of this
# environment reproduces the stored inputs before the new variant is trusted.
PIPELINE_CHECK_VARIANT = "filtered_zscore_v2"
PIPELINE_CHECK_TOLERANCE = 1e-4
SLEEPEDF_RECORDS = 153
SHHS_RECORDS = 500
MANIFEST_DIR = A2_ROOT / "manifests"
SLEEPEDF_MANIFEST = MANIFEST_DIR / "preprocess_sleepedf_a2.json"
SHHS_MANIFEST = MANIFEST_DIR / "preprocess_holdout_a2.json"
SHHS_PROCESSED = Path("data/processed/shhs_holdout_a2")
SHARED_FIELDS = ("record_key", "subject_id", "y", "valid_mask", "original_epoch_index")


def _environment() -> dict[str, str]:
    import pyedflib
    import scipy

    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pyedflib": pyedflib.__version__}


def _quartiles(values: list[float]) -> dict[str, float]:
    q1, median, q3 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
    return {"n": len(values), "median": float(median), "q1": float(q1), "q3": float(q3)}


def compare_with_reference(new_path: Path, reference_path: Path, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """Field-by-field identity of the epochs covered by two variants of one record."""
    with np.load(new_path, allow_pickle=False) as new, np.load(reference_path, allow_pickle=False) as ref:
        checks = {name: bool(np.array_equal(new[name], ref[name])) for name in SHARED_FIELDS + extra}
        checks["variant"] = str(new["preprocess_version"].item()) == VARIANT
        checks["finite"] = bool(np.isfinite(new["x"]).all())
        checks["shape"] = new["x"].shape == (len(new["y"]), 3000) and new["x"].dtype == np.float32
        scope = str(new["normalization_scope"].item()) if "normalization_scope" in new.files else ""
        checks["scope"] = scope == "centred_rolling_window_after_filter_clip"
        x_std = float(new["x"].astype(np.float64).std())
    return {"passed": all(checks.values()), "checks": checks, "x_std_in_window": x_std}


def _stored_x_std(path: Path) -> float:
    with np.load(path, allow_pickle=False) as z:
        return float(z["x"].astype(np.float64).std())


# --------------------------------------------------------------------------- Sleep-EDF
def _sleepedf_one(job: dict[str, Any]) -> dict[str, Any]:
    from .preprocessing import PreprocessConfig, process_record

    try:
        records = process_record(Path(job["psg"]), Path(job["hyp"]), [job["variant"]], Path(job["output_root"]),
                                 job["hashes"], PreprocessConfig(), overwrite=False)
        return {"key": job["key"], "ok": True, "records": records}
    except Exception as exc:  # reported, never silently skipped
        return {"key": job["key"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def prepare_sleepedf(workspace: Path, data_dir: Path, raw_manifest: Path, *, workers: int = 1) -> dict[str, Any]:
    """Write the A2 Sleep-EDF variant for all 153 records and check it against ``filtered_v2``."""
    from .preprocessing import PreprocessConfig, load_source_hashes, pair_files

    workspace = workspace.resolve()
    config, config_sha = load_a2_config(workspace)
    manifest_path = workspace / SLEEPEDF_MANIFEST
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; the A2 Sleep-EDF manifest is write-once")
    processed = workspace / "data" / "processed"
    pairs = pair_files(data_dir.resolve())
    hashes = load_source_hashes(raw_manifest.resolve())
    reference_keys = sorted(p.stem for p in (processed / SLEEPEDF_REFERENCE_VARIANT).glob("SC*.npz"))
    if sorted(pairs) != reference_keys or len(reference_keys) != SLEEPEDF_RECORDS:
        raise ValueError("Sleep-EDF source records differ from the existing filtered_v2 records")

    jobs = [{"key": key, "psg": str(pairs[key][0]), "hyp": str(pairs[key][1]), "hashes": hashes[key],
             "output_root": str(processed), "variant": VARIANT}
            for key in reference_keys if not (processed / VARIANT / f"{key}.npz").exists()]
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, result in enumerate(pool.map(_sleepedf_one, jobs), start=1):
            if not result["ok"]:
                failures[result["key"]] = result["error"]
                print(f"FAILED {result['key']}: {result['error']}", flush=True)
            if done % 10 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] Sleep-EDF records preprocessed", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} Sleep-EDF records failed: {failures}")

    records, e6_std, e6r_std = [], [], []
    for key in reference_keys:
        new = processed / VARIANT / f"{key}.npz"
        check = compare_with_reference(new, processed / SLEEPEDF_REFERENCE_VARIANT / f"{key}.npz")
        if not check["passed"]:
            raise ValueError(f"{key}: A2 variant differs from {SLEEPEDF_REFERENCE_VARIANT}: {check['checks']}")
        e6_std.append(_stored_x_std(processed / PIPELINE_CHECK_VARIANT / f"{key}.npz"))
        e6r_std.append(check["x_std_in_window"])
        records.append({"record_key": key, "output_sha256": sha256_file(new),
                        "reference_sha256": sha256_file(processed / SLEEPEDF_REFERENCE_VARIANT / f"{key}.npz"),
                        "x_std_in_window": check["x_std_in_window"]})

    reproduction = pipeline_reproduction(processed, pairs, hashes,
                                         [reference_keys[0], reference_keys[len(reference_keys) // 2],
                                          reference_keys[-1]])
    if not reproduction["passed"]:
        raise ValueError(f"Pipeline reproduction check failed: {reproduction}")
    manifest = {
        "schema_version": 1, "amendment": "A2", "status": "complete", "a2_config_sha256": config_sha,
        "dataset": "sleep-edf-expanded/sleep-cassette/1.0.0", "variant": VARIANT,
        "reference_variant": SLEEPEDF_REFERENCE_VARIANT, "raw_manifest_sha256": sha256_file(raw_manifest),
        "preprocess_config": asdict(PreprocessConfig()), "environment": _environment(),
        "identity_check": f"record_key, subject_id, y, valid_mask and original_epoch_index equal {SLEEPEDF_REFERENCE_VARIANT}",
        "pipeline_reproduction": reproduction,
        "input_scale_in_window": {"E6_filtered_zscore_v2": _quartiles(e6_std), "E6R": _quartiles(e6r_std)},
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, sort_keys=False)
    return manifest


def pipeline_reproduction(processed: Path, pairs: dict[str, tuple[Path, Path]], hashes: dict[str, Any],
                          keys: list[str]) -> dict[str, Any]:
    """Recompute ``filtered_zscore_v2`` for a few records and compare with the stored inputs."""
    from .preprocessing import PreprocessConfig, process_record

    rows = []
    with tempfile.TemporaryDirectory(prefix="a2_pipeline_check_") as tmp:
        for key in keys:
            process_record(pairs[key][0], pairs[key][1], [PIPELINE_CHECK_VARIANT], Path(tmp), hashes[key],
                           PreprocessConfig(), overwrite=False)
            with np.load(Path(tmp) / PIPELINE_CHECK_VARIANT / f"{key}.npz", allow_pickle=False) as new, \
                    np.load(processed / PIPELINE_CHECK_VARIANT / f"{key}.npz", allow_pickle=False) as old:
                same_epochs = bool(np.array_equal(new["original_epoch_index"], old["original_epoch_index"]))
                diff = float(np.abs(new["x"].astype(np.float64) - old["x"].astype(np.float64)).max()) \
                    if same_epochs else float("inf")
            rows.append({"record_key": key, "same_epochs": same_epochs, "max_abs_difference": diff})
    return {"variant": PIPELINE_CHECK_VARIANT, "tolerance": PIPELINE_CHECK_TOLERANCE, "records": rows,
            "passed": all(r["same_epochs"] and r["max_abs_difference"] <= PIPELINE_CHECK_TOLERANCE for r in rows)}


# --------------------------------------------------------------------------- SHHS1 holdout
def load_parent_manifest(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(path) != config["shhs"]["parent_preprocess_manifest_sha256"]:
        raise ValueError("Parent holdout preprocess manifest hash differs from the A2 config")
    parent = read_json(path)
    if parent.get("status") != "complete" or len(parent["analysis_cohort"]) != SHHS_RECORDS:
        raise ValueError("Parent holdout preprocess manifest is incomplete")
    return parent


def parent_entries(parent: dict[str, Any], variant: str) -> dict[str, dict[str, Any]]:
    rows = {r["record_key"]: r for r in parent["records"] if r["variant"] == variant}
    if len(rows) != SHHS_RECORDS:
        raise ValueError(f"Parent manifest has {len(rows)} {variant} records")
    return rows


def verified_parent_path(processed_root: Path, variant: str, entry: dict[str, Any]) -> Path:
    """Parent input under ``processed_root`` (manifest paths are machine-specific), hash-checked."""
    path = processed_root / variant / f"{entry['record_key']}.npz"
    if sha256_file(path) != entry["output_sha256"]:
        raise ValueError(f"{path}: hash differs from the parent preprocess manifest")
    return path


def _shhs_one(job: dict[str, Any]) -> dict[str, Any]:
    from .shhs_preprocessing import process_subject

    try:
        return {"key": job["key"], "ok": True, "records": process_subject(**job["kwargs"])}
    except Exception as exc:
        return {"key": job["key"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def prepare_shhs(workspace: Path, parent_manifest: Path, technical_audit: Path, preprocess_config: Path,
                 edf_dir: Path, xml_dir: Path, parent_processed_root: Path, *, workers: int = 1) -> dict[str, Any]:
    """Write the A2 variant for the 500 holdout participants and check it against ``bandpass_v2``."""
    from .shhs_preprocessing import load_locked_config

    workspace = workspace.resolve()
    config, config_sha = load_a2_config(workspace)
    manifest_path = workspace / SHHS_MANIFEST
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; the A2 holdout manifest is write-once")
    parent = load_parent_manifest(parent_manifest, config)
    audit_sha = sha256_file(technical_audit)
    if audit_sha != parent["technical_audit_sha256"]:
        raise ValueError("Technical audit differs from the one used by the parent preprocessing")
    audit = read_json(technical_audit)
    _, preprocess_sha, preprocess = load_locked_config(preprocess_config.resolve())
    if preprocess_sha != parent["config_sha256"]:
        raise ValueError("SHHS preprocessing config differs from the parent preprocessing")
    references = parent_entries(parent, SHHS_REFERENCE_VARIANT)
    output_root = workspace / SHHS_PROCESSED

    jobs = []
    for subject_id in parent["analysis_cohort"]:
        key = f"shhs1-{subject_id}"
        reference = verified_parent_path(parent_processed_root, SHHS_REFERENCE_VARIANT, references[key])
        with np.load(reference, allow_pickle=False) as z:
            item = {"subject_id": str(subject_id), "role": str(z["role"].item()),
                    "role_index": int(z["role_index"]), "pilot": bool(z["pilot"]),
                    "edf_filename": str(z["source_edf_name"].item()),
                    "annotation_filename": str(z["source_xml_name"].item())}
        jobs.append({"key": key, "kwargs": {
            "item": item, "audit_subject": audit["subjects"][str(subject_id)], "edf_dir": edf_dir.resolve(),
            "xml_dir": xml_dir.resolve(), "output_root": output_root, "variants": (VARIANT,),
            "config": preprocess, "config_sha256": preprocess_sha,
            "selection_manifest_sha256": parent["cohort_csv_sha256"], "technical_audit_sha256": audit_sha,
            "resume": True}})
    results: dict[str, list[dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, result in enumerate(pool.map(_shhs_one, jobs), start=1):
            if result["ok"]:
                results[result["key"]] = result["records"]
            else:
                failures[result["key"]] = result["error"]
                print(f"FAILED {result['key']}: {result['error']}", flush=True)
            if done % 25 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] holdout records preprocessed", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} holdout records failed (no replacement in A2): {failures}")

    records, e6_std, e6r_std = [], [], []
    zscore = parent_entries(parent, "filtered_zscore_v2")
    for job in jobs:
        key = job["key"]
        (summary,) = results[key]
        reference = verified_parent_path(parent_processed_root, SHHS_REFERENCE_VARIANT, references[key])
        check = compare_with_reference(output_root / VARIANT / f"{key}.npz", reference,
                                       extra=("raw_stage", "source_edf_sha256", "source_xml_sha256"))
        if not check["passed"]:
            raise ValueError(f"{key}: A2 variant differs from {SHHS_REFERENCE_VARIANT}: {check['checks']}")
        e6_std.append(float(zscore[key]["x_std"]))
        e6r_std.append(check["x_std_in_window"])
        records.append({**summary, "reference_variant_sha256": references[key]["output_sha256"]})
    manifest = {
        "schema_version": 1, "amendment": "A2", "status": "complete", "dataset": "SHHS Visit 1",
        "scope": "holdout_amendment_a2", "a2_config_sha256": config_sha,
        "parent_preprocess_manifest_sha256": sha256_file(parent_manifest),
        "protocol_sha256": parent["protocol_sha256"], "cohort_csv_sha256": parent["cohort_csv_sha256"],
        "technical_audit_sha256": audit_sha, "config_sha256": preprocess_sha, "variants": [VARIANT],
        "reference_variant": SHHS_REFERENCE_VARIANT, "environment": _environment(),
        "identity_check": ("record_key, subject_id, y, valid_mask, original_epoch_index, raw_stage and source "
                           f"hashes equal {SHHS_REFERENCE_VARIANT}"),
        "analysis_cohort": parent["analysis_cohort"],
        "input_scale_in_window": {"E6_filtered_zscore_v2": _quartiles(e6_std), "E6R": _quartiles(e6r_std)},
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, sort_keys=False)
    return manifest
