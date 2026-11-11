"""Amendment A4: one-edge filter variants built from the stored unfiltered evaluation-window epochs.

The raw recordings are not kept locally, but every stored ``paper_raw_v1`` record holds the
contiguous unfiltered evaluation window (calibrated microvolts, after the mandatory resampling of
SHHS1). Each A4 variant applies one edge of the locked 0.5-30 Hz band-pass to that window:

* ``highpass_v4``: zero-phase Butterworth high-pass, order 4, 0.5 Hz (second-order sections,
  forward-backward), no low-pass;
* ``lowpass_v4``: zero-phase Butterworth low-pass, order 4, 30 Hz, no high-pass.

``butter(4, [0.5, 30], 'bandpass')`` has fourth-order skirts at each edge, so each A4 variant keeps
one edge of the locked filter unchanged. Filtering the window instead of the whole recording differs
only near the window ends; the same window route is therefore checked to reproduce the stored
``bandpass_v2`` input of every record, and the epochs where it does not are reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .io.hashing import sha256_file
from .io.serialization import atomic_savez

FS = 100.0
ORDER = 4
VARIANTS = {
    "highpass_v4": ("highpass", 0.5),
    "lowpass_v4": ("lowpass", 30.0),
}
CHECK_VARIANT = "bandpass_v2"
CHECK_RELATIVE_TOLERANCE = 1e-4
COHORTS = {
    "sleep_edf": {"input": Path("data/processed"), "output": Path("data/processed"), "pattern": "SC*.npz"},
    "shhs1_holdout": {"input": Path("data/processed/shhs_holdout_v1"),
                      "output": Path("data/processed/shhs_holdout_a4"), "pattern": "shhs1-*.npz"},
    "telemetry": {"input": Path("data/processed/telemetry_a3"),
                  "output": Path("data/processed/telemetry_a4"), "pattern": "ST*.npz"},
}
MANIFEST = Path("runs/a4/manifests/a4_preprocess_manifest.json")


def _sos(kind: str, cutoff: float | tuple[float, float]) -> np.ndarray:
    return butter(ORDER, cutoff, btype=kind, fs=FS, output="sos")


def filter_window(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter the contiguous window (epochs x samples) as one signal and return it as float32 epochs."""
    return sosfiltfilt(sos, x.astype(np.float64).ravel()).reshape(x.shape).astype(np.float32)


def window_check(raw: dict[str, np.ndarray], stored_bandpass: np.ndarray) -> dict[str, Any]:
    """Rebuild ``bandpass_v2`` through the window route and compare it with the stored input."""
    rebuilt = filter_window(raw["x"], _sos("bandpass", (0.5, 30.0)))
    scale = np.maximum(stored_bandpass.std(axis=1), 1e-6)
    relative = np.abs(rebuilt.astype(np.float64) - stored_bandpass).max(axis=1) / scale
    differing = np.flatnonzero(relative > CHECK_RELATIVE_TOLERANCE)
    n = len(relative)
    interior = [int(i) for i in differing if 0 < i < n - 1]
    return {"epochs": n, "differing_epochs": [int(i) for i in differing], "interior_differing": interior,
            "differing_valid_epochs": int(raw["valid_mask"][differing].sum()),
            "max_relative_difference_interior": float(np.delete(relative, [0, n - 1]).max()) if n > 2 else 0.0}


def build_record(raw_path: Path, check_path: Path, output_root: Path) -> dict[str, Any]:
    with np.load(raw_path, allow_pickle=False) as npz:
        raw = {name: npz[name].copy() for name in npz.files}
    if str(raw["preprocess_version"].item()) != "paper_raw_v1":
        raise ValueError(f"{raw_path}: not a paper_raw_v1 record")
    if not np.all(np.diff(raw["original_epoch_index"]) == 1):
        raise ValueError(f"{raw_path}: evaluation window is not contiguous")
    with np.load(check_path, allow_pickle=False) as npz:
        if not np.array_equal(npz["original_epoch_index"], raw["original_epoch_index"]):
            raise ValueError(f"{raw_path.name}: bandpass_v2 covers other epochs")
        check = window_check(raw, npz["x"])
    if check["interior_differing"]:
        raise ValueError(f"{raw_path.name}: window route differs from bandpass_v2 inside the window: {check}")
    outputs = {}
    for variant, (kind, cutoff) in VARIANTS.items():
        record = dict(raw)
        record["x"] = filter_window(raw["x"], _sos(kind, cutoff))
        record["preprocess_version"] = np.array(variant)
        record["filter"] = np.array(f"butterworth_sosfiltfilt_order{ORDER}_{kind}_{cutoff}Hz_on_evaluation_window")
        record["normalization"] = np.array("none")
        record["normalization_scope"] = np.array("none")
        record["normalization_mean"] = np.float64(np.nan)
        record["normalization_std"] = np.float64(np.nan)
        path = output_root / variant / raw_path.name
        if path.exists():
            raise FileExistsError(f"{path} exists; A4 inputs are write-once")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_savez(path, record)
        outputs[variant] = {"path": path.as_posix(), "sha256": sha256_file(path)}
    return {"record_key": raw_path.stem, "raw_sha256": sha256_file(raw_path), "window_check": check,
            "outputs": outputs}


def prepare(workspace: Path, cohorts: list[str]) -> dict[str, Any]:
    workspace = workspace.resolve()
    manifest_path = workspace / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "schema_version": 1, "amendment": "A4", "filter_order": ORDER, "variants": {
            k: {"type": v[0], "cutoff_hz": v[1]} for k, v in VARIANTS.items()},
        "route": "stored contiguous paper_raw_v1 evaluation window filtered as one signal",
        "check": f"same route reproduces stored {CHECK_VARIANT} within relative {CHECK_RELATIVE_TOLERANCE} "
                 "except at the first and last window epoch", "cohorts": {}}
    for cohort in cohorts:
        spec = COHORTS[cohort]
        source = workspace / spec["input"]
        rows = []
        for raw_path in sorted((source / "paper_raw_v1").glob(spec["pattern"])):
            rows.append(build_record(raw_path, source / CHECK_VARIANT / raw_path.name, workspace / spec["output"]))
        edges = sum(len(r["window_check"]["differing_epochs"]) for r in rows)
        manifest["cohorts"][cohort] = {
            "records": len(rows), "window_edge_epochs_differing": edges,
            "window_edge_valid_epochs_differing": sum(r["window_check"]["differing_valid_epochs"] for r in rows),
            "scored_epochs": None, "rows": rows}
        print(f"{cohort}: {len(rows)} records, {edges} edge epochs differ from the full-record filter", flush=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
