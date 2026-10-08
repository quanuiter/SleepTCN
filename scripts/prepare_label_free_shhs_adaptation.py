"""Prepare the five locked adaptation recordings on CPU, without opening annotations.

Participant-level outputs must remain in ignored local data/cache storage. This
does not execute inference, EM, model selection, or evaluation on test labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyedflib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.cpu_followups import label_free_signal_variants
from sleeptcn.preprocessing import sha256_file
from sleeptcn.shhs_preprocessing import SHHSPreprocessConfig, normalize_uv_unit


def locked_adaptation_subjects(selection_raw: bytes, audit: dict) -> list[dict]:
    selection = json.loads(selection_raw)
    if selection.get("dataset") != "SHHS Visit 1" or selection.get("selection_seed") != 42:
        raise ValueError("expected locked SHHS Visit 1 selection seed 42")
    if (audit.get("status") != "passed"
            or audit.get("manifest_sha256") != hashlib.sha256(selection_raw).hexdigest()):
        raise ValueError("technical audit is not linked to this selection")
    selected = [s for s in selection["subjects"] if s["role"] == "adaptation"]
    if len(selected) != 5 or len({s["subject_id"] for s in selected}) != 5:
        raise ValueError("expected exactly five unique locked adaptation subjects")
    for subject in selected:
        record = audit["subjects"][subject["subject_id"]]
        filename = subject["edf_filename"]
        if (not record["passed"] or record["role"] != "adaptation"
                or record["edf_filename"] != filename or Path(filename).name != filename):
            raise ValueError("adaptation EDF identity/audit mismatch")
    return selected


def validate_output(path: Path, epochs: int) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"x", "original_epoch_index", "metadata_json"}:
            raise ValueError("unexpected fields: label-free output schema violated")
        meta = json.loads(str(z["metadata_json"]))
        if (meta["role"] != "adaptation" or meta["uses_sleep_annotations"]
                or meta["epoch_selection"] != "full_record_signal_length_no_labels"):
            raise ValueError("label-free output declaration mismatch")
        if (z["x"].shape != (epochs, 3000) or z["x"].dtype != np.float32
                or not np.isfinite(z["x"]).all()
                or not np.array_equal(z["original_epoch_index"], np.arange(epochs))):
            raise ValueError("full-record signal/alignment validation failed")
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--technical-audit", type=Path, required=True)
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    selection_raw = args.selection_manifest.read_bytes()
    audit = json.loads(args.technical_audit.read_bytes())
    selected = locked_adaptation_subjects(selection_raw, audit)
    cache = (ROOT / "data/cache").resolve()
    output = args.output_root.resolve()
    if not output.is_relative_to(cache) or output == cache:
        raise ValueError("participant outputs require a dedicated directory inside data/cache")
    if output.exists() or args.summary.exists():
        raise FileExistsError("refusing to overwrite an existing output directory/summary")
    for subject in selected:
        if not (args.edf_dir / subject["edf_filename"]).is_file():
            raise FileNotFoundError("a locked adaptation EDF is unavailable")
    cfg = replace(SHHSPreprocessConfig(), wake_edge_minutes=0,
                  trim_anchor_policy="full_record_signal_length_no_labels")
    code_paths = [Path(__file__).resolve(), ROOT / "src/sleeptcn/cpu_followups.py",
                  ROOT / "src/sleeptcn/preprocessing.py", ROOT / "src/sleeptcn/shhs_preprocessing.py"]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths}
    output.mkdir(parents=True, exist_ok=False)
    records, total_epochs, total_bytes, clipping_records, maximum_residual = [], 0, 0, 0, 0.
    for index, subject in enumerate(selected, 1):
        edf = args.edf_dir / subject["edf_filename"]
        source_sha = sha256_file(edf)
        if source_sha != audit["subjects"][subject["subject_id"]]["edf_sha256"]:
            raise ValueError("adaptation EDF hash differs from locked technical audit")
        reader = pyedflib.EdfReader(str(edf))
        try:
            channels = [s.strip() for s in reader.getSignalLabels()]
            if channels.count(cfg.source_channel) != 1:
                raise ValueError("expected one primary EEG channel")
            channel = channels.index(cfg.source_channel)
            if (reader.getSampleFrequency(channel) != cfg.source_sampling_rate_hz
                    or normalize_uv_unit(reader.getPhysicalDimension(channel)) != "uv"):
                raise ValueError("unexpected source sampling rate or physical unit")
            signal = reader.readSignal(channel)
            duration = reader.getFileDuration()
        finally:
            reader.close()
        if not np.isclose(signal.size / cfg.source_sampling_rate_hz, duration, atol=1e-6, rtol=0):
            raise ValueError("EDF duration differs from EEG sample duration")
        variants = label_free_signal_variants(signal, cfg)
        epochs = signal.size // cfg.source_samples_per_epoch
        record = {"subject_id": subject["subject_id"], "source_edf_sha256": source_sha,
                  "full_record_epochs": epochs, "variants": {}}
        for variant, (x, fraction, variant_metadata) in variants.items():
            folder = output / variant
            folder.mkdir(exist_ok=True)
            path = folder / (edf.stem + ".npz")
            meta = {"subject_id": subject["subject_id"], "role": "adaptation",
                    "uses_sleep_annotations": False,
                    "epoch_selection": "full_record_signal_length_no_labels",
                    "full_record_epochs": epochs, "source_edf_sha256": source_sha,
                    "selection_manifest_sha256": hashlib.sha256(selection_raw).hexdigest(),
                    "config": asdict(cfg), "code_sha256": code_hashes,
                    "data_variant": variant, "clip_threshold_exceedance_fraction": fraction,
                    "preprocessing_metadata": variant_metadata}
            with path.open("xb") as stream:
                np.savez_compressed(stream, x=x,
                                    original_epoch_index=np.arange(epochs, dtype=np.int64),
                                    metadata_json=np.asarray(json.dumps(meta, allow_nan=False)))
            info = validate_output(path, epochs)
            record["variants"][variant] = {"path": str(path), **info,
                                          "clip_threshold_exceedance_fraction": fraction}
            total_bytes += info["bytes"]
        x3, fraction, _ = variants["filtered_v2"]
        x4, _, _ = variants["bandpass_v2"]
        if fraction == 0:
            residual = np.abs(100 * x3.astype(np.float64) - x4)
            if not np.all(residual <= 2e-5 + 3e-7 * np.abs(x4)):
                raise ValueError("scale relationship failed float32 tolerance")
            maximum_residual = max(maximum_residual, float(residual.max()))
        else:
            clipping_records += 1
        records.append(record)
        total_epochs += epochs
        print(f"validated full-record label-free inputs: {index}/5", flush=True)
    summary = {"status": "complete_preprocessing_only_no_inference", "role": "adaptation",
               "subjects": len(records), "variant_files": 2 * len(records),
               "epochs_per_variant": total_epochs, "saved_bytes": total_bytes,
               "uses_sleep_annotations": False, "epoch_selection": cfg.trim_anchor_policy,
               "records_with_clipping": clipping_records,
               "maximum_scale_residual_uv_in_unclipped_records": maximum_residual,
               "selection_manifest_sha256": hashlib.sha256(selection_raw).hexdigest(),
               "technical_audit_sha256": sha256_file(args.technical_audit),
               "code_sha256": code_hashes, "config": asdict(cfg),
               "numpy_version": np.__version__, "pyedflib_version": pyedflib.__version__,
               "historical_prediction_compatibility": "new_full_record_context_requires_new_inference"}
    with (output / "private_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"summary": summary, "records": records}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
