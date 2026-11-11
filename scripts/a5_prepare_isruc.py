"""Amendment A5: preprocess ISRUC-Sleep subgroup I (channels C4-A1 and F4-A1) as specified in
configs/amendment_a5.json. No model and no prediction is involved.

Input: a directory with one folder per subject, N/N.rec (EDF) and N/N_1.txt, N/N_2.txt (scorer labels),
as unpacked from the subgroup I archives N.rar. If only N.rar exists, it is unpacked with `unrar` or `7z`
when one of them is installed.

  python scripts/a5_prepare_isruc.py inspect --raw-root /workspace/isruc   (formats only; writes nothing)
  python scripts/a5_prepare_isruc.py run --raw-root /workspace/isruc
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyedflib  # noqa: E402

from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.io.serialization import atomic_savez  # noqa: E402
from sleeptcn.preprocessing import PreprocessConfig, preprocess_signal_variant, trim_sleep_window  # noqa: E402

CONFIG = json.loads((ROOT / "configs/amendment_a5.json").read_text(encoding="utf8"))
COHORT = CONFIG["cohort"]
CHANNELS = {"C4": ("C4-A1", "C4-M1", "C4"), "F4": ("F4-A1", "F4-M1", "F4")}
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}  # ISRUC code -> internal (0 W, 1 N1, 2 N2, 3 N3, 4 REM)
VARIANTS = tuple(COHORT["variants"])
OUTPUT = ROOT / "data/processed/isruc_a5"
TARGET_FS = 100


def unpack(raw_root: Path, subject: str) -> None:
    archive = raw_root / f"{subject}.rar"
    if (raw_root / subject).is_dir() or not archive.exists():
        return
    tool = shutil.which("unrar") or shutil.which("7z")
    if tool is None:
        raise SystemExit(f"{archive.name}: install unrar or p7zip (7z), or unpack the archives first")
    command = [tool, "x", "-o+", str(archive), str(raw_root) + "/"] if "unrar" in tool else \
        [tool, "x", "-y", f"-o{raw_root}", str(archive)]
    subprocess.run(command, check=True, capture_output=True)


def find_subject_files(raw_root: Path, subject: str) -> tuple[Path, Path, Path]:
    folder = raw_root / subject
    rec = next(iter(sorted(folder.rglob(f"{subject}.rec")) + sorted(folder.rglob(f"{subject}.edf"))), None)
    labels = [next(iter(sorted(folder.rglob(f"{subject}_{s}.txt"))), None) for s in (1, 2)]
    if rec is None or None in labels:
        raise FileNotFoundError(f"subject {subject}: recording or label files not found in {folder}")
    return rec, labels[0], labels[1]


def open_edf(path: Path):
    """pyedflib needs an .edf name on some platforms; read a .rec through a temporary hard copy."""
    if path.suffix.lower() == ".edf":
        return pyedflib.EdfReader(str(path)), None
    tmp = Path(tempfile.mkdtemp()) / (path.stem + ".edf")
    shutil.copyfile(path, tmp)
    return pyedflib.EdfReader(str(tmp)), tmp


def read_labels(path: Path) -> np.ndarray:
    values = [line.strip() for line in path.read_text(encoding="latin-1").splitlines() if line.strip()]
    codes = np.array([int(float(v)) for v in values], dtype=np.int64)
    return np.array([LABEL_MAP.get(int(c), -1) for c in codes], dtype=np.int8), codes


def align(labels: np.ndarray, epochs: int) -> tuple[np.ndarray, int]:
    if len(labels) >= epochs:
        return labels[:epochs], len(labels) - epochs
    return np.concatenate([labels, np.full(epochs - len(labels), -1, dtype=np.int8)]), len(labels) - epochs


def read_subject(raw_root: Path, subject: str) -> dict:
    unpack(raw_root, subject)
    rec, lab1, lab2 = find_subject_files(raw_root, subject)
    reader, tmp = open_edf(rec)
    try:
        labels = reader.getSignalLabels()
        out = {"subject": subject, "rec": rec, "rec_sha256": sha256_file(rec), "signal_labels": labels,
               "channels": {}}
        for name, accepted in CHANNELS.items():
            match = [i for i, label in enumerate(labels) if label.strip().upper() in {a.upper() for a in accepted}]
            if not match:
                out["channels"][name] = {"error": "missing"}
                continue
            index = match[0]
            unit = reader.getPhysicalDimension(index).strip().lower().replace("μ", "u").replace("µ", "u")
            fs = float(reader.getSampleFrequency(index))
            signal = reader.readSignal(index).astype(np.float64)
            out["channels"][name] = {"label": labels[index], "unit": unit, "fs": fs, "signal": signal}
    finally:
        reader.close()
        if tmp is not None:
            shutil.rmtree(tmp.parent, ignore_errors=True)
    out["labels1"], out["codes1"] = read_labels(lab1)
    out["labels2"], out["codes2"] = read_labels(lab2)
    return out


def check(subject: dict) -> list[str]:
    problems = []
    for name, info in subject["channels"].items():
        if "error" in info:
            problems.append(f"{name}: channel missing")
        elif info["unit"] != "uv":
            problems.append(f"{name}: unit {info['unit']!r}, expected microvolts")
        elif not np.isfinite(info["signal"]).all() or info["signal"].std() < 1.0:
            problems.append(f"{name}: signal not finite or flat")
    return problems


def resample(signal: np.ndarray, fs: float) -> np.ndarray:
    if not float(fs).is_integer():
        raise ValueError(f"non-integer sampling rate {fs}")
    g = math.gcd(int(fs), TARGET_FS)
    up, down = TARGET_FS // g, int(fs) // g
    if up == down == 1:
        return signal
    return resample_poly(signal, up, down, window=("kaiser", 5.0), padtype="constant")


def write_subject(subject: dict, config: PreprocessConfig) -> dict:
    spe = config.samples_per_epoch
    rows = {}
    for name, info in subject["channels"].items():
        continuous = resample(info["signal"], info["fs"])
        epochs = continuous.size // spe
        continuous = continuous[: epochs * spe]  # trailing incomplete epoch dropped
        y1, adjust1 = align(subject["labels1"], epochs)
        y2, adjust2 = align(subject["labels2"], epochs)
        raw_epochs = continuous.reshape(epochs, spe)
        _, y_window, index, start, stop = trim_sleep_window(raw_epochs, y1, config.wake_edge_epochs)
        key = f"ISRUC_I_{int(subject['subject']):03d}"
        base = {"y": y_window.astype(np.int8), "valid_mask": y_window >= 0, "original_epoch_index": index,
                "y_scorer2": y2[start:stop].astype(np.int8), "record_key": np.array(key),
                "subject_id": np.array(key), "dataset": np.array("ISRUC-Sleep subgroup I"),
                "channel": np.array(info["label"]), "source_rec_sha256": np.array(subject["rec_sha256"]),
                "source_hashes_verified": np.bool_(False), "source_sampling_rate_hz": np.float32(info["fs"]),
                "sampling_rate_hz": np.float32(TARGET_FS), "epoch_seconds": np.int16(config.epoch_seconds),
                "samples_per_epoch": np.int16(spe), "trim_anchor_policy": np.array(config.trim_anchor_policy),
                "trim_start_epoch": np.int32(start), "trim_stop_epoch_exclusive": np.int32(stop)}
        for variant in VARIANTS:
            if variant == "paper_raw_v1":
                x = raw_epochs[start:stop].astype(np.float32)
                extra = {"filter": np.array("none_after_mandatory_resample_poly_antialias")}
            else:
                processed, _, metadata = preprocess_signal_variant(continuous, variant, config,
                                                                   window_epochs=(start, stop))
                x = processed.reshape(epochs, spe)[start:stop]
                extra = {"filter": np.array("butterworth_sosfiltfilt_order4_0.5-30.0Hz"),
                         "normalization": np.array(metadata["normalization"])}
            path = OUTPUT / name / variant / f"{key}.npz"
            if path.exists():
                raise FileExistsError(f"{path} exists; A5 inputs are write-once")
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_savez(path, {**base, **extra, "x": x, "preprocess_version": np.array(variant)})
        rows[name] = {"record_key": key, "label": info["label"], "fs": info["fs"], "epochs": int(epochs),
                      "window": [int(start), int(stop)], "scored_epochs_scorer1": int((y_window >= 0).sum()),
                      "label_minus_signal_epochs_scorer1": int(adjust1),
                      "label_minus_signal_epochs_scorer2": int(adjust2)}
    return rows


def subjects_in(raw_root: Path) -> list[str]:
    names = {p.stem for p in raw_root.glob("*.rar")} | {p.name for p in raw_root.iterdir() if p.is_dir()}
    return sorted((n for n in names if n.isdigit()), key=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["inspect", "run"])
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    subjects = subjects_in(args.raw_root)
    if not subjects:
        raise SystemExit(f"no subject folders or archives in {args.raw_root}")
    config = PreprocessConfig()
    included, excluded, rows = [], {}, {}
    for subject in subjects:
        data = read_subject(args.raw_root, subject)
        problems = check(data)
        summary = {n: {k: v for k, v in i.items() if k != "signal"} for n, i in data["channels"].items()}
        codes = sorted(set(np.unique(data["codes1"]).tolist()) | set(np.unique(data["codes2"]).tolist()))
        print(subject, summary, "label codes", codes, "labels", len(data["labels1"]), len(data["labels2"]),
              "problems", problems, flush=True)
        if problems:
            excluded[subject] = problems
            continue
        included.append(subject)
        if args.action == "run":
            rows[subject] = write_subject(data, config)
    if args.action == "inspect":
        print(f"INSPECTED {len(subjects)} subjects; would exclude {excluded}")
        return
    manifest = {"schema_version": 1, "amendment": "A5", "dataset": "ISRUC-Sleep subgroup I",
                "config": asdict(config), "variants": list(VARIANTS), "included_subjects": included,
                "excluded_subjects": excluded, "rows": rows,
                "a5_config_sha256": sha256_file(ROOT / "configs/amendment_a5.json")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "isruc_preprocess_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"PASS: {len(included)} subjects written, {len(excluded)} excluded")


if __name__ == "__main__":
    main()
