"""Import the Colab GPU results of amendment A1 into runs/holdout_amendment_a1 (NumPy only, no PyTorch).

The Colab archive holds probabilities by record index (r000..r499) only. The local mapping
(colab_mapping.json, never uploaded) restores record keys and input hashes. Fold-prediction files are
written in the same format and metadata schema as run_holdout_amendment_a1.py, with device "cuda".

    python scripts/import_colab_a1_results.py --mode prepare          # set aside partial CPU outputs, compare, copy AdaBN states
    python scripts/import_colab_a1_results.py --mode records --start 0 --stop 50   # ... up to 500
    python scripts/import_colab_a1_results.py --mode check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path, PureWindowsPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "holdout_amendment_a1"
ARCHIVE = OUT / "colab_results" / "SleepTCN_A1_Colab_Results.zip"
EXTRACT = OUT / "colab_results" / "extracted"
CPU_PARTIAL = OUT / "cpu_partial_run_20261012"
ROLE, VARIANT, ARM, SEED = "holdout_test", "bandpass_v2", "E4_s42", 42

_spec = importlib.util.spec_from_file_location("serialization", ROOT / "src" / "sleeptcn" / "io" / "serialization.py")
serialization = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serialization)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_path(windows_path: str) -> Path:
    parts = PureWindowsPath(windows_path).parts
    i = [p.lower() for p in parts].index("sleeptcn")
    return ROOT.joinpath(*parts[i + 1:])


def context():
    amendment_path = ROOT / "configs" / "shhs_holdout_amendment_a1.json"
    amendment = json.loads(amendment_path.read_bytes())
    pre_path = OUT / "colab_inputs" / "preprocess_holdout_v1.json"
    if sha_file(pre_path) != amendment["cohort"]["preprocess_manifest_sha256"]:
        raise ValueError("preprocess manifest differs")
    pre = {r["record_key"]: r for r in json.loads(pre_path.read_bytes())["records"] if r["variant"] == VARIANT}
    mapping = json.loads((OUT / "colab_mapping.json").read_bytes())["records"]
    for m in mapping:
        if pre[m["record_key"]]["output_sha256"] != m["input_npz_sha256"]:
            raise ValueError(f"mapping hash differs for {m['record_key']}")
    return {"amendment": amendment, "amendment_sha256": sha_file(amendment_path),
            "parent_protocol_sha256": amendment["parent_protocol"]["config_sha256"],
            "preprocess_manifest_sha256": amendment["cohort"]["preprocess_manifest_sha256"],
            "pre": pre, "mapping": mapping, "archive_sha256": sha_file(ARCHIVE)}


def units(ctx):
    rows = {}
    for path in sorted(EXTRACT.glob("unit_*.npz")):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"].item()))
        if meta["amendment_sha256"] != ctx["amendment_sha256"]:
            raise ValueError(f"{path.name}: other amendment")
        rows[(meta["condition"], meta["outer_fold"])] = (z, meta)
    expected = {(c, f) for c in ctx["amendment"]["conditions"] for f in range(10)}
    if set(rows) != expected:
        raise ValueError(f"unit set differs: missing {sorted(expected - set(rows))}")
    return rows


def fold_file(condition: str, fold: int, key: str) -> Path:
    return OUT / "fold_predictions" / condition / f"fold_{fold:02d}" / f"{key}.npz"


def prepare(ctx) -> None:
    if not EXTRACT.exists():
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(EXTRACT)
    gate = json.loads((EXTRACT / "colab_gate.json").read_bytes())
    assert all(c["max_abs_difference"] <= 1e-5 and c["argmax_agreement"] == 1.0 for c in gate["checks"])
    # Set aside the partial CPU run (kept for comparison, never mixed with GPU outputs).
    for name in ("fold_predictions", "adabn_state"):
        if (OUT / name).exists() and not (CPU_PARTIAL / name).exists():
            CPU_PARTIAL.mkdir(parents=True, exist_ok=True)
            shutil.move(str(OUT / name), str(CPU_PARTIAL / name))
    rows = units(ctx)
    by_key = {m["record_key"]: m["idx"] for m in ctx["mapping"]}
    diffs, agree, n = [], 0, 0
    for path in sorted((CPU_PARTIAL / "fold_predictions").glob("*/fold_*/*.npz")):
        condition, fold = path.parts[-3], int(path.parts[-2].split("_")[1])
        with np.load(path, allow_pickle=False) as c:
            cpu = c["probabilities"]
        gpu = rows[(condition, fold)][0][f"p{by_key[path.stem]:03d}"]
        diffs.append(float(np.abs(cpu - gpu).max()))
        agree += int((cpu.argmax(1) == gpu.argmax(1)).sum())
        n += len(cpu)
    comparison = {"cpu_fold_files_compared": len(diffs), "max_abs_difference": max(diffs) if diffs else None,
                  "median_max_abs_difference": float(np.median(diffs)) if diffs else None,
                  "argmax_agreement": agree / n if n else None, "epochs_compared": n}
    adabn = {}
    (OUT / "adabn_state").mkdir(parents=True, exist_ok=True)
    prep = json.loads((OUT / "prepare_manifest.json").read_bytes())
    for fold in range(10):
        src = EXTRACT / f"adabn_state_f{fold:02d}.npz"
        with np.load(src, allow_pickle=False) as z:
            if str(z["adabn_sample_sha256"].item()) != prep["adabn_sample_sha256"]:
                raise ValueError("AdaBN state built from another sample")
            arrays = {k: z[k] for k in z.files}
        dst = OUT / "adabn_state" / f"fold_{fold:02d}.npz"
        if not dst.exists():
            serialization.atomic_savez(dst, arrays)
        adabn[fold] = sha_file(dst)
    cpu_adabn = CPU_PARTIAL / "adabn_state" / "fold_00.npz"
    if cpu_adabn.exists():
        with np.load(cpu_adabn) as c, np.load(OUT / "adabn_state" / "fold_00.npz") as g:
            comparison["adabn_fold00_running_var_max_rel_difference"] = max(
                float(np.max(np.abs(c[k] - g[k]) / np.maximum(np.abs(c[k]), 1e-12)))
                for k in c.files if k.endswith("running_var"))
    report = {"archive_sha256": ctx["archive_sha256"], "colab_gate": gate, "cpu_vs_gpu": comparison,
              "adabn_state_sha256": adabn}
    (OUT / "colab_import_prepare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["cpu_vs_gpu"], indent=2))


def write_records(ctx, start: int, stop: int) -> None:
    rows = units(ctx)
    adabn = json.loads((OUT / "colab_import_prepare.json").read_bytes())["adabn_state_sha256"]
    conditions = ctx["amendment"]["conditions"]
    written = 0
    for m in ctx["mapping"][start:stop]:
        entry = ctx["pre"][m["record_key"]]
        path = local_path(entry["output_path"])
        if sha_file(path) != entry["output_sha256"]:
            raise ValueError(f"{path.name}: input hash differs")
        with np.load(path, allow_pickle=False) as z:
            index = z["original_epoch_index"].copy()
            n = len(z["x"])
        for (condition, fold), (unit, meta) in rows.items():
            dest = fold_file(condition, fold, m["record_key"])
            if dest.exists():
                continue
            p = unit[f"p{m['idx']:03d}"]
            if p.shape != (n, 5) or not np.isfinite(p).all():
                raise ValueError(f"{condition}/{fold}/{m['idx']}: invalid probabilities")
            spec = conditions[condition]
            metadata = {"schema_version": 1, "amendment": "A1", "amendment_sha256": ctx["amendment_sha256"],
                        "parent_protocol_sha256": ctx["parent_protocol_sha256"],
                        "preprocess_manifest_sha256": ctx["preprocess_manifest_sha256"], "arm": ARM,
                        "condition": condition, "condition_type": spec["type"], "input_factor": float(spec["factor"]),
                        "data_variant": VARIANT, "seed": SEED, "role": ROLE, "device": meta["device"],
                        "torch_version": meta["torch_version"], "gpu": meta["gpu"],
                        "extractor_sha256": meta["extractor_sha256"],
                        "sequence_checkpoint_sha256": meta["sequence_checkpoint_sha256"], "outer_fold": fold,
                        "execution": "colab_gpu_import", "colab_results_archive_sha256": ctx["archive_sha256"],
                        "artifact_type": "amendment_a1_fold_prediction", "record_key": m["record_key"],
                        "subject_id": m["subject_id"], "input_npz_sha256": entry["output_sha256"]}
            if spec["type"] == "adabn":
                metadata["adabn_state_sha256"] = adabn[str(fold)]
            serialization.atomic_savez(dest, {"metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
                                              "probabilities": p.astype(np.float32),
                                              "original_epoch_index": index})
            written += 1
    print(f"records {start}-{stop}: {written} fold files written")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("prepare", "records", "check"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=500)
    args = parser.parse_args()
    ctx = context()
    if args.mode == "prepare":
        prepare(ctx)
    elif args.mode == "records":
        write_records(ctx, args.start, args.stop)
    else:
        n = sum(1 for _ in (OUT / "fold_predictions").glob("*/fold_*/*.npz"))
        print(f"{n}/25000 fold files")


if __name__ == "__main__":
    main()
