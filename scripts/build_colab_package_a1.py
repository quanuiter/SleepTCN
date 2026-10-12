"""Build the label-free, identifier-free Colab package for amendment A1 (GPU execution of the same conditions).

Contents uploaded to Colab: bandpass_v2 signals only (records renamed r000..r499 in record-key order), the
AdaBN sample signals, the E4 seed-42 checkpoints, stored parent probabilities for the reproduction check,
model code and configuration. No labels, no record keys, no subject identifiers. The mapping from index to
record is written locally to ``colab_mapping.json`` and never packaged.

Runs with NumPy only (no PyTorch). Each call writes one part so that it fits a short time budget:
    python scripts/build_colab_package_a1.py --part code
    python scripts/build_colab_package_a1.py --part adabn
    python scripts/build_colab_package_a1.py --part records --index 0      # ... --index 19 (25 records each)
    python scripts/build_colab_package_a1.py --part manifest
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PureWindowsPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "holdout_amendment_a1"
PKG = OUT / "colab_package"
RECORDS_PER_PART = 25
VARIANT = "bandpass_v2"
CODE_FILES = ["src/sleeptcn/__init__.py", "src/sleeptcn/models.py", "configs/experiments_v2.json",
              "configs/shhs_holdout_amendment_a1.json", "scripts/run_holdout_amendment_a1.py",
              "scripts/colab_amendment_a1_driver.py"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_path(windows_path: str) -> Path:
    parts = PureWindowsPath(windows_path).parts
    index = [p.lower() for p in parts].index("sleeptcn")
    return ROOT.joinpath(*parts[index + 1:])


def entries(pre_path: Path, expected_sha: str) -> list[dict]:
    if sha_file(pre_path) != expected_sha:
        raise ValueError("preprocess manifest hash differs from the amendment")
    pre = json.loads(pre_path.read_bytes())
    rows = sorted((r for r in pre["records"] if r["variant"] == VARIANT), key=lambda r: r["record_key"])
    if len(rows) != 500:
        raise ValueError("expected 500 bandpass_v2 records")
    return rows


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_part(name: str, members: dict[str, bytes], info: dict) -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    path = PKG / name
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for member, data in members.items():
            z.writestr(member, data)
    tmp.replace(path)
    info.update({"part": name, "bytes": path.stat().st_size, "sha256": sha_file(path),
                 "members": {m: sha(d) for m, d in members.items()}})
    (PKG / f"{name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{name}: {info['bytes'] / 1e6:.1f} MB, sha256 {info['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True, choices=("code", "adabn", "records", "manifest"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--preprocess-manifest", type=Path, default=OUT / "colab_inputs" / "preprocess_holdout_v1.json")
    args = parser.parse_args()
    amendment = json.loads((ROOT / "configs" / "shhs_holdout_amendment_a1.json").read_bytes())
    rows = entries(args.preprocess_manifest, amendment["cohort"]["preprocess_manifest_sha256"])
    prep = json.loads((OUT / "prepare_manifest.json").read_bytes())

    if args.part == "code":
        members = {f: (ROOT / f).read_bytes() for f in CODE_FILES}
        restore = ROOT / "runs" / "holdout_ckpt_restore_20261010" / "runs" / "v2" / "full" / "E4"
        for fold in range(10):
            base = restore / f"fold_{fold:02d}" / "seed_42" / "checkpoints"
            members[f"checkpoints/fold_{fold:02d}/resnet1d.pt"] = (base / "resnet1d" / "best.pt").read_bytes()
            members[f"checkpoints/fold_{fold:02d}/tcn.pt"] = (base / "sequence" / "tcn" / "best.pt").read_bytes()
        reference = {}
        parent = ROOT / "runs" / "holdout_inference_v1" / "fold_predictions" / "E4_s42"
        for fold in (0, 9):
            for idx in (0, 1):
                with np.load(parent / f"fold_{fold:02d}" / f"{rows[idx]['record_key']}.npz", allow_pickle=False) as z:
                    reference[f"p_f{fold:02d}_r{idx:03d}"] = z["probabilities"].copy()
        members["gate_reference.npz"] = (lambda b: (np.savez(b, **reference), b.getvalue())[1])(io.BytesIO())
        write_part("part_code.zip", members, {"contains": "code, configs, E4 seed-42 checkpoints, reference probabilities"})
    elif args.part == "adabn":
        sample_path = OUT / "adabn_sample.npz"
        if sha_file(sample_path) != prep["adabn_sample_sha256"]:
            raise ValueError("AdaBN sample differs from prepare manifest")
        with np.load(sample_path, allow_pickle=False) as z:
            x = z["x"].copy()
        write_part("part_adabn.zip", {"adabn_sample_x.npy": npy_bytes(x)},
                   {"contains": "AdaBN sample signals only", "adabn_sample_sha256": prep["adabn_sample_sha256"]})
    elif args.part == "records":
        start = args.index * RECORDS_PER_PART
        members, records = {}, []
        for idx in range(start, min(start + RECORDS_PER_PART, len(rows))):
            row = rows[idx]
            path = local_path(row["output_path"])
            if sha_file(path) != row["output_sha256"]:
                raise ValueError(f"{path.name}: hash differs from preprocess manifest")
            with np.load(path, allow_pickle=False) as z:
                x = z["x"].copy()
            data = npy_bytes(x)
            members[f"records/r{idx:03d}.npy"] = data
            records.append({"idx": idx, "n_epochs": int(len(x)), "npy_sha256": sha(data),
                            "record_key": row["record_key"], "subject_id": row["subject_id"],
                            "input_npz_sha256": row["output_sha256"]})
        write_part(f"part_records_{args.index:02d}.zip", members, {"contains": "signals only", "local_only_records": records})
    else:
        infos = [json.loads(p.read_bytes()) for p in sorted(PKG.glob("part_*.zip.json"))]
        names = {i["part"] for i in infos}
        expected = {"part_code.zip", "part_adabn.zip"} | {f"part_records_{k:02d}.zip" for k in range(20)}
        if names != expected:
            raise ValueError(f"missing parts: {sorted(expected - names)}")
        mapping, records = [], []
        for info in infos:
            if sha_file(PKG / info["part"]) != info["sha256"]:
                raise ValueError(f"{info['part']}: changed after it was built")
            for r in info.get("local_only_records", []):
                mapping.append(r)
                records.append({"idx": r["idx"], "n_epochs": r["n_epochs"], "npy_sha256": r["npy_sha256"]})
        mapping.sort(key=lambda r: r["idx"])
        records.sort(key=lambda r: r["idx"])
        if [r["idx"] for r in records] != list(range(500)):
            raise ValueError("record indices incomplete")
        (OUT / "colab_mapping.json").write_text(json.dumps({"records": mapping}, indent=2), encoding="utf-8")
        manifest = {"package": "SleepTCN amendment A1 Colab package", "amendment_sha256": sha_file(
                        ROOT / "configs" / "shhs_holdout_amendment_a1.json"),
                    "labels_included": False, "record_keys_included": False, "participant_ids_included": False,
                    "adabn_sample_sha256": prep["adabn_sample_sha256"],
                    "parts": {i["part"]: {"bytes": i["bytes"], "sha256": i["sha256"],
                                          "members": i["members"]} for i in infos},
                    "records": records}
        (PKG / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("package_manifest.json sha256", sha_file(PKG / "package_manifest.json"))


if __name__ == "__main__":
    main()
