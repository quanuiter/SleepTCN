"""Pack and verify the inputs of the final GPU session (besides the A4 Sleep-EDF bundle).

Parts
  core     archived fold checkpoints, telemetry A4 inputs and the files of the device check
  holdout  SHHS1 holdout A4 inputs (NSRR data: check the data use agreement before uploading)
  isruc    preprocessed ISRUC subgroup I inputs (after scripts/a5_prepare_isruc.py)

  python scripts/final_bundle.py pack --part core --output C:/Users/ADMIN/SleepTCN_upload/final_core.tar
  python scripts/final_bundle.py verify --part core      (on the GPU machine, after unpacking at the repo root)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.io.hashing import sha256_file  # noqa: E402

CHECK_RECORDS = ("ST7011J", "ST7022J")


def part_files(part: str) -> list[Path]:
    if part == "core":
        files = sorted(p for p in (ROOT / "runs/holdout_ckpt_restore_20261010").rglob("*") if p.is_file())
        files += sorted(p for p in (ROOT / "data/processed/telemetry_a4").rglob("*.npz"))
        for key in CHECK_RECORDS:
            files += [ROOT / f"data/processed/telemetry_a3/{v}/{key}.npz" for v in ("paper_raw_v1", "bandpass_v2")]
            files += [ROOT / f"runs/a3/telemetry/ensemble/{a}/{key}.npz" for a in ("E0_s42", "E4_s42")]
        return files
    if part == "holdout":
        return sorted(p for p in (ROOT / "data/processed/shhs_holdout_a4").rglob("*.npz"))
    if part == "isruc":
        return sorted(p for p in (ROOT / "data/processed/isruc_a5").rglob("*") if p.is_file())
    raise ValueError(part)


def manifest_path(part: str) -> Path:
    return ROOT / "runs/final/manifests" / f"bundle_{part}.json"


def pack(args: argparse.Namespace) -> None:
    files = part_files(args.part)
    missing = [str(p) for p in files if not p.is_file()]
    if not files or missing:
        raise SystemExit(f"part {args.part}: no files or missing files, e.g. {missing[:3]}")
    manifest = {"part": args.part, "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p)}
                                            for p in files]}
    destination = Path(args.output).resolve()
    with tarfile.open(destination, "w") as archive:
        for p in files:
            archive.add(p, arcname=p.relative_to(ROOT).as_posix())
        payload = json.dumps(manifest, indent=1).encode("utf-8")
        info = tarfile.TarInfo(manifest_path(args.part).relative_to(ROOT).as_posix())
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(f"BUNDLE {destination} ({destination.stat().st_size / 1e9:.2f} GB, {len(files)} files)")


def verify(args: argparse.Namespace) -> None:
    path = manifest_path(args.part)
    if not path.is_file():
        raise SystemExit(f"{path} not found; unpack the {args.part} bundle at the repository root")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    bad = [f["path"] for f in manifest["files"]
           if not (ROOT / f["path"]).is_file() or sha256_file(ROOT / f["path"]) != f["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} files missing or changed, e.g. {bad[:3]}")
    print(f"BUNDLE VERIFIED ({args.part}): {len(manifest['files'])} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["pack", "verify"])
    parser.add_argument("--part", choices=["core", "holdout", "isruc"], required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.action == "pack" and not args.output:
        raise SystemExit("--output is required for pack (outside the repository)")
    {"pack": pack, "verify": verify}[args.action](args)


if __name__ == "__main__":
    main()
