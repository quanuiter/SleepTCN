"""Kiểm kê tên và khóa ghép cặp Sleep-EDF Expanded Sleep Cassette.

Script chỉ đọc dữ liệu nguồn. Dùng --hash để tính SHA-256 cho toàn bộ EDF.
Kiểm tra sâu metadata EDF sẽ được bổ sung ở bước tiếp theo với pyedflib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PREFIX_RE = re.compile(r"^(SC\d{4}[A-Z])")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_key(path: Path) -> str:
    match = PREFIX_RE.match(path.name)
    if match is None:
        raise ValueError(f"Tên tệp không đúng mẫu Sleep Cassette: {path.name}")
    return match.group(1)


def file_info(path: Path, with_hash: bool) -> dict[str, object]:
    key = record_key(path)
    result: dict[str, object] = {
        "name": path.name,
        "path": str(path.resolve()),
        "record_key": key,
        "subject_id": key[:5],
        "size_bytes": path.stat().st_size,
    }
    if with_hash:
        result["sha256"] = sha256_file(path)
    return result


def unique_map(paths: list[Path], kind: str) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for path in paths:
        key = record_key(path)
        if key in mapped:
            raise RuntimeError(
                f"Trùng khóa {kind} {key}: {mapped[key].name} và {path.name}"
            )
        mapped[key] = path
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash", action="store_true", dest="with_hash")
    parser.add_argument("--expected-subjects", type=int, default=78)
    parser.add_argument("--expected-records", type=int, default=153)
    args = parser.parse_args()

    source = args.data_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {source}")

    psg_paths = sorted(source.glob("*-PSG.edf"))
    hyp_paths = sorted(source.glob("*-Hypnogram.edf"))
    psg_map = unique_map(psg_paths, "PSG")
    hyp_map = unique_map(hyp_paths, "Hypnogram")

    psg_keys = set(psg_map)
    hyp_keys = set(hyp_map)
    missing_hyp = sorted(psg_keys - hyp_keys)
    missing_psg = sorted(hyp_keys - psg_keys)
    paired_keys = sorted(psg_keys & hyp_keys)
    subjects = sorted({key[:5] for key in paired_keys})

    report = {
        "schema_version": 1,
        "dataset": "sleep-edf-expanded/sleep-cassette/1.0.0",
        "source_readonly": str(source),
        "hashes_included": args.with_hash,
        "summary": {
            "psg_files": len(psg_paths),
            "hypnogram_files": len(hyp_paths),
            "paired_records": len(paired_keys),
            "subjects": len(subjects),
            "missing_hypnogram_keys": missing_hyp,
            "missing_psg_keys": missing_psg,
        },
        "subjects": subjects,
        "records": [
            {
                "record_key": key,
                "subject_id": key[:5],
                "psg": file_info(psg_map[key], args.with_hash),
                "hypnogram": file_info(hyp_map[key], args.with_hash),
            }
            for key in paired_keys
        ],
    }

    errors: list[str] = []
    if len(psg_paths) != args.expected_records:
        errors.append(f"PSG={len(psg_paths)}, kỳ vọng {args.expected_records}")
    if len(hyp_paths) != args.expected_records:
        errors.append(f"Hypnogram={len(hyp_paths)}, kỳ vọng {args.expected_records}")
    if len(paired_keys) != args.expected_records:
        errors.append(f"Cặp={len(paired_keys)}, kỳ vọng {args.expected_records}")
    if len(subjects) != args.expected_subjects:
        errors.append(f"Đối tượng={len(subjects)}, kỳ vọng {args.expected_subjects}")
    if missing_hyp or missing_psg:
        errors.append("Có khóa PSG/Hypnogram không ghép được")

    report["validation"] = {"passed": not errors, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Manifest: {args.output.resolve()}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: file naming and PSG-Hypnogram pairing are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
