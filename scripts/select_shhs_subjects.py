"""Chọn cố định đối tượng SHHS1 và sinh lệnh tải có lọc.

Script chỉ đọc metadata CSV. Nó không tải dữ liệu, không đọc nhãn epoch và
không chứa token NSRR. Danh sách đối tượng sinh ra phải được lưu ngoài Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.hashing import sha256_file  # noqa: E402


ROLE_COUNTS = {
    "adaptation": 5,
    "validation": 15,
    "test": 180,
    "reserve": 20,
}
QUALITY_VALUES = {3, 4, 5, 6, 7}
ID_PATTERN = re.compile(r"^(?:shhs1-)?(?P<id>\d+)(?:\.0)?$", re.IGNORECASE)


def stable_key(namespace: str, seed: int, subject_id: str) -> str:
    value = f"{namespace}|{seed}|{subject_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize_subject_id(value: str) -> str:
    text = value.strip()
    match = ID_PATTERN.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid SHHS1 subject identifier: {value!r}")
    return match.group("id")


def parse_quality(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid overall_shhs1 value: {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"Non-integer overall_shhs1 value: {value!r}")
    return int(number)


def read_eligible_subjects(
    metadata_csv: Path,
    id_column: str,
    quality_column: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = {id_column, quality_column}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                f"Metadata is missing required columns {missing}; "
                f"available columns include {sorted(columns)[:30]}"
            )

        eligible: list[dict[str, str]] = []
        seen: set[str] = set()
        counters: Counter[str] = Counter()
        for row_number, row in enumerate(reader, start=2):
            counters["metadata_rows"] += 1
            raw_id = (row.get(id_column) or "").strip()
            if not raw_id:
                counters["excluded_missing_id"] += 1
                continue
            subject_id = normalize_subject_id(raw_id)
            if subject_id in seen:
                raise ValueError(
                    f"Duplicate SHHS1 subject {subject_id} at CSV row {row_number}"
                )
            seen.add(subject_id)

            quality = parse_quality(row.get(quality_column) or "")
            if quality is None:
                counters["excluded_missing_quality"] += 1
                continue
            if quality not in QUALITY_VALUES:
                counters["excluded_outside_quality_domain"] += 1
                continue

            eligible.append(
                {
                    "subject_id": subject_id,
                    "overall_shhs1": str(quality),
                    "gender": (row.get("gender") or "").strip(),
                    "age_s1": (row.get("age_s1") or "").strip(),
                }
            )
            counters["eligible_rows"] += 1
    return eligible, dict(sorted(counters.items()))


def assign_subjects(
    eligible: Iterable[dict[str, str]], seed: int
) -> list[dict[str, object]]:
    required = sum(ROLE_COUNTS.values())
    ranked = sorted(
        eligible,
        key=lambda item: stable_key("shhs-v1-selection", seed, item["subject_id"]),
    )
    if len(ranked) < required:
        raise ValueError(f"Need at least {required} eligible subjects, found {len(ranked)}")

    selected = ranked[:required]
    selected.sort(
        key=lambda item: stable_key("shhs-v1-role", seed, item["subject_id"])
    )

    assignments: list[dict[str, object]] = []
    offset = 0
    for role, count in ROLE_COUNTS.items():
        role_items = selected[offset : offset + count]
        for role_index, item in enumerate(role_items, start=1):
            pilot = role == "adaptation" or (
                role == "validation" and role_index <= 5
            )
            assignments.append(
                {
                    **item,
                    "role": role,
                    "role_index": role_index,
                    "pilot": pilot,
                    "edf_filename": f"shhs1-{item['subject_id']}.edf",
                    "annotation_filename": (
                        f"shhs1-{item['subject_id']}-profusion.xml"
                    ),
                }
            )
        offset += count
    return assignments


def filename_regex(subject_ids: Iterable[str], suffix: str) -> str:
    ids = "|".join(re.escape(item) for item in sorted(subject_ids))
    return rf"^shhs1-({ids}){suffix}$"


def powershell_download_script(
    data_root: Path, assignments: list[dict[str, object]], pilot_only: bool
) -> str:
    chosen = [item for item in assignments if item["pilot"]] if pilot_only else assignments
    subject_ids = [str(item["subject_id"]) for item in chosen]
    edf_regex = filename_regex(subject_ids, r"\.edf")
    xml_regex = filename_regex(subject_ids, r"-profusion\.xml")
    label = "10 pilot subjects" if pilot_only else "220 selected subjects"
    return f'''# Generated by scripts/select_shhs_subjects.py; contains no token.
$ErrorActionPreference = "Stop"
$DataRoot = "{data_root}"
$Ruby = "C:\\Ruby33-x64\\bin\\ruby.exe"
$Nsrr = "C:\\Ruby33-x64\\bin\\nsrr"

if (-not (Test-Path -LiteralPath $Ruby)) {{
    throw "Ruby executable not found: $Ruby"
}}
if (-not (Test-Path -LiteralPath $Nsrr)) {{
    throw "NSRR Ruby launcher not found: $Nsrr"
}}
Set-Location -LiteralPath $DataRoot

Write-Host "Downloading EDF files for {label}"
& $Ruby $Nsrr download shhs/polysomnography/edfs/shhs1 '--file={edf_regex}'
if ($LASTEXITCODE -ne 0) {{ throw "EDF download failed" }}

Write-Host "Downloading Profusion XML files for {label}"
& $Ruby $Nsrr download shhs/polysomnography/annotations-events-profusion/shhs1 '--file={xml_regex}'
if ($LASTEXITCODE -ne 0) {{ throw "XML download failed" }}
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-column", default="nsrrid")
    parser.add_argument("--quality-column", default="overall_shhs1")
    args = parser.parse_args()

    metadata_csv = args.metadata_csv.resolve()
    output_dir = args.output_dir.resolve()
    data_root = args.data_root.resolve()
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)

    eligible, audit = read_eligible_subjects(
        metadata_csv, args.id_column, args.quality_column
    )
    assignments = assign_subjects(eligible, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "shhs1_subject_roles_seed42.csv"
    json_path = output_dir / "shhs1_subject_manifest_seed42.json"
    pilot_script = output_dir / "download_pilot.ps1"
    full_script = output_dir / "download_selected.ps1"

    fields = [
        "subject_id",
        "role",
        "role_index",
        "pilot",
        "overall_shhs1",
        "gender",
        "age_s1",
        "edf_filename",
        "annotation_filename",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignments)

    role_counts = Counter(str(item["role"]) for item in assignments)
    manifest = {
        "schema_version": 1,
        "status": "selected_not_downloaded_not_technically_validated",
        "dataset": "SHHS Visit 1",
        "selection_seed": args.seed,
        "selection_method": "sort_by_sha256('shhs-v1-selection|seed|subject_id')",
        "role_method": "sort_selected_by_sha256('shhs-v1-role|seed|subject_id')",
        "eligibility": {
            "id_column": args.id_column,
            "quality_column": args.quality_column,
            "accepted_quality_values": sorted(QUALITY_VALUES),
            "clinical_variables_used_for_selection": False,
            "technical_validation_pending": True,
        },
        "source_metadata": str(metadata_csv),
        "source_metadata_sha256": sha256_file(metadata_csv),
        "counts": {
            **audit,
            "selected_total": len(assignments),
            "pilot": sum(bool(item["pilot"]) for item in assignments),
            "roles": dict(sorted(role_counts.items())),
        },
        "subjects": assignments,
    }
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pilot_script.write_text(
        powershell_download_script(data_root, assignments, pilot_only=True),
        encoding="utf-8-sig",
    )
    full_script.write_text(
        powershell_download_script(data_root, assignments, pilot_only=False),
        encoding="utf-8-sig",
    )

    sidecar = json_path.with_suffix(json_path.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256_file(json_path)}  {json_path.name}\n", encoding="ascii"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Manifest: {json_path}")
    print(f"Roles:    {csv_path}")
    print(f"Pilot:    {pilot_script}")
    print(f"Full:     {full_script}")
    print("STATUS: SELECTED_NOT_DOWNLOADED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
