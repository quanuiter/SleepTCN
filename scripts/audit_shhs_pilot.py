"""Kiem dinh ky thuat bo pilot hoac toan bo tap SHHS1 da khoa.

Cong cu nay chi doc du lieu. No doi chieu danh sach pilot trong manifest, kiem
tra cap EDF--Profusion XML, header va payload EDF, kenh EEG C4-A1, epoch nhan
30 giay, mien nhan thô, thoi luong va doc mau tin hieu tai ba vi tri.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


EDF_PATTERN = re.compile(r"^shhs1-(?P<id>\d+)\.edf$", re.IGNORECASE)
XML_PATTERN = re.compile(
    r"^shhs1-(?P<id>\d+)-profusion\.xml$", re.IGNORECASE
)
ALLOWED_RAW_STAGES = {0, 1, 2, 3, 4, 5, 6, 9}
EXCLUDED_RAW_STAGES = {6, 9}
PRIMARY_CHANNEL = "EEG"
PRIMARY_SAMPLING_HZ = 125.0
EPOCH_SECONDS = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ascii_number(raw: bytes, cast: type[int] | type[float], name: str) -> Any:
    value = raw.decode("ascii", errors="strict").strip()
    if not value:
        raise ValueError(f"EDF field {name} is empty")
    try:
        return cast(value)
    except ValueError as exc:
        raise ValueError(f"Invalid EDF field {name}: {value!r}") from exc


def read_edf_header(path: Path) -> dict[str, Any]:
    """Doc header EDF ma khong nap toan bo tin hieu vao bo nho."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError("EDF fixed header is shorter than 256 bytes")
        header_bytes = _ascii_number(fixed[184:192], int, "header_bytes")
        data_records = _ascii_number(fixed[236:244], int, "data_records")
        record_seconds = _ascii_number(fixed[244:252], float, "record_seconds")
        signal_count = _ascii_number(fixed[252:256], int, "signal_count")
        expected_header_bytes = 256 + 256 * signal_count
        if header_bytes != expected_header_bytes:
            raise ValueError(
                f"EDF header_bytes={header_bytes}, expected {expected_header_bytes}"
            )
        if data_records <= 0 or record_seconds <= 0 or signal_count <= 0:
            raise ValueError("EDF record count, duration and signal count must be positive")
        signal_header = handle.read(header_bytes - 256)
        if len(signal_header) != header_bytes - 256:
            raise ValueError("EDF per-signal header is truncated")

    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    names = (
        "label",
        "transducer",
        "unit",
        "physical_min",
        "physical_max",
        "digital_min",
        "digital_max",
        "prefilter",
        "samples_per_record",
        "reserved",
    )
    columns: dict[str, list[str]] = {}
    offset = 0
    for name, width in zip(names, widths):
        columns[name] = [
            signal_header[offset + index * width : offset + (index + 1) * width]
            .decode("ascii", errors="strict")
            .strip()
            for index in range(signal_count)
        ]
        offset += width * signal_count

    signals = []
    for index in range(signal_count):
        samples_per_record = int(columns["samples_per_record"][index])
        signals.append(
            {
                "label": columns["label"][index],
                "unit": columns["unit"][index],
                "physical_min": float(columns["physical_min"][index]),
                "physical_max": float(columns["physical_max"][index]),
                "digital_min": int(columns["digital_min"][index]),
                "digital_max": int(columns["digital_max"][index]),
                "samples_per_record": samples_per_record,
                "sampling_hz": samples_per_record / record_seconds,
            }
        )

    expected_file_bytes = header_bytes + (
        data_records * 2 * sum(item["samples_per_record"] for item in signals)
    )
    return {
        "header_bytes": header_bytes,
        "data_records": data_records,
        "record_seconds": record_seconds,
        "duration_seconds": data_records * record_seconds,
        "signal_count": signal_count,
        "signals": signals,
        "expected_file_bytes": expected_file_bytes,
    }


def read_profusion_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    epoch_text = root.findtext(".//EpochLength")
    if epoch_text is None:
        raise ValueError("Profusion XML has no EpochLength")
    try:
        epoch_seconds = int(epoch_text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid EpochLength: {epoch_text!r}") from exc

    stage_nodes = root.findall(".//SleepStages/SleepStage")
    if not stage_nodes:
        raise ValueError("Profusion XML has no SleepStage entries")
    stages: list[int] = []
    for index, node in enumerate(stage_nodes):
        try:
            stages.append(int((node.text or "").strip()))
        except ValueError as exc:
            raise ValueError(f"Invalid SleepStage at index {index}") from exc
    return {
        "root_tag": root.tag,
        "epoch_seconds": epoch_seconds,
        "stage_count": len(stages),
        "stage_counts": dict(sorted(Counter(stages).items())),
        "duration_seconds": len(stages) * epoch_seconds,
        "stages": stages,
    }


def read_signal_windows(path: Path, channel: str) -> dict[str, Any]:
    """Mo EDF bang pyedflib va doc dau/giua/cuoi de phat hien tep hong."""
    try:
        import pyedflib
    except ImportError as exc:
        raise RuntimeError(
            "pyedflib is required for decoded signal validation"
        ) from exc

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader = pyedflib.EdfReader(str(path))
        try:
            labels = reader.getSignalLabels()
            if labels.count(channel) != 1:
                raise ValueError(
                    f"Expected exactly one {channel!r} channel, found {labels.count(channel)}"
                )
            index = labels.index(channel)
            sample_count = int(reader.getNSamples()[index])
            sampling_hz = float(reader.getSampleFrequency(index))
            window = max(1, int(round(sampling_hz)))
            starts = (0, sample_count // 2, max(0, sample_count - window))
            decoded = []
            for start in starts:
                count = min(window, sample_count - start)
                values = reader.readSignal(index, start, count)
                if len(values) != count:
                    raise ValueError(
                        f"Decoded {len(values)} samples, expected {count} at {start}"
                    )
                if not all(math.isfinite(float(value)) for value in values):
                    raise ValueError(f"Non-finite decoded value at sample {start}")
                decoded.append(
                    {
                        "start_sample": start,
                        "sample_count": count,
                        "minimum_uv": float(min(values)),
                        "maximum_uv": float(max(values)),
                    }
                )
            return {
                "sample_count": sample_count,
                "sampling_hz": sampling_hz,
                "windows": decoded,
                "warnings": [str(item.message) for item in caught],
            }
        finally:
            reader.close()


def expected_subjects(
    manifest: dict[str, Any], scope: str
) -> dict[str, dict[str, Any]]:
    if manifest.get("dataset") != "SHHS Visit 1":
        raise ValueError("Manifest is not the locked SHHS Visit 1 manifest")
    if manifest.get("selection_seed") != 42:
        raise ValueError("Pilot audit requires the locked selection seed 42")
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list):
        raise ValueError("Manifest subjects must be a list")
    chosen_items = [
        item
        for item in subjects
        if scope == "selected" or bool(item.get("pilot"))
    ]
    selected = {str(item["subject_id"]): item for item in chosen_items}
    if len(selected) != len(chosen_items):
        raise ValueError("Manifest contains duplicate selected subject identifiers")
    if scope == "pilot":
        if len(selected) != 10:
            raise ValueError(f"Expected 10 pilot subjects, found {len(selected)}")
        if any(
            item.get("role") not in {"adaptation", "validation"}
            for item in selected.values()
        ):
            raise ValueError("Pilot must contain adaptation/validation only")
    elif scope == "selected":
        expected_roles = {
            "adaptation": 5,
            "validation": 15,
            "test": 180,
            "reserve": 20,
        }
        actual_roles = Counter(str(item.get("role")) for item in selected.values())
        if len(selected) != 220 or dict(actual_roles) != expected_roles:
            raise ValueError(
                "Selected scope must contain the locked 220 subjects with roles "
                f"{expected_roles}; found {dict(actual_roles)}"
            )
    else:
        raise ValueError(f"Unknown audit scope: {scope!r}")
    return selected


def expected_pilot(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Tuong thich nguoc voi bai kiem tra pilot da co."""
    return expected_subjects(manifest, "pilot")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    selected = expected_subjects(manifest, args.scope)

    errors: list[str] = []
    warnings_out: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    total_stage_counts: Counter[int] = Counter()

    edf_on_disk = {
        match.group("id"): path
        for path in args.edf_dir.glob("*.edf")
        if (match := EDF_PATTERN.fullmatch(path.name))
    }
    xml_on_disk = {
        match.group("id"): path
        for path in args.xml_dir.glob("*.xml")
        if (match := XML_PATTERN.fullmatch(path.name))
    }
    unexpected_edf = sorted(set(edf_on_disk) - set(selected))
    unexpected_xml = sorted(set(xml_on_disk) - set(selected))
    if unexpected_edf or unexpected_xml:
        warnings_out.append(
            f"Download directories contain files outside the {args.scope} scope; "
            "they were not used by this audit"
        )

    for subject_id, item in sorted(selected.items()):
        edf_path = args.edf_dir / str(item["edf_filename"])
        xml_path = args.xml_dir / str(item["annotation_filename"])
        subject_errors: list[str] = []
        record: dict[str, Any] = {
            "role": item["role"],
            "edf_filename": edf_path.name,
            "xml_filename": xml_path.name,
        }
        try:
            if not edf_path.is_file() or edf_path.stat().st_size == 0:
                raise ValueError("EDF is missing or empty")
            if not xml_path.is_file() or xml_path.stat().st_size == 0:
                raise ValueError("XML is missing or empty")

            header = read_edf_header(edf_path)
            annotation = read_profusion_xml(xml_path)
            primary = [
                signal for signal in header["signals"] if signal["label"] == PRIMARY_CHANNEL
            ]
            if len(primary) != 1:
                subject_errors.append(
                    f"Expected exactly one {PRIMARY_CHANNEL!r} channel, found {len(primary)}"
                )
            else:
                eeg = primary[0]
                if not math.isclose(eeg["sampling_hz"], PRIMARY_SAMPLING_HZ):
                    subject_errors.append(
                        f"EEG sampling rate is {eeg['sampling_hz']}, expected 125 Hz"
                    )
                if eeg["unit"].strip().lower() not in {"uv", "µv", "μv"}:
                    subject_errors.append(
                        f"EEG physical unit is {eeg['unit']!r}, expected uV"
                    )
                if eeg["physical_min"] >= eeg["physical_max"]:
                    subject_errors.append("EEG physical range is invalid")
                if eeg["digital_min"] >= eeg["digital_max"]:
                    subject_errors.append("EEG digital range is invalid")

            actual_file_bytes = edf_path.stat().st_size
            if actual_file_bytes != header["expected_file_bytes"]:
                subject_errors.append(
                    f"EDF size is {actual_file_bytes}, expected {header['expected_file_bytes']}"
                )
            if annotation["epoch_seconds"] != EPOCH_SECONDS:
                subject_errors.append(
                    f"EpochLength is {annotation['epoch_seconds']}, expected 30"
                )
            raw_stage_values = set(annotation["stages"])
            unexpected_stages = sorted(raw_stage_values - ALLOWED_RAW_STAGES)
            if unexpected_stages:
                subject_errors.append(f"Unexpected raw sleep stages: {unexpected_stages}")
            if not math.isclose(
                header["duration_seconds"], annotation["duration_seconds"]
            ):
                subject_errors.append(
                    "EDF and SleepStages durations differ: "
                    f"{header['duration_seconds']} vs {annotation['duration_seconds']} seconds"
                )

            decoded = read_signal_windows(edf_path, PRIMARY_CHANNEL)
            if not math.isclose(decoded["sampling_hz"], PRIMARY_SAMPLING_HZ):
                subject_errors.append("Decoded EEG sampling rate is not 125 Hz")
            expected_samples = round(
                header["duration_seconds"] * PRIMARY_SAMPLING_HZ
            )
            if decoded["sample_count"] != expected_samples:
                subject_errors.append(
                    f"Decoded EEG has {decoded['sample_count']} samples, expected {expected_samples}"
                )
            if decoded["warnings"]:
                subject_errors.append(
                    "pyedflib emitted warnings: " + "; ".join(decoded["warnings"])
                )

            total_stage_counts.update(annotation["stages"])
            record.update(
                {
                    "edf_bytes": actual_file_bytes,
                    "edf_sha256": sha256_file(edf_path),
                    "xml_bytes": xml_path.stat().st_size,
                    "xml_sha256": sha256_file(xml_path),
                    "duration_seconds": header["duration_seconds"],
                    "epoch_seconds": annotation["epoch_seconds"],
                    "raw_stage_counts": annotation["stage_counts"],
                    "eeg_sampling_hz": decoded["sampling_hz"],
                    "eeg_samples": decoded["sample_count"],
                    "decoded_windows": decoded["windows"],
                }
            )
        except Exception as exc:  # report all subjects in one run
            subject_errors.append(f"{type(exc).__name__}: {exc}")

        record["passed"] = not subject_errors
        record["errors"] = subject_errors
        records[subject_id] = record
        errors.extend(f"subject {subject_id}: {message}" for message in subject_errors)

    excluded_epochs = sum(total_stage_counts[stage] for stage in EXCLUDED_RAW_STAGES)
    role_summary = {
        role: {
            "expected": sum(item.get("role") == role for item in selected.values()),
            "passed": sum(
                item.get("role") == role and records[subject_id]["passed"]
                for subject_id, item in selected.items()
            ),
            "failed": sum(
                item.get("role") == role and not records[subject_id]["passed"]
                for subject_id, item in selected.items()
            ),
        }
        for role in ("adaptation", "validation", "test", "reserve")
        if any(item.get("role") == role for item in selected.values())
    }
    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "gate": (
            "SHHS1_PILOT_TECHNICAL_AUDIT"
            if args.scope == "pilot"
            else "SHHS1_SELECTED_TECHNICAL_AUDIT"
        ),
        "scope": args.scope,
        "dataset": "SHHS Visit 1",
        "selection_seed": 42,
        "manifest_sha256": manifest_sha256,
        "locked_primary_channel": {
            "edf_label": PRIMARY_CHANNEL,
            "montage": "C4-A1",
            "sampling_hz": PRIMARY_SAMPLING_HZ,
            "physical_unit": "uV",
        },
        "label_policy": {
            "epoch_seconds": EPOCH_SECONDS,
            "raw_to_five_class": {
                "0": "W",
                "1": "N1",
                "2": "N2",
                "3": "N3",
                "4": "N3",
                "5": "REM",
                "6": "EXCLUDE_MOVEMENT",
                "9": "EXCLUDE_UNKNOWN",
            },
        },
        "summary": {
            "expected_subjects": len(selected),
            "passed_subjects": sum(item["passed"] for item in records.values()),
            "failed_subjects": sum(not item["passed"] for item in records.values()),
            "edf_files_in_directory": len(edf_on_disk),
            "xml_files_in_directory": len(xml_on_disk),
            "unexpected_edf_files": len(unexpected_edf),
            "unexpected_xml_files": len(unexpected_xml),
            "raw_stage_counts": dict(sorted(total_stage_counts.items())),
            "excluded_epochs": excluded_epochs,
            "valid_five_class_epochs": sum(total_stage_counts.values()) - excluded_epochs,
            "roles": role_summary,
        },
        "errors": errors,
        "warnings": warnings_out,
        "subjects": records,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("pilot", "selected"), default="pilot"
    )
    args = parser.parse_args()
    for path in (args.manifest, args.edf_dir, args.xml_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_sha256 = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{output_sha256}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"STATUS: {report['status'].upper()}")
    print(f"REPORT: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
