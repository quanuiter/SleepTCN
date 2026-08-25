"""Kiểm định sâu metadata và annotation của Sleep-EDF Expanded/SC.

Script chỉ đọc EDF. Nó không đọc toàn bộ tín hiệu vào RAM và không tạo NPZ.
Mọi giả định cần cho tiền xử lý 30 giây được kiểm tra trước khi xử lý dữ liệu.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyedflib


KEY_RE = re.compile(r"^(SC\d{4}[A-Z])")
KNOWN_ANNOTATIONS = {
    "Sleep stage W",
    "Sleep stage 1",
    "Sleep stage 2",
    "Sleep stage 3",
    "Sleep stage 4",
    "Sleep stage R",
    "Sleep stage ?",
    "Movement time",
}


def record_key(path: Path) -> str:
    match = KEY_RE.match(path.name)
    if match is None:
        raise ValueError(f"Invalid Sleep Cassette filename: {path.name}")
    return match.group(1)


def unique_map(paths: list[Path], kind: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        key = record_key(path)
        if key in result:
            raise RuntimeError(f"Duplicate {kind} record key: {key}")
        result[key] = path
    return result


def is_integer_multiple(value: float, unit: float, tolerance: float = 1e-6) -> bool:
    if value <= 0 or unit <= 0:
        return False
    quotient = value / unit
    return math.isclose(quotient, round(quotient), abs_tol=tolerance)


def inspect_pair(
    psg_path: Path,
    hyp_path: Path,
    channel: str,
    expected_fs: float,
    epoch_seconds: float,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    annotation_counts: Counter[str] = Counter()

    psg = pyedflib.EdfReader(str(psg_path))
    hyp = pyedflib.EdfReader(str(hyp_path))
    try:
        psg_start = psg.getStartdatetime()
        hyp_start = hyp.getStartdatetime()
        if psg_start != hyp_start:
            errors.append("start_datetime_mismatch")

        channel_names = psg.getSignalLabels()
        if channel not in channel_names:
            errors.append("channel_missing")
            channel_index = None
            fs = None
            signal_samples = None
            signal_epochs = None
        else:
            channel_index = channel_names.index(channel)
            fs = float(psg.getSampleFrequency(channel_index))
            physical_dimension = str(psg.getPhysicalDimension(channel_index)).strip()
            signal_samples = int(psg.getNSamples()[channel_index])
            normalized_dimension = physical_dimension.lower().replace("μ", "u").replace("µ", "u")
            if normalized_dimension != "uv":
                errors.append("physical_dimension_not_uv")
            if not math.isclose(fs, expected_fs, abs_tol=1e-9):
                errors.append("sampling_rate_unexpected")
            samples_per_epoch_float = fs * epoch_seconds
            if not is_integer_multiple(samples_per_epoch_float, 1.0):
                errors.append("samples_per_epoch_noninteger")
                signal_epochs = None
            else:
                samples_per_epoch = int(round(samples_per_epoch_float))
                if signal_samples % samples_per_epoch != 0:
                    errors.append("signal_not_divisible_into_epochs")
                    signal_epochs = None
                else:
                    signal_epochs = signal_samples // samples_per_epoch

        onsets, durations, annotations = hyp.readAnnotations()
        annotation_epochs = 0
        expected_onset = 0.0
        unknown_annotations: set[str] = set()
        discontinuities = 0
        invalid_durations = 0

        for onset_raw, duration_raw, annotation_raw in zip(
            onsets, durations, annotations, strict=True
        ):
            onset = float(onset_raw)
            duration = float(duration_raw)
            annotation = str(annotation_raw)
            annotation_counts[annotation] += 1
            if annotation not in KNOWN_ANNOTATIONS:
                unknown_annotations.add(annotation)
            if not math.isclose(onset, expected_onset, abs_tol=1e-6):
                discontinuities += 1
            if duration <= 0 or not is_integer_multiple(duration, epoch_seconds):
                invalid_durations += 1
            else:
                annotation_epochs += int(round(duration / epoch_seconds))
            expected_onset = onset + duration

        if unknown_annotations:
            errors.append("unknown_annotation")
        if discontinuities:
            errors.append("annotation_timeline_discontinuous")
        if invalid_durations:
            errors.append("annotation_duration_not_30s_multiple")
        if len(annotations) == 0:
            errors.append("annotations_empty")

        epoch_difference = (
            None if signal_epochs is None else annotation_epochs - signal_epochs
        )
        if epoch_difference is not None:
            if epoch_difference < 0:
                errors.append("annotations_shorter_than_signal")
            elif epoch_difference > 0:
                warnings.append("annotations_longer_than_signal")

        signal_seconds = (
            None if signal_samples is None or fs is None else signal_samples / fs
        )
        psg_file_seconds = float(psg.getFileDuration())
        if signal_seconds is not None and not math.isclose(
            signal_seconds, psg_file_seconds, abs_tol=1e-6
        ):
            errors.append("channel_duration_differs_from_psg_duration")

        return {
            "record_key": record_key(psg_path),
            "subject_id": record_key(psg_path)[:5],
            "psg_name": psg_path.name,
            "hypnogram_name": hyp_path.name,
            "psg_start": psg_start.isoformat(),
            "hypnogram_start": hyp_start.isoformat(),
            "psg_file_seconds": psg_file_seconds,
            "psg_datarecord_seconds": float(psg.datarecord_duration),
            "psg_datarecords": int(psg.datarecords_in_file),
            "channel": channel,
            "sampling_rate_hz": fs,
            "physical_dimension": (
                None if channel_index is None else physical_dimension
            ),
            "signal_samples": signal_samples,
            "signal_epochs_30s": signal_epochs,
            "annotation_items": int(len(annotations)),
            "annotation_epochs_30s": annotation_epochs,
            "annotation_minus_signal_epochs": epoch_difference,
            "annotation_end_seconds": expected_onset,
            "annotation_discontinuities": discontinuities,
            "invalid_annotation_durations": invalid_durations,
            "unknown_annotations": sorted(unknown_annotations),
            "annotation_counts": dict(sorted(annotation_counts.items())),
            "warnings": sorted(set(warnings)),
            "errors": sorted(set(errors)),
            "passed": not errors,
        }
    finally:
        psg.close()
        hyp.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--channel", default="EEG Fpz-Cz")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--epoch-seconds", type=float, default=30.0)
    parser.add_argument("--expected-records", type=int, default=153)
    parser.add_argument("--expected-subjects", type=int, default=78)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    psg_map = unique_map(sorted(data_dir.glob("*-PSG.edf")), "PSG")
    hyp_map = unique_map(sorted(data_dir.glob("*-Hypnogram.edf")), "Hypnogram")
    missing_hyp = sorted(set(psg_map) - set(hyp_map))
    missing_psg = sorted(set(hyp_map) - set(psg_map))
    if missing_hyp or missing_psg:
        raise RuntimeError(
            f"Unpaired files: missing_hyp={missing_hyp}, missing_psg={missing_psg}"
        )

    keys = sorted(set(psg_map) & set(hyp_map))
    records = [
        inspect_pair(
            psg_map[key],
            hyp_map[key],
            args.channel,
            args.sampling_rate,
            args.epoch_seconds,
        )
        for key in keys
    ]

    subjects = sorted({record["subject_id"] for record in records})
    error_counts = Counter(
        error for record in records for error in record["errors"]
    )
    warning_counts = Counter(
        warning for record in records for warning in record["warnings"]
    )
    global_errors: list[str] = []
    if len(records) != args.expected_records:
        global_errors.append("record_count_unexpected")
    if len(subjects) != args.expected_subjects:
        global_errors.append("subject_count_unexpected")

    report = {
        "schema_version": 2,
        "dataset": "sleep-edf-expanded/sleep-cassette/1.0.0",
        "source_readonly": "sleep-edf-expanded/sleep-cassette/1.0.0",
        "assumptions": {
            "channel": args.channel,
            "sampling_rate_hz": args.sampling_rate,
            "epoch_seconds": args.epoch_seconds,
        },
        "summary": {
            "records": len(records),
            "subjects": len(subjects),
            "records_passed": sum(record["passed"] for record in records),
            "records_failed": sum(not record["passed"] for record in records),
            "error_counts": dict(sorted(error_counts.items())),
            "warning_counts": dict(sorted(warning_counts.items())),
            "global_errors": global_errors,
        },
        "subjects": subjects,
        "records": records,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_fields = [
        "record_key",
        "subject_id",
        "psg_name",
        "hypnogram_name",
        "psg_start",
        "psg_file_seconds",
        "psg_datarecord_seconds",
        "sampling_rate_hz",
        "physical_dimension",
        "signal_samples",
        "signal_epochs_30s",
        "annotation_items",
        "annotation_epochs_30s",
        "annotation_minus_signal_epochs",
        "annotation_discontinuities",
        "invalid_annotation_durations",
        "warnings",
        "errors",
        "passed",
    ]
    with args.csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in csv_fields}
            row["warnings"] = "|".join(record["warnings"])
            row["errors"] = "|".join(record["errors"])
            writer.writerow(row)

    print(json.dumps(report["summary"], indent=2))
    print(f"JSON: {args.json_output.resolve()}")
    print(f"CSV:  {args.csv_output.resolve()}")
    passed = not global_errors and not error_counts
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
