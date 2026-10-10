"""Technical audit of the fresh SHHS1 holdout cohort (read-only).

Applies the same EDF/XML checks as ``audit_shhs_pilot.py`` to the 500 subjects of
``configs/shhs_holdout_confirmatory_v1.json`` plus the pre-specified spares, then
applies the protocol's replacement rule: a cohort subject that fails a technical
check is replaced by the next passing spare in rank order. Labels are only used
for the protocol's "at least one scored N1-REM epoch" rule; no model is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_shhs_pilot import (  # noqa: E402
    ALLOWED_RAW_STAGES,
    EPOCH_SECONDS,
    EXCLUDED_RAW_STAGES,
    PRIMARY_CHANNEL,
    PRIMARY_SAMPLING_HZ,
    read_edf_header,
    read_profusion_xml,
    read_signal_windows,
)
from sleeptcn.io.hashing import sha256_file  # noqa: E402

SLEEP_RAW_STAGES = {1, 2, 3, 4, 5}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    if protocol.get("status") != "locked_before_holdout_preprocessing":
        raise ValueError("Holdout protocol is not in its locked state")
    return protocol, sha256_bytes(raw)


def load_cohort(csv_path: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    raw = csv_path.read_bytes()
    digest = sha256_bytes(raw)
    expected = protocol["cohort"]["final_list_sha256"]
    if digest != expected:
        raise ValueError(f"Cohort CSV SHA-256 {digest} differs from protocol {expected}")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    ids = [row["subject_id"] for row in rows]
    if len(ids) != protocol["cohort"]["n_target"] or len(set(ids)) != len(ids):
        raise ValueError("Cohort CSV must contain the protocol's unique target subjects")
    return rows, digest


def audit_subject(subject_id: str, edf_path: Path, xml_path: Path) -> dict[str, Any]:
    """Same checks as audit_shhs_pilot.audit, plus the protocol sleep-epoch rule."""
    errors: list[str] = []
    record: dict[str, Any] = {"edf_filename": edf_path.name, "xml_filename": xml_path.name}
    try:
        if not edf_path.is_file() or edf_path.stat().st_size == 0:
            raise ValueError("EDF is missing or empty")
        if not xml_path.is_file() or xml_path.stat().st_size == 0:
            raise ValueError("XML is missing or empty")
        header = read_edf_header(edf_path)
        annotation = read_profusion_xml(xml_path)
        primary = [s for s in header["signals"] if s["label"] == PRIMARY_CHANNEL]
        if len(primary) != 1:
            errors.append(f"Expected exactly one {PRIMARY_CHANNEL!r} channel, found {len(primary)}")
        else:
            eeg = primary[0]
            if not math.isclose(eeg["sampling_hz"], PRIMARY_SAMPLING_HZ):
                errors.append(f"EEG sampling rate is {eeg['sampling_hz']}, expected 125 Hz")
            if eeg["unit"].strip().lower() not in {"uv", "µv", "μv"}:
                errors.append(f"EEG physical unit is {eeg['unit']!r}, expected uV")
            if eeg["physical_min"] >= eeg["physical_max"]:
                errors.append("EEG physical range is invalid")
            if eeg["digital_min"] >= eeg["digital_max"]:
                errors.append("EEG digital range is invalid")
            record["eeg_physical_range_uv"] = [eeg["physical_min"], eeg["physical_max"]]
        actual_bytes = edf_path.stat().st_size
        if actual_bytes != header["expected_file_bytes"]:
            errors.append(f"EDF size is {actual_bytes}, expected {header['expected_file_bytes']}")
        if annotation["epoch_seconds"] != EPOCH_SECONDS:
            errors.append(f"EpochLength is {annotation['epoch_seconds']}, expected 30")
        unexpected = sorted(set(annotation["stages"]) - ALLOWED_RAW_STAGES)
        if unexpected:
            errors.append(f"Unexpected raw sleep stages: {unexpected}")
        if not math.isclose(header["duration_seconds"], annotation["duration_seconds"]):
            errors.append(
                "EDF and SleepStages durations differ: "
                f"{header['duration_seconds']} vs {annotation['duration_seconds']} seconds"
            )
        if not SLEEP_RAW_STAGES & set(annotation["stages"]):
            errors.append("No scored N1, N2, N3 or REM epoch")
        decoded = read_signal_windows(edf_path, PRIMARY_CHANNEL)
        if not math.isclose(decoded["sampling_hz"], PRIMARY_SAMPLING_HZ):
            errors.append("Decoded EEG sampling rate is not 125 Hz")
        expected_samples = round(header["duration_seconds"] * PRIMARY_SAMPLING_HZ)
        if decoded["sample_count"] != expected_samples:
            errors.append(f"Decoded EEG has {decoded['sample_count']} samples, expected {expected_samples}")
        if decoded["warnings"]:
            errors.append("pyedflib emitted warnings: " + "; ".join(decoded["warnings"]))
        record.update(
            {
                "edf_bytes": actual_bytes,
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
    except Exception as exc:  # report every subject in one run
        errors.append(f"{type(exc).__name__}: {exc}")
    record["passed"] = not errors
    record["errors"] = errors
    return {"subject_id": subject_id, **record}


def apply_replacement_rule(
    cohort_ids: list[str], spare_ids: list[str], records: dict[str, dict[str, Any]]
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Replace failed cohort subjects by passing spares, both in rank order."""
    available = [sid for sid in spare_ids if records[sid]["passed"]]
    final: list[str] = []
    replacements: list[dict[str, str]] = []
    unresolved: list[str] = []
    for sid in cohort_ids:
        if records[sid]["passed"]:
            final.append(sid)
        elif available:
            spare = available.pop(0)
            final.append(spare)
            replacements.append({"replaced": sid, "by": spare})
        else:
            unresolved.append(sid)
    return final, replacements, unresolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    protocol, protocol_sha256 = load_protocol(args.protocol)
    rows, cohort_sha256 = load_cohort(args.cohort_csv, protocol)
    cohort_ids = [row["subject_id"] for row in rows]
    spare_ids = [str(s) for s in protocol["cohort"]["spares_in_rank_order"]]
    if set(spare_ids) & set(cohort_ids):
        raise ValueError("Spares overlap the cohort")
    all_ids = cohort_ids + spare_ids

    jobs = [
        (sid, args.edf_dir / f"shhs1-{sid}.edf", args.xml_dir / f"shhs1-{sid}-profusion.xml")
        for sid in all_ids
    ]
    records: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(audit_subject, *job) for job in jobs]
        for index, future in enumerate(futures, start=1):
            result = future.result()
            records[result.pop("subject_id")] = result
            if index % 25 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] audited", flush=True)

    final_ids, replacements, unresolved = apply_replacement_rule(cohort_ids, spare_ids, records)
    rank = {row["subject_id"]: int(row["selection_rank"]) for row in rows}
    stage_totals: Counter[int] = Counter()
    for sid in final_ids:
        stage_totals.update({int(k): v for k, v in records[sid].get("raw_stage_counts", {}).items()})
    excluded = sum(stage_totals[s] for s in EXCLUDED_RAW_STAGES)
    failed_cohort = [sid for sid in cohort_ids if not records[sid]["passed"]]

    report = {
        "schema_version": 1,
        "gate": "SHHS1_HOLDOUT_TECHNICAL_AUDIT",
        "status": "passed" if not unresolved else "needs_more_spares",
        "dataset": "SHHS Visit 1",
        "protocol_path": str(args.protocol),
        "protocol_sha256": protocol_sha256,
        "cohort_csv_sha256": cohort_sha256,
        "locked_primary_channel": {
            "edf_label": PRIMARY_CHANNEL,
            "montage": "C4-A1",
            "sampling_hz": PRIMARY_SAMPLING_HZ,
            "physical_unit": "uV",
        },
        "summary": {
            "cohort_subjects": len(cohort_ids),
            "spares_audited": len(spare_ids),
            "cohort_failed": len(failed_cohort),
            "replacements": len(replacements),
            "unresolved": len(unresolved),
            "analysis_cohort_size": len(final_ids),
            "raw_stage_counts_analysis_cohort": dict(sorted(stage_totals.items())),
            "excluded_epochs_analysis_cohort": excluded,
            "valid_five_class_epochs_analysis_cohort": sum(stage_totals.values()) - excluded,
        },
        "failed_cohort_subjects": {sid: records[sid]["errors"] for sid in failed_cohort},
        "replacements": replacements,
        "unresolved": unresolved,
        "analysis_cohort": final_ids,
        "analysis_cohort_ranks": {sid: rank.get(sid, "spare") for sid in final_ids},
        "subjects": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if replacements:
        print("REPLACEMENTS:", json.dumps(replacements))
    if failed_cohort:
        print("FAILED:", json.dumps(report["failed_cohort_subjects"], ensure_ascii=False))
    print(f"STATUS: {report['status'].upper()}")
    print(f"REPORT: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
