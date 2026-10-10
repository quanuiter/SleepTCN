"""Preprocess the fresh SHHS1 holdout cohort with the locked v1 pipeline.

Reuses ``sleeptcn.shhs_preprocessing.process_subject`` unchanged (same resampling,
label mapping, evaluation window and variant definitions as v1) and adds the
``bandpass_v2`` variant required by the E4 arms of the holdout protocol.
A subject whose preprocessing raises an error is replaced by the next unused spare
that passed the technical audit, as pre-specified in the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy  # noqa: E402
import pyedflib  # noqa: E402
import scipy  # noqa: E402

from sleeptcn.preprocessing import sha256_file  # noqa: E402
from sleeptcn.shhs_preprocessing import load_locked_config, process_subject  # noqa: E402

HOLDOUT_VARIANTS = ("paper_raw_v1", "filtered_v2", "filtered_zscore_v2", "bandpass_v2")
ROLE = "holdout_test"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw.decode("utf-8"))
    if protocol.get("status") != "locked_before_holdout_preprocessing":
        raise ValueError("Holdout protocol is not in its locked state")
    protocol_sha256 = sha256_bytes(protocol_raw)

    _, config_sha256, config = load_locked_config(args.config.resolve())
    if config_sha256 != protocol["preprocessing"]["config_sha256"]:
        raise ValueError("Preprocessing config SHA-256 differs from the holdout protocol")
    if tuple(protocol["preprocessing"]["variants"]) != HOLDOUT_VARIANTS:
        raise ValueError("Protocol variants differ from the implemented holdout variants")

    audit_raw = args.technical_audit.read_bytes()
    audit = json.loads(audit_raw.decode("utf-8"))
    audit_sha256 = sha256_bytes(audit_raw)
    if audit.get("gate") != "SHHS1_HOLDOUT_TECHNICAL_AUDIT" or audit.get("status") != "passed":
        raise ValueError("Holdout technical audit has not passed")
    if audit.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Technical audit was produced under a different protocol file")
    if audit.get("cohort_csv_sha256") != protocol["cohort"]["final_list_sha256"]:
        raise ValueError("Technical audit refers to a different cohort list")
    cohort = [str(s) for s in audit["analysis_cohort"]]
    if len(cohort) != protocol["cohort"]["n_target"] or len(set(cohort)) != len(cohort):
        raise ValueError("Analysis cohort size differs from protocol")
    used = set(cohort)
    spares = [
        str(s)
        for s in protocol["cohort"]["spares_in_rank_order"]
        if str(s) not in used and audit["subjects"].get(str(s), {}).get("passed")
    ]
    return {
        "protocol_sha256": protocol_sha256,
        "config": config,
        "config_sha256": config_sha256,
        "audit": audit,
        "audit_sha256": audit_sha256,
        "cohort": cohort,
        "spares": spares,
    }


def make_item(subject_id: str, role_index: int) -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "role": ROLE,
        "role_index": role_index,
        "pilot": False,
        "edf_filename": f"shhs1-{subject_id}.edf",
        "annotation_filename": f"shhs1-{subject_id}-profusion.xml",
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    try:
        records = process_subject(**job["kwargs"])
        return {"subject_id": job["subject_id"], "ok": True, "records": records}
    except Exception as exc:  # recorded and handled by the replacement rule
        return {"subject_id": job["subject_id"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_job(subject_id: str, role_index: int, ctx: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "kwargs": {
            "item": make_item(subject_id, role_index),
            "audit_subject": ctx["audit"]["subjects"][subject_id],
            "edf_dir": args.edf_dir.resolve(),
            "xml_dir": args.xml_dir.resolve(),
            "output_root": args.output_root.resolve(),
            "variants": HOLDOUT_VARIANTS,
            "config": ctx["config"],
            "config_sha256": ctx["config_sha256"],
            "selection_manifest_sha256": ctx["audit"]["cohort_csv_sha256"],
            "technical_audit_sha256": ctx["audit_sha256"],
            "resume": args.resume,
        },
    }


def quartiles(values: list[float]) -> dict[str, float]:
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "q1": q[0], "q3": q[2], "n": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--technical-audit", type=Path, required=True)
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_manifest}")

    ctx = load_inputs(args)
    order = {sid: index for index, sid in enumerate(ctx["cohort"], start=1)}
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, build_job(sid, order[sid], ctx, args)) for sid in ctx["cohort"]]
        for done, future in enumerate(futures, start=1):
            result = future.result()
            if result["ok"]:
                results[result["subject_id"]] = result
            else:
                failures[result["subject_id"]] = result["error"]
                print(f"FAILED {result['subject_id']}: {result['error']}", flush=True)
            if done % 10 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] preprocessed", flush=True)

    # Protocol replacement rule: failed subjects are replaced by unused passing spares in rank order.
    final_cohort = list(ctx["cohort"])
    replacements: list[dict[str, str]] = []
    spares = list(ctx["spares"])
    for failed in [sid for sid in ctx["cohort"] if sid in failures]:
        replaced = False
        while spares and not replaced:
            spare = spares.pop(0)
            result = run_one(build_job(spare, order[failed], ctx, args))
            if result["ok"]:
                results[spare] = result
                final_cohort[final_cohort.index(failed)] = spare
                replacements.append({"replaced": failed, "by": spare})
                replaced = True
            else:
                failures[spare] = result["error"]
        if not replaced:
            print(f"UNRESOLVED {failed}: no passing spare left", flush=True)

    unresolved = [sid for sid in final_cohort if sid not in results]
    records = [record for sid in final_cohort if sid in results for record in results[sid]["records"]]
    zscore_std = [r["x_std"] for r in records if r["variant"] == "filtered_zscore_v2"]
    report = {
        "schema_version": 1,
        "status": "complete" if not unresolved else "needs_more_spares",
        "dataset": "SHHS Visit 1",
        "scope": "holdout_confirmatory_v1",
        "protocol_sha256": ctx["protocol_sha256"],
        "cohort_csv_sha256": ctx["audit"]["cohort_csv_sha256"],
        "technical_audit_sha256": ctx["audit_sha256"],
        "config_path": str(args.config.resolve()),
        "config_sha256": ctx["config_sha256"],
        "config": asdict(ctx["config"]),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pyedflib": pyedflib.__version__,
        },
        "variants": list(HOLDOUT_VARIANTS),
        "preprocessing_failures": failures,
        "replacements": replacements,
        "unresolved": unresolved,
        "analysis_cohort": final_cohort,
        "summary": {
            "subjects": len(final_cohort) - len(unresolved),
            "outputs": len(records),
            "records_per_variant": {v: sum(r["variant"] == v for r in records) for v in HOLDOUT_VARIANTS},
            "epochs_per_variant": {v: sum(r["epochs"] for r in records if r["variant"] == v) for v in HOLDOUT_VARIANTS},
            "valid_epochs_per_variant": {v: sum(r["valid_epochs"] for r in records if r["variant"] == v) for v in HOLDOUT_VARIANTS},
            "ignored_epochs_per_variant": {v: sum(r["ignored_epochs"] for r in records if r["variant"] == v) for v in HOLDOUT_VARIANTS},
            "max_clip_fraction": max((r["clip_fraction"] for r in records if r["variant"] != "paper_raw_v1"), default=None),
            "label_counts_filtered_v2": dict(
                sum((Counter(r["label_counts"]) for r in records if r["variant"] == "filtered_v2"), Counter())
            ),
            "s6_zscore_input_std_in_window": quartiles(zscore_std) if len(zscore_std) >= 2 else None,
        },
        "records": records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256_file(args.output_manifest)
    args.output_manifest.with_suffix(args.output_manifest.suffix + ".sha256").write_text(
        f"{digest}  {args.output_manifest.name}\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], indent=2))
    if replacements:
        print("REPLACEMENTS:", json.dumps(replacements))
    print(f"STATUS: {report['status'].upper()}\nMANIFEST: {args.output_manifest.resolve()}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
