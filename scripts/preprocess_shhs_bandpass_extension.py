"""Create the SHHS band-pass-only extension from the locked raw cohort.

This deliberately does not alter the locked primary SHHS preprocessing protocol.
It reuses the same subject manifest, technical audit, resampling, trimming and
source-hash checks, and writes only the additional ``bandpass_v2`` variant.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy
import pyedflib
import scipy

from sleeptcn.preprocessing import sha256_file
from sleeptcn.shhs_preprocessing import (
    load_locked_config,
    load_verified_sources,
    process_subject,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--technical-audit", type=Path, required=True)
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "primary"), default="primary")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_manifest = args.output_manifest.resolve()
    if output_manifest.exists() and not args.resume:
        raise FileExistsError(output_manifest)

    locked, config_sha256, config = load_locked_config(args.config.resolve())
    subjects, audit, manifest_sha256, audit_sha256 = load_verified_sources(
        args.manifest.resolve(),
        args.technical_audit.resolve(),
        locked,
        args.scope,
    )
    variant = "bandpass_v2"
    records = []
    for index, item in enumerate(subjects, start=1):
        print(
            f"[{index}/{len(subjects)}] role={item['role']} "
            f"subject_index={item['role_index']}",
            flush=True,
        )
        records.extend(
            process_subject(
                item=item,
                audit_subject=audit["subjects"][str(item["subject_id"])],
                edf_dir=args.edf_dir.resolve(),
                xml_dir=args.xml_dir.resolve(),
                output_root=args.output_root.resolve(),
                variants=(variant,),
                config=config,
                config_sha256=config_sha256,
                selection_manifest_sha256=manifest_sha256,
                technical_audit_sha256=audit_sha256,
                resume=args.resume,
            )
        )

    roles = Counter(str(item["role"]) for item in subjects)
    report = {
        "schema_version": 1,
        "status": "complete",
        "extension_type": "same_cohort_bandpass_only",
        "dataset": "SHHS Visit 1",
        "scope": args.scope,
        "selection_seed": 42,
        "selection_manifest_sha256": manifest_sha256,
        "technical_audit_sha256": audit_sha256,
        "config_path": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "config": asdict(config),
        "variant": variant,
        "source_protocol": "configs/shhs_v1_protocol.json primary cohort",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pyedflib": pyedflib.__version__,
        },
        "summary": {
            "subjects": len(subjects),
            "roles": dict(sorted(roles.items())),
            "outputs": len(records),
            "epochs": sum(record["epochs"] for record in records),
            "valid_epochs": sum(record["valid_epochs"] for record in records),
            "ignored_epochs": sum(record["ignored_epochs"] for record in records),
            "subjects_with_nonzero_clip_fraction": sum(
                float(record.get("clip_fraction", 0.0) or 0.0) > 0.0
                for record in records
            ),
        },
        "records": records,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    digest = sha256_file(output_manifest)
    output_manifest.with_suffix(output_manifest.suffix + ".sha256").write_text(
        f"{digest}  {output_manifest.name}\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"STATUS: COMPLETE\nMANIFEST: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
