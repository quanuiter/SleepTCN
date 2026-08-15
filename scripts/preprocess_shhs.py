"""CLI preprocessing SHHS1 theo giao thuc ngoai mien da khoa."""

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

from sleeptcn.shhs_preprocessing import (
    SHHS_VARIANTS,
    load_locked_config,
    load_verified_sources,
    process_subject,
)
from sleeptcn.preprocessing import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--technical-audit", type=Path, required=True)
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "primary"), required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=SHHS_VARIANTS, default=list(SHHS_VARIANTS)
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_manifest.exists() and not args.resume:
        raise FileExistsError(args.output_manifest)
    variants = tuple(args.variants)
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate preprocessing variant")
    if variants != SHHS_VARIANTS:
        raise ValueError(
            f"Locked SHHS preprocessing requires variants in order {SHHS_VARIANTS}"
        )

    locked, config_sha256, config = load_locked_config(args.config.resolve())
    subjects, audit, manifest_sha256, audit_sha256 = load_verified_sources(
        args.manifest.resolve(),
        args.technical_audit.resolve(),
        locked,
        args.scope,
    )
    records = []
    for index, item in enumerate(subjects, start=1):
        print(
            f"[{index}/{len(subjects)}] role={item['role']} subject_index={item['role_index']}",
            flush=True,
        )
        records.extend(
            process_subject(
                item=item,
                audit_subject=audit["subjects"][str(item["subject_id"])],
                edf_dir=args.edf_dir.resolve(),
                xml_dir=args.xml_dir.resolve(),
                output_root=args.output_root.resolve(),
                variants=variants,
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
        "dataset": "SHHS Visit 1",
        "scope": args.scope,
        "selection_seed": 42,
        "selection_manifest_sha256": manifest_sha256,
        "technical_audit_sha256": audit_sha256,
        "config_path": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "config": asdict(config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pyedflib": pyedflib.__version__,
        },
        "variants": list(variants),
        "summary": {
            "subjects": len(subjects),
            "roles": dict(sorted(roles.items())),
            "outputs": len(records),
            "records_per_variant": {
                variant: sum(record["variant"] == variant for record in records)
                for variant in variants
            },
            "epochs_per_variant": {
                variant: sum(record["epochs"] for record in records if record["variant"] == variant)
                for variant in variants
            },
            "valid_epochs_per_variant": {
                variant: sum(record["valid_epochs"] for record in records if record["variant"] == variant)
                for variant in variants
            },
            "ignored_epochs_per_variant": {
                variant: sum(record["ignored_epochs"] for record in records if record["variant"] == variant)
                for variant in variants
            },
        },
        "records": records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    digest = sha256_file(args.output_manifest)
    args.output_manifest.with_suffix(args.output_manifest.suffix + ".sha256").write_text(
        f"{digest}  {args.output_manifest.name}\n", encoding="ascii"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"STATUS: COMPLETE\nMANIFEST: {args.output_manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
