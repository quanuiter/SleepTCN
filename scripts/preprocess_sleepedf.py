"""CLI sinh paper_raw_v1 và filtered_v2 từ Sleep-EDF Expanded/SC."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy
import pyedflib
import scipy

from sleeptcn.preprocessing import (
    PreprocessConfig,
    VALID_VARIANTS,
    load_source_hashes,
    pair_files,
    process_record,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["paper_raw_v1", "filtered_v2"],
        choices=sorted(VALID_VARIANTS),
    )
    parser.add_argument("--record-key", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = PreprocessConfig()
    pairs = pair_files(args.data_dir.resolve())
    source_hashes = load_source_hashes(args.raw_manifest.resolve())
    selected_keys = sorted(set(args.record_key)) if args.record_key else sorted(pairs)
    missing = sorted(set(selected_keys) - set(pairs))
    if missing:
        raise ValueError(f"Requested record keys do not exist: {missing}")
    if set(selected_keys) - set(source_hashes):
        raise ValueError("Raw manifest is missing selected record hashes")

    records = []
    for index, key in enumerate(selected_keys, start=1):
        print(f"[{index}/{len(selected_keys)}] {key}", flush=True)
        psg_path, hyp_path = pairs[key]
        records.extend(
            process_record(
                psg_path=psg_path,
                hyp_path=hyp_path,
                variants=args.variants,
                output_root=args.output_root.resolve(),
                source_hashes=source_hashes[key],
                config=config,
                overwrite=args.overwrite,
            )
        )

    report = {
        "schema_version": 1,
        "dataset": "sleep-edf-expanded/sleep-cassette/1.0.0",
        "source_readonly": str(args.data_dir.resolve()),
        "raw_manifest": str(args.raw_manifest.resolve()),
        "config": asdict(config),
        "generation_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pyedflib": pyedflib.__version__,
        },
        "variants": args.variants,
        "selected_record_keys": selected_keys,
        "summary": {
            "source_records": len(selected_keys),
            "outputs": len(records),
            "total_epochs_by_variant": {
                variant: sum(
                    record["epochs"] for record in records if record["variant"] == variant
                )
                for variant in args.variants
            },
        },
        "records": records,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Manifest: {args.manifest_output.resolve()}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
