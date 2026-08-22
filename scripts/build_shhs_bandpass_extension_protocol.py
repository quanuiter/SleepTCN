"""Build a locked, same-cohort SHHS E4 extension protocol.

The primary SHHS protocol remains unchanged.  This file combines its already
validated E0/E2/E3/E6 inputs with the separately generated E4 band-pass-only
inputs and records the seed-123 checkpoint campaign used for the extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = {
    "E0": "paper_raw_v1",
    "E2": "paper_raw_v1",
    "E3": "filtered_v2",
    "E4": "bandpass_v2",
    "E6": "filtered_zscore_v2",
}


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-seed", type=int, default=123)
    args = parser.parse_args()

    primary, primary_sha256 = read_json(args.primary_manifest.resolve())
    extension, extension_sha256 = read_json(args.extension_manifest.resolve())
    if primary.get("status") != "complete" or primary.get("scope") != "primary":
        raise ValueError("Primary SHHS preprocessing manifest is not complete/primary")
    if extension.get("status") != "complete" or extension.get("scope") != "primary":
        raise ValueError("E4 extension manifest is not complete/primary")
    if extension.get("variant") != "bandpass_v2":
        raise ValueError("Extension manifest is not bandpass_v2")
    if primary.get("selection_manifest_sha256") != extension.get("selection_manifest_sha256"):
        raise ValueError("Primary and E4 manifests use different subject manifests")
    if primary.get("technical_audit_sha256") != extension.get("technical_audit_sha256"):
        raise ValueError("Primary and E4 manifests use different technical audits")

    primary_records = list(primary.get("records", []))
    extension_records = list(extension.get("records", []))
    records = primary_records + extension_records
    expected = {variant: 200 for variant in set(EXPERIMENTS.values())}
    observed = {
        variant: sum(record.get("variant") == variant for record in records)
        for variant in expected
    }
    if observed != expected:
        raise ValueError(f"Unexpected records per variant: {observed}")
    keys = [(record["record_key"], record["variant"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate record/variant in combined manifest")

    output_manifest = args.output_manifest.resolve()
    output_protocol = args.output_protocol.resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "schema_version": 1,
        "status": "complete",
        "extension_type": "same_cohort_paired_extension",
        "dataset": "SHHS Visit 1",
        "scope": "primary",
        "selection_seed": 42,
        "primary_manifest_sha256": primary_sha256,
        "bandpass_extension_manifest_sha256": extension_sha256,
        "selection_manifest_sha256": primary["selection_manifest_sha256"],
        "technical_audit_sha256": primary["technical_audit_sha256"],
        "variants": sorted(set(EXPERIMENTS.values())),
        "summary": {
            "subjects": 200,
            "records": len(records),
            "records_per_variant": observed,
        },
        "records": records,
    }
    output_manifest.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    combined_raw = output_manifest.read_bytes()
    combined_sha256 = hashlib.sha256(combined_raw).hexdigest()
    output_manifest.with_suffix(output_manifest.suffix + ".sha256").write_text(
        f"{combined_sha256}  {output_manifest.name}\n", encoding="ascii"
    )

    protocol = {
        "schema_version": 1,
        "status": "locked_before_validation_inference",
        "analysis_scope": "same_cohort_paired_shhs_extension",
        "interpretation": (
            "A CPU-only extension on the fixed SHHS Visit 1 cohort; it compares "
            "the same subjects with seed-123 fold checkpoints and does not alter "
            "the locked primary E0/E3/E6 results."
        ),
        "dataset": "SHHS Visit 1",
        "selection_seed": 42,
        "checkpoint_seed": args.checkpoint_seed,
        "target_subjects": 200,
        "roles": {"adaptation": 5, "validation": 15, "test": 180, "reserve": 20},
        "experiments": {
            "E0": {
                "data_variant": "paper_raw_v1",
                "extractor": "15-CNN",
                "sequence_model": "BiLSTM",
            },
            "E2": {
                "data_variant": "paper_raw_v1",
                "extractor": "ResNet-1D",
                "sequence_model": "TCN",
            },
            "E3": {
                "data_variant": "filtered_v2",
                "extractor": "ResNet-1D",
                "sequence_model": "TCN",
            },
            "E4": {
                "data_variant": "bandpass_v2",
                "extractor": "ResNet-1D",
                "sequence_model": "TCN",
            },
            "E6": {
                "data_variant": "filtered_zscore_v2",
                "extractor": "ResNet-1D",
                "sequence_model": "TCN",
            },
        },
        "checkpoint_policy": {
            "outer_folds": list(range(10)),
            "checkpoint_filename": "best.pt",
            "use_latest_checkpoint": False,
            "rank_or_select_fold_by_validation_metric": False,
            "mix_components_between_folds": False,
            "use_all_folds": True,
        },
        "ensemble": {
            "per_fold_transform": "softmax_logits",
            "aggregation": "arithmetic_mean_probability",
            "accumulator_dtype": "float64",
            "fold_order": list(range(10)),
            "prediction": "argmax_mean_probability",
            "tie_break": "lowest_class_index",
        },
        "roles_policy": {
            "validation_subjects": 15,
            "test_subjects": 180,
            "zero_shot_weight_updates": 0,
            "complete_all_experiments_regardless_of_intermediate_results": True,
        },
        "preprocessing_provenance": {
            "manifest_sha256": combined_sha256,
            "combined_manifest_sha256": combined_sha256,
            "primary_manifest_sha256": primary_sha256,
            "bandpass_extension_manifest_sha256": extension_sha256,
            "selection_manifest_sha256": primary["selection_manifest_sha256"],
            "technical_audit_sha256": primary["technical_audit_sha256"],
        },
        "execution": {
            "device": "cpu",
            "gradient_enabled": False,
            "model_mode": "eval",
            "deterministic_algorithms": True,
            "checkpoint_seed": args.checkpoint_seed,
        },
        "comparisons": [
            "E4-E2: band-pass effect relative to raw input with ResNet+TCN",
            "E3-E4: divide-by-100 effect after band-pass",
            "E3-E6: divide-by-100 versus per-record z-score",
            "E3-E0: primary filtered pipeline versus paper baseline",
        ],
    }
    output_protocol.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    protocol_sha256 = hashlib.sha256(output_protocol.read_bytes()).hexdigest()
    output_protocol.with_suffix(output_protocol.suffix + ".sha256").write_text(
        f"{protocol_sha256}  {output_protocol.name}\n", encoding="ascii"
    )
    print(json.dumps({
        "status": "complete",
        "combined_manifest": str(output_manifest),
        "combined_manifest_sha256": combined_sha256,
        "protocol": str(output_protocol),
        "protocol_sha256": protocol_sha256,
        "records_per_variant": observed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
