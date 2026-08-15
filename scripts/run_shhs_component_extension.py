"""Run the locked E1/E2 secondary SHHS zero-shot extension on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sleeptcn.shhs_zero_shot import run_role


EXPERIMENT_VARIANTS = {"E1": "paper_raw_v1", "E2": "paper_raw_v1"}
PROTOCOL_STATUS = "locked_before_component_inference"
TEST_CONFIRMATION = "OPEN-SHHS-E1-E2-TEST-ONCE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--role", choices=("validation", "test"), required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-gate", type=Path)
    parser.add_argument("--confirm", help=f"test only: {TEST_CONFIRMATION}")
    args = parser.parse_args()
    result = run_role(
        workspace=args.workspace.resolve(),
        processed_root=args.processed_root.resolve(),
        preprocess_manifest_path=args.preprocess_manifest.resolve(),
        protocol_path=args.protocol.resolve(),
        inventory_path=args.checkpoint_inventory.resolve(),
        output_root=args.output_root.resolve(),
        role=args.role,
        threads=args.threads,
        batch_size=args.batch_size,
        resume=args.resume,
        validation_gate_path=(
            None if args.validation_gate is None else args.validation_gate.resolve()
        ),
        confirmation=args.confirm,
        experiment_variants=EXPERIMENT_VARIANTS,
        expected_protocol_status=PROTOCOL_STATUS,
        expected_inventory_checkpoints=180,
        test_confirmation=TEST_CONFIRMATION,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "role": result["role"],
                "summary": result["summary"],
                "subject_macro_f1_mean": {
                    experiment: values["subject_macro_f1_mean"]
                    for experiment, values in result["metrics"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
