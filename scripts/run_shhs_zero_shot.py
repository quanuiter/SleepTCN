"""Run locked SHHS zero-shot validation or test inference on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sleeptcn.shhs_zero_shot import TEST_CONFIRMATION, run_role


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
        validation_gate_path=(None if args.validation_gate is None else args.validation_gate.resolve()),
        confirmation=args.confirm,
    )
    aggregate_metrics = {
        experiment: {
            "subject_macro_f1_mean": values["subject_macro_f1_mean"],
            "pooled": values["pooled"],
        }
        for experiment, values in result["metrics"].items()
    }
    print(
        json.dumps(
            {
                "status": result["status"],
                "role": result["role"],
                "summary": result["summary"],
                "metrics": aggregate_metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
