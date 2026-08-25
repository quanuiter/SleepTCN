"""Create a full E0-E6 config with one locked ResNet tuning candidate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.serialization import atomic_write_json, read_json


def build_locked_config(
    base_config: dict[str, object],
    search_config: dict[str, object],
    candidate_id: str,
) -> dict[str, object]:
    candidates = search_config.get("candidates")
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        raise ValueError(f"unknown tuning candidate: {candidate_id}")
    candidate = candidates[candidate_id]
    if not isinstance(candidate, dict) or not isinstance(
        candidate.get("resnet1d"), dict
    ):
        raise ValueError("candidate must contain a resnet1d mapping")
    components = base_config.get("components")
    if not isinstance(components, dict) or not isinstance(
        components.get("resnet1d"), dict
    ):
        raise ValueError("base config has no components.resnet1d mapping")

    locked = copy.deepcopy(base_config)
    locked_components = locked["components"]
    assert isinstance(locked_components, dict)
    resnet_config = copy.deepcopy(locked_components["resnet1d"])
    assert isinstance(resnet_config, dict)
    resnet_config.update(copy.deepcopy(candidate))
    locked_components["resnet1d"] = resnet_config
    locked["status"] = "resnet_v3_locked_candidate"
    locked["resnet_tuning"] = {
        "campaign_id": search_config.get("campaign_id"),
        "candidate_id": candidate_id,
        "selection_role": "validation_only",
        "test_policy": "candidate_selected_before_test",
        "source_search_config": "configs/tuning/resnet_v3_search.json",
    }
    return locked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locked = build_locked_config(
        read_json(args.base_config),
        read_json(args.search_config),
        args.candidate,
    )
    atomic_write_json(args.output, locked)
    print(
        json.dumps(
            {"output": str(args.output), "candidate": args.candidate}, indent=2
        )
    )


if __name__ == "__main__":
    main()
