"""Replay predictions across the interruption boundary from selected checkpoints."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.demo import load_demo_models, validate_asset_manifest
from sleeptcn.models import SleepTCN
from sleeptcn.preprocessing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--shhs-root", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads((args.pilot / "aggregate_results.json").read_bytes())
    verification = json.loads((args.pilot / "verification.json").read_bytes())
    if verification["status"] != "passed" or sha256_file(args.pilot / "aggregate_results.json") != verification["aggregate_results_sha256"]:
        raise ValueError("weighted pilot must be verified first")
    manifest_path = args.pilot / "private_inference_manifest.json"
    if sha256_file(manifest_path) != result["recovery"]["inference_manifest_sha256"]:
        raise ValueError("target manifest changed")
    entries = json.loads(manifest_path.read_bytes())["records"]
    spec = importlib.util.spec_from_file_location("weighted_signal_replay", ROOT / "scripts/run_recovered_e3_cpu_pilot.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(4)
    assets = ROOT / "demo/assets"
    encoder = load_demo_models(assets, "E3", validated_manifest=validate_asset_manifest(assets)).extractor.eval()
    models = {}
    for arm, selected in result["selections"].items():
        path = args.pilot / arm / "best.pt"
        if sha256_file(path) != selected["checkpoint_sha256"]:
            raise ValueError("checkpoint changed")
        models[arm] = SleepTCN(input_dim=128).eval()
        models[arm].load_state_dict(torch.load(path, weights_only=True, map_location="cpu")["model_state"])
    checks = []
    for ordinal in [0, 75, 76, 179]:
        entry = entries[ordinal]
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("prediction changed")
        edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
        x, _ = helper.read_full_record(edf, entry["source_edf_sha256"])
        with torch.inference_mode(), np.load(entry["path"], allow_pickle=False) as expected:
            features = torch.cat([encoder.extract_features(torch.from_numpy(x[i:i+64]).unsqueeze(1))
                                  for i in range(0, len(x), 64)]).unsqueeze(0)
            row = {"zero_based_manifest_ordinal": ordinal, "epochs": len(x), "arms": {}}
            for arm, model in models.items():
                actual = torch.softmax(model(features, padding_mask=None), -1).squeeze(0).numpy()
                row["arms"][arm] = {"max_probability_difference": float(np.max(np.abs(actual - expected[arm]))),
                    "argmax_disagreements": int(np.count_nonzero(actual.argmax(1) != expected[arm].argmax(1)))}
                # Float32 CPU kernels are not guaranteed bit-identical across
                # process restarts. Bound absolute residual and require exactly
                # identical decisions; never alter stored probabilities to fit.
                if (row["arms"][arm]["max_probability_difference"] > 1e-6
                        or row["arms"][arm]["argmax_disagreements"] != 0):
                    raise ValueError("prediction replay exceeds float32 tolerance or changes decisions")
            checks.append(row)
    document = {"status": "passed", "checks": checks,
                "absolute_probability_tolerance": 1e-6, "required_argmax_disagreements": 0,
                "inference_manifest_sha256": sha256_file(manifest_path),
                "verification_script_sha256": sha256_file(Path(__file__))}
    with (args.pilot / "recovery_replay_verification.json").open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
