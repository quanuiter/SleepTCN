"""Export only verified cohort-level results; exclude private manifests and record identifiers."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import write_once_json


CAMPAIGNS = {
    "E4_seed42": ("runs/teacher_revision_cpu_20261002/e4_seed42_extension", "paired_difference", 10),
    "calibration_seed123": ("runs/teacher_revision_cpu_20261003/calibration_10fold_seed123", "comparisons", 10),
    "weighted_seed123_fold0": ("runs/teacher_revision_cpu_20261001/weighted_e3_fold0", "weighted_minus_unweighted", 1),
    "ADAST_seed123_fold0": ("runs/teacher_revision_cpu_20261001/adast_fold0", "adast_minus_source_only", 1),
}


def main():
    public = {}
    for name, (relative, comparison, folds) in CAMPAIGNS.items():
        base = ROOT / relative
        result = json.loads((base / "aggregate_results.json").read_bytes())
        verification = json.loads((base / "verification.json").read_bytes())
        digest = sha256_file(base / "aggregate_results.json")
        if verification["status"] != "passed" or verification["aggregate_results_sha256"] != digest:
            raise ValueError("campaign is not verified")
        if result["subjects"] != 180 or result["valid_epochs"] != 169012:
            raise ValueError("expected shared benchmark support")
        document = {"subjects": result["subjects"], "valid_epochs": result["valid_epochs"],
                    "fold_models_per_arm": folds, "seed": 42 if name == "E4_seed42" else 123,
                    "target_context": "historical_benchmark_window" if name == "E4_seed42" else "full_record",
                    "evaluation": "historical_169012_scored_epoch_mask", "arms": copy.deepcopy(result["arms"]),
                    "verification_status": "passed", "private_aggregate_sha256": digest,
                    "comparison": copy.deepcopy(result[comparison])}
        public[name] = document
    output = ROOT / "Reports/analysis/teacher_revision_20261003.json"
    write_once_json(output, {"schema_version": 1, "date": "2026-10-03", "class_order": ["W", "N1", "N2", "N3", "REM"],
        "scope": "Additional comparisons on the previously examined SHHS cohort; fixed fitted models, paired participant bootstrap.",
        "privacy": "Only cohort-level metrics, pooled confusion matrices and bootstrap summaries. No participant identifiers, record paths, per-person metrics or probabilities.",
        "bootstrap_resamples": 10000, "bootstrap_seed": 2031, "campaigns": public})
    print(json.dumps({"status": "exported", "campaigns": len(public), "path": str(output)}, indent=2))


if __name__ == "__main__":
    main()
