"""Validate the final deterministic Gate-8 publication package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_OUTPUTS = {
    "performance_csv": "table_performance.csv",
    "comparisons_csv": "table_statistical_comparisons.csv",
    "complexity_csv": "table_complexity_speed.csv",
    "silhouette_csv": "table_feature_silhouette.csv",
    "ablation_csv": "table_context_ablation.csv",
    "ablation_comparisons_csv": "table_context_ablation_comparisons.csv",
    "tables_markdown": "TABLES.md",
    "evidence_markdown": "CLAIM_EVIDENCE_MATRIX.md",
    "evidence_json": "claim_evidence_matrix.json",
    "manuscript_draft": "MANUSCRIPT_DRAFT_VI.md",
    "author_checklist": "AUTHOR_CHECKLIST.md",
    "primary_effects_figure": "figure_primary_effects.png",
    "tradeoff_figure": "figure_performance_speed_tradeoff.png",
    "silhouette_figure": "figure_feature_silhouette.png",
    "ablation_figure": "figure_context_ablation_effects.png",
}
EXPECTED_COUNTS = {
    "experiments": 6,
    "comparisons": 5,
    "primary_comparisons": 4,
    "feature_folds": 10,
    "gate8_conditions": 4,
    "gate8_comparisons": 3,
    "claims": 12,
}
EXPECTED_EXPERIMENTS = {"E0", "E1", "E2", "E3", "E4", "E6"}
EXPECTED_CONDITIONS = {"FULL_CPN", "C", "CP", "CN"}
EXPECTED_GATE8_COMPARISONS = {"FULL_CPN-C", "FULL_CPN-CP", "FULL_CPN-CN"}
EXPECTED_CLAIM_STATUS = {
    "C01": "supported",
    "C02": "supported",
    "C03": "not_supported",
    "C04": "not_supported",
    "C05": "supported_with_tradeoff",
    "C06": "contradicted_by_measurement",
    "C07": "not_evaluated",
    "C08": "limited",
    "C09": "not_supported",
    "C10": "withdrawn_unsupported",
    "C11": "not_established",
    "C12": "supported",
}
REQUIRED_MANUSCRIPT_SECTIONS = (
    "## Tóm tắt",
    "## 1. Đặt vấn đề",
    "## 2. Phương pháp",
    "### 2.6. Ablation nhóm đặc trưng C/P/N",
    "## 3. Kết quả",
    "### 3.4. Ablation nhóm đặc trưng C/P/N",
    "## 4. Thảo luận",
    "### 4.1. Ý nghĩa của kết quả Gate 8",
    "## 5. Hạn chế",
    "## 6. Kết luận",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(package_dir: Path) -> dict[str, Any]:
    manifest = read_json(package_dir / "publication_manifest.json")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "scope": "Gate8_final_publication_package",
        "counts": EXPECTED_COUNTS,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Gate-8 manifest mismatch: {mismatches}")
    commit = manifest.get("git_commit", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("manifest git_commit is not a full SHA-1")
    if set(manifest.get("input_sha256", {})) != {
        "gate5",
        "latency",
        "parameters",
        "feature",
        "gate6_validation",
        "gate8_analysis",
        "gate8_validation_campaign",
        "gate8_test_campaign",
    }:
        raise ValueError("Gate-8 manifest input set mismatch")
    environment = manifest.get("environment", {})
    if set(environment) != {"python", "numpy", "matplotlib"} or any(
        not str(value).strip() for value in environment.values()
    ):
        raise ValueError("Gate-8 manifest lacks rendering environment")
    hashes = manifest.get("output_sha256", {})
    if set(hashes) != set(EXPECTED_OUTPUTS):
        raise ValueError("Gate-8 manifest output set mismatch")
    verified = {}
    for key, filename in EXPECTED_OUTPUTS.items():
        path = package_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != hashes[key]:
            raise ValueError(f"{filename}: SHA-256 mismatch")
        verified[key] = observed
    return {"passed": True, "git_commit": commit, "verified_outputs": verified}


def validate_tables(package_dir: Path) -> dict[str, Any]:
    performance = read_csv(package_dir / EXPECTED_OUTPUTS["performance_csv"])
    ablation = read_csv(package_dir / EXPECTED_OUTPUTS["ablation_csv"])
    comparisons = read_csv(package_dir / EXPECTED_OUTPUTS["ablation_comparisons_csv"])
    if {row["experiment"] for row in performance} != EXPECTED_EXPERIMENTS:
        raise ValueError("Gate-8 package experiment set mismatch")
    if {row["condition"] for row in ablation} != EXPECTED_CONDITIONS:
        raise ValueError("Gate-8 ablation condition set mismatch")
    if {row["comparison"] for row in comparisons} != EXPECTED_GATE8_COMPARISONS:
        raise ValueError("Gate-8 ablation comparison set mismatch")
    if not all(float(row["holm_p"]) == 1.0 for row in comparisons):
        raise ValueError("Gate-8 Holm results differ from locked analysis")
    if not all(float(row["ci95_low"]) < 0 < float(row["ci95_high"]) for row in comparisons):
        raise ValueError("Gate-8 confidence interval direction changed")
    by_condition = {row["condition"]: row for row in ablation}
    if not abs(float(by_condition["FULL_CPN"]["overall_macro_f1"]) - 0.7802296650249438) < 1e-15:
        raise ValueError("Full CPN Macro-F1 changed")
    markdown = (package_dir / EXPECTED_OUTPUTS["tables_markdown"]).read_text(encoding="utf-8")
    for phrase in ("# Bảng Gate 8", "ablation C/P/N", "không thiết lập tương đương"):
        if phrase not in markdown:
            raise ValueError(f"Gate-8 table boundary missing: {phrase}")
    return {
        "passed": True,
        "experiments": len(performance),
        "gate8_conditions": len(ablation),
        "gate8_comparisons": len(comparisons),
    }


def validate_claims(package_dir: Path) -> dict[str, Any]:
    claims = read_json(package_dir / EXPECTED_OUTPUTS["evidence_json"])
    if not isinstance(claims, list) or len(claims) != 12:
        raise ValueError("Gate-8 claim-evidence matrix must contain 12 claims")
    statuses = {row["claim_id"]: row["status"] for row in claims}
    if statuses != EXPECTED_CLAIM_STATUS:
        raise ValueError(f"Gate-8 claim status mismatch: {statuses}")
    required = {
        "claim_id",
        "claim",
        "status",
        "evidence",
        "source",
        "allowed_wording",
        "prohibited_wording",
    }
    for row in claims:
        if set(row) != required or any(not str(row[key]).strip() for key in required):
            raise ValueError(f"{row.get('claim_id')}: incomplete claim evidence")
    prohibited = " ".join(row["prohibited_wording"] for row in claims)
    if "12% thông tin" not in prohibited or "tương đương" not in prohibited:
        raise ValueError("Gate-8 prohibited wording is incomplete")
    return {"passed": True, "claims": len(claims), "statuses": statuses}


def validate_manuscript(package_dir: Path) -> dict[str, Any]:
    text = (package_dir / EXPECTED_OUTPUTS["manuscript_draft"]).read_text(encoding="utf-8")
    missing = [section for section in REQUIRED_MANUSCRIPT_SECTIONS if section not in text]
    if missing:
        raise ValueError(f"Gate-8 manuscript sections missing: {missing}")
    required_boundaries = (
        "3.76 lần",
        "4.37 lần số tham số",
        "một training seed",
        "chưa có SHHS",
        "không phải phép đo phần trăm thông tin",
        "không thiết lập tương đương",
        "P/N không có thông tin",
    )
    absent = [phrase for phrase in required_boundaries if phrase not in text]
    if absent:
        raise ValueError(f"Gate-8 manuscript boundaries missing: {absent}")
    prohibited_assertions = (
        "P/N chỉ chứa 12% thông tin",
        "P/N hoàn toàn vô dụng",
        "đã chứng minh khả năng zero-shot",
        "tiết kiệm tham số và VRAM",
    )
    present = [phrase for phrase in prohibited_assertions if phrase in text]
    if present:
        raise ValueError(f"unsupported Gate-8 manuscript assertions: {present}")
    return {"passed": True, "required_sections": len(REQUIRED_MANUSCRIPT_SECTIONS)}


def validate_figures(package_dir: Path) -> dict[str, Any]:
    figure_keys = (
        "primary_effects_figure",
        "tradeoff_figure",
        "silhouette_figure",
        "ablation_figure",
    )
    sizes = {}
    for key in figure_keys:
        path = package_dir / EXPECTED_OUTPUTS[key]
        data = path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 10_000:
            raise ValueError(f"{path.name}: invalid or unexpectedly small PNG")
        sizes[path.name] = len(data)
    return {"passed": True, "png_bytes": sizes}


def validate(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "Gate8_final_publication_package",
        "manifest": validate_manifest(package_dir),
        "tables": validate_tables(package_dir),
        "claims": validate_claims(package_dir),
        "manuscript": validate_manuscript(package_dir),
        "figures": validate_figures(package_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.package_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
