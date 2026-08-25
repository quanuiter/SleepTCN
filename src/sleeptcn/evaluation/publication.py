"""Validation contracts for the locked Gate-7 and Gate-8 publication packages.

The publication packages are immutable inputs to these validators.  This module
owns the validation logic so the command-line scripts remain thin adapters and
the same contract can be exercised without importing the CLI modules.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from sleeptcn.io.hashing import sha256_file
from sleeptcn.io.serialization import read_json


# Gate 7 contract ---------------------------------------------------------

GATE7_EXPECTED_OUTPUTS = {
    "performance_csv": "table_performance.csv",
    "comparisons_csv": "table_statistical_comparisons.csv",
    "complexity_csv": "table_complexity_speed.csv",
    "silhouette_csv": "table_feature_silhouette.csv",
    "tables_markdown": "TABLES.md",
    "evidence_markdown": "CLAIM_EVIDENCE_MATRIX.md",
    "evidence_json": "claim_evidence_matrix.json",
    "manuscript_draft": "MANUSCRIPT_DRAFT_VI.md",
    "author_checklist": "AUTHOR_CHECKLIST.md",
    "primary_effects_figure": "figure_primary_effects.png",
    "tradeoff_figure": "figure_performance_speed_tradeoff.png",
    "silhouette_figure": "figure_feature_silhouette.png",
}
GATE7_EXPECTED_COUNTS = {
    "experiments": 6,
    "comparisons": 5,
    "primary_comparisons": 4,
    "feature_folds": 10,
    "claims": 8,
}
GATE7_EXPECTED_EXPERIMENTS = {"E0", "E1", "E2", "E3", "E4", "E6"}
GATE7_EXPECTED_COMPARISONS = {"E1-E0", "E2-E1", "E3-E2", "E3-E6", "E4-E2"}
GATE7_EXPECTED_CLAIM_STATUS = {
    "C01": "supported",
    "C02": "supported",
    "C03": "not_supported",
    "C04": "not_supported",
    "C05": "supported_with_tradeoff",
    "C06": "contradicted_by_measurement",
    "C07": "not_evaluated",
    "C08": "limited",
}
GATE7_REQUIRED_MANUSCRIPT_SECTIONS = (
    "## Tóm tắt",
    "## 1. Đặt vấn đề",
    "## 2. Phương pháp",
    "## 3. Kết quả",
    "## 4. Thảo luận",
    "## 5. Hạn chế",
    "## 6. Kết luận",
)


# Gate 8 contract ---------------------------------------------------------

GATE8_EXPECTED_OUTPUTS = {
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
GATE8_EXPECTED_COUNTS = {
    "experiments": 6,
    "comparisons": 5,
    "primary_comparisons": 4,
    "feature_folds": 10,
    "gate8_conditions": 4,
    "gate8_comparisons": 3,
    "claims": 12,
}
GATE8_EXPECTED_EXPERIMENTS = {"E0", "E1", "E2", "E3", "E4", "E6"}
GATE8_EXPECTED_CONDITIONS = {"FULL_CPN", "C", "CP", "CN"}
GATE8_EXPECTED_COMPARISONS = {"FULL_CPN-C", "FULL_CPN-CP", "FULL_CPN-CN"}
GATE8_EXPECTED_CLAIM_STATUS = {
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
GATE8_REQUIRED_MANUSCRIPT_SECTIONS = (
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


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV artifact using its first row as the header."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_gate7_manifest(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "publication_manifest.json"
    manifest = read_json(manifest_path)
    expected_top = {
        "schema_version": 1,
        "status": "complete",
        "scope": "Gate7_publication_tables_figures_and_claim_evidence",
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_top.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"manifest schema/status/scope mismatch: {mismatches}")
    commit = manifest.get("git_commit", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("manifest git_commit is not a full SHA-1")
    if manifest.get("counts") != GATE7_EXPECTED_COUNTS:
        raise ValueError(f"manifest counts mismatch: {manifest.get('counts')}")
    environment = manifest.get("environment", {})
    if set(environment) != {"python", "numpy", "matplotlib"} or any(
        not str(value).strip() for value in environment.values()
    ):
        raise ValueError("manifest lacks complete rendering environment")
    hashes = manifest.get("output_sha256", {})
    if set(hashes) != set(GATE7_EXPECTED_OUTPUTS):
        raise ValueError("manifest output set mismatch")
    verified: dict[str, str] = {}
    for key, filename in GATE7_EXPECTED_OUTPUTS.items():
        path = package_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != hashes[key]:
            raise ValueError(f"{filename}: SHA-256 mismatch")
        verified[key] = observed
    return {
        "passed": True,
        "git_commit": commit,
        "environment": environment,
        "verified_outputs": verified,
    }


def validate_gate7_tables(package_dir: Path) -> dict[str, Any]:
    performance = read_csv(package_dir / GATE7_EXPECTED_OUTPUTS["performance_csv"])
    comparisons = read_csv(package_dir / GATE7_EXPECTED_OUTPUTS["comparisons_csv"])
    complexity = read_csv(package_dir / GATE7_EXPECTED_OUTPUTS["complexity_csv"])
    silhouettes = read_csv(package_dir / GATE7_EXPECTED_OUTPUTS["silhouette_csv"])
    if {row["experiment"] for row in performance} != GATE7_EXPECTED_EXPERIMENTS:
        raise ValueError("performance experiment set mismatch")
    if {row["comparison"] for row in comparisons} != GATE7_EXPECTED_COMPARISONS:
        raise ValueError("statistical comparison set mismatch")
    if {row["experiment"] for row in complexity} != GATE7_EXPECTED_EXPERIMENTS:
        raise ValueError("complexity experiment set mismatch")
    if len(silhouettes) != 10 or {int(row["fold"]) for row in silhouettes} != set(range(10)):
        raise ValueError("silhouette table must cover folds 0..9 exactly once")
    if not all(float(row["E2_minus_E1"]) < 0 for row in silhouettes):
        raise ValueError("silhouette direction differs from locked Gate-6 result")
    primary = [row for row in comparisons if row["family"] == "primary"]
    if len(primary) != 4:
        raise ValueError("primary comparison family must contain four tests")
    significant = {
        row["comparison"] for row in primary if float(row["holm_p"]) < 0.05
    }
    if significant != {"E3-E6"}:
        raise ValueError(f"unexpected Holm-significant comparisons: {significant}")
    return {
        "passed": True,
        "experiments": len(performance),
        "comparisons": len(comparisons),
        "primary_comparisons": len(primary),
        "feature_folds": len(silhouettes),
        "holm_significant": sorted(significant),
    }


def validate_gate7_claims(package_dir: Path) -> dict[str, Any]:
    claims = read_json(package_dir / GATE7_EXPECTED_OUTPUTS["evidence_json"])
    if not isinstance(claims, list) or len(claims) != 8:
        raise ValueError("claim-evidence matrix must contain eight claims")
    statuses = {row["claim_id"]: row["status"] for row in claims}
    if statuses != GATE7_EXPECTED_CLAIM_STATUS:
        raise ValueError(f"claim status mismatch: {statuses}")
    required_fields = {
        "claim_id",
        "claim",
        "status",
        "evidence",
        "source",
        "allowed_wording",
        "prohibited_wording",
    }
    for row in claims:
        if set(row) != required_fields or any(
            not str(row[key]).strip() for key in required_fields
        ):
            raise ValueError(f"{row.get('claim_id')}: incomplete claim evidence")
    return {"passed": True, "claims": len(claims), "statuses": statuses}


def validate_gate7_manuscript(package_dir: Path) -> dict[str, Any]:
    text = (package_dir / GATE7_EXPECTED_OUTPUTS["manuscript_draft"]).read_text(
        encoding="utf-8"
    )
    missing = [section for section in GATE7_REQUIRED_MANUSCRIPT_SECTIONS if section not in text]
    if missing:
        raise ValueError(f"manuscript sections missing: {missing}")
    required_boundaries = (
        "3.76 lần",
        "4.37 lần số tham số",
        "peak VRAM cao hơn",
        "chỉ áp dụng in-domain",
        "một training seed",
        "chưa có SHHS",
        "10/10 fold",
    )
    absent = [phrase for phrase in required_boundaries if phrase not in text]
    if absent:
        raise ValueError(f"manuscript evidence boundaries missing: {absent}")
    prohibited_assertions = (
        "nhanh hơn 8,2 lần",
        "TCN vượt trội BiLSTM",
        "đã chứng minh khả năng zero-shot",
        "tiết kiệm tham số và VRAM",
    )
    present = [phrase for phrase in prohibited_assertions if phrase in text]
    if present:
        raise ValueError(f"unsupported manuscript assertions found: {present}")
    return {
        "passed": True,
        "required_sections": len(GATE7_REQUIRED_MANUSCRIPT_SECTIONS),
        "scope_boundaries": len(required_boundaries),
    }


def validate_gate7_figures(package_dir: Path) -> dict[str, Any]:
    figure_keys = (
        "primary_effects_figure",
        "tradeoff_figure",
        "silhouette_figure",
    )
    sizes: dict[str, int] = {}
    for key in figure_keys:
        path = package_dir / GATE7_EXPECTED_OUTPUTS[key]
        data = path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 10_000:
            raise ValueError(f"{path.name}: invalid or unexpectedly small PNG")
        sizes[path.name] = len(data)
    return {"passed": True, "png_bytes": sizes}


def validate_gate7_package(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "Gate7_publication_package",
        "manifest": validate_gate7_manifest(package_dir),
        "tables": validate_gate7_tables(package_dir),
        "claims": validate_gate7_claims(package_dir),
        "manuscript": validate_gate7_manuscript(package_dir),
        "figures": validate_gate7_figures(package_dir),
    }


def validate_gate8_manifest(package_dir: Path) -> dict[str, Any]:
    manifest = read_json(package_dir / "publication_manifest.json")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "scope": "Gate8_final_publication_package",
        "counts": GATE8_EXPECTED_COUNTS,
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
    if set(hashes) != set(GATE8_EXPECTED_OUTPUTS):
        raise ValueError("Gate-8 manifest output set mismatch")
    verified: dict[str, str] = {}
    for key, filename in GATE8_EXPECTED_OUTPUTS.items():
        path = package_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != hashes[key]:
            raise ValueError(f"{filename}: SHA-256 mismatch")
        verified[key] = observed
    return {"passed": True, "git_commit": commit, "verified_outputs": verified}


def validate_gate8_tables(package_dir: Path) -> dict[str, Any]:
    performance = read_csv(package_dir / GATE8_EXPECTED_OUTPUTS["performance_csv"])
    ablation = read_csv(package_dir / GATE8_EXPECTED_OUTPUTS["ablation_csv"])
    comparisons = read_csv(package_dir / GATE8_EXPECTED_OUTPUTS["ablation_comparisons_csv"])
    if {row["experiment"] for row in performance} != GATE8_EXPECTED_EXPERIMENTS:
        raise ValueError("Gate-8 package experiment set mismatch")
    if {row["condition"] for row in ablation} != GATE8_EXPECTED_CONDITIONS:
        raise ValueError("Gate-8 ablation condition set mismatch")
    if {row["comparison"] for row in comparisons} != GATE8_EXPECTED_COMPARISONS:
        raise ValueError("Gate-8 ablation comparison set mismatch")
    if not all(float(row["holm_p"]) == 1.0 for row in comparisons):
        raise ValueError("Gate-8 Holm results differ from locked analysis")
    if not all(float(row["ci95_low"]) < 0 < float(row["ci95_high"]) for row in comparisons):
        raise ValueError("Gate-8 confidence interval direction changed")
    by_condition = {row["condition"]: row for row in ablation}
    if not abs(float(by_condition["FULL_CPN"]["overall_macro_f1"]) - 0.7802296650249438) < 1e-15:
        raise ValueError("Full CPN Macro-F1 changed")
    markdown = (package_dir / GATE8_EXPECTED_OUTPUTS["tables_markdown"]).read_text(
        encoding="utf-8"
    )
    for phrase in ("# Bảng Gate 8", "ablation C/P/N", "không thiết lập tương đương"):
        if phrase not in markdown:
            raise ValueError(f"Gate-8 table boundary missing: {phrase}")
    return {
        "passed": True,
        "experiments": len(performance),
        "gate8_conditions": len(ablation),
        "gate8_comparisons": len(comparisons),
    }


def validate_gate8_claims(package_dir: Path) -> dict[str, Any]:
    claims = read_json(package_dir / GATE8_EXPECTED_OUTPUTS["evidence_json"])
    if not isinstance(claims, list) or len(claims) != 12:
        raise ValueError("Gate-8 claim-evidence matrix must contain 12 claims")
    statuses = {row["claim_id"]: row["status"] for row in claims}
    if statuses != GATE8_EXPECTED_CLAIM_STATUS:
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


def validate_gate8_manuscript(package_dir: Path) -> dict[str, Any]:
    text = (package_dir / GATE8_EXPECTED_OUTPUTS["manuscript_draft"]).read_text(encoding="utf-8")
    missing = [section for section in GATE8_REQUIRED_MANUSCRIPT_SECTIONS if section not in text]
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
    return {"passed": True, "required_sections": len(GATE8_REQUIRED_MANUSCRIPT_SECTIONS)}


def validate_gate8_figures(package_dir: Path) -> dict[str, Any]:
    figure_keys = (
        "primary_effects_figure",
        "tradeoff_figure",
        "silhouette_figure",
        "ablation_figure",
    )
    sizes: dict[str, int] = {}
    for key in figure_keys:
        path = package_dir / GATE8_EXPECTED_OUTPUTS[key]
        data = path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 10_000:
            raise ValueError(f"{path.name}: invalid or unexpectedly small PNG")
        sizes[path.name] = len(data)
    return {"passed": True, "png_bytes": sizes}


def validate_gate8_package(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "Gate8_final_publication_package",
        "manifest": validate_gate8_manifest(package_dir),
        "tables": validate_gate8_tables(package_dir),
        "claims": validate_gate8_claims(package_dir),
        "manuscript": validate_gate8_manuscript(package_dir),
        "figures": validate_gate8_figures(package_dir),
    }


__all__ = [
    "read_json",
    "read_csv",
    "GATE7_EXPECTED_OUTPUTS",
    "GATE7_EXPECTED_COUNTS",
    "GATE7_EXPECTED_EXPERIMENTS",
    "GATE7_EXPECTED_COMPARISONS",
    "GATE7_EXPECTED_CLAIM_STATUS",
    "GATE7_REQUIRED_MANUSCRIPT_SECTIONS",
    "validate_gate7_manifest",
    "validate_gate7_tables",
    "validate_gate7_claims",
    "validate_gate7_manuscript",
    "validate_gate7_figures",
    "validate_gate7_package",
    "GATE8_EXPECTED_OUTPUTS",
    "GATE8_EXPECTED_COUNTS",
    "GATE8_EXPECTED_EXPERIMENTS",
    "GATE8_EXPECTED_CONDITIONS",
    "GATE8_EXPECTED_COMPARISONS",
    "GATE8_EXPECTED_CLAIM_STATUS",
    "GATE8_REQUIRED_MANUSCRIPT_SECTIONS",
    "validate_gate8_manifest",
    "validate_gate8_tables",
    "validate_gate8_claims",
    "validate_gate8_manuscript",
    "validate_gate8_figures",
    "validate_gate8_package",
]
