"""Validate the deterministic Gate-7 publication package."""

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
    "tables_markdown": "TABLES.md",
    "evidence_markdown": "CLAIM_EVIDENCE_MATRIX.md",
    "evidence_json": "claim_evidence_matrix.json",
    "manuscript_draft": "MANUSCRIPT_DRAFT_VI.md",
    "author_checklist": "AUTHOR_CHECKLIST.md",
    "primary_effects_figure": "figure_primary_effects.png",
    "tradeoff_figure": "figure_performance_speed_tradeoff.png",
    "silhouette_figure": "figure_feature_silhouette.png",
}
EXPECTED_COUNTS = {
    "experiments": 6,
    "comparisons": 5,
    "primary_comparisons": 4,
    "feature_folds": 10,
    "claims": 8,
}
EXPECTED_EXPERIMENTS = {"E0", "E1", "E2", "E3", "E4", "E6"}
EXPECTED_COMPARISONS = {"E1-E0", "E2-E1", "E3-E2", "E3-E6", "E4-E2"}
EXPECTED_CLAIM_STATUS = {
    "C01": "supported",
    "C02": "supported",
    "C03": "not_supported",
    "C04": "not_supported",
    "C05": "supported_with_tradeoff",
    "C06": "contradicted_by_measurement",
    "C07": "not_evaluated",
    "C08": "limited",
}
REQUIRED_MANUSCRIPT_SECTIONS = (
    "## Tóm tắt",
    "## 1. Đặt vấn đề",
    "## 2. Phương pháp",
    "## 3. Kết quả",
    "## 4. Thảo luận",
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
    if manifest.get("counts") != EXPECTED_COUNTS:
        raise ValueError(f"manifest counts mismatch: {manifest.get('counts')}")
    environment = manifest.get("environment", {})
    if set(environment) != {"python", "numpy", "matplotlib"} or any(
        not str(value).strip() for value in environment.values()
    ):
        raise ValueError("manifest lacks complete rendering environment")
    hashes = manifest.get("output_sha256", {})
    if set(hashes) != set(EXPECTED_OUTPUTS):
        raise ValueError("manifest output set mismatch")
    verified: dict[str, str] = {}
    for key, filename in EXPECTED_OUTPUTS.items():
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


def validate_tables(package_dir: Path) -> dict[str, Any]:
    performance = read_csv(package_dir / EXPECTED_OUTPUTS["performance_csv"])
    comparisons = read_csv(package_dir / EXPECTED_OUTPUTS["comparisons_csv"])
    complexity = read_csv(package_dir / EXPECTED_OUTPUTS["complexity_csv"])
    silhouettes = read_csv(package_dir / EXPECTED_OUTPUTS["silhouette_csv"])
    if {row["experiment"] for row in performance} != EXPECTED_EXPERIMENTS:
        raise ValueError("performance experiment set mismatch")
    if {row["comparison"] for row in comparisons} != EXPECTED_COMPARISONS:
        raise ValueError("statistical comparison set mismatch")
    if {row["experiment"] for row in complexity} != EXPECTED_EXPERIMENTS:
        raise ValueError("complexity experiment set mismatch")
    if len(silhouettes) != 10 or {int(row["fold"]) for row in silhouettes} != set(range(10)):
        raise ValueError("silhouette table must cover folds 0..9 exactly once")
    if not all(float(row["E2_minus_E1"]) < 0 for row in silhouettes):
        raise ValueError("silhouette direction differs from locked Gate-6 result")
    primary = [row for row in comparisons if row["family"] == "primary"]
    if len(primary) != 4:
        raise ValueError("primary comparison family must contain four tests")
    significant = {
        row["comparison"]
        for row in primary
        if float(row["holm_p"]) < 0.05
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


def validate_claims(package_dir: Path) -> dict[str, Any]:
    claims = read_json(package_dir / EXPECTED_OUTPUTS["evidence_json"])
    if not isinstance(claims, list) or len(claims) != 8:
        raise ValueError("claim-evidence matrix must contain eight claims")
    statuses = {row["claim_id"]: row["status"] for row in claims}
    if statuses != EXPECTED_CLAIM_STATUS:
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
        if set(row) != required_fields or any(not str(row[key]).strip() for key in required_fields):
            raise ValueError(f"{row.get('claim_id')}: incomplete claim evidence")
    return {"passed": True, "claims": len(claims), "statuses": statuses}


def validate_manuscript(package_dir: Path) -> dict[str, Any]:
    text = (package_dir / EXPECTED_OUTPUTS["manuscript_draft"]).read_text(
        encoding="utf-8"
    )
    missing = [section for section in REQUIRED_MANUSCRIPT_SECTIONS if section not in text]
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
        "required_sections": len(REQUIRED_MANUSCRIPT_SECTIONS),
        "scope_boundaries": len(required_boundaries),
    }


def validate_figures(package_dir: Path) -> dict[str, Any]:
    figure_keys = (
        "primary_effects_figure",
        "tradeoff_figure",
        "silhouette_figure",
    )
    sizes: dict[str, int] = {}
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
        "scope": "Gate7_publication_package",
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
