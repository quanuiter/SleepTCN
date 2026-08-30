"""Build the final Gate-8 publication package from locked Gate 5-8 artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.io.serialization import read_json  # noqa: E402
from sleeptcn.workflows.provenance import clean_git_commit as _require_clean_git  # noqa: E402

file_sha256 = sha256_file  # compatibility alias for existing builder imports


SCHEMA_VERSION = 1
CONDITIONS = ("FULL_CPN", "C", "CP", "CN")
COMPARISONS = ("FULL_CPN-C", "FULL_CPN-CP", "FULL_CPN-CN")


def clean_git_commit(workspace: Path) -> str:
    return _require_clean_git(
        workspace,
        dirty_message="official Gate-8 build requires a clean Git worktree",
    )


def load_gate7_builder(workspace: Path) -> Any:
    path = workspace / "scripts" / "build_gate7_publication_package.py"
    spec = importlib.util.spec_from_file_location("gate7_publication_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Gate-7 publication builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gate7_publication_base"] = module
    spec.loader.exec_module(module)
    return module


def validate_gate8_inputs(
    analysis: dict[str, Any],
    validation_campaign: dict[str, Any],
    test_campaign: dict[str, Any],
) -> None:
    if (
        analysis.get("schema_version") != 1
        or analysis.get("status") != "complete"
        or analysis.get("gate") != "GATE_8_CONTEXT_GROUP_ABLATION"
    ):
        raise ValueError("Gate-8 analysis is incomplete or has the wrong schema")
    coverage = analysis.get("input_coverage", {})
    if (coverage.get("subjects"), coverage.get("records"), coverage.get("valid_epochs")) != (
        78,
        153,
        195_469,
    ):
        raise ValueError("Gate-8 analysis coverage differs from the locked protocol")
    if set(coverage.get("predictions", {})) != set(CONDITIONS):
        raise ValueError("Gate-8 analysis does not cover all four conditions")
    observed_comparisons = tuple(
        item["comparison"] for item in analysis.get("comparisons", [])
    )
    if observed_comparisons != COMPARISONS:
        raise ValueError("Gate-8 comparison order differs from the frozen protocol")
    for item in analysis["comparisons"]:
        test = item["transition_radius_1_macro_f1"]["subject_wilcoxon"]
        if test.get("holm_family_size") != 3:
            raise ValueError("Gate-8 Holm family must contain exactly three comparisons")
    expected_campaign = {
        "validation": (validation_campaign, "validation_complete"),
        "test": (test_campaign, "complete"),
    }
    expected_targets = {
        f"{condition}/fold_{fold:02d}"
        for condition in ("CP", "CN", "C")
        for fold in range(10)
    }
    for name, (campaign, status) in expected_campaign.items():
        if campaign.get("status") != status or campaign.get("target_count") != 30:
            raise ValueError(f"Gate-8 {name} campaign is incomplete")
        targets = campaign.get("targets", {})
        if set(targets) != expected_targets or any(
            item.get("state") != "complete" for item in targets.values()
        ):
            raise ValueError(f"Gate-8 {name} campaign target set is incomplete")


def ablation_condition_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition in CONDITIONS:
        item = analysis["descriptive"][condition]
        rows.append(
            {
                "condition": condition,
                "overall_macro_f1": item["overall"]["macro_f1"],
                "overall_accuracy": item["overall"]["accuracy"],
                "n1_f1": item["overall"]["n1_f1"],
                "n1_recall": item["overall"]["n1_recall"],
                "transition_radius1_epochs": item["transition_radius_1"]["epochs"],
                "transition_radius1_macro_f1": item["transition_radius_1"][
                    "macro_f1"
                ],
                "transition_radius1_n1_recall": item["transition_radius_1"][
                    "n1_recall"
                ],
                "transition_radius2_epochs": item["transition_radius_2"]["epochs"],
                "transition_radius2_macro_f1": item["transition_radius_2"][
                    "macro_f1"
                ],
                **{
                    f"transition_{name}_macro_f1": item["transition_types"][name][
                        "macro_f1"
                    ]
                    for name in ("W_N1", "N1_N2", "N1_REM", "N2_N3")
                },
            }
        )
    return rows


def ablation_comparison_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in analysis["comparisons"]:
        bootstrap = item["transition_radius_1_macro_f1"]["cluster_bootstrap"]
        wilcoxon = item["transition_radius_1_macro_f1"]["subject_wilcoxon"]
        rows.append(
            {
                "comparison": item["comparison"],
                "role": item["role"],
                "delta_transition_macro_f1": bootstrap["observed_difference"],
                "ci95_low": bootstrap["ci95_low"],
                "ci95_high": bootstrap["ci95_high"],
                "wilcoxon_p": wilcoxon["p_value"],
                "holm_p": wilcoxon["holm_adjusted_p_value"],
                "median_subject_delta": wilcoxon["median_subject_difference"],
                "wins": wilcoxon["wins"],
                "ties": wilcoxon["ties"],
                "losses": wilcoxon["losses"],
                **item["supporting_overall"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty Gate-8 table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def append_ablation_tables(
    path: Path,
    gate7: Any,
    conditions: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    text = path.read_text(encoding="utf-8").replace(
        "# Bảng Gate 7 sinh tự động", "# Bảng Gate 8 — kết quả định lượng"
    )
    text += "\n## Gate 8 — ablation C/P/N\n\n"
    text += gate7.markdown_table(
        ["Điều kiện", "Macro-F1", "F1 N1", "Recall N1", "Macro-F1 chuyển pha", "Recall N1 chuyển pha"],
        [
            [
                row["condition"],
                fmt(row["overall_macro_f1"]),
                fmt(row["n1_f1"]),
                fmt(row["n1_recall"]),
                fmt(row["transition_radius1_macro_f1"]),
                fmt(row["transition_radius1_n1_recall"]),
            ]
            for row in conditions
        ],
        ["---", "---:", "---:", "---:", "---:", "---:"],
    )
    text += "\n## Gate 8 — so sánh vùng chuyển pha ±1\n\n"
    text += gate7.markdown_table(
        ["So sánh", "Δ Macro-F1", "CI 95%", "p Holm", "Thắng/Hòa/Thua"],
        [
            [
                row["comparison"],
                fmt(row["delta_transition_macro_f1"]),
                f"[{fmt(row['ci95_low'])}; {fmt(row['ci95_high'])}]",
                fmt(row["holm_p"]),
                f"{row['wins']}/{row['ties']}/{row['losses']}",
            ]
            for row in comparisons
        ],
        ["---", "---:", "---:", "---:", "---:"],
    )
    text += (
        "\nGate 8 là phân tích cơ chế bổ sung với một seed. Giá trị p lớn không thiết lập tương đương; "
        "ablation không phải phép đo phần trăm thông tin.\n"
    )
    path.write_text(text, encoding="utf-8")


def gate8_evidence_rows(
    base: list[dict[str, str]], comparisons: list[dict[str, Any]]
) -> list[dict[str, str]]:
    by_name = {row["comparison"]: row for row in comparisons}
    full_c = by_name["FULL_CPN-C"]
    all_non_significant = all(row["holm_p"] >= 0.05 for row in comparisons)
    if not all_non_significant:
        raise ValueError("unexpected significant Gate-8 comparison")
    additional = [
        {
            "claim_id": "C09",
            "claim": "P/N mang lại lợi ích tăng thêm có ý nghĩa cho vùng chuyển pha.",
            "status": "not_supported",
            "evidence": (
                f"Full CPN−C={full_c['delta_transition_macro_f1']:.6f}, "
                f"CI95%=[{full_c['ci95_low']:.6f}; {full_c['ci95_high']:.6f}], "
                "ba p Holm đều bằng 1,000."
            ),
            "source": "Gate 8 analysis, paired subject-cluster bootstrap and Wilcoxon-Holm.",
            "allowed_wording": "Chưa quan sát thấy đóng góp tăng thêm có ý nghĩa của P/N cho Macro-F1 vùng chuyển pha.",
            "prohibited_wording": "P/N chắc chắn không có tác dụng.",
        },
        {
            "claim_id": "C10",
            "claim": "Có thể quy P/N thành một tỷ lệ phần trăm thông tin.",
            "status": "withdrawn_unsupported",
            "evidence": "Ablation chỉ đo hiệu ứng dự báo có điều kiện trong một quy trình; không đo lượng thông tin hay quan hệ nhân quả.",
            "source": "Gate 8 protocol claim boundary and group ablation design.",
            "allowed_wording": "Báo cáo chênh lệch dự báo, CI và kiểm định bắt cặp theo từng nhóm.",
            "prohibited_wording": "P/N chỉ chứa hoặc đóng góp 12% thông tin.",
        },
        {
            "claim_id": "C11",
            "claim": "Full CPN tương đương C, CP hoặc CN.",
            "status": "not_established",
            "evidence": "Không có biên tương đương/không thua kém được định trước; kiểm định khác biệt không có ý nghĩa không chứng minh tương đương.",
            "source": "Gate 8 protocol and confidence intervals crossing zero.",
            "allowed_wording": "Chưa phát hiện khác biệt; tương đương chưa được kiểm định.",
            "prohibited_wording": "Các điều kiện đã được chứng minh tương đương.",
        },
        {
            "claim_id": "C12",
            "claim": "Gate 8 hoàn tất và có thể truy nguyên artifact.",
            "status": "supported",
            "evidence": "30/30 validation, 30/30 test, 30 checkpoint, 30 vector train và prediction thẳng hàng; local audit passed.",
            "source": "Gate 8 validation/test campaign journals, manifests and SHA-256 audit.",
            "allowed_wording": "Gate 8 hoàn tất kỹ thuật và artifact đã được kiểm toán.",
            "prohibited_wording": "Hoàn tất kỹ thuật đồng nghĩa kết luận có giá trị lâm sàng.",
        },
    ]
    return [*base, *additional]


def integrate_gate8_manuscript(
    path: Path,
    conditions: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    condition = {row["condition"]: row for row in conditions}
    comparison = {row["comparison"]: row for row in comparisons}
    text = path.read_text(encoding="utf-8").replace(
        "# Bản nháp bài viết/khóa luận — Gate 7",
        "# Bản thảo nghiên cứu — Gate 8",
    )
    method = """
### 2.6. Ablation nhóm đặc trưng C/P/N

Gate 8 đánh giá đóng góp dự báo có điều kiện của ba nhóm ngữ cảnh: epoch hiện tại, epoch liền trước và
epoch liền sau. Điều kiện đầy đủ được đối chiếu với các điều kiện lần lượt loại nhóm ngữ cảnh trước hoặc
sau; nhóm bị loại được thay bằng giá trị trung bình ước lượng từ dữ liệu huấn luyện của từng fold. Tiêu
chí chính là Macro-F1 tại vùng chuyển pha. Các so sánh dùng bootstrap cụm bắt cặp, Wilcoxon theo đối
tượng và hiệu chỉnh Holm. Gate 8 là phân tích cơ chế bổ sung với một training seed. Phân tích này không phải phép đo phần trăm thông tin và không thiết lập tương đương.

"""
    result = f"""
### 3.4. Ablation nhóm đặc trưng C/P/N

Macro-F1 toàn bộ của Full CPN, C, CP và CN lần lượt là
{condition['FULL_CPN']['overall_macro_f1']:.6f}, {condition['C']['overall_macro_f1']:.6f},
{condition['CP']['overall_macro_f1']:.6f} và {condition['CN']['overall_macro_f1']:.6f}. Tại vùng chuyển
pha ±1, các giá trị tương ứng là {condition['FULL_CPN']['transition_radius1_macro_f1']:.6f},
{condition['C']['transition_radius1_macro_f1']:.6f}, {condition['CP']['transition_radius1_macro_f1']:.6f}
và {condition['CN']['transition_radius1_macro_f1']:.6f}.

Full CPN−C tại vùng chuyển pha là {comparison['FULL_CPN-C']['delta_transition_macro_f1']:.6f}, CI 95%
[{comparison['FULL_CPN-C']['ci95_low']:.6f}; {comparison['FULL_CPN-C']['ci95_high']:.6f}]. Full CPN−CP
và Full CPN−CN lần lượt là {comparison['FULL_CPN-CP']['delta_transition_macro_f1']:.6f} và
{comparison['FULL_CPN-CN']['delta_transition_macro_f1']:.6f}; cả ba p Holm đều bằng 1,000. Vì vậy
chưa có bằng chứng thống kê rằng P/N tạo lợi ích tăng thêm cho Macro-F1 vùng chuyển pha trong thiết kế
hiện tại. Kết quả không thiết lập tương đương và không cho phép kết luận P/N không có thông tin.

"""
    discussion = """
### 4.1. Ý nghĩa của kết quả Gate 8

Ablation theo nhóm không xác nhận cách diễn giải cũ rằng các nhóm ngữ cảnh chỉ đóng góp một tỷ lệ thông
tin cố định. Hiệu ứng tại vùng chuyển pha nhỏ, khoảng tin cậy chứa 0 và không nhất quán theo đối tượng.
Cách trình bày phù hợp là báo cáo hiệu ứng tăng thêm có điều kiện và độ bất định, không quy đổi thành
phần trăm thông tin. Việc không bác bỏ giả thuyết không cũng không chứng minh các điều kiện tương đương.

"""
    text = text.replace("## 3. Kết quả", method + "## 3. Kết quả")
    text = text.replace("## 4. Thảo luận", result + "## 4. Thảo luận")
    text = text.replace("## 5. Hạn chế", discussion + "## 5. Hạn chế")
    text = text.replace(
        "N1 vẫn là lớp",
        "Gate 8 được thiết kế sau khi đã xem E0–E6 và chỉ dùng một seed; do đó đây là phân tích cơ chế bổ sung, không phải xác nhận độc lập. Không có kiểm định tương đương hoặc không thua kém. N1 vẫn là lớp",
    )
    text = text.replace(
        "## 7. Hướng dẫn sử dụng bảng và hình",
        "Gate 8 không tìm thấy lợi ích tăng thêm có ý nghĩa của P/N tại vùng chuyển pha, nhưng cũng không chứng minh các ablation tương đương hoặc P/N vô dụng.\n\n## 7. Hướng dẫn sử dụng bảng và hình",
    )
    text += (
        "\n- Bảng ablation: `table_context_ablation.csv` và "
        "`table_context_ablation_comparisons.csv`.\n"
        "- Hình Gate 8: `figure_context_ablation_effects.png`.\n"
    )
    path.write_text(text, encoding="utf-8")


def append_gate8_checklist(path: Path) -> None:
    text = path.read_text(encoding="utf-8").replace("Gate 7", "Gate 8")
    text += """

## Gate 8 — ablation nhóm đặc trưng C/P/N

- [ ] Ghi rõ Full CPN tái sử dụng E1, còn CP/CN/C được huấn luyện lại TCN.
- [ ] Ghi rõ vector thay thế chỉ được tính từ dữ liệu train hợp lệ trong từng fold.
- [ ] Tiêu chí chính là Macro-F1 vùng chuyển pha ±1; Holm gồm đúng ba so sánh Gate 8.
- [ ] Không dùng cụm “12% thông tin” hoặc bất kỳ phần trăm thông tin nào.
- [ ] Không diễn giải p lớn thành tương đương hoặc không thua kém.
- [ ] Ghi rõ Gate 8 là phân tích cơ chế bổ sung với một training seed.
- [ ] Không tuyên bố P/N vô dụng hoặc C đủ thay thế CPN.
"""
    path.write_text(text, encoding="utf-8")


def plot_ablation_effects(path: Path, rows: list[dict[str, Any]]) -> None:
    y = np.arange(len(rows))
    effect = np.asarray([row["delta_transition_macro_f1"] * 100 for row in rows])
    low = np.asarray([row["ci95_low"] * 100 for row in rows])
    high = np.asarray([row["ci95_high"] * 100 for row in rows])
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
    for index, row in enumerate(rows):
        axis.errorbar(
            effect[index],
            y[index],
            xerr=[[effect[index] - low[index]], [high[index] - effect[index]]],
            fmt="o",
            color="#4c78a8",
            capsize=4,
            markersize=7,
        )
        axis.text(high[index] + 0.025, y[index], f"p Holm={row['holm_p']:.3f}", va="center")
    axis.set_yticks(y, [row["comparison"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(float(low.min()) - 0.1, float(high.max()) + 0.3)
    axis.set_xlabel("Chênh lệch Macro-F1 vùng chuyển pha (điểm phần trăm)")
    axis.set_title("Gate 8: hiệu ứng ablation C/P/N và CI 95% theo đối tượng")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, metadata={"Software": "SleepTCN Gate 8"})
    plt.close(fig)


def build(workspace: Path, output_dir: Path) -> dict[str, Any]:
    workspace, output_dir = workspace.resolve(), output_dir.resolve()
    commit = clean_git_commit(workspace)
    gate7 = load_gate7_builder(workspace)
    inputs = {
        "gate5": workspace / "runs/v2/analysis/gate5_paired_results_seed42.json",
        "latency": workspace / "runs/v2/analysis/gate6_latency_fold00_seed42.json",
        "parameters": workspace / "runs/v2/analysis/gate6_parameters_fold00_seed42.json",
        "feature": workspace / "runs/v2/analysis/gate6_feature_space/feature_space_report.json",
        "gate6_validation": workspace / "runs/v2/analysis/gate6_validation_report.json",
        "gate8_analysis": workspace / "runs/v2/gate8/analysis_seed42.json",
        "gate8_validation_campaign": workspace / "runs/v2/gate8/validation_campaign_seed42.json",
        "gate8_test_campaign": workspace / "runs/v2/gate8/test_campaign_seed42.json",
    }
    reports = {name: read_json(path) for name, path in inputs.items()}
    gate7.validate_inputs(
        reports["gate5"],
        reports["latency"],
        reports["parameters"],
        reports["feature"],
        reports["gate6_validation"],
    )
    validate_gate8_inputs(
        reports["gate8_analysis"],
        reports["gate8_validation_campaign"],
        reports["gate8_test_campaign"],
    )
    performance = gate7.performance_rows(reports["gate5"])
    comparisons = gate7.comparison_rows(reports["gate5"])
    complexity = gate7.complexity_rows(reports["latency"], reports["parameters"])
    silhouettes = gate7.silhouette_rows(reports["feature"])
    ablation = ablation_condition_rows(reports["gate8_analysis"])
    ablation_comparisons = ablation_comparison_rows(reports["gate8_analysis"])
    evidence = gate8_evidence_rows(
        gate7.evidence_rows(performance, comparisons, complexity, silhouettes),
        ablation_comparisons,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "performance_csv": output_dir / "table_performance.csv",
        "comparisons_csv": output_dir / "table_statistical_comparisons.csv",
        "complexity_csv": output_dir / "table_complexity_speed.csv",
        "silhouette_csv": output_dir / "table_feature_silhouette.csv",
        "ablation_csv": output_dir / "table_context_ablation.csv",
        "ablation_comparisons_csv": output_dir / "table_context_ablation_comparisons.csv",
        "tables_markdown": output_dir / "TABLES.md",
        "evidence_markdown": output_dir / "CLAIM_EVIDENCE_MATRIX.md",
        "evidence_json": output_dir / "claim_evidence_matrix.json",
        "manuscript_draft": output_dir / "MANUSCRIPT_DRAFT_VI.md",
        "author_checklist": output_dir / "AUTHOR_CHECKLIST.md",
        "primary_effects_figure": output_dir / "figure_primary_effects.png",
        "tradeoff_figure": output_dir / "figure_performance_speed_tradeoff.png",
        "silhouette_figure": output_dir / "figure_feature_silhouette.png",
        "ablation_figure": output_dir / "figure_context_ablation_effects.png",
    }
    gate7.write_csv(files["performance_csv"], performance)
    gate7.write_csv(files["comparisons_csv"], comparisons)
    gate7.write_csv(files["complexity_csv"], complexity)
    gate7.write_csv(files["silhouette_csv"], silhouettes)
    write_csv(files["ablation_csv"], ablation)
    write_csv(files["ablation_comparisons_csv"], ablation_comparisons)
    gate7.write_tables_markdown(
        files["tables_markdown"], performance, comparisons, complexity, silhouettes
    )
    append_ablation_tables(files["tables_markdown"], gate7, ablation, ablation_comparisons)
    gate7.write_evidence_markdown(files["evidence_markdown"], evidence)
    files["evidence_json"].write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate7.write_manuscript_draft(
        files["manuscript_draft"], performance, comparisons, complexity, silhouettes
    )
    integrate_gate8_manuscript(files["manuscript_draft"], ablation, ablation_comparisons)
    gate7.write_author_checklist(files["author_checklist"])
    append_gate8_checklist(files["author_checklist"])
    gate7.plot_primary_effects(files["primary_effects_figure"], comparisons)
    gate7.plot_tradeoff(files["tradeoff_figure"], performance, complexity)
    gate7.plot_silhouette(files["silhouette_figure"], silhouettes)
    plot_ablation_effects(files["ablation_figure"], ablation_comparisons)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "git_commit": commit,
        "scope": "Gate8_final_publication_package",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "output_sha256": {name: sha256_file(path) for name, path in files.items()},
        "counts": {
            "experiments": len(performance),
            "comparisons": len(comparisons),
            "primary_comparisons": sum(row["family"] == "primary" for row in comparisons),
            "feature_folds": len(silhouettes),
            "gate8_conditions": len(ablation),
            "gate8_comparisons": len(ablation_comparisons),
            "claims": len(evidence),
        },
    }
    (output_dir / "publication_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.workspace, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
