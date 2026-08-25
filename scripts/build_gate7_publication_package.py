"""Build deterministic Gate-7 tables, figures and evidence metadata."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.io.serialization import read_json  # noqa: E402
from sleeptcn.workflows.provenance import clean_git_commit as _require_clean_git  # noqa: E402

file_sha256 = sha256_file  # compatibility alias for existing builder imports

EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E6")
PRIMARY_COMPARISONS = ("E1-E0", "E2-E1", "E3-E2", "E3-E6")
SCHEMA_VERSION = 1
LABELS = {
    "E0": "15CNN + BiLSTM, raw",
    "E1": "15CNN + TCN, raw",
    "E2": "ResNet-1D + TCN, raw",
    "E3": "ResNet-1D + TCN, filtered/chia 100",
    "E4": "ResNet-1D + TCN, band-pass",
    "E6": "ResNet-1D + TCN, filtered/z-score",
}


def clean_git_commit(workspace: Path) -> str:
    return _require_clean_git(
        workspace,
        dirty_message="official Gate-7 build requires a clean Git worktree",
    )


def validate_inputs(
    gate5: dict[str, Any],
    latency: dict[str, Any],
    parameters: dict[str, Any],
    feature: dict[str, Any],
    gate6_validation: dict[str, Any],
) -> None:
    if gate5.get("schema_version") != 2 or gate5.get("status") != "complete":
        raise ValueError("Gate-5 report is not complete schema v2")
    if [item["comparison"] for item in gate5.get("primary_results", [])] != list(
        PRIMARY_COMPARISONS
    ):
        raise ValueError("Gate-5 primary comparison family changed")
    if [item["comparison"] for item in gate5.get("secondary_results", [])] != [
        "E4-E2"
    ]:
        raise ValueError("Gate-5 secondary comparison changed")
    coverage = gate5.get("input_coverage", {})
    if set(coverage) != set(EXPERIMENTS) or any(
        (item["subjects"], item["records"], item["valid_epochs"])
        != (78, 153, 195_469)
        for item in coverage.values()
    ):
        raise ValueError("Gate-5 input coverage differs from the locked protocol")
    expected_gate6 = {
        "latency": latency,
        "parameters": parameters,
    }
    for name, report in expected_gate6.items():
        if report.get("schema_version") != 2 or report.get("status") != "complete":
            raise ValueError(f"Gate-6 {name} report is incomplete")
        if set(report.get("models", {})) != set(EXPERIMENTS):
            raise ValueError(f"Gate-6 {name} experiment set differs")
    if latency.get("mode") != "latency" or parameters.get("mode") != "parameters":
        raise ValueError("Gate-6 report modes are wrong")
    expected_protocol = {
        "fold": 0,
        "seed": 42,
        "batch_records": 1,
        "sequence_length": 100,
        "warmup": 20,
        "repeats": 100,
        "rounds": 3,
    }
    if any(latency.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("Gate-6 latency protocol differs from the frozen values")
    if feature.get("schema_version") != 3 or feature.get("status") != "complete":
        raise ValueError("Gate-6 feature-space report is incomplete")
    if feature.get("folds") != list(range(10)) or feature.get(
        "total_sample_count"
    ) != 10_000:
        raise ValueError("Gate-6 feature-space coverage differs")
    if gate6_validation.get("status") != "passed" or any(
        not gate6_validation[part]["passed"]
        for part in ("parameters", "latency", "feature_space")
    ):
        raise ValueError("Gate-6 validation report did not pass")


def performance_rows(gate5: dict[str, Any]) -> list[dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for result in gate5["primary_results"] + gate5["secondary_results"]:
        for role in ("proposed", "reference"):
            experiment = result[role]
            metrics = result["descriptive"][role]
            if experiment in observed and observed[experiment] != metrics:
                raise ValueError(f"inconsistent descriptive metrics for {experiment}")
            observed[experiment] = metrics
    if set(observed) != set(EXPERIMENTS):
        raise ValueError("descriptive metrics do not cover all experiments")
    return [
        {
            "experiment": experiment,
            "description": LABELS[experiment],
            "macro_f1": observed[experiment]["macro_f1"],
            "accuracy": observed[experiment]["accuracy"],
            "cohen_kappa": observed[experiment]["cohen_kappa"],
            "f1_W": observed[experiment]["per_class"]["W"]["f1"],
            "f1_N1": observed[experiment]["per_class"]["N1"]["f1"],
            "f1_N2": observed[experiment]["per_class"]["N2"]["f1"],
            "f1_N3": observed[experiment]["per_class"]["N3"]["f1"],
            "f1_REM": observed[experiment]["per_class"]["REM"]["f1"],
        }
        for experiment in EXPERIMENTS
    ]


def comparison_rows(gate5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family, results in (
        ("primary", gate5["primary_results"]),
        ("secondary", gate5["secondary_results"]),
    ):
        for result in results:
            bootstrap = result["cluster_bootstrap_macro_f1"]
            wilcoxon = result["subject_wilcoxon_macro_f1"]
            rows.append(
                {
                    "family": family,
                    "comparison": result["comparison"],
                    "delta_macro_f1": bootstrap["observed_difference"],
                    "ci95_low": bootstrap["ci95_low"],
                    "ci95_high": bootstrap["ci95_high"],
                    "wilcoxon_p": wilcoxon["p_value"],
                    "holm_p": wilcoxon["holm_adjusted_p_value"],
                    "median_subject_delta": wilcoxon["median_subject_difference"],
                    "wins": wilcoxon["wins"],
                    "ties": wilcoxon["ties"],
                    "losses": wilcoxon["losses"],
                    **{
                        f"delta_f1_{stage}": result["descriptive"][
                            "difference_proposed_minus_reference"
                        ]["per_class_f1"][stage]
                        for stage in ("W", "N1", "N2", "N3", "REM")
                    },
                }
            )
    return rows


def complexity_rows(
    latency: dict[str, Any], parameters: dict[str, Any]
) -> list[dict[str, Any]]:
    baseline_latency = latency["models"]["E0"]["latency"][
        "all_samples_ms_median"
    ]
    baseline_parameters = parameters["models"]["E0"]["parameters"]
    baseline_vram = latency["models"]["E0"]["latency"][
        "maximum_peak_allocated_bytes"
    ]
    rows = []
    for experiment in EXPERIMENTS:
        parameter_model = parameters["models"][experiment]
        latency_model = latency["models"][experiment]
        if parameter_model["parameters"] != latency_model["parameters"]:
            raise ValueError(f"parameter reports disagree for {experiment}")
        timing = latency_model["latency"]
        rows.append(
            {
                "experiment": experiment,
                "description": LABELS[experiment],
                "component_models": parameter_model["component_models"],
                "parameters": parameter_model["parameters"],
                "parameter_ratio_vs_E0": parameter_model["parameters"]
                / baseline_parameters,
                "latency_ms_median": timing["all_samples_ms_median"],
                "latency_ms_p95": timing["all_samples_ms_p95"],
                "throughput_epochs_per_second": timing[
                    "throughput_epochs_per_second_from_all_sample_median"
                ],
                "peak_allocated_mib": timing["maximum_peak_allocated_bytes"]
                / 2**20,
                "peak_reserved_mib": timing["maximum_peak_reserved_bytes"] / 2**20,
                "speedup_vs_E0": baseline_latency
                / timing["all_samples_ms_median"],
                "peak_allocated_ratio_vs_E0": timing[
                    "maximum_peak_allocated_bytes"
                ]
                / baseline_vram,
            }
        )
    return rows


def silhouette_rows(feature: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fold in range(10):
        item = feature["fold_results"][f"fold_{fold:02d}"]
        e1 = item["representations"]["E1"]["silhouette_score_pca"]
        e2 = item["representations"]["E2"]["silhouette_score_pca"]
        difference = item["silhouette_difference_E2_minus_E1"]
        if not np.isclose(e2 - e1, difference, rtol=0.0, atol=1e-12):
            raise ValueError(f"fold {fold}: silhouette difference mismatch")
        rows.append(
            {
                "fold": fold,
                "E1_silhouette": e1,
                "E2_silhouette": e2,
                "E2_minus_E1": difference,
                "sample_count": item["sample_count"],
                "subjects_represented": item["subjects_represented"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(
    headers: list[str], rows: Iterable[Iterable[Any]], align: list[str] | None = None
) -> str:
    row_list = [list(row) for row in rows]
    if align is None:
        align = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in row_list)
    return "\n".join(lines) + "\n"


def format_decimal(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def write_tables_markdown(
    path: Path,
    performance: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    complexity: list[dict[str, Any]],
    silhouettes: list[dict[str, Any]],
) -> None:
    primary = [row for row in comparisons if row["family"] == "primary"]
    sections = ["# Bảng Gate 7 sinh tự động\n"]
    sections.append("## Hiệu năng test out-of-fold\n")
    sections.append(
        markdown_table(
            ["E", "Macro-F1", "Accuracy", "Kappa", "F1 N1"],
            [
                [
                    row["experiment"],
                    format_decimal(row["macro_f1"], 6),
                    format_decimal(row["accuracy"], 6),
                    format_decimal(row["cohen_kappa"], 6),
                    format_decimal(row["f1_N1"], 6),
                ]
                for row in performance
            ],
            ["---", "---:", "---:", "---:", "---:"],
        )
    )
    sections.append("\n## So sánh thống kê chính\n")
    sections.append(
        markdown_table(
            ["So sánh", "Δ Macro-F1", "CI 95%", "p Holm", "Thắng/Hòa/Thua"],
            [
                [
                    row["comparison"],
                    format_decimal(row["delta_macro_f1"], 6),
                    f"[{format_decimal(row['ci95_low'], 6)}; {format_decimal(row['ci95_high'], 6)}]",
                    format_decimal(row["holm_p"], 6),
                    f"{row['wins']}/{row['ties']}/{row['losses']}",
                ]
                for row in primary
            ],
            ["---", "---:", "---:", "---:", "---:"],
        )
    )
    sections.append("\n## Độ phức tạp và tốc độ\n")
    sections.append(
        markdown_table(
            ["E", "Tham số", "Latency (ms)", "p95 (ms)", "Epoch/s", "Peak MiB", "Nhanh hơn E0"],
            [
                [
                    row["experiment"],
                    f"{row['parameters']:,}".replace(",", "."),
                    format_decimal(row["latency_ms_median"], 4),
                    format_decimal(row["latency_ms_p95"], 4),
                    f"{row['throughput_epochs_per_second']:,.0f}".replace(",", "."),
                    format_decimal(row["peak_allocated_mib"], 2),
                    f"{format_decimal(row['speedup_vs_E0'], 2)}×",
                ]
                for row in complexity
            ],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
    )
    sections.append("\n## Silhouette theo fold\n")
    sections.append(
        markdown_table(
            ["Fold", "E1", "E2", "E2−E1"],
            [
                [
                    f"{row['fold']:02d}",
                    format_decimal(row["E1_silhouette"], 6),
                    format_decimal(row["E2_silhouette"], 6),
                    format_decimal(row["E2_minus_E1"], 6),
                ]
                for row in silhouettes
            ],
            ["---:", "---:", "---:", "---:"],
        )
    )
    path.write_text("\n".join(sections), encoding="utf-8")


def plot_primary_effects(path: Path, comparisons: list[dict[str, Any]]) -> None:
    rows = [row for row in comparisons if row["family"] == "primary"]
    y = np.arange(len(rows))
    effect = np.asarray([row["delta_macro_f1"] * 100 for row in rows])
    low = np.asarray([row["ci95_low"] * 100 for row in rows])
    high = np.asarray([row["ci95_high"] * 100 for row in rows])
    colors = ["#168a45" if row["holm_p"] < 0.05 else "#4c78a8" for row in rows]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
    for index, row in enumerate(rows):
        axis.errorbar(
            effect[index],
            y[index],
            xerr=[[effect[index] - low[index]], [high[index] - effect[index]]],
            fmt="o",
            color=colors[index],
            capsize=4,
            markersize=7,
        )
        axis.text(
            high[index] + 0.08,
            y[index],
            f"p Holm={row['holm_p']:.4f}",
            va="center",
            fontsize=9,
        )
    axis.set_yticks(y, [row["comparison"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(min(-0.4, float(low.min()) - 0.1), float(high.max()) + 0.65)
    axis.set_xlabel("Chênh lệch Macro-F1 (điểm phần trăm)")
    axis.set_title("Hiệu ứng chính và CI 95% bootstrap theo cụm đối tượng")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, metadata={"Software": "SleepTCN Gate 7"})
    plt.close(fig)


def plot_tradeoff(
    path: Path,
    performance: list[dict[str, Any]],
    complexity: list[dict[str, Any]],
) -> None:
    perf = {row["experiment"]: row for row in performance}
    comp = {row["experiment"]: row for row in complexity}
    fig, axis = plt.subplots(figsize=(9, 6.5))
    for experiment in EXPERIMENTS:
        x = comp[experiment]["latency_ms_median"]
        y = perf[experiment]["macro_f1"]
        size = comp[experiment]["parameters"] / 2_000
        axis.scatter(x, y, s=size, alpha=0.75, edgecolor="black", linewidth=0.6)
        offsets = {
            "E0": (5, 7),
            "E1": (5, -14),
            "E2": (5, -14),
            "E3": (5, 8),
            "E4": (5, 2),
            "E6": (5, 7),
        }
        axis.annotate(experiment, (x, y), xytext=offsets[experiment], textcoords="offset points")
    axis.set_xlabel("Latency trung vị cho 100 epoch (ms, thấp hơn tốt hơn)")
    axis.set_ylabel("Macro-F1 test out-of-fold (cao hơn tốt hơn)")
    axis.set_title("Đánh đổi hiệu năng – tốc độ – số tham số")
    axis.grid(alpha=0.25)
    axis.text(
        0.99,
        0.02,
        "Kích thước điểm ∝ số tham số",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, metadata={"Software": "SleepTCN Gate 7"})
    plt.close(fig)


def plot_silhouette(path: Path, rows: list[dict[str, Any]]) -> None:
    folds = np.asarray([row["fold"] for row in rows])
    e1 = np.asarray([row["E1_silhouette"] for row in rows])
    e2 = np.asarray([row["E2_silhouette"] for row in rows])
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for _, left, right in zip(folds, e1, e2, strict=True):
        axis.plot([0, 1], [left, right], color="#999999", alpha=0.65, linewidth=1)
    axis.scatter(np.zeros_like(e1), e1, color="#4c78a8", label="E1 — 15CNN softmax")
    axis.scatter(np.ones_like(e2), e2, color="#f58518", label="E2 — ResNet-1D")
    axis.set_xticks([0, 1], ["E1", "E2"])
    axis.set_ylabel("Silhouette Score sau StandardScaler + PCA 20D")
    axis.set_title("So sánh Silhouette bắt cặp trên 10 test fold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, metadata={"Software": "SleepTCN Gate 7"})
    plt.close(fig)


def evidence_rows(
    performance: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    complexity: list[dict[str, Any]],
    silhouettes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    perf = {row["experiment"]: row for row in performance}
    comp = {row["experiment"]: row for row in complexity}
    compare = {row["comparison"]: row for row in comparisons}
    e3e6 = compare["E3-E6"]
    return [
        {
            "claim_id": "C01",
            "status": "supported",
            "claim": "E3 có Macro-F1 test mô tả cao nhất trong sáu cấu hình.",
            "evidence": f"E3={perf['E3']['macro_f1']:.6f}; các E còn lại thấp hơn.",
            "source": "gate5_paired_results_seed42.json",
            "allowed_wording": "E3 đạt Macro-F1 out-of-fold cao nhất trong thí nghiệm Sleep-EDF seed 42.",
            "prohibited_wording": "E3 là mô hình tốt nhất một cách tổng quát.",
        },
        {
            "claim_id": "C02",
            "status": "supported",
            "claim": "E3 tốt hơn E6 theo cả CI bootstrap và Wilcoxon sau Holm.",
            "evidence": (
                f"Δ={e3e6['delta_macro_f1']:.6f}; CI=[{e3e6['ci95_low']:.6f}, "
                f"{e3e6['ci95_high']:.6f}]; p Holm={e3e6['holm_p']:.6f}."
            ),
            "source": "gate5_paired_results_seed42.json",
            "allowed_wording": "Chia hằng số bảo toàn quan hệ biên độ tốt hơn z-score theo bản ghi trong giao thức hiện tại.",
            "prohibited_wording": "Chuẩn hóa biên độ giải quyết domain shift hoặc luôn tốt hơn z-score.",
        },
        {
            "claim_id": "C03",
            "status": "not_supported",
            "claim": "TCN tốt hơn BiLSTM có ý nghĩa thống kê.",
            "evidence": f"E1−E0 p Holm={compare['E1-E0']['holm_p']:.6f}; CI chứa 0.",
            "source": "gate5_paired_results_seed42.json",
            "allowed_wording": "E1 tăng mô tả nhẹ so với E0 nhưng chưa đủ bằng chứng sau Holm.",
            "prohibited_wording": "TCN vượt trội BiLSTM.",
        },
        {
            "claim_id": "C04",
            "status": "not_supported",
            "claim": "ResNet-1D tốt hơn 15CNN có ý nghĩa thống kê về Macro-F1.",
            "evidence": f"E2−E1 p Holm={compare['E2-E1']['holm_p']:.6f}; CI chứa 0.",
            "source": "gate5_paired_results_seed42.json",
            "allowed_wording": "E2 tăng mô tả nhẹ, chưa có ý nghĩa thống kê sau Holm.",
            "prohibited_wording": "ResNet-1D cải thiện độ chính xác một cách chắc chắn.",
        },
        {
            "claim_id": "C05",
            "status": "supported_with_tradeoff",
            "claim": "ResNet-1D + TCN đơn giản hơn về vận hành và nhanh hơn E0.",
            "evidence": (
                f"2 so với 16 mô hình thành phần; speedup E2={comp['E2']['speedup_vs_E0']:.3f}×; "
                f"tham số={comp['E2']['parameter_ratio_vs_E0']:.3f}×; peak VRAM="
                f"{comp['E2']['peak_allocated_ratio_vs_E0']:.3f}×."
            ),
            "source": "gate6_latency_fold00_seed42.json; gate6_parameters_fold00_seed42.json",
            "allowed_wording": "Pipeline ít mô hình thành phần hơn và suy luận nhanh hơn, đổi lại nhiều tham số và VRAM hơn.",
            "prohibited_wording": "Mô hình nhẹ, tiết kiệm tham số hoặc nhanh hơn 8,2×.",
        },
        {
            "claim_id": "C06",
            "status": "contradicted_by_measurement",
            "claim": "Embedding ResNet tách năm lớp tốt hơn 15CNN softmax.",
            "evidence": f"E2−E1 Silhouette < 0 ở {sum(row['E2_minus_E1'] < 0 for row in silhouettes)}/10 fold.",
            "source": "gate6_feature_space/feature_space_report.json",
            "allowed_wording": "Phân tích hỗ trợ không cho thấy Silhouette của E2 cao hơn E1.",
            "prohibited_wording": "15CNN giàu thông tin hơn hoặc ResNet kém hơn một cách tổng quát.",
        },
        {
            "claim_id": "C07",
            "status": "not_evaluated",
            "claim": "Mô hình giải quyết domain shift hoặc zero-shot trên SHHS.",
            "evidence": "Không có thí nghiệm SHHS trong giao thức E0–E6.",
            "source": "EXPERIMENT_PROTOCOL_V2.md",
            "allowed_wording": "Kết luận hiện tại chỉ áp dụng in-domain trên Sleep-EDF Expanded.",
            "prohibited_wording": "Đã chứng minh khả năng zero-shot/domain adaptation/lâm sàng.",
        },
        {
            "claim_id": "C08",
            "status": "limited",
            "claim": "Kết quả ổn định theo khởi tạo ngẫu nhiên.",
            "evidence": "Huấn luyện chính thức mới dùng seed 42.",
            "source": "experiments_v2.json; Gate-5 report",
            "allowed_wording": "Đây là kết quả của một training seed; cần thêm seed để đánh giá độ ổn định.",
            "prohibited_wording": "Kết quả bền vững theo random seed.",
        },
    ]


def write_evidence_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    sections = [
        "# Ma trận tuyên bố – bằng chứng\n",
        "Tài liệu này là chốt biên tập: mọi Abstract, Kết luận và slide phải đối chiếu trước khi sử dụng.\n",
    ]
    sections.append(
        markdown_table(
            ["ID", "Trạng thái", "Tuyên bố", "Bằng chứng", "Nguồn"],
            [
                [row["claim_id"], row["status"], row["claim"], row["evidence"], row["source"]]
                for row in rows
            ],
        )
    )
    for row in rows:
        sections.extend(
            [
                f"\n## {row['claim_id']} — {row['status']}\n",
                f"- Câu chữ được phép: {row['allowed_wording']}\n",
                f"- Câu chữ không được phép: {row['prohibited_wording']}\n",
            ]
        )
    path.write_text("\n".join(sections), encoding="utf-8")


def write_manuscript_draft(
    path: Path,
    performance: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    complexity: list[dict[str, Any]],
    silhouettes: list[dict[str, Any]],
) -> None:
    perf = {row["experiment"]: row for row in performance}
    compare = {row["comparison"]: row for row in comparisons}
    comp = {row["experiment"]: row for row in complexity}
    e3e6 = compare["E3-E6"]
    silhouette_mean = float(np.mean([row["E2_minus_E1"] for row in silhouettes]))
    draft = f"""# Bản nháp bài viết/khóa luận — Gate 7

> Trạng thái: bản nháp khoa học dựa trên artifact đã khóa. Cần bổ sung trích dẫn thư mục tài liệu tham
> khảo, thông tin hội đồng/tác giả và định dạng theo nơi nộp. Không thay các con số bằng kết quả chạy tay.

## Tiêu đề đề xuất

**Đánh giá bắt cặp ResNet-1D và mạng tích chập theo thời gian cho phân giai đoạn giấc ngủ một kênh trên
Sleep-EDF Expanded**

Tiêu đề thay thế ngắn hơn:

**Đơn giản hóa pipeline phân giai đoạn giấc ngủ một kênh: đánh đổi giữa hiệu năng, tốc độ và độ phức tạp**

## Tóm tắt

Phân giai đoạn giấc ngủ tự động từ điện não đồ một kênh có tiềm năng giảm chi phí xử lý đa ký giấc ngủ,
nhưng các so sánh mô hình thường bị ảnh hưởng bởi cách chia đối tượng, lựa chọn checkpoint và khác biệt
tiền xử lý. Nghiên cứu này tái triển khai một baseline 15CNN–BiLSTM và đánh giá tuần tự việc thay BiLSTM
bằng TCN, thay 15CNN bằng ResNet-1D, cùng các biến thể tiền xử lý trên Sleep-EDF Expanded Sleep Cassette.
Tất cả cấu hình dùng cùng split 10-fold theo đối tượng, seed huấn luyện 42 và checkpoint được chọn chỉ từ
validation; mỗi đối tượng xuất hiện đúng một lần trong test out-of-fold. Chỉ số chính là Macro-F1. Độ bất
định được ước lượng bằng bootstrap bắt cặp theo cụm đối tượng 10.000 lần; Wilcoxon signed-rank theo đối
tượng được hiệu chỉnh Holm trên bốn so sánh chính. Trong 78 đối tượng, 153 bản ghi và 195.469 epoch hợp
lệ, cấu hình E3 đạt Macro-F1 {perf['E3']['macro_f1']:.4f}, cao nhất trong sáu cấu hình. So với z-score
theo bản ghi (E6), E3 cải thiện {e3e6['delta_macro_f1']:.4f}, CI 95%
[{e3e6['ci95_low']:.4f}; {e3e6['ci95_high']:.4f}], p Holm={e3e6['holm_p']:.4f}. Thay BiLSTM bằng TCN
và thay 15CNN bằng ResNet-1D chỉ tạo mức tăng mô tả nhỏ, chưa có ý nghĩa sau Holm. Trên Tesla V100,
ResNet-1D–TCN suy luận nhanh hơn baseline khoảng {comp['E2']['speedup_vs_E0']:.2f} lần, nhưng có
{comp['E2']['parameter_ratio_vs_E0']:.2f} lần số tham số và peak VRAM cao hơn
{(comp['E2']['peak_allocated_ratio_vs_E0'] - 1) * 100:.1f}%. Kết quả cho thấy lợi ích chính đến từ
lựa chọn xử lý biên độ, trong khi pipeline ResNet-1D–TCN mang lại sự đơn giản hóa vận hành và tăng tốc
suy luận với đánh đổi về tham số và bộ nhớ. Kết luận hiện chỉ áp dụng in-domain trên Sleep-EDF và một
training seed.

**Từ khóa:** phân giai đoạn giấc ngủ; EEG một kênh; ResNet-1D; TCN; Sleep-EDF; thiết kế thực nghiệm bắt cặp.

## 1. Đặt vấn đề

Phân giai đoạn giấc ngủ là bước nền tảng trong phân tích đa ký giấc ngủ. Việc chấm thủ công đòi hỏi thời
gian và chuyên môn, tạo động lực cho các phương pháp học sâu tự động. Tuy nhiên, đánh giá trong tín hiệu
y sinh phải kiểm soát biến thiên giữa người bệnh: nếu các mô hình dùng split khác nhau, chênh lệch có thể
phản ánh thành phần đối tượng thay vì kiến trúc. Ngoài ra, tuyên bố “đơn giản” cần phân biệt số mô hình
thành phần, thời gian suy luận, số tham số và bộ nhớ.

Nghiên cứu này giải quyết ba câu hỏi. Thứ nhất, TCN có cải thiện baseline BiLSTM khi giữ nguyên 15CNN
hay không? Thứ hai, ResNet-1D có thay thế 15CNN hiệu quả khi dùng chung TCN hay không? Thứ ba, các lựa
chọn lọc và biến đổi biên độ ảnh hưởng thế nào đến hiệu năng? Điểm trọng tâm là một giao thức bắt cặp
theo đối tượng, khóa test cho đến khi hoàn tất lựa chọn checkpoint và công bố đầy đủ đánh đổi tính toán.

Các đóng góp chính gồm: (1) tái triển khai pipeline baseline và pipeline ResNet-1D–TCN trong cùng giao
thức; (2) ablation tuần tự tách thay đổi mô hình chuỗi, bộ trích đặc trưng và tiền xử lý; (3) phân tích
thống kê bắt cặp ở mức đối tượng; và (4) benchmark có kiểm soát về latency, throughput, tham số và VRAM.

## 2. Phương pháp

### 2.1. Dữ liệu và nhãn

Nghiên cứu sử dụng Sleep-EDF Expanded, phân tập Sleep Cassette, gồm 78 đối tượng và 153 bản ghi. Kênh
EEG Fpz-Cz được lấy mẫu ở 100 Hz và chia thành epoch 30 giây. Năm lớp đánh giá là W, N1, N2, N3 và REM.
Movement/Unknown được giữ trong chuỗi với nhãn −1 để bảo toàn vị trí thời gian nhưng bị mask khỏi loss và
metrics. Tổng cộng có 195.469 epoch hợp lệ.

### 2.2. Chia dữ liệu và phòng tránh rò rỉ

Split 10-fold được tạo theo đối tượng với seed 42; hai đêm của cùng một người luôn cùng vai trò. Trong
mỗi outer fold, tập test là một fold đối tượng, validation là fold kế tiếp theo modulo 10 và phần còn lại
là train. Cùng một split được dùng cho mọi cấu hình. Mỗi đối tượng xuất hiện đúng một lần trong test
out-of-fold. Checkpoint tốt nhất được chọn bằng validation Macro-F1; test bị khóa trong suốt huấn luyện
và chỉ được mở một lần sau khi đủ 60 run validation-only.

### 2.3. Cấu hình thí nghiệm

- E0: 15CNN tạo 75 xác suất từ epoch hiện tại/liền trước/liền sau, sau đó BiLSTM.
- E1: giữ 15CNN của E0 và thay BiLSTM bằng TCN chung.
- E2: thay 15CNN bằng ResNet-1D tạo embedding 128 chiều, giữ TCN và dữ liệu raw.
- E3: ResNet-1D–TCN với gói lọc và chia hằng số 100.
- E4: ResNet-1D–TCN chỉ với lọc dải.
- E6: ResNet-1D–TCN với lọc và z-score theo bản ghi.

E5 bị loại vì dữ liệu của biến thể clipping trùng bitwise với E4 (`clip_fraction=0`). Vì vậy không tạo
p-value E5−E4.

### 2.4. Chỉ số và thống kê

Macro-F1 gộp trên toàn bộ dự đoán test out-of-fold là chỉ số chính; Accuracy, Cohen's kappa và F1 từng
lớp là chỉ số hỗ trợ. CI 95% cho chênh lệch Macro-F1 được tính bằng bootstrap bắt cặp theo cụm đối tượng
10.000 lần, giữ toàn bộ epoch của đối tượng được lấy mẫu. Wilcoxon signed-rank hai phía dùng Macro-F1
từng đối tượng. Holm được áp dụng chỉ cho bốn so sánh định trước: E1−E0, E2−E1, E3−E2 và E3−E6.
E4−E2 là phân tích cơ chế thứ cấp.

### 2.5. Benchmark và phân tích đặc trưng

Benchmark dùng checkpoint thật fold 00 trên Tesla V100 16 GB, input `(1,100,1,3000)`, ba vòng xáo
thứ tự; mỗi mô hình có 20 lượt làm nóng và 100 lượt đo/vòng với đồng bộ CUDA. Phép đo không gồm I/O,
preprocessing, cache và training. Phân tích không gian đặc trưng so sánh E1/E2 trên cùng epoch test:
200 epoch/lớp/fold, StandardScaler, PCA 20 chiều và Silhouette Score; t-SNE fold 00 chỉ dùng mô tả.

## 3. Kết quả

### 3.1. Hiệu năng test

E0, E1, E2, E3, E4 và E6 lần lượt đạt Macro-F1
{perf['E0']['macro_f1']:.4f}, {perf['E1']['macro_f1']:.4f}, {perf['E2']['macro_f1']:.4f},
{perf['E3']['macro_f1']:.4f}, {perf['E4']['macro_f1']:.4f} và {perf['E6']['macro_f1']:.4f}. E3 có
giá trị mô tả cao nhất; E6 thấp nhất trong nhóm ResNet-1D–TCN.

E1−E0 tăng {compare['E1-E0']['delta_macro_f1']:.4f}, nhưng CI chứa 0 và p Holm=
{compare['E1-E0']['holm_p']:.4f}. E2−E1 tăng {compare['E2-E1']['delta_macro_f1']:.4f}, CI chứa 0 và
p Holm={compare['E2-E1']['holm_p']:.4f}. Do đó chưa đủ bằng chứng kết luận TCN tốt hơn BiLSTM hoặc
ResNet-1D tốt hơn 15CNN về Macro-F1 với seed hiện tại.

E3−E2 tăng Macro-F1 gộp {compare['E3-E2']['delta_macro_f1']:.4f} với CI vừa vượt 0, nhưng Wilcoxon
không có ý nghĩa và số đối tượng thắng/thua là {compare['E3-E2']['wins']}/{compare['E3-E2']['losses']}.
Điều này cho thấy lợi ích gộp không đồng đều theo đối tượng. E3−E6 là kết quả nhất quán nhất, với chênh
lệch {e3e6['delta_macro_f1']:.4f}, CI hoàn toàn dương và p Holm={e3e6['holm_p']:.4f}.

### 3.2. Độ phức tạp và tốc độ

E0 có {comp['E0']['parameters']:,} tham số; E1 có {comp['E1']['parameters']:,}; E2–E6 có
{comp['E2']['parameters']:,}. ResNet-1D–TCN giảm từ 16 xuống 2 mô hình thành phần và có latency trung
vị khoảng {comp['E2']['latency_ms_median']:.3f} ms/100 epoch, so với
{comp['E0']['latency_ms_median']:.3f} ms của E0. Tuy nhiên, số tham số tăng
{comp['E2']['parameter_ratio_vs_E0']:.2f} lần và peak VRAM tăng
{(comp['E2']['peak_allocated_ratio_vs_E0'] - 1) * 100:.1f}%.

### 3.3. Không gian đặc trưng

Silhouette E2 thấp hơn E1 trong 10/10 fold; chênh lệch E2−E1 trung bình là {silhouette_mean:.4f}.
Do đó phép đo không hỗ trợ giả thuyết embedding ResNet tự động tạo cụm lớp tốt hơn softmax 15CNN.
Kết quả này chỉ mang tính hỗ trợ và không thay thế đánh giá dự đoán chuỗi.

## 4. Thảo luận

Kết quả cho thấy cần tách hai loại đóng góp. Về chất lượng dự đoán, thay đổi kiến trúc chuỗi và bộ trích
đặc trưng đem lại mức tăng mô tả nhỏ nhưng chưa vượt qua ngưỡng suy luận sau hiệu chỉnh đa kiểm định.
Ngược lại, cách xử lý biên độ E3 so với z-score E6 tạo hiệu ứng lớn hơn và nhất quán ở cả bootstrap lẫn
Wilcoxon. Điều này gợi ý việc bảo toàn quan hệ biên độ có liên quan đến hiệu năng trong dữ liệu hiện tại,
nhưng chưa chứng minh quan hệ nhân quả hoặc khả năng khái quát sang cơ sở dữ liệu khác.

Về vận hành, ResNet-1D–TCN thay 15 mô hình CNN bằng một bộ trích đặc trưng, giúp pipeline ít thành phần
hơn và nhanh hơn khoảng {comp['E2']['speedup_vs_E0']:.2f} lần. Đổi lại, mô hình có nhiều tham số và sử
dụng peak VRAM cao hơn. Do đó “đơn giản hóa” nên được dùng theo nghĩa kiến trúc vận hành, không đồng nhất
với tiết kiệm tài nguyên.

Sự khác biệt giữa Silhouette và Macro-F1 nhấn mạnh rằng biểu đồ t-SNE hoặc độ gọn cụm không thể thay thế
đánh giá tác vụ. Softmax 15CNN được tối ưu trực tiếp theo nhãn và có cấu trúc 15×5 xác suất, trong khi
embedding ResNet có thể mã hóa thông tin phục vụ TCN mà không tạo cụm Euclid gọn.

## 5. Hạn chế

Nghiên cứu mới dùng một training seed 42, nên chưa định lượng độ ổn định theo khởi tạo. Benchmark chỉ
thực hiện trên một Tesla V100, một batch và một độ dài chuỗi, không gồm I/O hay preprocessing. Dữ liệu
chỉ là Sleep-EDF Expanded in-domain; chưa có SHHS, domain shift, zero-shot, đa kênh hoặc xác nhận lâm
sàng. Phân tích không gian đặc trưng được thực hiện sau Gate 5 và chỉ là bằng chứng hỗ trợ. N1 vẫn là lớp
khó và EEG một kênh không chứa đầy đủ thông tin chuyển động mắt.

## 6. Kết luận

Trong giao thức bắt cặp theo đối tượng, E3 đạt hiệu năng mô tả cao nhất và tốt hơn E6 một cách nhất quán,
cho thấy lựa chọn biến đổi biên độ là yếu tố đáng chú ý nhất. ResNet-1D–TCN mang lại pipeline ít mô hình
thành phần hơn và suy luận nhanh hơn, nhưng không tiết kiệm tham số hoặc VRAM. Các kết luận chỉ áp dụng
cho Sleep-EDF Expanded và seed 42; bước xác nhận tiếp theo nên đánh giá thêm seed và dữ liệu ngoài miền
theo một giao thức đăng ký trước riêng biệt.

## 7. Hướng dẫn sử dụng bảng và hình

- Bảng hiệu năng: `TABLES.md`, phần “Hiệu năng test out-of-fold”.
- Hình hiệu ứng: `figure_primary_effects.png`.
- Hình đánh đổi: `figure_performance_speed_tradeoff.png`.
- Hình đặc trưng: `figure_feature_silhouette.png`; t-SNE gốc nằm ở artifact Gate 6.
- Trước khi sửa Abstract/Kết luận, đối chiếu `CLAIM_EVIDENCE_MATRIX.md`.
"""
    path.write_text(draft, encoding="utf-8")


def write_author_checklist(path: Path) -> None:
    path.write_text(
        """# Danh sách kiểm tra trước khi nộp

## Nội dung khoa học

- [ ] Mọi số liệu trong Abstract khớp bảng sinh tự động.
- [ ] Chỉ gọi Macro-F1 là chỉ số chính; Accuracy/kappa là hỗ trợ.
- [ ] Nêu rõ 78 đối tượng, 153 bản ghi, 195.469 epoch hợp lệ.
- [ ] Nêu rõ một training seed 42.
- [ ] Không coi 10 fold là 10 mẫu độc lập.
- [ ] Holm chỉ gồm bốn so sánh chính.
- [ ] E4−E2 được ghi là phân tích thứ cấp.
- [ ] Không tạo p-value E5−E4.
- [ ] Báo cáo đồng thời tốc độ, tham số và VRAM.
- [ ] Không sử dụng tuyên bố 8,2×.
- [ ] Không tuyên bố domain shift, zero-shot, SHHS hoặc giá trị lâm sàng.

## Trình bày

- [ ] Thêm trích dẫn paper gốc và các công trình liên quan bằng nguồn chính thức.
- [ ] Định nghĩa mọi chữ viết tắt khi xuất hiện lần đầu.
- [ ] Mọi bảng/hình có caption, đơn vị và phạm vi đo.
- [ ] Hình t-SNE được ghi rõ chỉ mang tính mô tả.
- [ ] Sơ đồ pipeline phân biệt preprocessing, extractor và sequence model.
- [ ] Phụ lục ghi commit, config/split hash và môi trường phần mềm.

## Tái lập

- [ ] Nhánh/commit công bố đã được gắn tag.
- [ ] Không đưa dataset, cache hoặc metadata mức epoch lên kho công khai.
- [ ] Lệnh tái tạo bảng/hình Gate 7 được ghi trong README/runbook.
- [ ] Tất cả kiểm thử và manifest xuất bản trả trạng thái đạt.
""",
        encoding="utf-8",
    )


def build(workspace: Path, output_dir: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    commit = clean_git_commit(workspace)
    inputs = {
        "gate5": workspace / "runs/v2/analysis/gate5_paired_results_seed42.json",
        "latency": workspace / "runs/v2/analysis/gate6_latency_fold00_seed42.json",
        "parameters": workspace / "runs/v2/analysis/gate6_parameters_fold00_seed42.json",
        "feature": workspace
        / "runs/v2/analysis/gate6_feature_space/feature_space_report.json",
        "gate6_validation": workspace / "runs/v2/analysis/gate6_validation_report.json",
    }
    reports = {name: read_json(path) for name, path in inputs.items()}
    validate_inputs(**reports)
    performance = performance_rows(reports["gate5"])
    comparisons = comparison_rows(reports["gate5"])
    complexity = complexity_rows(reports["latency"], reports["parameters"])
    silhouettes = silhouette_rows(reports["feature"])
    evidence = evidence_rows(performance, comparisons, complexity, silhouettes)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "performance_csv": output_dir / "table_performance.csv",
        "comparisons_csv": output_dir / "table_statistical_comparisons.csv",
        "complexity_csv": output_dir / "table_complexity_speed.csv",
        "silhouette_csv": output_dir / "table_feature_silhouette.csv",
        "tables_markdown": output_dir / "TABLES.md",
        "evidence_markdown": output_dir / "CLAIM_EVIDENCE_MATRIX.md",
        "evidence_json": output_dir / "claim_evidence_matrix.json",
        "manuscript_draft": output_dir / "MANUSCRIPT_DRAFT_VI.md",
        "author_checklist": output_dir / "AUTHOR_CHECKLIST.md",
        "primary_effects_figure": output_dir / "figure_primary_effects.png",
        "tradeoff_figure": output_dir / "figure_performance_speed_tradeoff.png",
        "silhouette_figure": output_dir / "figure_feature_silhouette.png",
    }
    write_csv(files["performance_csv"], performance)
    write_csv(files["comparisons_csv"], comparisons)
    write_csv(files["complexity_csv"], complexity)
    write_csv(files["silhouette_csv"], silhouettes)
    write_tables_markdown(
        files["tables_markdown"], performance, comparisons, complexity, silhouettes
    )
    write_evidence_markdown(files["evidence_markdown"], evidence)
    files["evidence_json"].write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manuscript_draft(
        files["manuscript_draft"], performance, comparisons, complexity, silhouettes
    )
    write_author_checklist(files["author_checklist"])
    plot_primary_effects(files["primary_effects_figure"], comparisons)
    plot_tradeoff(files["tradeoff_figure"], performance, complexity)
    plot_silhouette(files["silhouette_figure"], silhouettes)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "git_commit": commit,
        "scope": "Gate7_publication_tables_figures_and_claim_evidence",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "output_sha256": {
            name: sha256_file(path) for name, path in files.items()
        },
        "counts": {
            "experiments": len(performance),
            "comparisons": len(comparisons),
            "primary_comparisons": sum(
                row["family"] == "primary" for row in comparisons
            ),
            "feature_folds": len(silhouettes),
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
