"""Render English supplement figures from existing aggregate results only.

This script performs no inference, model fitting, statistical testing or resampling.
It leaves the original Vietnamese figures and scientific artifacts untouched.
Run from the repository root: python scripts/build_english_supplement_figures.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_verified(path: Path, expected: str) -> dict:
    contents = path.read_bytes()
    actual = hashlib.sha256(contents).hexdigest()
    if actual != expected:
        raise ValueError(f"Scientific input hash mismatch: {path.name}")
    return json.loads(contents)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    feature_path = root / "runs/v2/analysis/gate6_feature_space/feature_space_report.json"
    context_path = root / "runs/v2/gate8/analysis_seed42.json"
    # Pinned identities of the unchanged aggregates used by this manuscript.
    feature = read_verified(feature_path, "a8f27b333876fcff2d5092e28a6b0aa7a3ec311e872e8e3bf0c2fbce16eb5d46")
    context = read_verified(context_path, "25465db4b81d02d638aa8e974b7aa4f3ec51fcb9d4190287c27b88cb3e5380e1")
    output = root / "Reports/paper_en/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    fig, axis = plt.subplots(figsize=(4.9, 3.7))
    for fold in range(10):
        item = feature["fold_results"][f"fold_{fold:02d}"]["representations"]
        e1 = item["E1"]["silhouette_score_pca"]
        e2 = item["E2"]["silhouette_score_pca"]
        axis.plot([0, 1], [e1, e2], color="#999999", alpha=0.65, linewidth=1)
        axis.scatter(0, e1, color="#356AA0", zorder=3)
        axis.scatter(1, e2, color="#D27A20", zorder=3)
    axis.set_xlim(-0.30, 1.30)
    axis.set_xticks([0, 1], ["E1: CNN15", "E2: ResNet-1D"])
    axis.set_ylabel("Silhouette coefficient")
    axis.set_title("Standardised features, 20-component PCA", fontsize=10)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"gate8_feature_silhouette_en.{suffix}", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.9, 3.7))
    axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
    for index, item in enumerate(context["comparisons"]):
        result = item["transition_radius_1_macro_f1"]
        interval = result["cluster_bootstrap"]
        effect = 100 * interval["observed_difference"]
        low = 100 * interval["ci95_low"]
        high = 100 * interval["ci95_high"]
        axis.errorbar(
            effect, index, xerr=[[effect - low], [high - effect]],
            fmt="o", color="#356AA0", capsize=4, markersize=6,
        )
        axis.text(
            1.17, index,
            f"{result['subject_wilcoxon']['holm_adjusted_p_value']:.3f}",
            ha="center", va="center", fontsize=10,
        )
    axis.text(1.17, -0.58, "Holm p", ha="center", va="center", fontsize=10)
    axis.set_yticks(
        range(len(context["comparisons"])),
        [item["comparison"].replace("FULL_CPN", "CPN").replace("-", " - ")
         for item in context["comparisons"]],
    )
    axis.set_ylim(2.65, -0.9)
    axis.set_xlim(-0.65, 1.45)
    axis.set_xticks([-0.5, 0, 0.5, 1])
    axis.set_xlabel("Boundary macro-F1 difference\n(percentage points)")
    axis.set_title("Paired subject-cluster 95% intervals", fontsize=10)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"gate8_context_ablation_effects_en.{suffix}", dpi=300)
    plt.close(fig)
    print("Rendered two English figures from hash-verified stored aggregates.")
    print("No scientific analysis, training or predictions were rerun.")


if __name__ == "__main__":
    main()
