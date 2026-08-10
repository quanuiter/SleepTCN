"""Create a balanced t-SNE view and quantitative feature-space summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


STAGES = ("W", "N1", "N2", "N3", "REM")


def load_features(
    feature_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    features, labels, subjects = [], [], []
    seen: set[tuple[str, str, int]] = set()
    dimensions: set[int] = set()
    extractor_hashes: set[str] = set()
    outer_folds: set[int] = set()
    seeds: set[int] = set()
    for path in sorted(feature_root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as npz:
            required = {"metadata_json", "features", "labels", "original_epoch_index"}
            if not required.issubset(npz.files):
                continue
            metadata = json.loads(str(npz["metadata_json"].item()))
            extractor_hashes.add(str(metadata["extractor_sha256"]))
            outer_folds.add(int(metadata["outer_fold"]))
            seeds.add(int(metadata["seed"]))
            matrix = npz["features"]
            target = npz["labels"]
            original = npz["original_epoch_index"]
            valid = (target >= 0) & (target < 5)
            dimensions.add(int(matrix.shape[1]))
            selected_indices = np.flatnonzero(valid)
            keep = []
            for index in selected_indices:
                key = (
                    str(metadata["subject_id"]),
                    str(metadata["record_key"]),
                    int(original[index]),
                )
                if key in seen:
                    continue
                seen.add(key)
                keep.append(int(index))
            if keep:
                features.append(matrix[keep].astype(np.float32, copy=False))
                labels.append(target[keep].astype(np.int8, copy=False))
                subjects.append(
                    np.full(len(keep), str(metadata["subject_id"]), dtype="U5")
                )
    if not features:
        raise ValueError(f"no feature artifacts found below {feature_root}")
    if len(dimensions) != 1:
        raise ValueError(f"mixed feature dimensions: {sorted(dimensions)}")
    if len(extractor_hashes) != 1 or len(outer_folds) != 1 or len(seeds) != 1:
        raise ValueError(
            "feature analysis must use one extractor checkpoint, fold and seed; "
            f"found hashes={len(extractor_hashes)}, folds={sorted(outer_folds)}, "
            f"seeds={sorted(seeds)}"
        )
    metadata_summary: dict[str, object] = {
        "extractor_sha256": next(iter(extractor_hashes)),
        "outer_fold": next(iter(outer_folds)),
        "seed": next(iter(seeds)),
    }
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(subjects),
        metadata_summary,
    )


def balanced_sample(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    available = [int(np.sum(labels == label)) for label in range(5)]
    target_count = min(per_class, min(available))
    chosen = []
    for label in range(5):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) == 0:
            raise ValueError(f"class {label} is absent")
        chosen.append(
            rng.choice(candidates, size=target_count, replace=False)
        )
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return features[indices], labels[indices], subjects[indices]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-per-class", type=int, default=1000)
    parser.add_argument("--silhouette-sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_per_class <= 0 or args.silhouette_sample <= 0:
        raise ValueError("sample sizes must be positive")
    features, labels, subjects, artifact_metadata = load_features(
        args.feature_root.resolve()
    )
    features, labels, subjects = balanced_sample(
        features, labels, subjects, args.sample_per_class, args.seed
    )
    standardized = StandardScaler().fit_transform(features)
    sample_size = min(args.silhouette_sample, len(labels))
    silhouette = float(
        silhouette_score(
            standardized,
            labels,
            sample_size=sample_size,
            random_state=args.seed,
        )
    )
    perplexity = min(30.0, max(5.0, (len(labels) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=args.seed,
    ).fit_transform(standardized)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tsne_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "stage", "stage_id", "subject_id"])
        for point, label, subject in zip(embedding, labels, subjects, strict=True):
            writer.writerow([float(point[0]), float(point[1]), STAGES[label], int(label), subject])
    figure, axis = plt.subplots(figsize=(9, 7))
    for label, stage in enumerate(STAGES):
        selected = labels == label
        axis.scatter(
            embedding[selected, 0],
            embedding[selected, 1],
            s=8,
            alpha=0.55,
            label=stage,
        )
    axis.set_title("t-SNE của không gian đặc trưng (lấy mẫu cân bằng lớp)")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(markerscale=2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "tsne.png", dpi=200)
    plt.close(figure)
    report = {
        "schema_version": 1,
        "feature_root": str(args.feature_root.resolve()),
        "feature_dimension": int(features.shape[1]),
        "artifact_metadata": artifact_metadata,
        "balanced_samples": int(len(labels)),
        "samples_per_class": {
            STAGES[label]: int(np.sum(labels == label)) for label in range(5)
        },
        "silhouette_score_standardized_original_features": silhouette,
        "silhouette_sample_size": sample_size,
        "tsne_perplexity": perplexity,
        "seed": args.seed,
        "warning": "t-SNE is descriptive; it is not a causal performance test.",
    }
    (args.output_dir / "feature_space_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
