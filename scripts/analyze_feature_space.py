"""Gate-6 paired feature-space analysis for locked E1/E2 extractors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.hashing import sha256_file
from sleeptcn.experiment import _selected_records, build_context
from sleeptcn.features import MANIPULATIONS, MANIPULATION_PREFIX, STAGE_NAMES
from sleeptcn.run_validation import validate_run
from sleeptcn.test_gate import _load_extractor
from sleeptcn.training_data import resolve_fold_partitions
from sleeptcn.workflows.provenance import clean_git_commit


STAGES = ("W", "N1", "N2", "N3", "REM")
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SampleEpoch:
    subject_id: str
    record_key: str
    position: int
    original_epoch_index: int
    label: int


def _clean_git_commit(workspace: Path) -> str:
    return clean_git_commit(
        workspace,
        dirty_message="official feature analysis requires a clean Git worktree",
    )


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def select_balanced_epochs(
    records: Sequence[Any], per_class: int, seed: int
) -> tuple[SampleEpoch, ...]:
    """Balance classes and distribute each class across subjects round-robin."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    candidates: dict[int, dict[str, list[SampleEpoch]]] = {
        label: defaultdict(list) for label in range(5)
    }
    for record in records:
        for position in np.flatnonzero(record.valid_mask):
            label = int(record.y[position])
            candidates[label][record.info.subject_id].append(
                SampleEpoch(
                    subject_id=record.info.subject_id,
                    record_key=record.info.record_key,
                    position=int(position),
                    original_epoch_index=int(record.original_epoch_index[position]),
                    label=label,
                )
            )
    rng = np.random.default_rng(seed)
    selected: list[SampleEpoch] = []
    for label in range(5):
        groups = candidates[label]
        available = sum(len(items) for items in groups.values())
        if available < per_class:
            raise ValueError(
                f"class {label} has {available} epochs, fewer than requested {per_class}"
            )
        subjects = sorted(groups)
        rng.shuffle(subjects)
        for subject in subjects:
            rng.shuffle(groups[subject])
        pointers = {subject: 0 for subject in subjects}
        class_selected: list[SampleEpoch] = []
        while len(class_selected) < per_class:
            progressed = False
            for subject in subjects:
                pointer = pointers[subject]
                if pointer < len(groups[subject]):
                    class_selected.append(groups[subject][pointer])
                    pointers[subject] += 1
                    progressed = True
                    if len(class_selected) == per_class:
                        break
            if not progressed:
                raise AssertionError("balanced sampler exhausted unexpectedly")
        selected.extend(class_selected)
    order = rng.permutation(len(selected))
    result = tuple(selected[int(index)] for index in order)
    keys = {
        (item.subject_id, item.record_key, item.original_epoch_index)
        for item in result
    }
    if len(keys) != len(result):
        raise AssertionError("feature sample contains duplicate epochs")
    return result


def _fold_records(workspace: Path, fold: int, seed: int) -> tuple[Any, ...]:
    if fold not in range(10) or seed != 42:
        raise ValueError("Gate 6 feature analysis requires folds 00-09 and seed 42")
    context = build_context(
        workspace,
        "E1",
        fold,
        seed,
        "cpu",
        smoke=False,
        allow_test_evaluation=True,
        num_workers=0,
        resume=True,
    )
    report = validate_run(workspace, context.run_root)
    if not report.get("passed") or "test" not in report.get("roles", {}):
        raise ValueError(f"E1 fold {fold:02d} has not passed Gate 4")
    partitions = resolve_fold_partitions(
        workspace / "data" / "processed",
        workspace / "data" / "splits" / "sleepedf_sc_10fold_seed42_v2.json",
        fold,
        "paper_raw_v1",
    )
    return _selected_records(partitions.test, "paper_raw_v1", None)


def prepare_sample_manifest(
    workspace: Path, *, seed: int, per_class_per_fold: int
) -> dict[str, Any]:
    workspace = workspace.resolve()
    commit = _clean_git_commit(workspace)
    folds: dict[str, Any] = {}
    global_subjects: set[str] = set()
    for fold in range(10):
        records = _fold_records(workspace, fold, seed)
        samples = select_balanced_epochs(
            records, per_class_per_fold, seed + fold
        )
        subject_counts: dict[str, int] = defaultdict(int)
        for sample in samples:
            subject_counts[sample.subject_id] += 1
            global_subjects.add(sample.subject_id)
        folds[f"fold_{fold:02d}"] = {
            "sample_count": len(samples),
            "class_counts": {
                STAGES[label]: sum(sample.label == label for sample in samples)
                for label in range(5)
            },
            "subject_counts": dict(sorted(subject_counts.items())),
            "samples": [asdict(sample) for sample in samples],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "purpose": "paired_10fold_supportive_E1_15CNN_vs_E2_ResNet_feature_space",
        "selection_timing": "after_Gate5_supportive_not_confirmatory",
        "source_git_commit": commit,
        "folds": list(range(10)),
        "role": "test",
        "seed": seed,
        "data_variant": "paper_raw_v1",
        "sample_per_class_per_fold": per_class_per_fold,
        "total_sample_count": 10 * 5 * per_class_per_fold,
        "subjects_represented_across_folds": len(global_subjects),
        "config_sha256": sha256_file(workspace / "configs/experiments_v2.json"),
        "split_sha256": sha256_file(
            workspace / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
        ),
        "fold_samples": folds,
    }


def _load_sample_manifest(
    path: Path,
) -> tuple[dict[str, Any], dict[int, tuple[SampleEpoch, ...]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "folds": list(range(10)),
        "role": "test",
        "seed": 42,
        "data_variant": "paper_raw_v1",
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"sample manifest mismatch: {mismatches}")
    folds: dict[int, tuple[SampleEpoch, ...]] = {}
    for fold in range(10):
        fold_payload = manifest["fold_samples"][f"fold_{fold:02d}"]
        samples = tuple(SampleEpoch(**item) for item in fold_payload["samples"])
        if len(samples) != fold_payload["sample_count"]:
            raise ValueError(f"fold {fold:02d}: sample count mismatch")
        if any(
            sum(sample.label == label for sample in samples)
            != manifest["sample_per_class_per_fold"]
            for label in range(5)
        ):
            raise ValueError(f"fold {fold:02d}: sample is not class-balanced")
        folds[fold] = samples
    if sum(map(len, folds.values())) != manifest["total_sample_count"]:
        raise ValueError("total sample count mismatch")
    return manifest, folds


def _signals_for_samples(
    records: Sequence[Any], samples: Sequence[SampleEpoch], manipulation: str
) -> np.ndarray:
    by_key = {record.info.record_key: record for record in records}
    parts = []
    for sample in samples:
        record = by_key.get(sample.record_key)
        if record is None:
            raise ValueError(f"sample escaped fold records: {sample.record_key}")
        position = sample.position
        if (
            record.info.subject_id != sample.subject_id
            or int(record.original_epoch_index[position]) != sample.original_epoch_index
            or int(record.y[position]) != sample.label
        ):
            raise ValueError(f"sample/source mismatch: {sample}")
        if manipulation == "current":
            source = position
        elif manipulation == "previous":
            source = max(0, position - 1)
        elif manipulation == "next":
            source = min(len(record.y) - 1, position + 1)
        else:
            raise ValueError(manipulation)
        parts.append(record.x[source])
    matrix = np.stack(parts).astype(np.float32, copy=False)
    if matrix.shape != (len(samples), 3000):
        raise AssertionError("sampled signal matrix has wrong shape")
    return matrix


@torch.no_grad()
def _extract_cnn15(
    records: Sequence[Any],
    samples: Sequence[SampleEpoch],
    models: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    columns = []
    for manipulation in MANIPULATIONS:
        signals = torch.from_numpy(
            _signals_for_samples(records, samples, manipulation)
        ).unsqueeze(1)
        prefix = MANIPULATION_PREFIX[manipulation]
        for stage in STAGE_NAMES:
            model = models[f"{prefix}_{stage}"].to(device).eval()
            batches = []
            for start in range(0, len(signals), batch_size):
                logits = model(signals[start : start + batch_size].to(device))
                batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
            columns.append(np.concatenate(batches))
    result = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
    if result.shape != (len(samples), 75) or not np.isfinite(result).all():
        raise ValueError("invalid 15CNN feature matrix")
    return result


@torch.no_grad()
def _extract_resnet(
    records: Sequence[Any],
    samples: Sequence[SampleEpoch],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    signals = torch.from_numpy(
        _signals_for_samples(records, samples, "current")
    ).unsqueeze(1)
    model = model.to(device).eval()
    batches = []
    for start in range(0, len(signals), batch_size):
        batches.append(
            model.extract_features(signals[start : start + batch_size].to(device))
            .cpu()
            .numpy()
        )
    result = np.concatenate(batches).astype(np.float32, copy=False)
    if result.shape != (len(samples), 128) or not np.isfinite(result).all():
        raise ValueError("invalid ResNet feature matrix")
    return result


def _embedding_summary(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pca_dimensions: int,
    silhouette_sample: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if pca_dimensions <= 0 or pca_dimensions >= min(features.shape):
        raise ValueError("invalid PCA dimensions")
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(n_components=pca_dimensions, svd_solver="full")
    reduced = pca.fit_transform(standardized)
    sample_size = min(silhouette_sample, len(labels))
    silhouette = float(
        silhouette_score(
            reduced,
            labels,
            sample_size=sample_size,
            random_state=seed,
            metric="euclidean",
        )
    )
    perplexity = 30.0
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(reduced)
    return embedding, {
        "feature_dimension": int(features.shape[1]),
        "pca_dimensions": pca_dimensions,
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "silhouette_score_pca": silhouette,
        "silhouette_sample_size": sample_size,
        "tsne_perplexity": perplexity,
    }


def _silhouette_summary(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pca_dimensions: int,
    silhouette_sample: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if pca_dimensions <= 0 or pca_dimensions >= min(features.shape):
        raise ValueError("invalid PCA dimensions")
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(n_components=pca_dimensions, svd_solver="full")
    reduced = pca.fit_transform(standardized)
    sample_size = min(silhouette_sample, len(labels))
    silhouette = float(
        silhouette_score(
            reduced,
            labels,
            sample_size=sample_size,
            random_state=seed,
            metric="euclidean",
        )
    )
    return reduced, {
        "feature_dimension": int(features.shape[1]),
        "pca_dimensions": pca_dimensions,
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "silhouette_score_pca": silhouette,
        "silhouette_sample_size": sample_size,
    }


def analyze(
    workspace: Path,
    sample_manifest_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    pca_dimensions: int,
    silhouette_sample: int,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    commit = _clean_git_commit(workspace)
    if batch_size <= 0 or silhouette_sample <= 0:
        raise ValueError("batch/sample sizes must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sample_manifest, fold_samples = _load_sample_manifest(sample_manifest_path)
    if sample_manifest["config_sha256"] != sha256_file(
        workspace / "configs/experiments_v2.json"
    ):
        raise ValueError("configuration changed after sample preparation")
    if sample_manifest["split_sha256"] != sha256_file(
        workspace / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    ):
        raise ValueError("split changed after sample preparation")
    fold_results: dict[str, Any] = {}
    fold00_features: dict[str, np.ndarray] = {}
    fold00_labels: np.ndarray | None = None
    fold00_subjects: np.ndarray | None = None
    for fold in range(10):
        records = _fold_records(workspace, fold, 42)
        samples = fold_samples[fold]
        labels = np.asarray([sample.label for sample in samples], dtype=np.int8)
        subjects = np.asarray([sample.subject_id for sample in samples])
        experiment_results: dict[str, Any] = {}
        for experiment in ("E1", "E2"):
            context = build_context(
                workspace,
                experiment,
                fold,
                42,
                str(device),
                smoke=False,
                allow_test_evaluation=True,
                num_workers=0,
                resume=True,
            )
            validation = validate_run(workspace, context.run_root)
            if not validation.get("passed") or "test" not in validation.get(
                "roles", {}
            ):
                raise ValueError(
                    f"{experiment}/fold_{fold:02d}: Gate-4 validation failed"
                )
            kind, extractor, extractor_hash = _load_extractor(context)
            if experiment == "E1" and kind == "cnn15":
                matrix = _extract_cnn15(
                    records, samples, extractor, device, batch_size
                )
            elif experiment == "E2" and kind == "resnet1d":
                matrix = _extract_resnet(
                    records, samples, extractor, device, batch_size
                )
            else:
                raise ValueError(f"unexpected extractor for {experiment}: {kind}")
            _, summary = _silhouette_summary(
                matrix,
                labels,
                pca_dimensions=pca_dimensions,
                silhouette_sample=silhouette_sample,
                seed=42 + fold,
            )
            summary.update(
                {
                    "extractor_kind": kind,
                    "extractor_sha256": extractor_hash,
                    "run_manifest_sha256": validation["manifest_sha256"],
                }
            )
            experiment_results[experiment] = summary
            if fold == 0:
                fold00_features[experiment] = matrix
        fold_results[f"fold_{fold:02d}"] = {
            "sample_count": len(samples),
            "subjects_represented": int(len(np.unique(subjects))),
            "representations": experiment_results,
            "silhouette_difference_E2_minus_E1": (
                experiment_results["E2"]["silhouette_score_pca"]
                - experiment_results["E1"]["silhouette_score_pca"]
            ),
        }
        if fold == 0:
            fold00_labels = labels
            fold00_subjects = subjects
    if fold00_labels is None or fold00_subjects is None:
        raise AssertionError("fold 00 was not analyzed")
    embeddings: dict[str, np.ndarray] = {}
    for experiment in ("E1", "E2"):
        reduced, _ = _silhouette_summary(
            fold00_features[experiment],
            fold00_labels,
            pca_dimensions=pca_dimensions,
            silhouette_sample=silhouette_sample,
            seed=42,
        )
        embeddings[experiment] = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=30.0,
            random_state=42,
        ).fit_transform(reduced)
    differences = np.asarray(
        [fold_results[f"fold_{fold:02d}"]["silhouette_difference_E2_minus_E1"] for fold in range(10)],
        dtype=np.float64,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tsne_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["representation", "x", "y", "stage", "stage_id", "subject_id"]
        )
        for experiment in ("E1", "E2"):
            for point, label, subject in zip(
                embeddings[experiment], fold00_labels, fold00_subjects, strict=True
            ):
                writer.writerow(
                    [
                        experiment,
                        float(point[0]),
                        float(point[1]),
                        STAGES[int(label)],
                        int(label),
                        subject,
                    ]
                )
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    titles = {"E1": "E1: 15CNN softmax (75D)", "E2": "E2: ResNet-1D (128D)"}
    for axis, experiment in zip(axes, ("E1", "E2"), strict=True):
        for label, stage in enumerate(STAGES):
            selected = fold00_labels == label
            axis.scatter(
                embeddings[experiment][selected, 0],
                embeddings[experiment][selected, 1],
                s=7,
                alpha=0.5,
                label=stage,
            )
        axis.set_title(titles[experiment])
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
    axes[1].legend(markerscale=2)
    figure.suptitle("Cùng epoch test fold 00, lấy mẫu cân bằng lớp")
    figure.tight_layout()
    figure.savefig(output_dir / "tsne_E1_vs_E2.png", dpi=200)
    plt.close(figure)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "supportive_post_Gate5_feature_representation_analysis",
        "git_commit": commit,
        "sample_manifest_sha256": _lf_sha256(sample_manifest_path),
        "total_sample_count": sample_manifest["total_sample_count"],
        "sample_per_class_per_fold": sample_manifest["sample_per_class_per_fold"],
        "subjects_represented_across_folds": sample_manifest[
            "subjects_represented_across_folds"
        ],
        "folds": list(range(10)),
        "role": "test",
        "seed": 42,
        "data_variant": "paper_raw_v1",
        "fold_results": fold_results,
        "silhouette_difference_E2_minus_E1_summary": {
            "mean": float(differences.mean()),
            "median": float(np.median(differences)),
            "minimum": float(differences.min()),
            "maximum": float(differences.max()),
            "folds_E2_greater_than_E1": int(np.sum(differences > 0)),
            "folds_equal": int(np.sum(differences == 0)),
            "folds_E2_less_than_E1": int(np.sum(differences < 0)),
        },
        "tsne_scope": "fold_00_only_descriptive",
        "warnings": [
            "This analysis was executed after Gate 5 and is supportive, not confirmatory.",
            "t-SNE is descriptive and must not be interpreted causally.",
            "Silhouette depends on scaling, PCA and the locked sampling rule.",
        ],
    }
    (output_dir / "feature_space_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "analyze"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-class-per-fold", type=int, default=200)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pca-dimensions", type=int, default=20)
    parser.add_argument("--silhouette-sample", type=int, default=5000)
    args = parser.parse_args()
    if args.mode == "prepare":
        report = prepare_sample_manifest(
            args.workspace,
            seed=args.seed,
            per_class_per_fold=args.sample_per_class_per_fold,
        )
        args.sample_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.sample_manifest.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required in analyze mode")
        report = analyze(
            args.workspace,
            args.sample_manifest,
            args.output_dir,
            device_name=args.device,
            batch_size=args.batch_size,
            pca_dimensions=args.pca_dimensions,
            silhouette_sample=args.silhouette_sample,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
