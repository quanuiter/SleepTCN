"""Create a provenance-preserving summary for one completed validation fold."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeptcn.io.serialization import read_json  # noqa: E402


EXPERIMENTS = tuple(f"E{index}" for index in range(7))


def checkpoint_summary(checkpoint_root: Path) -> dict[str, Any]:
    files = sorted(checkpoint_root.rglob("*.pt")) if checkpoint_root.exists() else []
    complete = sorted(checkpoint_root.rglob("complete.json")) if checkpoint_root.exists() else []
    lfs_pointers = 0
    lfs_declared_bytes = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.startswith("version https://git-lfs.github.com/spec/v1\n"):
            lfs_pointers += 1
            for line in text.splitlines():
                if line.startswith("size "):
                    lfs_declared_bytes += int(line.removeprefix("size "))
                    break
    return {
        "checkpoint_files": len(files),
        "completion_markers": len(complete),
        "git_lfs_pointer_files": lfs_pointers,
        "git_lfs_declared_bytes": lfs_declared_bytes,
    }


def monitoring_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    maximum_memory_mib: int | None = None
    samples = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0] == "timestamp":
                continue
            samples += 1
            value = row[4].replace(" MiB", "").strip()
            maximum_memory_mib = max(maximum_memory_mib or 0, int(value))
    return {"samples": samples, "peak_memory_mib_observed": maximum_memory_mib}


def time_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if "elapsed_seconds" in values:
        values["elapsed_seconds"] = int(values["elapsed_seconds"])
    return values


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Báo cáo validation {report['fold_label']} — seed {report['seed']}",
        "",
        "## Phạm vi",
        "",
        "Đây là kết quả validation của một fold đầy đủ; test vẫn bị khóa và không được dùng để kết luận khoa học.",
        "",
        "## Tính toàn vẹn",
        "",
        f"- Tất cả artifact validation đạt: **{report['all_artifacts_passed']}**.",
        f"- Test đã mở: **{report['any_test_evaluation']}** (phải là `False`).",
        f"- Config hash đồng nhất: **{report['provenance']['config_hashes_consistent']}**.",
        f"- Split hash đồng nhất: **{report['provenance']['split_hashes_consistent']}**.",
        f"- Runner-code hash đồng nhất: **{report['provenance']['runner_hashes_consistent']}**.",
        "",
        "## Chỉ số validation",
        "",
        "| E | Dữ liệu | Macro-F1 | Δ so với E0 | Accuracy | Kappa | Epoch hợp lệ | Artifact |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["experiments"]:
        metric = item["metrics"]
        delta = item["macro_f1_delta_vs_e0"]
        delta_text = "—" if delta is None else f"{delta:+.4f}"
        lines.append(
            "| {id} | {variant} | {f1:.4f} | {delta} | {accuracy:.4f} | {kappa:.4f} | {epochs} | {passed} |".format(
                id=item["experiment_id"],
                variant=item["data_variant"],
                f1=metric["macro_f1"],
                delta=delta_text,
                accuracy=metric["accuracy"],
                kappa=metric["cohen_kappa"],
                epochs=metric["n_valid_epochs"],
                passed="PASS" if item["artifact_passed"] else "FAIL",
            )
        )
    lines.extend(["", "## Checkpoint và tài nguyên", ""])
    for item in report["experiments"]:
        checkpoint = item["checkpoints"]
        lines.append(
            f"- {item['experiment_id']}: {checkpoint['checkpoint_files']} checkpoint `.pt`, "
            f"{checkpoint['completion_markers']} marker hoàn tất, "
            f"{checkpoint['git_lfs_pointer_files']} LFS pointer."
        )
    timing = report.get("timing")
    if timing:
        lines.extend(["", "### E0 monitoring", ""])
        if timing.get("elapsed_seconds") is not None:
            lines.append(f"- Thời gian: {timing['elapsed_seconds']} giây.")
        monitor = timing.get("gpu_monitoring")
        if monitor and monitor.get("peak_memory_mib_observed") is not None:
            lines.append(
                f"- Peak VRAM quan sát được: {monitor['peak_memory_mib_observed']} MiB "
                f"trên {monitor['samples']} mẫu."
            )
    lines.extend(
        [
            "",
            "## Diễn giải giới hạn",
            "",
            "Không dùng bảng này để công bố hiệu quả hay kiểm định thống kê: nó chỉ là validation của một fold. "
            "Cần khóa dependency/code, huấn luyện đủ 10 fold trước, rồi mới mở test một lần và chạy phân tích paired.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.fold not in range(10) or args.seed < 0:
        raise ValueError("invalid fold or seed")

    root = args.workspace.resolve()
    items = []
    for experiment in EXPERIMENTS:
        run_root = root / "runs/v2/full" / experiment / f"fold_{args.fold:02d}" / f"seed_{args.seed}"
        manifest = read_json(run_root / "run_manifest.json")
        validation = read_json(run_root / "validation_report.json")
        metric_payload = read_json(run_root / "metrics/validation.json")
        if manifest["status"] != "complete" or manifest["smoke"]:
            raise ValueError(f"not a completed full run: {run_root}")
        if metric_payload["metadata"]["role"] != "validation":
            raise ValueError(f"unexpected metric role: {run_root}")
        items.append(
            {
                "experiment_id": experiment,
                "data_variant": manifest["data_variant"],
                "run_root": str(run_root.relative_to(root)),
                "artifact_passed": bool(validation["passed"]),
                "test_evaluation_enabled": bool(manifest["allow_test_evaluation"]),
                "metrics": metric_payload["metrics"],
                "provenance": {
                    key: manifest[key]
                    for key in ("git_commit", "config_sha256", "split_sha256", "runner_code_sha256")
                },
                "checkpoints": checkpoint_summary(run_root / "checkpoints"),
            }
        )
    baseline = items[0]["metrics"]["macro_f1"]
    for item in items:
        item["macro_f1_delta_vs_e0"] = (
            None if item["experiment_id"] == "E0" else item["metrics"]["macro_f1"] - baseline
        )
    provenance = {
        "config_hashes_consistent": len({item["provenance"]["config_sha256"] for item in items}) == 1,
        "split_hashes_consistent": len({item["provenance"]["split_sha256"] for item in items}) == 1,
        "runner_hashes_consistent": len({item["provenance"]["runner_code_sha256"] for item in items}) == 1,
        "git_commits": {item["experiment_id"]: item["provenance"]["git_commit"] for item in items},
    }
    timing_path = root / "runs/v2/monitoring" / f"E0_fold{args.fold:02d}_seed{args.seed}_time.txt"
    monitor_path = root / "runs/v2/monitoring" / f"E0_fold{args.fold:02d}_seed{args.seed}_gpu.csv"
    timing = time_summary(timing_path)
    if timing is not None:
        timing["gpu_monitoring"] = monitoring_summary(monitor_path)
    report = {
        "schema_version": 1,
        "scope": "completed_full_validation_fold_only_not_test_or_statistical_analysis",
        "fold": args.fold,
        "fold_label": f"fold {args.fold:02d}",
        "seed": args.seed,
        "all_artifacts_passed": all(item["artifact_passed"] for item in items),
        "any_test_evaluation": any(item["test_evaluation_enabled"] for item in items),
        "provenance": provenance,
        "experiments": items,
        "timing": timing,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_markdown": str(args.output_markdown), "all_artifacts_passed": report["all_artifacts_passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
