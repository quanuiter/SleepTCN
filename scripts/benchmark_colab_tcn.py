"""Source-only CUDA smoke test. No campaign checkpoint or target-data access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    import numpy as np
    import torch
    from sleeptcn.models import SleepTCN
    from sleeptcn.training import collate_feature_sequences, masked_cross_entropy

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "backend": args.backend, "torch": torch.__version__,
              "python": platform.python_version(), "cuda": torch.version.cuda,
              "cpu_threads": 4, "float32_only": True, "tf32": False,
              "deterministic_algorithms": True, "target_data_access": False,
              "campaign_checkpoint_access": False,
              "script_sha256": sha256(Path(__file__)),
              "models_sha256": sha256(ROOT / "src/sleeptcn/models.py"),
              "training_sha256": sha256(ROOT / "src/sleeptcn/training.py"),
              "sample_sha256": sha256(Path(args.sample)), "cases": []}

    def save():
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    save()
    try:
        if args.backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; do not silently fall back to CPU")
        device = torch.device(args.backend)
        if args.backend == "cuda":
            result.update(gpu=torch.cuda.get_device_name(0),
                          gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        with np.load(args.sample, allow_pickle=False) as z:
            records = [(torch.from_numpy(z[f"features_{i:02}"].copy()),
                        torch.from_numpy(z[f"labels_{i:02}"].astype(np.int64))) for i in range(8)]
            weights = torch.from_numpy(z["train_class_weights"].copy())
        for name, selected in [("short_diagnostic", [(x[:256], y[:256]) for x, y in records[:2]]),
                               ("actual_eight_record_batch", records)]:
            batch = collate_feature_sequences(selected)
            case = {"name": name, "shape": list(batch.features.shape),
                    "valid_labels": int(batch.valid_target_mask.sum()), "status": "checking"}
            result["cases"].append(case)
            print(f"START {name} {case['shape']} {device}", flush=True)
            save()
            torch.manual_seed(731)
            reference = SleepTCN(input_dim=128).eval()
            initial = {k: v.detach().clone() for k, v in reference.state_dict().items()}
            candidate = SleepTCN(input_dim=128).to(device).eval()
            candidate.load_state_dict(initial)
            x, y, mask, w = batch.features.to(device), batch.targets.to(device), batch.padding_mask.to(device), weights.to(device)
            expected = reference(batch.features, padding_mask=batch.padding_mask)
            expected_loss = masked_cross_entropy(expected, batch.targets, weights)
            expected_loss.backward()
            actual = candidate(x, padding_mask=mask)
            loss = masked_cross_entropy(actual, y, w)
            loss.backward()
            actual_cpu = actual.detach().cpu()
            torch.testing.assert_close(actual_cpu, expected.detach(), rtol=1e-3, atol=1e-4)
            torch.testing.assert_close(loss.detach().cpu(), expected_loss.detach(), rtol=1e-3, atol=1e-4)
            gradients = []
            for (key, param), (_, other) in zip(reference.named_parameters(), candidate.named_parameters()):
                gradient = other.grad.detach().cpu()
                gradients.append({"parameter": key,
                    "max_abs_residual": float((gradient - param.grad).abs().max()),
                    "passed": bool(torch.isfinite(gradient).all() and torch.allclose(gradient, param.grad, rtol=1e-2, atol=1e-4))})
            case["numerical_check"] = {"forward_max_abs_residual": float((actual_cpu - expected.detach()).abs().max()),
                "loss_cpu": float(expected_loss.detach()), "loss_candidate": float(loss.detach().cpu()),
                "gradient_max_abs_residual": max(row["max_abs_residual"] for row in gradients),
                "gradients": gradients, "dropout_disabled_for_comparison_only": True}
            save()
            if not all(row["passed"] for row in gradients):
                raise ValueError("Gradient comparison failed; no timing/training campaign permitted")
            optimizers = [torch.optim.Adam(model.parameters(), lr=0.0005) for model in [reference, candidate]]
            for model, optimizer in zip([reference, candidate], optimizers):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            parameter_residual = 0.
            for param, other in zip(reference.parameters(), candidate.parameters()):
                value = other.detach().cpu()
                torch.testing.assert_close(value, param.detach(), rtol=1e-3, atol=1e-3)
                parameter_residual = max(parameter_residual, float((value - param.detach()).abs().max()))
            case["numerical_check"].update(passed=True, first_adam_step_max_parameter_residual=parameter_residual)
            candidate.load_state_dict(initial)
            candidate.train()
            optimizer = torch.optim.Adam(candidate.parameters(), lr=0.0005)
            if args.backend == "cuda":
                torch.cuda.reset_peak_memory_stats()

            def step(transfer):
                bx, by, bm = (batch.features.to(device), batch.targets.to(device), batch.padding_mask.to(device)) if transfer else (x, y, mask)
                optimizer.zero_grad(set_to_none=True)
                output_logits = candidate(bx, padding_mask=bm)
                objective = masked_cross_entropy(output_logits, by, w)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
                optimizer.step()
                value = float(objective.detach().cpu())
                if args.backend == "cuda":
                    torch.cuda.synchronize()
                if not np.isfinite(value):
                    raise ValueError("Nonfinite loss")

            for transfer in [False, True]:
                for _ in range(3):
                    step(transfer)
                timings = []
                for _ in range(15):
                    tick = time.perf_counter()
                    step(transfer)
                    timings.append(time.perf_counter() - tick)
                case["with_transfer" if transfer else "resident"] = {
                    "seconds": timings, "median_seconds": statistics.median(timings),
                    "warmups": 3, "repetitions": 15, "normal_training_dropout": 0.2,
                    "includes_forward_backward_clip_adam_and_synchronization": True}
                save()
            if args.backend == "cuda":
                case["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            case["status"] = "passed"
            save()
            print(f"DONE {name} {case['with_transfer']['median_seconds']:.5f} seconds/step", flush=True)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    save()
    print("FINAL_STATUS", result["status"], flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bounded-child", action="store_true")
    args = parser.parse_args()
    if args.bounded_child:
        try:
            completed = subprocess.run([sys.executable, __file__, "--backend", args.backend,
                "--sample", args.sample, "--output", args.output], timeout=300)
            sys.exit(completed.returncode)
        except subprocess.TimeoutExpired:
            path = Path(args.output)
            result = json.loads(path.read_bytes()) if path.exists() else {}
            result.update(status="stopped_300_second_budget", backend=args.backend)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            sys.exit(2)
    sys.exit(run(args))
