"""Pinned ADAST/source-only matched-budget pilot; private artifacts stay in runs.

Uses upstream model classes and similarity penalty unchanged. The orchestration
adapts data, logging, persistence and a source-only control to the locked split.
No target label is accessed until both final models have predicted all records.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.cpu_followups import confusion_diagnostics, paired_diagnostic_bootstrap
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.preprocessing import sha256_file

PROTOCOL = ROOT / "configs/teacher_revision_adast_fold0_pilot_v1.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def model_digest(models):
    h = hashlib.sha256()
    for name, model in models.items():
        h.update(name.encode())
        for key, value in model.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_models(module, config, seed):
    torch.manual_seed(seed)
    # Match the upstream order, including the otherwise unused discriminator
    # and target attention in the source-only initialization fingerprint.
    return {"encoder": module.cnn_feature_extractor(config.base_model),
            "head1": module.Classifier(config.base_model),
            "head2": module.Classifier(config.base_model),
            "discriminator": module.Discriminator(config.base_model),
            "source_attention": module.Self_Attn(config.base_model.final_out_channels),
            "target_attention": module.Self_Attn(config.base_model.final_out_channels)}


def features_and_logits(models, x, domain):
    f = models[domain + "_attention"](models["encoder"](x))
    return f, models["head1"](f), models["head2"](f)


@torch.inference_mode()
def predict(models, x, domain, pseudo=False, batch_size=128):
    for model in models.values():
        model.eval()
    values = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.asarray(x[start:start+batch_size], dtype=np.float32).copy()).unsqueeze(1)
        _, first, second = features_and_logits(models, batch, domain)
        logits = (first + second) / 2 if pseudo else torch.maximum(first, second)
        values.append(logits.argmax(1).numpy() if pseudo else logits.numpy())
    return np.concatenate(values)


def update(models, optimizer, disc_optimizer, utils, sx, sy, tx, pseudo, round_index, cfg, arm):
    for model in models.values():
        model.train()
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    sf, sl1, sl2 = features_and_logits(models, sx, "source")
    source_loss = ce(sl1, sy) + ce(sl2, sy)
    loss = (cfg["source_loss_weights_by_round"][round_index] * source_loss
            + cfg["similarity_weight"] * utils.calc_similiar_penalty(models["head1"], models["head2"]))
    if arm == "adast":
        tf, tl1, tl2 = features_and_logits(models, tx, "target")
        discriminator = models["discriminator"]
        for param in discriminator.parameters():
            param.requires_grad = True
        disc_output = discriminator(torch.cat([sf, tf]).detach()).squeeze(-1)
        labels = torch.cat([torch.ones(len(sf)), torch.zeros(len(tf))])
        disc_loss = bce(disc_output, labels)
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        for param in discriminator.parameters():
            param.requires_grad = False
        fake_output = torch.cat([discriminator(tf).squeeze(-1), discriminator(sf).squeeze(-1)])
        fake_labels = torch.cat([torch.ones(len(tf)), torch.zeros(len(sf))])
        loss = (loss + cfg["adversarial_weight"] * bce(fake_output, fake_labels)
                + cfg["target_loss_weights_by_round"][round_index] * (ce(tl1, pseudo) + ce(tl2, pseudo)))
    optimizer.zero_grad()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                      for model in models.values() for p in model.parameters()):
        raise ValueError("nonfinite loss or gradient")
    optimizer.step()
    return float(loss.detach())


def prepare_inputs(args, output, split):
    manifest = output / "private_inputs.json"
    if manifest.exists():
        saved = json.loads(manifest.read_bytes())
        for item in saved["cached_arrays"].values():
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("cached input changed")
        return saved
    train_keys = sorted(split["train"]["record_keys"])
    source_entries, count = [], 0
    for key in train_keys:
        path = ROOT / "data/processed/filtered_v2" / (key + ".npz")
        with np.load(path, allow_pickle=False) as z:
            n = int((z["y"] >= 0).sum())
        source_entries.append({"record_key": key, "path": str(path), "sha256": sha256_file(path), "epochs": n})
        count += n
    if count != 157200:
        raise ValueError("source training support differs")
    x_path, y_path = output / "source_train_x.npy", output / "source_train_y.npy"
    if x_path.exists() or y_path.exists():
        raise FileExistsError("unverified partial input cache; inspect before recovery")
    x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(count, 3000))
    y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64, shape=(count,))
    offset = 0
    for item in source_entries:
        with np.load(item["path"], allow_pickle=False) as z:
            valid = z["y"] >= 0
            n = item["epochs"]
            x[offset:offset+n], y[offset:offset+n] = z["x"][valid], z["y"][valid]
            offset += n
    x.flush()
    y.flush()
    del x, y
    adaptation_manifest = args.adaptation_cache / "private_manifest.json"
    adaptation = json.loads(adaptation_manifest.read_bytes())
    target_parts, target_entries = [], []
    if len(adaptation["records"]) != 5 or len({e["subject_id"] for e in adaptation["records"]}) != 5:
        raise ValueError("expected five locked adaptation subjects")
    for record in sorted(adaptation["records"], key=lambda e: e["subject_id"]):
        entry = record["variants"]["filtered_v2"]
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("adaptation input changed")
        with np.load(entry["path"], allow_pickle=False) as z:
            if set(z.files) != {"x", "original_epoch_index", "metadata_json"}:
                raise ValueError("target input contains unexpected fields or labels")
            if not np.array_equal(z["original_epoch_index"], np.arange(len(z["x"]))):
                raise ValueError("target input must cover full record")
            target_parts.append(z["x"])
        target_entries.append({"subject_id": record["subject_id"], **entry})
    target = np.concatenate(target_parts)
    if target.shape != (4989, 3000):
        raise ValueError("adaptation support differs")
    target_path = output / "adaptation_x.npy"
    with target_path.open("xb") as stream:
        np.save(stream, target)
    result = {"source_records": source_entries, "adaptation_records": target_entries,
              "adaptation_manifest_sha256": sha256_file(adaptation_manifest),
              "cached_arrays": {key: {"path": str(path), "sha256": sha256_file(path)}
                                for key, path in [("source_x", x_path), ("source_y", y_path), ("target_x", target_path)]}}
    write_json(manifest, result)
    return result


def train_arm(arm, models, utils, x, y, target, cfg, folder):
    folder.mkdir(exist_ok=True)
    initial = model_digest(models)
    selection_path = folder / "selection.json"
    final_path = folder / "final.pt"
    if selection_path.exists():
        selection = json.loads(selection_path.read_bytes())
        if sha256_file(final_path) != selection["checkpoint_sha256"] or initial != selection["initial_state_sha256"]:
            raise ValueError("existing model changed")
        payload = torch.load(final_path, map_location="cpu", weights_only=True)
        for name, model in models.items():
            model.load_state_dict(payload["models"][name])
        return selection
    train_names = ["encoder", "head1", "head2", "source_attention", "target_attention"]
    opt = cfg["optimizer"]
    optimizer = torch.optim.Adam([p for name in train_names for p in models[name].parameters()],
                                 lr=opt["lr"], betas=tuple(opt["betas"]), weight_decay=opt["weight_decay"])
    disc_optimizer = torch.optim.Adam(models["discriminator"].parameters(), lr=opt["lr"],
                                      betas=tuple(opt["betas"]), weight_decay=opt["weight_decay"])
    done, elapsed, history = 0, 0., []
    latest = folder / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        for name, model in models.items():
            model.load_state_dict(checkpoint["models"][name])
        optimizer.load_state_dict(checkpoint["optimizer"])
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        torch.set_rng_state(checkpoint["rng_state"])
        done, elapsed, history = checkpoint["epochs_completed"], checkpoint["elapsed_seconds"], checkpoint["history"]
    for round_index in range(cfg["rounds"]):
        pseudo_path = folder / f"pseudo_round_{round_index}.npy"
        if arm == "adast" and done < (round_index + 1) * cfg["epochs_per_round"]:
            if pseudo_path.exists():
                pseudo = np.load(pseudo_path, allow_pickle=False)
            else:
                if done != round_index * cfg["epochs_per_round"]:
                    raise ValueError("missing frozen round pseudo-labels")
                pseudo = predict(models, target, "target", pseudo=True)
                with pseudo_path.open("xb") as stream:
                    np.save(stream, pseudo)
        for epoch in range(cfg["epochs_per_round"]):
            global_epoch = round_index * cfg["epochs_per_round"] + epoch
            if global_epoch < done:
                continue
            tick = time.perf_counter()
            # Dedicated per-epoch generators guarantee identical source examples
            # across arms and recover order independently of dropout RNG.
            source_order = np.random.default_rng(cfg["seed"] + global_epoch).permutation(len(x))[:cfg["steps_per_epoch"]*cfg["batch_size"]]
            target_order = np.random.default_rng(cfg["seed"] + 10000 + global_epoch).permutation(len(target))[:len(source_order)]
            losses = []
            for start in range(0, len(source_order), cfg["batch_size"]):
                si, ti = source_order[start:start+cfg["batch_size"]], target_order[start:start+cfg["batch_size"]]
                sx, sy = torch.from_numpy(x[si].copy()).unsqueeze(1), torch.from_numpy(y[si].copy())
                tx = torch.from_numpy(target[ti].copy()).unsqueeze(1) if arm == "adast" else None
                py = torch.from_numpy(pseudo[ti].astype(np.int64)) if arm == "adast" else None
                losses.append(update(models, optimizer, disc_optimizer, utils, sx, sy, tx, py, round_index, cfg, arm))
            # Exact learning-rate effect of upstream StepLR(10,.1), stepped
            # only after round-0 epochs. This explicit counter is restart-safe.
            if round_index == 0 and (epoch + 1) % 10 == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= .1
            seconds = time.perf_counter() - tick
            elapsed += seconds
            row = {"round": round_index, "epoch": epoch+1, "updates": len(losses),
                   "source_order_sha256": hashlib.sha256(source_order.tobytes()).hexdigest(),
                   "loss_mean": float(np.mean(losses)), "seconds": seconds,
                   "encoder_lr_after_epoch": optimizer.param_groups[0]["lr"]}
            history.append(row)
            checkpoint = {"models": {name: model.state_dict() for name, model in models.items()},
                          "optimizer": optimizer.state_dict(), "disc_optimizer": disc_optimizer.state_dict(),
                          "rng_state": torch.get_rng_state(), "epochs_completed": global_epoch+1,
                          "elapsed_seconds": elapsed, "history": history, "initial_state_sha256": initial,
                          "protocol_sha256": sha256_file(PROTOCOL)}
            torch.save(checkpoint, folder / "latest.tmp")
            (folder / "latest.tmp").replace(latest)
            done = global_epoch + 1
            print(f"{arm} round {round_index} epoch {epoch+1}/{cfg['epochs_per_round']}: loss={row['loss_mean']:.4f}, {seconds:.1f}s", flush=True)
    if not final_path.exists():
        shutil.copyfile(latest, final_path)
    selection = {"initial_state_sha256": initial, "checkpoint_sha256": sha256_file(final_path),
                 "selection": cfg["selection"], "updates": sum(e["updates"] for e in history),
                 "training_seconds": elapsed, "history": history}
    write_json(selection_path, selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--adaptation-cache", type=Path, required=True)
    parser.add_argument("--verified-pilot", type=Path, required=True)
    parser.add_argument("--shhs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("outputs must be inside ignored runs")
    if (output / "aggregate_results.json").exists():
        raise FileExistsError("completed pilot must not be overwritten")
    cfg = json.loads(PROTOCOL.read_bytes())
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != cfg["upstream_commit"] or subprocess.check_output(
            ["git", "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("upstream revision modified")
    module = load_module("adast_models", args.upstream / "models/models.py")
    upstream_cfg = load_module("adast_config", args.upstream / "config_files/configs.py").Config()
    utils = load_module("adast_utils", args.upstream / "utils.py")
    helper = load_module("adast_signal_helpers", ROOT / "scripts/run_recovered_e3_cpu_pilot.py")
    split_path = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    split = json.loads(split_path.read_bytes())["outer_runs"][0]
    source_roles = [{str(s) for s in split[role]["subject_ids"]} for role in ["train", "validation", "test"]]
    if any(source_roles[i] & source_roles[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("source subject overlap")
    recovered_result = json.loads((args.verified_pilot / "aggregate_results.json").read_bytes())
    target_manifest = args.verified_pilot / "private_inference_manifest.json"
    if sha256_file(target_manifest) != recovered_result["inference_manifest_sha256"]:
        raise ValueError("locked inference manifest changed")
    original = json.loads(target_manifest.read_bytes())["records"]
    if len(original) != 180 or len({e["subject_id"] for e in original}) != 180:
        raise ValueError("expected 180 target test subjects")
    output.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), PROTOCOL, ROOT / "scripts/run_recovered_e3_cpu_pilot.py",
                  ROOT / "src/sleeptcn/cpu_followups.py", ROOT / "src/sleeptcn/metrics.py"]
    provenance = {"protocol": cfg, "protocol_sha256": sha256_file(PROTOCOL),
                  "source_split_sha256": sha256_file(split_path),
                  "target_manifest_sha256": sha256_file(target_manifest),
                  "adaptation_manifest_sha256": sha256_file(args.adaptation_cache / "private_manifest.json"),
                  "code_sha256": {str(p.resolve().relative_to(ROOT)): sha256_file(p) for p in code_paths},
                  "upstream_files_sha256": {str(p.relative_to(args.upstream)): sha256_file(p) for p in
                       [args.upstream / "models/models.py", args.upstream / "utils.py", args.upstream / "config_files/configs.py"]},
                  "torch_version": str(torch.__version__), "cpu_threads": 4}
    spec_path = output / "execution_specification.json"
    if spec_path.exists():
        if json.loads(spec_path.read_bytes()) != provenance:
            raise ValueError("execution specification changed during resume")
    else:
        write_json(spec_path, provenance)
        for path in code_paths:
            dest = output / "code_snapshot" / path.resolve().relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    inputs = prepare_inputs(args, output, split)
    if {e["subject_id"] for e in inputs["adaptation_records"]} & {e["subject_id"] for e in original}:
        raise ValueError("adaptation/test overlap")
    x, y, target = [np.load(inputs["cached_arrays"][key]["path"], mmap_mode="r", allow_pickle=False)
                    for key in ["source_x", "source_y", "target_x"]]
    torch.set_num_threads(4)
    models, selections = {}, {}
    for arm in cfg["arms"]:
        models[arm] = build_models(module, upstream_cfg, cfg["seed"])
        selections[arm] = train_arm(arm, models[arm], utils, x, y, target, cfg, output / arm)
    if selections["source_only"]["initial_state_sha256"] != selections["adast"]["initial_state_sha256"]:
        raise ValueError("initial model states differ")
    if ([r["source_order_sha256"] for r in selections["source_only"]["history"]]
            != [r["source_order_sha256"] for r in selections["adast"]["history"]]):
        raise ValueError("source sample exposure differs")
    frozen = output / "both_arms_selected.json"
    if not frozen.exists():
        write_json(frozen, selections)
    elif json.loads(frozen.read_bytes()) != selections:
        raise ValueError("frozen selection changed")
    selected_hash = sha256_file(frozen)
    pred_folder = output / "target_predictions"
    pred_folder.mkdir(exist_ok=True)
    targets = []
    for i, entry in enumerate(original, 1):
        path = pred_folder / (entry["record_key"] + ".npz")
        if not path.exists():
            edf = args.shhs_root / "shhs/polysomnography/edfs/shhs1" / (entry["record_key"] + ".edf")
            signal, _ = helper.read_full_record(edf, entry["source_edf_sha256"])
            logits = {arm: predict(model, signal, "source" if arm == "source_only" else "target")
                      for arm, model in models.items()}
            with path.open("xb") as stream:
                np.savez_compressed(stream, **logits, original_epoch_index=np.arange(len(signal)),
                    metadata_json=np.array(json.dumps({"subject_id": entry["subject_id"],
                        "source_edf_sha256": entry["source_edf_sha256"], "selected_sha256": selected_hash})))
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"]))
            if (meta["selected_sha256"] != selected_hash or meta["subject_id"] != entry["subject_id"]
                    or not np.array_equal(z["original_epoch_index"], np.arange(entry["epochs"]))
                    or any(z[arm].shape != (entry["epochs"], 5) or not np.isfinite(z[arm]).all() for arm in models)):
                raise ValueError("target prediction identity/alignment mismatch")
        targets.append({**entry, "path": str(path), "sha256": sha256_file(path)})
        if i % 20 == 0:
            print(f"ADAST pair full-record inference {i}/180", flush=True)
    manifest_path = output / "private_inference_manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {"status": "complete", "records": targets})
    # No target reference labels above this point.
    confusions = {arm: [] for arm in models}
    for entry in targets:
        reference = args.shhs_root / "processed_v1/filtered_v2" / (entry["record_key"] + ".npz")
        with np.load(reference, allow_pickle=False) as z, np.load(entry["path"], allow_pickle=False) as pred:
            if (str(z["subject_id"]) != entry["subject_id"] or str(z["role"]) != "test"
                    or str(z["source_edf_sha256"]) != entry["source_edf_sha256"]):
                raise ValueError("target reference identity mismatch")
            positions = helper.benchmark_positions(z["original_epoch_index"], entry["epochs"])
            for arm in models:
                confusions[arm].append(confusion_matrix_5(z["y"], pred[arm][positions].argmax(1)))
    cm = {arm: np.stack(values) for arm, values in confusions.items()}
    if any(m.sum() != 169012 for m in cm.values()):
        raise ValueError("benchmark support mismatch")
    contrast = paired_diagnostic_bootstrap(cm["adast"], cm["source_only"])
    differences = confusion_diagnostics(cm["adast"])["macro_f1"] - confusion_diagnostics(cm["source_only"])["macro_f1"]
    draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(0, 180, (cfg["bootstrap_resamples"], 180), dtype=np.int32)
    contrast["subject_mean_macro_f1"] = {"difference": float(differences.mean()),
        "ci95": np.quantile(differences[draws].mean(1), [.025, .975]).tolist()}
    results = {"status": "complete_exploratory_single_fold_harmonized_ADAST_pilot", "provenance": provenance,
               "subjects": 180, "valid_epochs": 169012, "selections": selections,
               "input_manifest_sha256": sha256_file(output / "private_inputs.json"),
               "inference_manifest_sha256": sha256_file(manifest_path),
               "arms": {arm: {"subject_mean_macro_f1": float(confusion_diagnostics(matrix)["macro_f1"].mean()),
                               "pooled": metrics_from_confusion(matrix.sum(0))} for arm, matrix in cm.items()},
               "adast_minus_source_only": contrast}
    with (output / "private_subject_confusions.npz").open("xb") as stream:
        np.savez_compressed(stream, **cm)
    write_json(output / "aggregate_results.json", results)
    print(json.dumps({"status": results["status"], "arms": results["arms"],
                      "subject_mean_contrast": contrast["subject_mean_macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
