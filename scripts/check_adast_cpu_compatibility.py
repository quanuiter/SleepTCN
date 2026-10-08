"""CPU model/loss compatibility check for a pinned upstream ADAST revision.

This exercises original model classes and two locally orchestrated update steps
on real source/adaptation inputs. It is not a training campaign or accuracy result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.preprocessing import sha256_file

EXPECTED_COMMIT = "e0fb503544ddd38f71027c09e3401b900f3dabc3"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--adaptation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != EXPECTED_COMMIT or subprocess.check_output(
            ["git", "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("upstream commit changed or tracked source was modified")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite compatibility output")
    module = load_module("adast_original_models", args.upstream / "models/models.py")
    config_module = load_module("adast_original_config", args.upstream / "config_files/configs.py")
    utils = load_module("adast_original_utils", args.upstream / "utils.py")
    cfg = config_module.Config()
    split = json.loads((ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json").read_bytes())
    first = sorted(split["outer_runs"][0]["train"]["record_keys"])[0]
    source_path = ROOT / "data/processed/filtered_v2" / (first + ".npz")
    with np.load(source_path, allow_pickle=False) as z:
        positions = np.flatnonzero(z["y"] >= 0)[:8]
        source_x, source_y = torch.from_numpy(z["x"][positions]).unsqueeze(1), torch.from_numpy(z["y"][positions].astype(np.int64))
    manifest_path = args.adaptation_cache / "private_manifest.json"
    adaptation = json.loads(manifest_path.read_bytes())
    item = sorted(adaptation["records"], key=lambda r: r["subject_id"])[0]["variants"]["filtered_v2"]
    if sha256_file(Path(item["path"])) != item["sha256"]:
        raise ValueError("adaptation input hash mismatch")
    with np.load(item["path"], allow_pickle=False) as z:
        if set(z.files) != {"x", "original_epoch_index", "metadata_json"}:
            raise ValueError("target input contains unexpected schema or labels")
        target_x = torch.from_numpy(z["x"][:8]).unsqueeze(1)
    torch.set_num_threads(2)
    torch.manual_seed(123)
    encoder = module.cnn_feature_extractor(cfg.base_model)
    src_att, trg_att = module.Self_Attn(128), module.Self_Attn(128)
    head1, head2 = module.Classifier(cfg.base_model), module.Classifier(cfg.base_model)
    discriminator = module.Discriminator(cfg.base_model)
    networks = [encoder, src_att, trg_att, head1, head2]
    optimizer = torch.optim.Adam([p for model in networks for p in model.parameters()], lr=cfg.lr,
                                 betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr,
                                      betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    steps = []
    for round_index in range(2):
        for model in networks:
            model.eval()
        with torch.no_grad():
            feature = trg_att(encoder(target_x))
            pseudo = ((head1(feature) + head2(feature)) / 2).argmax(1)
        for model in networks:
            model.train()
        discriminator.train()
        for param in discriminator.parameters():
            param.requires_grad = True
        source_features, target_features = src_att(encoder(source_x)), trg_att(encoder(target_x))
        disc_logits = discriminator(torch.cat([source_features, target_features]).detach()).squeeze(-1)
        disc_loss = bce(disc_logits, torch.cat([torch.ones(8), torch.zeros(8)]))
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        for param in discriminator.parameters():
            param.requires_grad = False
        adversarial = bce(torch.cat([discriminator(target_features).squeeze(-1),
                                     discriminator(source_features).squeeze(-1)]),
                          torch.cat([torch.ones(8), torch.zeros(8)]))
        source_loss = ce(head1(source_features), source_y) + ce(head2(source_features), source_y)
        target_loss = ce(head1(target_features), pseudo) + ce(head2(target_features), pseudo)
        penalty = utils.calc_similiar_penalty(head1, head2)
        total = (cfg.adast_params.disc_wt * adversarial
                 + cfg.adast_params.src_clf_wt * (1. if round_index == 0 else .1) * source_loss
                 + cfg.adast_params.similarity_wt * penalty
                 + (0. if round_index == 0 else cfg.adast_params.trg_clf_wt) * target_loss)
        optimizer.zero_grad()
        total.backward()
        if not torch.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                           for model in networks for p in model.parameters()):
            raise ValueError("ADAST CPU update produced nonfinite loss/gradient")
        optimizer.step()
        steps.append({"self_training_round": round_index, "loss_finite": True,
                      "feature_shape": list(source_features.shape), "logits_shape": list(head1(source_features).shape)})
    result = {"status": "passed_component_and_loss_compatibility_not_full_training",
              "upstream_url": "https://github.com/emadeldeen24/ADAST", "upstream_commit": revision,
              "upstream_source_modified": False, "target_true_labels_used": False,
              "source_input_sha256": sha256_file(source_path), "adaptation_input_sha256": item["sha256"],
              "torch_version": torch.__version__, "steps": steps,
              "upstream_files_sha256": {str(p.relative_to(args.upstream)): sha256_file(p)
                                        for p in [args.upstream / "models/models.py", args.upstream / "config_files/configs.py",
                                                  args.upstream / "trainer/ADAST.py", args.upstream / "trainer/training_evaluation.py",
                                                  args.upstream / "utils.py", args.upstream / "LICENSE"]},
              "check_script_sha256": sha256_file(Path(__file__).resolve()),
              "limitations": "Original model classes and penalty function; locally orchestrated two update steps. No full trainer, convergence, accuracy, source-only counterpart, or UDA benchmark validated."}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
