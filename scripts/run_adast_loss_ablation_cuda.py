"""Four fresh full-source CUDA arms: reference and three single-factor interventions."""
import argparse
import copy
import json
from pathlib import Path
import zipfile

import torch
from torch import nn
import run_adast_development_cuda as base


EXPECTED_ARMS = {
    'adast_reference_full_source': {},
    'adast_keep_source_ce_full_source': {'source_loss_weights_by_round': [1., 1.]},
    'adast_no_pseudo_full_source': {'target_loss_weights_by_round': [0., 0.]},
    'adast_no_adversarial_full_source': {'adversarial_weight': 0.},
}
original_update = base.measured_update
original_train = base.train_development_arm


def effective_config(cfg, key):
    if cfg['arms'] != list(EXPECTED_ARMS) or cfg['arm_overrides'] != EXPECTED_ARMS:
        raise ValueError('Only the frozen three single-factor changes are allowed')
    result = copy.deepcopy(cfg)
    result.update(cfg['arm_overrides'][key])
    return result


def measured_update(models, optimizer, disc_optimizer, utils, sx, sy, tx, pseudo, round_index, cfg, arm):
    if cfg['adversarial_weight'] != 0.:
        # Reference, retained source CE, and zero pseudo CE keep the original RNG/computation order.
        return original_update(models, optimizer, disc_optimizer, utils, sx, sy, tx, pseudo, round_index, cfg, arm)
    if arm != 'adast':
        raise ValueError('This intervention requires ADAST with target forward retained')
    for model in models.values():
        model.train()
    ce = nn.CrossEntropyLoss()
    _, sl1, sl2 = base.reference.logits(models, sx, 'source')
    source_loss = ce(sl1, sy) + ce(sl2, sy)
    similarity = utils.calc_similiar_penalty(models['head1'], models['head2'])
    loss = cfg['source_loss_weights_by_round'][round_index] * source_loss + cfg['similarity_weight'] * similarity
    _, tl1, tl2 = base.reference.logits(models, tx, 'target')
    target_loss = ce(tl1, pseudo) + ce(tl2, pseudo)
    loss = loss + cfg['target_loss_weights_by_round'][round_index] * target_loss
    optimizer.zero_grad()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                      for model in models.values() for p in model.parameters()):
        raise ValueError('Nonfinite loss/gradient')
    optimizer.step()
    return {'total': float(loss.detach()), 'source_ce': float(source_loss.detach()),
            'similarity': float(similarity.detach()), 'target_pseudo_ce': float(target_loss.detach()),
            'adversarial': 0., 'discriminator': 0.}


def train_arm(key, models, utils, arrays, cfg, folder, identity, guard, device):
    return original_train(key, models, utils, arrays, effective_config(cfg, key), folder, identity, guard, device)


def export(output):
    paths = sorted(p for p in output.rglob('*') if p.is_file() and p.suffix in {'.pt', '.json', '.npz', '.npy', '.log'})
    archive = output.parent / 'SleepTCN_ADAST_Loss_Ablation_Results_20261004.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths:
            z.write(p, p.relative_to(output).as_posix())
        z.writestr('export_manifest.json', json.dumps({p.relative_to(output).as_posix(): base.reference.digest(p) for p in paths}, indent=2))
    print('RESULT_ARCHIVE', archive, 'SHA256', base.reference.digest(archive), flush=True)


def main(args):
    cfg = json.loads((Path(__file__).resolve().parent / 'protocol.json').read_bytes())
    effective_config(cfg, cfg['arms'][0])
    if cfg['campaign_kind'] != 'full_source_single_factor_loss_ablation':
        raise ValueError('Unexpected campaign purpose')
    base.measured_update = measured_update
    base.train_development_arm = train_arm
    base.export = export
    try:
        base.main(args)
    finally:
        base.measured_update = original_update
        base.train_development_arm = original_train
        base.export = original_export


original_export = base.export
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=18000)
    main(parser.parse_args())
