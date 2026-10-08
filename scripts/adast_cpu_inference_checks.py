"""Inference-only CPU helpers; preserve checkpoints, batch order and full validation support."""
import numpy as np
import torch


@torch.inference_mode()
def predict_both_attention(models, x, guard=lambda: None, batch_size=128):
    """Reuse identical eval encoder features for the two attention paths of each batch."""
    for model in models.values():
        model.eval()
    output = {'source': [], 'target': []}
    for start in range(0, len(x), batch_size):
        guard()
        batch = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32).copy()).unsqueeze(1)
        features = models['encoder'](batch)
        for domain in output:
            attended = models[domain + '_attention'](features)
            logits = torch.maximum(models['head1'](attended), models['head2'](attended))
            output[domain].append(logits.cpu().numpy())
    return {domain: np.concatenate(parts) for domain, parts in output.items()}
