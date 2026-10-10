"""Measure CPU inference only on source validation; never learn or score SHHS."""
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
import verify_adast_fullsource_completion_results as check
from adast_cpu_inference_checks import predict_both_attention

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/adast_fullsource_retry_20261007/cpu_verification_profile.json'
ARCHIVE = Path('C:/Users/ADMIN/Downloads/SleepTCN_ADAST_Fullsource_Completion_Results_20261006.zip')
EXPECTED = '59cd81bca2a1bb67149b8d3348415d63c8524642240a32b608723ce56c0dd295'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve inference profile; do not repeat automatically')
    ref = check.shared.reference
    if ref.digest(ARCHIVE) != EXPECTED:
        raise ValueError('Result archive differs')
    work = Path(check.shared.read_local_json(check.INPUT / 'storage.json')['payload_root'])
    with np.load(work / 'data/fold_02_roles.npz', allow_pickle=False) as z:
        indices = z['validation'][:256]
    x = np.load(work / 'data/source_x.npy', mmap_mode='r', allow_pickle=False)[indices]
    module = ref.load_module('profile_adast_models', work / 'upstream/models.py')
    cfg = ref.load_module('profile_adast_config', work / 'upstream/configs.py').Config()
    torch.set_flush_denormal(False)
    with zipfile.ZipFile(ARCHIVE) as z:
        value = torch.load(io.BytesIO(z.read('fold_02/adast_full_source/final.pt')), map_location='cpu', weights_only=True)
    tiny = np.finfo(np.float32).tiny
    subnormal = sum(int(((v.numpy() != 0) & (np.abs(v.numpy()) < tiny)).sum()) for state in value['models'].values()
                    for v in state.values() if v.dtype == torch.float32)
    models = ref.build_models(module, cfg, 123, 'cpu')
    for name, model in models.items():
        model.load_state_dict(value['models'][name], strict=True)
    before = ref.state_digest(models)
    rows, baseline = [], None
    for threads, flush in [(2, False), (4, False), (4, True)]:
        torch.set_num_threads(threads)
        supported = torch.set_flush_denormal(flush)
        start = time.perf_counter()
        separate = {d: ref.predict(models, x, d, 'cpu') for d in ('source', 'target')}
        seconds_separate = time.perf_counter() - start
        start = time.perf_counter()
        joint = predict_both_attention(models, x)
        seconds_joint = time.perf_counter() - start
        for domain in joint:
            np.testing.assert_array_equal(joint[domain], separate[domain])
        if baseline is None:
            baseline = separate
        rows.append({'threads': threads, 'flush_denormal': flush, 'flush_supported': supported,
                     'separate_seconds': seconds_separate, 'joint_seconds': seconds_joint,
                     'joint_exactly_matches_separate_logits': True,
                     'argmax_changes_vs_original': {d: int((separate[d].argmax(1) != baseline[d].argmax(1)).sum()) for d in separate},
                     'max_abs_logits_delta_vs_original': {d: float(np.abs(separate[d] - baseline[d]).max()) for d in separate}})
    torch.set_flush_denormal(False)
    if ref.state_digest(models) != before:
        raise ValueError('Inference changed model state')
    report = {'status': 'complete_inference_profile_no_training', 'source_validation_epochs': 256,
              'source_fold': 2, 'checkpoint': 'adast_full_source/final', 'archive_sha256': EXPECTED,
              'subnormal_float32_model_values': subnormal, 'model_state_unchanged': True,
              'SHHS_access': False, 'training': False, 'rows': rows}
    ref.write_once(OUT, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
