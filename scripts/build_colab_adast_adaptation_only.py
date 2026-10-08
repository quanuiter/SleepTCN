"""Only the five expressly authorized, ID-free, label-free adaptation signals."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/colab_adast_20261004'


def main():
    signal = OUT / 'payload/data/adaptation_x.npy'
    x = np.load(signal, mmap_mode='r', allow_pickle=False)
    if x.shape != (4989, 3000) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Adaptation-only shape/dtype/finiteness differs')
    source_manifest = json.loads((OUT / 'payload/manifest.json').read_bytes())
    sha = hashlib.sha256(signal.read_bytes()).hexdigest()
    if sha != source_manifest['files']['data/adaptation_x.npy']:
        raise ValueError('Adaptation-only values changed')
    manifest = {'scope': 'five_locked_SHHS_adaptation_subjects_only', 'subjects': 5, 'epochs': 4989,
                'participant_ids_included': False, 'true_labels_included': False,
                'source_sleepedf_data_included': False, 'target_test_data_included': False,
                'full_record_label_independent': True,
                'files': {'adaptation_x.npy': sha}}
    archive = OUT / 'SleepTCN_ADAST_AdaptationOnly_20261004.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(signal, arcname='adaptation_x.npy')
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != {'adaptation_x.npy', 'manifest.json'} or z.testzip() is not None:
            raise ValueError('Adaptation-only archive differs')
    result = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'archive_bytes': archive.stat().st_size, **manifest}
    (OUT / 'adaptation_only_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
