# 1. Check the environment; no fold training starts in this notebook.
import json, platform, torch

info = {'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda}
if info['cuda_available']:
    info.update(gpu=torch.cuda.get_device_name(0),
                gpu_memory_GiB=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print(json.dumps(info, indent=2))
assert info['cuda_available'], 'Select a GPU runtime before this diagnostic.'

