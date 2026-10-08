# 3. Bounded device diagnostic only: no fold training and no target scoring.
import json, pathlib, subprocess, sys, time

work = pathlib.Path('/content/sleeptcn_t4_smoke_20261004')
output_dir = work / 'results'
output_dir.mkdir(exist_ok=True)
progress_path = output_dir / 'progress.json'
started = time.monotonic()
for backend in ['cpu', 'cuda']:
    progress_path.write_text(json.dumps({'status': 'running', 'phase': backend,
        'elapsed_seconds': time.monotonic() - started}), encoding='utf-8')
    command = [sys.executable, str(work / 'scripts/benchmark_colab_tcn.py'),
               '--backend', backend, '--sample', str(work / 'sample_records.npz'),
               '--output', str(output_dir / f'{backend}.json'), '--bounded-child']
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=320)
    except subprocess.TimeoutExpired:
        progress_path.write_text(json.dumps({'status': 'stopped_resource_budget',
            'phase': backend}), encoding='utf-8')
        raise
    (output_dir / f'{backend}.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
    print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)
    result = json.loads((output_dir / f'{backend}.json').read_bytes())
    if completed.returncode != 0 or result['status'] != 'passed':
        progress_path.write_text(json.dumps({'status': 'failed', 'phase': backend}), encoding='utf-8')
        raise RuntimeError(f'{backend} failed: inspect its saved JSON/log before any campaign.')
cpu = json.loads((output_dir / 'cpu.json').read_bytes())
gpu = json.loads((output_dir / 'cuda.json').read_bytes())
assert cpu['sample_sha256'] == gpu['sample_sha256']
assert cpu['models_sha256'] == gpu['models_sha256']
assert cpu['training_sha256'] == gpu['training_sha256']
assert cpu['torch'] == gpu['torch']
rows = []
for c, g in zip(cpu['cases'], gpu['cases']):
    assert c['shape'] == g['shape'] and c['name'] == g['name']
    assert c['numerical_check']['passed'] and g['numerical_check']['passed']
    rows.append({'case': c['name'], 'shape': c['shape'],
        'cpu_seconds_per_step': c['with_transfer']['median_seconds'],
        'gpu_seconds_per_step': g['with_transfer']['median_seconds'],
        'cpu_over_gpu_speed_ratio': c['with_transfer']['median_seconds'] / g['with_transfer']['median_seconds'],
        'gpu_gradient_max_abs_residual': g['numerical_check']['gradient_max_abs_residual'],
        'gpu_peak_allocated_MiB': g.get('peak_memory_allocated_bytes', 0) / 2**20})
summary = {'status': 'passed', 'gpu': gpu['gpu'], 'torch': gpu['torch'], 'cases': rows,
    'bounded_device_diagnostic_only': True, 'no_campaign_started': True,
    'elapsed_seconds': time.monotonic() - started}
(output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
progress_path.write_text(json.dumps({'status': 'completed', 'no_campaign_started': True}), encoding='utf-8')
print(json.dumps(summary, indent=2))

