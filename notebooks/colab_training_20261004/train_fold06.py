# Train the real matched source pair, not another benchmark.
import hashlib, importlib.util, json, pathlib, subprocess, sys, zipfile
from google.colab import files
package = pathlib.Path('/content/SleepTCN_Train_Fold06_20261004.zip')
assert hashlib.sha256(package.read_bytes()).hexdigest() == 'fbc0f5aee8bf14829e9ffb5485187788aaf9bdca31331b5d968cbfec0d127dec'
work = pathlib.Path('/content/sleeptcn_training_20261004/fold_06')
with zipfile.ZipFile(package) as z:
    manifest = json.loads(z.read('manifest.json'))
    assert manifest['fold'] == 6 and not manifest['target_data_included']
    assert not manifest['participant_ids_included'] and not manifest['raw_eeg_included']
    assert len(z.namelist()) == len(manifest['files']) + 1
    assert set(z.namelist()) == set(manifest['files']) | {'manifest.json'}
    assert z.testzip() is None
    for name in z.namelist():
        relative = pathlib.PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        data = z.read(name)
        if name != 'manifest.json':
            assert hashlib.sha256(data).hexdigest() == manifest['files'][name]
        target = work / name
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
if importlib.util.find_spec('pyedflib') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'pyedflib==0.1.42'], check=True)
output = pathlib.Path('/content/sleeptcn_training_20261004/results/fold_06')
output.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-u', str(work / 'scripts/run_colab_source_training.py'),
           '--output', str(output), '--max-seconds', '18000']
print('STARTING REAL CUDA TRAINING: fold 6, unweighted + weighted, full source roles', flush=True)
with (output / 'run.log').open('a', encoding='utf-8') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / 'process.json').write_text(json.dumps({'pid': process.pid, 'command': command}), encoding='utf-8')
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    returncode = process.wait()
archive = output.parent / 'SleepTCN_Fold06_CUDA_Checkpoints.zip'
if archive.exists():
    files.download(str(archive))
assert returncode == 0, 'Training stopped/failed; checkpoints retained, inspect log.'
print('FOLD 6 MATCHED CUDA PAIR COMPLETE. No SHHS scoring performed.', flush=True)
