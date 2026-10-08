# 4. Preserve diagnostics outside the temporary runtime (download, no public sharing).
import hashlib, json, pathlib, zipfile
from google.colab import files

work = pathlib.Path('/content/sleeptcn_t4_smoke_20261004')
output_dir = work / 'results'
summary = json.loads((output_dir / 'summary.json').read_bytes())
assert summary['status'] == 'passed'
expected = ['cpu.json', 'cuda.json', 'cpu.log', 'cuda.log', 'summary.json', 'progress.json']
manifest = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in expected}
(output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
destination = pathlib.Path('/content/SleepTCN_T4_Smoke_Results_20261004.zip')
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in expected + ['manifest.json']:
        archive.write(output_dir / name, arcname=name)
print('RESULT_ARCHIVE_SHA256', hashlib.sha256(destination.read_bytes()).hexdigest())
files.download(str(destination))

