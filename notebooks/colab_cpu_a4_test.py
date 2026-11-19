# Paste into ONE cell of a fresh Colab notebook with a CPU runtime (Runtime -> Change runtime type -> CPU),
# MAIN account. Do this only after the GPU session printed "ALL 40 A4 RUNS TRAINED" and ran sync_now().
# It replaces cells 1-5 of final_colab_a4.ipynb on CPU: restore from SleepTCN_a4_clean, check all 40 runs,
# CPU device check, open the Sleep-EDF test folds once, Telemetry + SHHS1 inference, export to Drive.
# Safe to rerun after a disconnect: the test journal and the inference outputs resume where they stopped.
import os, subprocess, json, shutil
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
DRIVE_INPUTS = Path("/content/drive/MyDrive/SleepTCN_upload")
BACKUP = Path("/content/drive/MyDrive/SleepTCN_a4_clean")
REPO = Path("/content/SleepTCN")
PY = str(REPO / ".venv" / "bin" / "python")
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""}
BUNDLES = {"a4_inputs_sleepedf.tar": "runs/a4/inputs_bundle_sleepedf.json",
           "final_core.tar": "runs/final/manifests/bundle_core.json",
           "final_holdout.tar": "runs/final/manifests/bundle_holdout.json"}


def run(args, expect=None, cwd=None):
    print("$", " ".join(map(str, args)), flush=True)
    p = subprocess.Popen(list(map(str, args)), cwd=cwd or REPO, env=ENV, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if p.wait() != 0:
        raise RuntimeError(f"command failed (exit {p.returncode})")
    if expect and expect not in "".join(lines):
        raise RuntimeError(f"expected {expect!r} in the output")
    return "".join(lines)


def sh(command):
    return run(["bash", "-lc", command], cwd="/content")


def sync_now():
    sh(f"for d in runs/a4 runs/final; do [ -d {REPO}/$d ] && mkdir -p {BACKUP}/$d && rsync -a {REPO}/$d/ {BACKUP}/$d/; done")


missing = [n for n in BUNDLES if not (DRIVE_INPUTS / n).is_file()]
if missing or not (BACKUP / "runs/a4/full").is_dir():
    raise RuntimeError(f"missing inputs {missing} or {BACKUP}/runs/a4/full")
print("CPUs:", os.cpu_count())

# code, environment (the CUDA wheels of the lock file also run on CPU), data
sh("command -v rsync >/dev/null || (apt-get -qq update && apt-get -qq install -y rsync)")
if not (REPO / ".git").is_dir():
    sh(f"git clone https://github.com/quanuiter/SleepTCN.git {REPO}")
if run(["git", "cat-file", "-t", "8af1d12"]).strip() != "commit":
    raise RuntimeError("repository history incomplete")
if not Path(PY).exists():
    sh("pip install --quiet 'uv==0.5.0'")
    sh(f"cd {REPO} && python -m uv venv --python-preference only-managed --python 3.11 --seed .venv")
    sh(f"cd {REPO} && {PY} -m pip install --quiet --upgrade pip && "
       f"{PY} -m pip install --quiet -r requirements/lock-cu121.txt && {PY} -m pip install --quiet -e . --no-deps")
for name, manifest in BUNDLES.items():
    if not (REPO / manifest).is_file():
        print("extracting", name, flush=True)
        sh(f"tar -xf '{DRIVE_INPUTS / name}' -C {REPO}")
run([PY, "scripts/run_amendment_a4.py", "verify-bundle"], expect="BUNDLE VERIFIED")
run([PY, "scripts/final_bundle.py", "verify", "--part", "core"], expect="BUNDLE VERIFIED")
run([PY, "scripts/final_bundle.py", "verify", "--part", "holdout"], expect="BUNDLE VERIFIED")

# restore the trained runs
for sub in ("runs/a4", "runs/final"):
    if (BACKUP / sub).is_dir():
        sh(f"mkdir -p {REPO / sub} && rsync -a '{BACKUP / sub}/' '{REPO / sub}/'")
snippet = ("import json,sys; sys.path.insert(0,'src'); from pathlib import Path; "
           "from sleeptcn.amendment_a4 import load_a4_config,training_targets,training_state; "
           "w=Path('.').resolve(); c,_=load_a4_config(w); n={}; "
           "[n.__setitem__(s,n.get(s,0)+1) for s in (training_state(w,t) for t in training_targets(c))]; print(json.dumps(n))")
states = json.loads(run([PY, "-c", snippet]).strip().splitlines()[-1])
if states.get("trained", 0) + states.get("tested", 0) != 40:
    raise RuntimeError(f"not all 40 runs are trained in {BACKUP}: {states}; finish training on the GPU first")
run(["git", "status", "--short"])
run([PY, "scripts/run_amendment_a4.py", "env-check", "--device", "cpu"], expect="ENVIRONMENT OK")

# CPU device check (the GPU report is kept as device_check_cuda.json)
check = REPO / "runs/final/device_check.json"
if check.is_file() and json.loads(check.read_text()).get("device", "").startswith("cuda"):
    shutil.copy(check, check.with_name("device_check_cuda.json"))
if not (check.is_file() and json.loads(check.read_text()).get("device") == "cpu"
        and json.loads(check.read_text())["passed"]):
    run([PY, "scripts/final_infer.py", "device-check", "--device", "cpu"], expect="DEVICE CHECK PASSED")
sync_now()

# background sync every 5 minutes
SYNC = f'''while true; do
  for d in runs/a4 runs/final; do [ -d "{REPO}/$d" ] && mkdir -p "{BACKUP}/$d" && rsync -a "{REPO}/$d/" "{BACKUP}/$d/"; done
  date > "{BACKUP}/last_sync.txt"; sleep 300
done'''
if subprocess.run(["pgrep", "-f", "last_sync.txt"], capture_output=True).returncode != 0:
    subprocess.Popen(["bash", "-c", SYNC], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# open the Sleep-EDF test folds once
if states.get("tested", 0) < 40:
    run([PY, "scripts/run_amendment_a4.py", "test"], expect="Preflight passed")
    run([PY, "scripts/run_amendment_a4.py", "test", "--device", "cpu", "--confirm", "OPEN-A4-LOCKED-TEST-ONCE"],
        expect="A4 TEST CAMPAIGN: complete")
    sync_now()

# Telemetry + SHHS1 zero-shot inference on CPU (protocol device)
run([PY, "scripts/final_infer.py", "a4", "--device", "cpu"])
sync_now()

archive = Path("/content/final_results.tar.gz")
members = ["runs/a4/full", "runs/a4/monitoring", "runs/a4/test_campaign_a4.json", "runs/a4/reproduction_check.json",
           "runs/final"]
sh(f"cd {REPO} && tar -czf {archive} " + " ".join(m for m in members if (REPO / m).exists()))
shutil.copy(archive, BACKUP / archive.name)
print(sh(f"sha256sum {archive} {BACKUP / archive.name}"))
print("DONE: final_results.tar.gz is in", BACKUP)
