# Paste into ONE Colab cell on the MAIN account, after cells 1-3 of final_colab_a4.ipynb have run
# (BACKUP must point at SleepTCN_a4_clean). Deletes nothing on Drive except stale files inside the repaired
# run folders of SleepTCN_a4_clean; SleepTCN_final_backup is only read.
#
# Why: two Colab accounts sometimes trained the same run at the same time, and the merge combined files
# from different attempts (for example best.pt from one attempt, complete.json from the other), which gives
# "completed stage checkpoint mismatch". This cell checks every A4 run locally, and for each run that fails
# it downloads every Drive copy of that run separately, checks each copy on its own and installs the best
# self-consistent one: a complete copy that passes validate_run, otherwise a partial copy whose stage
# markers match their checkpoints (training resumes from it), otherwise nothing (training starts that run
# from scratch). Replaced local folders are moved to /content/a4_broken, not deleted.
import io, json, shutil, subprocess, time
from pathlib import Path
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import os
REPO = Path("/content/SleepTCN")
PY = str(REPO / ".venv" / "bin" / "python")
ENV = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
BACKUP = Path("/content/drive/MyDrive/SleepTCN_a4_clean")
if not (Path(PY).exists() and (REPO / "runs/a4/full").is_dir() and BACKUP.is_dir()):
    raise RuntimeError("new runtime: first run cells 1, 2, 3 (with BACKUP = SleepTCN_a4_clean), then this cell")

CHECKER = Path("/content/a4_check.py")
CHECKER.write_text('''
import json, sys
sys.path.insert(0, "src")
from pathlib import Path
from sleeptcn.amendment_a4 import load_a4_config, training_targets, run_root
from sleeptcn.io.hashing import sha256_file
from sleeptcn.run_validation import validate_run
w = Path(".").resolve()
if sys.argv[1] == "targets":
    c, _ = load_a4_config(w)
    print(json.dumps([[t.key, run_root(w, t).relative_to(w).as_posix()] for t in training_targets(c)]))
    raise SystemExit
out = {}
for d in map(Path, sys.argv[2:]):
    r = {"exists": d.is_dir(), "bad_markers": []}
    for m in d.glob("checkpoints/**/complete.json"):
        best = m.parent / "best.pt"
        if not (best.is_file() and sha256_file(best) == json.loads(m.read_text()).get("best_checkpoint_sha256")):
            r["bad_markers"].append(m.parent.relative_to(d).as_posix())
    man = d / "run_manifest.json"
    r["status"] = json.loads(man.read_text()).get("status") if man.is_file() else "missing"
    r["valid"] = False
    if r["status"] == "complete" and (d / "validation_report.json").is_file():
        try:
            validate_run(w, d); r["valid"] = True
        except Exception as e:
            r["error"] = repr(e)[:300]
    r["kind"] = ("good" if r["valid"] and not r["bad_markers"] else
                 "resumable" if r["exists"] and r["status"] != "complete" and not r["bad_markers"] else
                 "empty" if not r["exists"] else "broken")
    out[str(d)] = r
print(json.dumps(out))
''')


def check(*dirs):
    res = subprocess.run([PY, str(CHECKER), "check", *map(str, dirs)], cwd=REPO, env=ENV,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stdout + res.stderr)
    return json.loads(res.stdout.strip().splitlines()[-1])


res = subprocess.run([PY, str(CHECKER), "targets"], cwd=REPO, env=ENV, capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError(res.stdout + res.stderr)
targets = json.loads(res.stdout.strip().splitlines()[-1])
local = check(*(REPO / rel for _, rel in targets))
bad = [(key, rel) for key, rel in targets if local[str(REPO / rel)]["kind"] != "good"]
print(f"{len(targets) - len(bad)} runs good locally; to repair:")
for key, rel in bad:
    print("  ", key, local[str(REPO / rel)])

if bad:
    auth.authenticate_user()
    svc = build("drive", "v3")
    FOLDER = "application/vnd.google-apps.folder"

    def children(fid):
        out, tok = [], None
        while True:
            r = svc.files().list(q=f"'{fid}' in parents and trashed=false", pageSize=1000, pageToken=tok,
                                 fields="nextPageToken,files(id,name,mimeType,modifiedTime)").execute()
            out += r["files"]
            tok = r.get("nextPageToken")
            if not tok:
                return out

    def tree(fid, rel=()):
        files = {}
        for c in children(fid):
            r = rel + (c["name"],)
            if c["mimeType"] == FOLDER:
                for k, v in tree(c["id"], r).items():
                    if k not in files or v["modifiedTime"] > files[k]["modifiedTime"]:
                        files[k] = v
            elif not c["mimeType"].startswith("application/vnd.google-apps"):
                if r not in files or c["modifiedTime"] > files[r]["modifiedTime"]:
                    files[r] = c
        return files

    def download(fid, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            d, done = MediaIoBaseDownload(fh, svc.files().get_media(fileId=fid), chunksize=64 << 20), False
            while not done:
                _, done = d.next_chunk()

    roots = svc.files().list(q=f"name='SleepTCN_final_backup' and mimeType='{FOLDER}' and 'me' in owners "
                               "and trashed=false", fields="files(id)").execute()["files"]
    assert len(roots) == 1, roots
    stamp = time.strftime("%Y%m%d_%H%M%S")
    repaired = []
    for key, rel in bad:
        level = [roots[0]["id"]]
        for name in rel.split("/"):      # runs/a4/full/<arm>/fold_XX/seed_YYY, duplicates at any level
            level = [c["id"] for f in level for c in children(f) if c["name"] == name and c["mimeType"] == FOLDER]
        cands = []
        for i, fid in enumerate(level):
            files = tree(fid)
            dest = Path(f"/content/a4_candidates/{key.replace('/', '_')}/{i}")
            shutil.rmtree(dest, ignore_errors=True)
            for r, f in files.items():
                download(f["id"], dest.joinpath(*r))
            newest = max((f["modifiedTime"] for f in files.values()), default="")
            cands.append((dest, newest))
        verdict = check(*(d for d, _ in cands)) if cands else {}
        for d, newest in cands:
            v = verdict[str(d)]
            print(f"   {key} copy {d.name}: {v['kind']} status={v['status']} newest={newest} "
                  f"bad_markers={v['bad_markers']} {v.get('error', '')}")
        good = sorted((n, d) for d, n in cands if verdict[str(d)]["kind"] == "good")
        resum = sorted((n, d) for d, n in cands if verdict[str(d)]["kind"] == "resumable")
        target = REPO / rel
        lk = local[str(target)]["kind"]
        if good:
            choice, why = good[-1][1], "complete copy that passes validation"
        elif lk == "resumable":
            choice, why = None, "keep local partial copy (resumes)"
        elif resum:
            choice, why = resum[-1][1], "partial copy with consistent checkpoints (resumes)"
        else:
            choice, why = "scratch", "no consistent copy: this run trains from scratch"
        if choice is not None and target.exists():
            aside = Path(f"/content/a4_broken/{stamp}/{rel}")
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(aside))
        if isinstance(choice, Path):
            shutil.copytree(choice, target)
        print(f"==> {key}: {why}", flush=True)
        repaired.append(rel)

    for rel in repaired:   # make the clean Drive copy match, so the next restore gets the same files
        src, dst = REPO / rel, BACKUP / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{dst}/"], check=True)
        elif dst.is_dir():
            empty = Path("/content/a4_empty")
            empty.mkdir(exist_ok=True)
            subprocess.run(["rsync", "-a", "--delete", f"{empty}/", f"{dst}/"], check=True)

after = check(*(REPO / rel for _, rel in targets))
print("\nafter repair:", {k: sum(v["kind"] == k for v in after.values()) for k in ("good", "resumable", "empty", "broken")})
print("now rerun cell 4" if not any(v["kind"] == "broken" for v in after.values()) else "STILL BROKEN: send this output")
