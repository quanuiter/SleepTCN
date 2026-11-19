# Paste into ONE Colab cell, logged in with the MAIN account (the owner of SleepTCN_final_backup).
# Stop every other Colab session first. Deletes nothing on Drive.
# Builds one clean copy of runs/ from SleepTCN_final_backup, where the second account's sync left
# several folders with the same name (for example 22 copies of E4HP/fold_02/seed_123). For each run it
# keeps the newest copy whose run_manifest.json says "complete" (otherwise the newest partial copy for
# resuming), downloads it to /content/a4_merged/runs, then copies that to MyDrive/SleepTCN_a4_clean/runs.
import io, json, shutil, subprocess
from collections import defaultdict
from pathlib import Path
from google.colab import auth, drive
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

auth.authenticate_user()
drive.mount("/content/drive")
svc = build("drive", "v3")
FOLDER = "application/vnd.google-apps.folder"
OUT = Path("/content/a4_merged/runs")
CLEAN = Path("/content/drive/MyDrive/SleepTCN_a4_clean/runs")


def children(fid):
    out, tok = [], None
    while True:
        r = svc.files().list(q=f"'{fid}' in parents and trashed=false", pageSize=1000, pageToken=tok,
                             fields="nextPageToken,files(id,name,mimeType,modifiedTime,size)").execute()
        out += r["files"]
        tok = r.get("nextPageToken")
        if not tok:
            return out


def one(parent, name):
    found = [c for c in children(parent) if c["name"] == name and c["mimeType"] == FOLDER]
    if len(found) != 1:
        raise RuntimeError(f"{name}: {len(found)} folders")
    return found[0]["id"]


def download(fid, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        d, done = MediaIoBaseDownload(fh, svc.files().get_media(fileId=fid), chunksize=64 << 20), False
        while not done:
            _, done = d.next_chunk()


def tree(fid, rel=()):
    files = {}
    for c in children(fid):
        r = rel + (c["name"],)
        if c["mimeType"] == FOLDER:
            files.update(tree(c["id"], r))
        elif not c["mimeType"].startswith("application/vnd.google-apps"):
            if r not in files or c["modifiedTime"] > files[r]["modifiedTime"]:
                files[r] = c
    return files


root = svc.files().list(q=f"name='SleepTCN_final_backup' and mimeType='{FOLDER}' and 'me' in owners and trashed=false",
                        fields="files(id)").execute()["files"]
assert len(root) == 1, root
runs_id = one(root[0]["id"], "runs")

run_copies = defaultdict(list)   # ('a4','full',arm,fold,seed) -> folder ids
loose = {}                       # every other file: newest copy per path


def walk(fid, rel):
    for c in children(fid):
        r = rel + (c["name"],)
        if c["mimeType"] == FOLDER:
            if len(r) == 5 and r[:2] == ("a4", "full"):
                run_copies[r].append(c["id"])
            else:
                walk(c["id"], r)
        elif not c["mimeType"].startswith("application/vnd.google-apps"):
            if r not in loose or c["modifiedTime"] > loose[r]["modifiedTime"]:
                loose[r] = c


walk(runs_id, ())
print(len(run_copies), "runs,", sum(map(len, run_copies.values())), "folder copies,", len(loose), "other files")

report = {}
for rel, ids in sorted(run_copies.items()):
    best = None
    for fid in ids:
        files = tree(fid)
        man = files.get(("run_manifest.json",))
        status = "missing"
        if man:
            buf = io.BytesIO()
            d, done = MediaIoBaseDownload(buf, svc.files().get_media(fileId=man["id"])), False
            while not done:
                _, done = d.next_chunk()
            status = json.loads(buf.getvalue()).get("status", "?")
        complete = status == "complete" and ("validation_report.json",) in files
        stamp = max((f["modifiedTime"] for f in files.values()), default="")
        score = (complete, stamp)
        if best is None or score > best[0]:
            best = (score, files, status)
    (complete, _), files, status = best
    for r, f in files.items():
        download(f["id"], OUT.joinpath(*rel, *r))
    report["/".join(rel[2:])] = "complete" if complete else f"partial ({status})"
    print("/".join(rel[2:]), report["/".join(rel[2:])], f"from {len(ids)} copies", flush=True)

for r, f in loose.items():
    download(f["id"], OUT.joinpath(*r))

done = sum(v == "complete" for v in report.values())
print(f"\n{done} complete runs (repro check counted separately if present)")
CLEAN.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["rsync", "-a", f"{OUT}/", f"{CLEAN}/"], check=True)
print("clean copy written to", CLEAN)
