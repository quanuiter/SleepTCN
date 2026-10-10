"""Restore the locked zero-shot checkpoints for the SHHS1 holdout protocol from Git history.

Reads blobs with ``git cat-file`` (no checkout, no change to the working tree) into a
separate git-ignored workspace under ``runs/``. Every ``best.pt`` is verified against the
SHA-256 recorded in the historical checkpoint audit named by the holdout protocol, and
every stage marker (``complete.json``) is checked to reference that same checkpoint.
The script is resumable: files that already exist are re-hashed, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {(e, s) for e in ("E0", "E2", "E3", "E4", "E6") for s in (42, 123)}
EXPECTED_BEST_PER_ARM = {"E0": 160, "E2": 20, "E3": 20, "E4": 20, "E6": 20}
SUPPORT_FILES = ("configs/experiments_v2.json", "data/splits/sleepedf_sc_10fold_seed42_v2.json")
PATTERN = re.compile(r"^runs/v2/full/(E\d)/fold_(\d\d)/seed_(\d+)/checkpoints/.+/best\.pt$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobReader:
    """Streams blobs through one ``git cat-file --batch`` process."""

    def __init__(self, revision: str) -> None:
        self.revision = revision
        self.proc = subprocess.Popen(
            ["git", "-C", str(ROOT), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def read(self, path: str) -> bytes:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(f"{self.revision}:{path}\n".encode())
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().decode().strip()
        if header.endswith("missing"):
            raise FileNotFoundError(f"{path} not found at {self.revision}")
        size = int(header.split()[2])
        data = self.proc.stdout.read(size)
        self.proc.stdout.read(1)
        return data

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "shhs_holdout_confirmatory_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "holdout_ckpt_restore_20261010")
    parser.add_argument("--max-seconds", type=float, default=0, help="stop early and resume later (0 = no limit)")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Restore directory must be under the git-ignored runs/ folder")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    gate = protocol["checkpoint_gate"]["reference_hash_sources"]
    audit_path = ROOT / gate["git_history_audit"]
    audit_raw = audit_path.read_bytes()
    if sha256_bytes(audit_raw) != gate["git_history_audit_sha256"]:
        raise ValueError("Checkpoint history audit differs from the protocol's recorded hash")
    audit = json.loads(audit_raw)
    if audit.get("status") != "passed":
        raise ValueError("Checkpoint history audit has not passed")
    revision = audit["revision"]

    expected: dict[str, str] = {}
    for entry in audit["checkpoints"]:
        match = PATTERN.match(entry["git_path"])
        if match and (match.group(1), int(match.group(3))) in ARMS:
            expected[entry["git_path"]] = entry["sha256"]
    counts = Counter(
        (PATTERN.match(p).group(1), int(PATTERN.match(p).group(3))) for p in expected  # type: ignore[union-attr]
    )
    for experiment, seed in sorted(ARMS):
        if counts[(experiment, seed)] != EXPECTED_BEST_PER_ARM[experiment]:
            raise ValueError(f"{experiment} seed {seed}: audit lists {counts[(experiment, seed)]} checkpoints")

    paths = sorted(expected) + sorted(p.rsplit("/", 1)[0] + "/complete.json" for p in expected) + list(SUPPORT_FILES)
    reader = BlobReader(revision)
    started = time.monotonic()
    records = []
    stopped_early = False
    try:
        for index, path in enumerate(paths, start=1):
            dest = (output / path).resolve()
            if not dest.is_relative_to(output):
                raise ValueError("Path escaped the restore directory")
            if dest.exists():
                data = dest.read_bytes()
            else:
                if args.max_seconds and time.monotonic() - started > args.max_seconds:
                    stopped_early = True
                    break
                data = reader.read(path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
            digest = sha256_bytes(data)
            if path in expected and digest != expected[path]:
                raise ValueError(f"{path}: SHA-256 differs from the audited checkpoint")
            if path.endswith("complete.json"):
                marker = json.loads(data)
                best = path.rsplit("/", 1)[0] + "/best.pt"
                if marker.get("best_checkpoint_sha256") != expected[best]:
                    raise ValueError(f"{path}: marker does not reference the audited checkpoint")
            records.append({"git_path": path, "sha256": digest, "bytes": len(data), "audited_checkpoint": path in expected})
            if index % 50 == 0:
                print(f"[{index}/{len(paths)}] restored/verified", flush=True)
    finally:
        reader.close()

    if stopped_early:
        print(f"PARTIAL: {len(records)}/{len(paths)} files; run again to continue")
        return 2

    per_arm = {}
    for experiment, seed in sorted(ARMS):
        prefix = f"runs/v2/full/{experiment}/"
        hashes = sorted(r["sha256"] for r in records if r["audited_checkpoint"] and r["git_path"].startswith(prefix) and f"/seed_{seed}/" in r["git_path"])
        per_arm[f"{experiment}_s{seed}"] = {"best_checkpoints": len(hashes), "combined_sha256": sha256_bytes("".join(hashes).encode())}
    result = {
        "status": "complete",
        "revision": revision,
        "protocol_sha256": sha256_bytes(args.protocol.read_bytes()),
        "audit_sha256": gate["git_history_audit_sha256"],
        "workspace": str(output),
        "best_checkpoints": len(expected),
        "files": len(records),
        "arms": per_arm,
        "records": records,
    }
    manifest = output / "restoration_manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    print("STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
