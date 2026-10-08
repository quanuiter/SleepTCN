"""Restore audited E3/E4 artifacts into a separate ignored directory, never checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("require private runs subdirectory")
    audit = json.loads(args.audit.read_bytes())
    if audit["status"] != "passed" or audit["metadata_and_marker_checks_E3_E4"] != 80:
        raise ValueError("require completed historical audit")
    expected = {e["git_path"]: e["sha256"] for e in audit["checkpoints"] if e.get("metadata_verified")}
    paths = set(expected)
    for path in list(paths):
        parts = path.split("/")
        base = "/".join(parts[:6])
        paths.update(["/".join(parts[:-1]) + "/complete.json", base + "/run_manifest.json",
                      base + "/predictions/validation.npz", base + "/predictions/test.npz"])
    paths.update(["configs/experiments_v2.json", "data/splits/sleepedf_sc_10fold_seed42_v2.json"])
    output.mkdir(parents=True, exist_ok=True)
    restored = []
    for path in sorted(paths):
        dest = (output / path).resolve()
        if not dest.is_relative_to(output):
            raise ValueError("path outside restore directory")
        raw = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", audit["revision"] + ":" + path])
        digest = hashlib.sha256(raw).hexdigest()
        if path in expected and digest != expected[path]:
            raise ValueError("audited checkpoint changed")
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                raise ValueError("existing restore file differs; will not overwrite")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as stream:
                stream.write(raw)
        restored.append({"git_path": path, "path": str(dest), "sha256": digest, "bytes": len(raw),
                         "audited_checkpoint": path in expected})
    result = {"status": "complete", "revision": audit["revision"],
              "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
              "checkpoints": len(expected), "files": len(restored), "records": restored,
              "restored_bytes": sum(e["bytes"] for e in restored)}
    manifest = output / "restoration_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_bytes()) != result:
            raise ValueError("restore manifest changed")
    else:
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
