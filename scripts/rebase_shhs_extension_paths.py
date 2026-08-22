"""Rebase absolute SHHS manifest paths for a different runtime filesystem.

The Windows preprocessing run writes ``E:\\research\\Dataset\\SHHS_v1`` paths.
Before running inside Linux/Docker, this script rewrites only each record's
``output_path`` and updates the combined-manifest hash recorded by the protocol.
It never changes arrays, labels, source hashes or subject membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-protocol", type=Path, required=True)
    args = parser.parse_args()

    old = args.old_prefix.rstrip("/\\")
    new = args.new_prefix.rstrip("/\\")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    changed = 0
    for record in manifest.get("records", []):
        path = str(record["output_path"])
        if not path.startswith(old):
            raise ValueError(f"Path does not start with old prefix: {path}")
        record["output_path"] = new + path[len(old):].replace("\\", "/")
        changed += 1
    output_manifest = args.output_manifest.resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_sha256 = digest(output_manifest)
    output_manifest.with_suffix(output_manifest.suffix + ".sha256").write_text(
        f"{manifest_sha256}  {output_manifest.name}\n", encoding="ascii"
    )

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    provenance = protocol.setdefault("preprocessing_provenance", {})
    provenance["manifest_sha256"] = manifest_sha256
    provenance["combined_manifest_sha256"] = manifest_sha256
    output_protocol = args.output_protocol.resolve()
    output_protocol.parent.mkdir(parents=True, exist_ok=True)
    output_protocol.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    protocol_sha256 = digest(output_protocol)
    output_protocol.with_suffix(output_protocol.suffix + ".sha256").write_text(
        f"{protocol_sha256}  {output_protocol.name}\n", encoding="ascii"
    )
    print(json.dumps({
        "status": "complete",
        "records_rebased": changed,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "output_manifest": str(output_manifest),
        "output_protocol": str(output_protocol),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
