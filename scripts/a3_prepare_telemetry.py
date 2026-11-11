"""Amendment A3, step 1: download Sleep-EDF Telemetry from PhysioNet and preprocess it like Sleep-EDF SC.

Downloads the 44 PSG/hypnogram pairs listed in the PhysioNet SHA256SUMS file, verifies every SHA-256,
and writes the paper_raw_v1, bandpass_v2, filtered_v2 and filtered_zscore_v2 variants with the unchanged
Sleep-EDF preprocessing function. No model and no prediction is involved. See docs/AMENDMENT_A3.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn import preprocessing  # noqa: E402

CONFIG = json.loads((ROOT / "configs/amendment_a3.json").read_text(encoding="utf8"))
COHORT = CONFIG["cohorts"]["sleep_edf_telemetry"]
BASE = COHORT["source"]
KEY_RE = re.compile(r"^(ST\d{4}[A-Z])")
USER_AGENT = "python-urllib (research data download)"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw_root: Path) -> dict[str, str]:
    listing = fetch(COHORT["checksums"]).decode("utf8")
    expected = {}
    for line in listing.splitlines():
        digest, _, name = line.strip().partition(" ")
        name = name.strip()
        if name.startswith("sleep-telemetry/ST") and name.endswith(".edf"):
            expected[name.split("/", 1)[1]] = digest
    if len(expected) != 88:
        raise ValueError(f"expected 88 telemetry files in the checksum list, found {len(expected)}")
    raw_root.mkdir(parents=True, exist_ok=True)
    for index, (name, digest) in enumerate(sorted(expected.items()), start=1):
        path = raw_root / name
        if not path.exists() or sha256_file(path) != digest:
            print(f"[{index}/88] downloading {name}", flush=True)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(fetch(BASE + name))
            temporary.replace(path)
        if sha256_file(path) != digest:
            raise ValueError(f"{name}: SHA-256 differs from the PhysioNet checksum list")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / COHORT["raw_root"])
    parser.add_argument("--output-root", type=Path, default=ROOT / COHORT["processed_root"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/a3/telemetry_preprocess_manifest.json")
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError("Preserve the existing manifest; do not overwrite")
    expected = download(args.raw_root)

    # The locked preprocessing module recognises Sleep Cassette names only; the telemetry
    # recordings use the same file layout with the prefix ST, so only the key pattern changes.
    preprocessing.KEY_RE = KEY_RE
    pairs = preprocessing.pair_files(args.raw_root.resolve())
    if len(pairs) != 44:
        raise ValueError(f"expected 44 telemetry recordings, found {len(pairs)}")
    config = preprocessing.PreprocessConfig()
    if config.channel != COHORT["channel"]:
        raise ValueError("preprocessing channel differs from the locked A3 channel")
    records = []
    for index, (key, (psg_path, hyp_path)) in enumerate(pairs.items(), start=1):
        print(f"[{index}/44] {key}", flush=True)
        records.extend(preprocessing.process_record(
            psg_path=psg_path, hyp_path=hyp_path, variants=list(COHORT["variants"]),
            output_root=args.output_root.resolve(),
            source_hashes={"psg_sha256": expected[psg_path.name], "hypnogram_sha256": expected[hyp_path.name]},
            config=config, overwrite=False))
    subjects = sorted({key[:5] for key in pairs})
    report = {
        "schema_version": 1, "amendment": "A3", "dataset": "sleep-edf-expanded/sleep-telemetry/1.0.0",
        "source": BASE, "config": asdict(config), "variants": list(COHORT["variants"]),
        "recordings": len(pairs), "subjects": len(subjects), "source_sha256": expected,
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"PASS: {len(pairs)} recordings, {len(subjects)} subjects")


if __name__ == "__main__":
    main()
