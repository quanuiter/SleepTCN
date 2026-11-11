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


class TalHypnogram:
    """Minimal EDF+ annotation reader used where pyedflib rejects a telemetry hypnogram header.

    Seven telemetry hypnograms start after midnight but carry the previous day in the EDF+
    'Startdate' recording field, which pyedflib refuses. This reader takes the start time from the
    fixed header fields and parses the time-stamped annotation lists (TALs) of the data records.
    """

    def __init__(self, path: str):
        raw = Path(path).read_bytes()
        header = raw[:256].decode("ascii")
        ns = int(header[252:256])
        self.records = int(header[236:244])
        signal_header = raw[256:256 + ns * 256].decode("ascii")
        labels = [signal_header[i * 16:(i + 1) * 16].strip() for i in range(ns)]
        offset = ns * 216
        samples = [int(signal_header[offset + i * 8: offset + (i + 1) * 8]) for i in range(ns)]
        if labels != ["EDF Annotations"]:
            raise ValueError(f"{path}: unexpected hypnogram signals {labels}")
        day, month, year = (int(v) for v in header[168:176].split("."))
        hour, minute, second = (int(v) for v in header[176:184].split("."))
        import datetime
        self.start = datetime.datetime(1900 + year if year >= 85 else 2000 + year, month, day,
                                       hour, minute, second)
        body = raw[256 + ns * 256:]
        record_bytes = samples[0] * 2
        self.onsets, self.durations, self.texts = [], [], []
        for index in range(self.records):
            chunk = body[index * record_bytes:(index + 1) * record_bytes]
            for tal in chunk.split(b"\x00"):
                if not tal:
                    continue
                parts = tal.split(b"\x14")
                stamp, notes = parts[0].decode("ascii"), [p.decode("latin-1") for p in parts[1:] if p]
                onset_text, _, duration_text = stamp.partition("\x15")
                for note in notes:
                    self.onsets.append(float(onset_text))
                    self.durations.append(float(duration_text) if duration_text else -1.0)
                    self.texts.append(note)

    def getStartdatetime(self):  # noqa: N802 - pyedflib API name
        return self.start

    def readAnnotations(self):  # noqa: N802 - pyedflib API name
        import numpy as np
        return np.array(self.onsets), np.array(self.durations), np.array(self.texts, dtype=object)

    def close(self):
        return None


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
    # Deviation D1 (recorded in the manifest): telemetry files are stored in 10 s data records, so
    # 29 of the 44 recordings end with 10 or 20 s that do not fill a 30 s epoch. The trailing
    # incomplete epoch is dropped before preprocessing; the locked module itself is not changed.
    import numpy as np
    samples_per_epoch = preprocessing.PreprocessConfig().samples_per_epoch
    original_reader = preprocessing.pyedflib.EdfReader
    truncated: dict[str, int] = {}
    fallback_files: list[str] = []
    padded_seconds: dict[str, float] = {}

    class TruncatingReader(original_reader):
        def readSignal(self, chn, *a, **k):  # noqa: N802 - pyedflib API name
            signal = super().readSignal(chn, *a, **k)
            if signal.size != self.getNSamples()[chn] or not np.any(signal) or signal.std() < 1.0:
                raise ValueError(f"{self.file_name}: signal read failed or is flat")
            extra = signal.size % samples_per_epoch
            if extra:
                truncated[Path(self.file_name).name] = int(extra)
                signal = signal[: signal.size - extra]
            return signal

    # Scored span of every PSG after D1, used by D3 below.
    psg_seconds = {}
    for path in sorted(args.raw_root.glob("*-PSG.edf")):
        reader = original_reader(str(path))
        channel = reader.getSignalLabels().index(preprocessing.PreprocessConfig().channel)
        psg_seconds[KEY_RE.match(path.name).group(1)] = (
            reader.getNSamples()[channel] // samples_per_epoch * preprocessing.PreprocessConfig().epoch_seconds)
        reader.close()

    # Deviation D2: the TAL fallback must reproduce pyedflib exactly on every hypnogram both can read.
    # A failed pyedflib open corrupts reads of a PSG that is open at the same time, so files that
    # pyedflib rejects are identified here, while nothing else is open, and never reopened with it.
    rejected: set[str] = set()
    for path in sorted(args.raw_root.glob("*-Hypnogram.edf")):
        try:
            reference = original_reader(str(path))
        except OSError:
            rejected.add(path.name)
            continue
        onsets, durations, texts = reference.readAnnotations()
        start = reference.getStartdatetime()
        reference.close()
        tal = TalHypnogram(str(path))
        o, d, t = tal.readAnnotations()
        if not (np.allclose(o, onsets) and np.allclose(d, durations) and list(t) == list(texts)
                and tal.getStartdatetime() == start):
            raise ValueError(f"{path.name}: TAL fallback differs from pyedflib")

    class PaddedHypnogram:
        """Deviation D3: unscored signal after the last annotation is marked 'Sleep stage ?'."""

        def __init__(self, inner, key: str, name: str):
            self.inner, self.key, self.name = inner, key, name

        def getStartdatetime(self):  # noqa: N802
            return self.inner.getStartdatetime()

        def readAnnotations(self):  # noqa: N802
            onsets, durations, texts = self.inner.readAnnotations()
            out_o, out_d, out_t, cursor, filled = [], [], [], 0.0, 0.0
            for onset, duration, text in zip(onsets, durations, texts):
                gap = float(onset) - cursor
                if gap < -1e-6:
                    raise ValueError(f"{self.name}: overlapping annotations at {onset}")
                if gap > 1e-6:
                    out_o.append(cursor); out_d.append(gap); out_t.append("Sleep stage ?")
                    filled += gap
                out_o.append(float(onset)); out_d.append(float(duration)); out_t.append(str(text))
                cursor = float(onset) + float(duration)
            gap = psg_seconds[self.key] - cursor
            if gap > 1e-6:
                out_o.append(cursor); out_d.append(gap); out_t.append("Sleep stage ?")
                filled += gap
            if filled:
                padded_seconds[self.name] = filled
            return np.array(out_o), np.array(out_d), np.array(out_t, dtype=object)

        def close(self):
            self.inner.close()

    def reader_factory(path, *a, **k):
        name = Path(path).name
        if name.endswith("-Hypnogram.edf"):
            if name in rejected:
                fallback_files.append(name)
                inner = TalHypnogram(path)
            else:
                inner = original_reader(path, *a, **k)
            return PaddedHypnogram(inner, KEY_RE.match(name).group(1), name)
        return TruncatingReader(path, *a, **k)

    preprocessing.pyedflib.EdfReader = reader_factory
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
        "deviations": [{"id": "D1", "description": "trailing samples that do not fill a 30 s epoch were "
                        "dropped before preprocessing (10 s data records)",
                        "truncated_samples_by_file": truncated},
                       {"id": "D2", "description": "hypnograms whose EDF+ Startdate field names the previous "
                        "day (recording starts after midnight) were read with a minimal TAL parser that "
                        "reproduced pyedflib on every hypnogram pyedflib can read",
                        "files": sorted(set(fallback_files))},
                       {"id": "D3", "description": "signal not covered by any hypnogram annotation (before the "
                        "first, between annotations or after the last) was marked 'Sleep stage ?' (unknown, "
                        "excluded from metrics), as unscored Sleep Cassette epochs are",
                        "padded_seconds_by_file": padded_seconds}],
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"PASS: {len(pairs)} recordings, {len(subjects)} subjects")


if __name__ == "__main__":
    main()
