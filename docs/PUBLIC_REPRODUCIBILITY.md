# Public source, historical replay and delivery build

## Clean checkout and default tests

Use Python 3.10 or 3.11. The base environment intentionally pins PyTorch 2.5.1;
it is not the Colab CUDA training environment used for the ADAST additions.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[test]"
.venv/Scripts/python.exe -m pytest -q
```

Default tests require neither EEG/checkpoints nor optimizer steps. The retained private
integration tests and synthetic optimizer regression tests are reported as skipped.
They require explicit `--run-private` and/or `--run-optimizer-tests`. These flags do not
download data or grant permission to run a research campaign. Never enable optimizer
tests when the agreed scope prohibits CPU training. Some unit tests compute gradients
to test loss masking; they do not update model weights.

The default suite also blocks optimizer `.step()` calls so an unmarked new test cannot
silently train. The source-only test import uses the same adapter and metric bytes as the historical
payload, directly from `scripts/run_colab_adast_training.py` and `src/sleeptcn/metrics.py`.
There is no need to publish a private payload to make collection work.

## Three different uses

1. **Inspect/code-test/build paper:** public source, small aggregate evidence and TeX
   assets are sufficient. No datasets or model weights are shipped.
2. **Replay a historical campaign:** obtain the authorized local data/checkpoints and
   original proof/pointer files separately. Dated runners intentionally bind original
   splits, recipe, upstream bytes and proof hashes. They are not generic train commands.
   Private storage paths must be mapped in a separate local replay setup, not replaced
   in a frozen manifest. Do not remove hash checks to make a replay pass.
3. **New training:** create a separate protocol/bundle and obtain data/cloud permission.
   ADAST full-source training used PyTorch `2.11.0+cu130` on Colab T4 and rejects CPU
   fallback. Preserve the base environment rather than upgrading historical dependencies
   globally. The launcher verifies the actual GPU, library version and input hashes.

ADAST upstream is `https://github.com/emadeldeen24/ADAST`, pinned by
`configs/teacher_revision_adast_fold0_pilot_v1.json` to commit
`e0fb503544ddd38f71027c09e3401b900f3dabc3`. Authorized replay preparation uses the upstream
`models/models.py`, `config_files/configs.py`, `utils.py` and `LICENSE`; preserve that
license with copies. The repository does not grant a new license to upstream code.
No root project license is selected by this maintenance change; the authors must choose it.
Sleep-EDF/SHHS access terms apply separately from code licensing.

## Operational v2 (no automatic rerun)

`scripts/run_adast_fullsource_local_evaluation_v2.py` uses the reusable
`sleeptcn.runtime_io` progress writer and per-operation hash cache. It writes to a new
output directory and binds these new source bytes in its own execution specification.
The original runner, training core and campaign helpers are unchanged so old proofs
remain verifiable. Do not use v2 to overwrite/re-label historical results.

The new writer retries Windows sharing locks for a bounded interval, then retains a
complete uniquely named `.pending` status snapshot with a warning. Other I/O failures
remain errors. Checkpoints/results still use fail-closed writes. Pending status never
means a completed experiment; monitors must inspect the retained snapshot and process.
The wall-clock deadline is unchanged. Hash caching detects stat changes and still checks
each expected digest; final independent verification rereads all bound files without cache.

## Build the current paper (no training)

Prerequisites: `pdflatex`, `xelatex` and `bibtex` on PATH, with the packages listed in
`Reports/paper_en/BUILD.md`. The current XeLaTeX translation uses the installed
Times New Roman font; provide that font under its license rather than silently
substituting a different face and changing pagination.
This is a multi-file LaTeX project, not a standalone editor document.

```powershell
python scripts/build_public_delivery.py
python scripts/build_public_delivery.py --verify-only
# Equivalent Windows entrypoint:
powershell -File scripts/rebuild_bspc_delivery.ps1 -Python .venv/Scripts/python.exe
```

The build uses a fresh staging directory without old aux/bbl files. It builds the
supplement, English paper and **current BSPC Vietnamese translation**, then independently
builds the portable English source ZIP. It does not rebuild or overwrite older Vietnamese
reports. Approved vector figures are used as inputs; the script does not redesign figures.

Output: `SleepTCN_Scientific_Article_EN.pdf`, `SleepTCN_BSPC_Ban_dich_Tieng_Viet.pdf`,
`SleepTCN_Supplement_EN.pdf` in `Reports/output/pdf/`, and the English manuscript source
ZIP in `Reports/output/source/`. The Vietnamese source remains in the repository; the
journal source ZIP is English. `--stage-only` builds without publishing. `--package-only`
only verifies the existing release and refuses stale hashes; it cannot bless old PDFs
after source changes.

`Reports/publication/evidence.sha256` locks public aggregate/protocol evidence. The builder
checks it before and after compilation and never updates its hashes. Publication manifests
are regenerated only after a full successful build; they contain no `tmp/` or private
storage pointers. Prior delivery files/manifests are backed up in the local build directory.
Publication replacement is atomic per file, not a multi-file transaction; an interrupted
publish requires inspection and cannot pass verification against a stale manifest.

Visual QA remains required after any manuscript/figure edit: render every page, inspect
tables/references/glyphs, and compare text/numerical claims. Build/hash success alone is not
scientific or submission approval. PDF render tools are optional development dependencies,
not requirements for training; CI does not claim PDF visual review.

## Git hygiene and byte identity

Research sources/configs, signed aggregates and paper sources use `-text` attributes to
preserve historical bytes across Windows/Linux Git checkout. New Python files still use
LF. The CUDA 12.1 environment lock and its included freeze use canonical LF, matching
their existing declared checksums; these two text files had been converted to CRLF on
Windows and are restored to the originally hashed bytes. Do not run `git add --renormalize`
over frozen research artifacts. Hash verification
must use the checkout bytes, not only a local pre-commit copy.

Keep private data, model weights, individual predictions, operational pointers/logs,
Colab upload ZIPs, QA renders, `.venv`, `.miktex` and `tmp/` local. Only allowlisted
aggregates belong in a public release. Do not use `git add .`; review the final staged
allowlist and existing metadata permissions. Removing generated files from Git tracking
does not delete local files and does not erase them from prior Git history.
