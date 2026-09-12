# Building the SleepTCN English manuscript source bundle

This is a LaTeX source bundle, not a complete experimental-data archive. It contains the manuscript,
standalone supplement, bibliography, figure assets, vector plot source and editable highlights text.
No raw data, participant identifiers, predictions or internal audit documents are included.

The manuscript retains author-supplied declarations; these do not independently verify ethics,
CRediT or figure provenance. It identifies archived repository snapshot `f7c22e7`, not a newly
verified execution revision. Before submitting, push the
final editorial revision and verify journal-specific upload
requirements against the current BSPC Guide for Authors.

## Build order

Run these commands from the bundle root. Build the supplement first because `main.tex` imports its
table labels with `xr-hyper`:

The processing schematic and speed--performance plot are supplied as PDFs with editable TikZ sources.
From `figure_sources/`, run the following commands to rebuild them (the quoted output-directory
argument also works in PowerShell). No experimental results are recalculated.

```powershell
pdflatex -interaction=nonstopmode -halt-on-error '-output-directory=../figures' gate8_pipeline_overview_en.tex
pdflatex -interaction=nonstopmode -halt-on-error '-output-directory=../figures' gate8_primary_speedup_f1_en.tex
```

```powershell
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item -LiteralPath supplement.pdf -Destination SleepTCN_Supplement_EN.pdf
Copy-Item -LiteralPath main.pdf -Destination SleepTCN_Scientific_Article_EN.pdf
```

Check the exit code of each command. Place the two named delivery PDFs together so that the main
manuscript's supplementary-table links resolve. When a submission service builds only `main.tex`,
provide the freshly generated `supplement.aux` as a cross-reference dependency if the service permits
it, or follow the venue's process for building the two documents. Do not silently omit the supplement
or accept undefined references.

Standard TeX Live/MiKTeX packages are used, including `newtxtext`, `newtxmath`, `authblk`, `titlesec`,
`abstract`, `booktabs`, `tabularx`, `subcaption`, `placeins`, `float`, `microtype`, `xurl`, `xr-hyper`,
`fancyhdr` and `hyperref`. English figure dependencies are under `figures/`; no parent-directory
assets are required by the active documents.
The local `ieeetr-doi.bst` is included and must remain beside `main.tex`; it preserves numeric
citation order while adding available DOI links. TikZ/PGFPlots is needed only to rebuild the vector sources.

## Final checks

- Inspect all pages, table numbers, citations and external supplement references.
- Check every source change against a fresh build; do not upload a stale PDF.
- The four bullets in `highlights.txt` are source text; convert to the required editable upload
  format and recheck the journal's character limit.
- Do not interpret a successful PDF build as author approval, proof of ethics authorisation,
  clinical validation or end-to-end scientific reproducibility.

The supplementary figure source is `scripts/build_english_supplement_figures.py` in the project
repository. It reads two hash-pinned aggregate JSON files named in the script. Those inputs and the
script are not part of this typesetting ZIP; use the repository for plotting provenance. This bundle
does not rerun a model or recalculate experimental statistics.

## Submission items (Guide for Authors supplied on 11 September 2026)

- Upload the English manuscript source, figure files and the English supplement. The Vietnamese
  documents are synchronised internal reports, not additional manuscripts for BSPC.
- Upload `highlights.txt` separately as an editable highlights file, or convert it to an editable
  format accepted by the portal. It contains four bullets, each under 85 characters.
- Complete Elsevier's declarations tool and upload its resulting `.doc`/`.docx` file even if no
  competing interests are declared. A manuscript sentence does not replace that workflow.
- Supply a concise caption for the supplementary file, for example: "Supplementary methods,
  paired statistics, seed sensitivity, and supporting error and representation analyses."
- Graphical abstract is encouraged, not required. BSPC uses single-anonymized review.
- Final author confirmation, the funder's role, the basis for the ethics statement, data-use
  compliance and approval of the current revision must be settled before Submit.
