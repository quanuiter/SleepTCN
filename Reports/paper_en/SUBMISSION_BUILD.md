# Building the SleepTCN English manuscript source bundle

This is a LaTeX source bundle, not a complete experimental-data archive. It contains the manuscript,
standalone supplement, bibliography, two English figure PDFs and editable highlights text.
No raw data, participant identifiers, predictions or internal audit documents are included.

The manuscript records the author confirmations for CRediT, ethics/access, funding, competing interests
and AI disclosure. Before submitting, verify the exact repository revision/tag and journal-specific upload
requirements against the current BSPC Guide for Authors.

## Build order

Run these commands from the bundle root. Build the supplement first because `main.tex` imports its
table labels with `xr-hyper`:

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

## Final checks

- Inspect all pages, table numbers, citations and external supplement references.
- Check every source change against a fresh build; do not upload a stale PDF.
- The four bullets in `highlights.txt` are source text; convert to the required editable upload
  format and recheck the journal's character limit.
- Do not interpret a successful PDF build as author approval, proof of ethics authorisation,
  clinical validation or end-to-end scientific reproducibility.

Figure data and plotting provenance are maintained separately in the project repository. This bundle
uses the verified figure outputs; it does not rerun a model or recalculate experimental statistics.
