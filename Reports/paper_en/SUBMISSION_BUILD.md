# Building the SleepTCN English manuscript source bundle

This is a LaTeX source bundle, not a complete experimental-data archive. It contains the manuscript,
standalone supplement, bibliography, figure assets, vector plot source and editable highlights text.
No raw data, participant identifiers, predictions or internal audit documents are included.

Revision, 7 October 2026: the abstract and interpretation distinguish the
primary E3 comparisons from the same-seed E4 extensions at seeds 42 and 123. Additional
intervention experiments are reported with their own matched controls and ensemble sizes.
Class weighting now reports a verified matched ten-fold ensemble, with CPU-trained folds 0--5
and CUDA-trained folds 6--9. The earlier single-fold result remains separate. ADAST now reports
a verified matched ten-fold CUDA-trained comparison, evaluated locally on all 180 SHHS
participants and all source outer-test subjects at two matched budgets. With 1,140 updates,
ADAST improves target N3 recall but reduces class-balanced performance and produces no N1
predictions. With 30 complete source passes, epoch-30 ADAST improves subject-mean macro-F1
by 0.0445 and N3 recall by 0.3679 over its matched control, while reducing N1/N2/REM F1
and source outer-test performance. The paired difference of budget effects is 0.0897.
The source-validation-selected secondary evaluation retains the same direction of
N3 benefit and stage/source-performance costs: SHHS subject-mean macro-F1 improves
by 0.0322. Supplementary Tables S32-S33 report aggregate and all-class results;
epoch 30 remains primary. The earlier CPU pilot remains separate. New CUDA training times do not replace the
historical benchmark.
Two coauthors have been appended
with the confirmed Information Systems affiliation; the corresponding author is unchanged.
This is a working manuscript revision with limited-budget and full-source multi-fold ADAST comparisons, not
a comprehensive UDA benchmark or a final
submission approval. Author contribution statements still require all coauthors' confirmation.

The current revision also reports the full-source budget and loss-component studies, including
the three-arm fold-1 comparison. Results distinguish selected and final checkpoints and source-
validation attention paths. A numerical and claim review preserved all table values and replaced
repeated procedural caveats with direct descriptions of measured effects and experimental scope.

The manuscript retains author-supplied declarations; these do not independently verify ethics,
CRediT or figure provenance. It identifies archived repository snapshot `f7c22e7`, not a newly
verified execution revision. Before submitting, push the
final editorial revision and verify journal-specific upload
requirements against the current BSPC Guide for Authors.

## Build order

Run these commands from the bundle root. Build the supplement first because `main.tex` imports its
table labels with `xr-hyper`:

The study-design schematic and speed--performance plot are supplied as vector PDFs. The schematic
also has an editable SVG and a Matplotlib builder; the speed plot retains its TikZ source.
From `figure_sources/`, run the following commands to rebuild them (the quoted output-directory
argument also works in PowerShell). The schematic builder requires Python, Matplotlib and Arial
(or a locally selected substitute font). No experimental results are recalculated.

```powershell
python study_design.py
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
first-citation order, adds available DOI or URL links, and distinguishes article identifiers
from page ranges. TikZ/PGFPlots is needed only to rebuild the speed-plot source.

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
