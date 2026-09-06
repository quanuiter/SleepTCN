# Building the English manuscript

For the 06 September 2026 delivery, see `../BSPC_FINAL_READINESS_REPORT.md` and
`../BSPC_AUTHOR_CONFIRMATIONS_VI.md`. The independently build-checked source ZIP is
`../output/source/SleepTCN_BSPC_Manuscript_Source.zip`; its build instructions are in
`SUBMISSION_BUILD.md`. The author confirmations have been incorporated; before submission, identify the
exact pushed/tagged repository revision represented by the manuscript.

From `Reports/paper_en`, build the supplement first so that the manuscript can import its table
labels through `xr-hyper`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Suggested delivery path: `Reports/output/pdf/SleepTCN_Scientific_Article_EN.pdf`.

Build the standalone supplementary material with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

Suggested delivery path: `Reports/output/pdf/SleepTCN_Supplement_EN.pdf`.

Figure dependencies must be included with any portable source bundle. English supplementary figures
must use English labels; preserve the Vietnamese figure variants used by the Vietnamese reports.
Check the current `\includegraphics` paths rather than assuming all figures are shared unchanged.

## Required LaTeX packages

Standard TeX Live / MiKTeX packages: `newtxtext`, `newtxmath`, `authblk`, `titlesec`, `abstract`,
`booktabs`, `tabularx`, `subcaption`, `placeins`, `microtype`, `xurl`, `xr-hyper`, `fancyhdr`, `hyperref`.

If `newtxtext`/`newtxmath` are unavailable in your distribution, replace

```latex
\usepackage{newtxtext,newtxmath}
```

with

```latex
\usepackage{mathptmx}
```

or simply `\usepackage{lmodern}`. Nothing else in the document depends on the font choice.

## What changed relative to `Reports/paper` (the Vietnamese manuscript)

### Front matter

The Vietnamese draft used a Vietnamese primary title with an English subtitle, `\author{... \and ...}`
(which places authors in separate columns and attaches the affiliation to the last one only), and a
prose date line. The English manuscript uses a conventional journal front page: a single bold title,
an `authblk` author block with numbered affiliations on one line, an affiliation block, a
correspondence line, and a horizontal rule separating front matter from the abstract. Section headings
were reduced to journal scale via `titlesec`. The abstract is now structured
(Background / Methods / Results / Conclusions). Confirm the target journal's current abstract
requirements before upload.

**Author confirmations received:** Ngo Nhat Quan is the first and corresponding author, with email
`23521258@gm.uit.edu.vn`; Pham Thai Son is listed second. Both authors approved the final manuscript.
The role-based CRediT statement, SHHS/NSRR access, no-ethics-approval statement, student-support funding,
no-competing-interests statement and AI disclosure are recorded in the manuscript. Do not add roles,
supervision, grant codes or approval numbers that the authors have not verified. ORCID identifiers are
optional unless the submission system requires them; enter only verified identifiers.

### Scientific scope (unchanged rules, inherited from `Reports/paper/BUILD.md`)

- Pre-specified Sleep-EDF comparisons: E1−E0, E2−E1, E3−E2, E3−E6. No public
  pre-registration is claimed.
- E3−E0 on Sleep-EDF and E3−E2 on SHHS1 must always be labelled post-hoc.
- The E1/E2 analysis on SHHS1 is secondary evidence, because the cohort was already opened for E0/E3/E6.
- Seed 42 is the primary campaign; seed 123 is a full post-protocol sensitivity repeat on the same
  split. The seeds are reported separately and their p-values are never pooled.
- Do not claim equivalence, non-inferiority, that P/N are useless, that ResNet is universally better,
  or that the model is parameter-efficient.
- E1−E0 reuses the E0 encoder/cache but replaces the sequence-model configuration **and** its training
  recipe: learning rate 0.01→0.0005, batch size 4→8 recordings, maximum epochs 1000→300 and patience
  10→30. It does not isolate an architecture-only effect.
- E2−E1 replaces the 75-dimensional C/P/N package with a 128-dimensional current-epoch ResNet-1D
  representation. Encoder objective, optimizer, training budget and checkpoint selection also differ;
  this is not an isolated encoder contrast. Shared partitions and evaluation remain valuable controls.
- E6 estimates mean and standard deviation from the complete unlabelled target recording. It is
  label-free target-record normalisation with a transductive dependency, not purely inductive zero-shot
  inference.
- E5 is an audit-only identifier: 153/153 E4--E5 input records were bitwise identical and the clipping
  fraction was zero. It was excluded before performance analysis and must never be described as a
  score-based negative result.

### Current scientific narrative

The manuscript answers three practical questions rather than claiming a new state-of-the-art model:

1. Do the specified sequence-model and feature/context replacements provide stable incremental value under a shared subject-wise
   protocol? They do not establish a stable predictive advantage; the supported benefit is operational:
   a measured forward-pass speed-up and shorter observed training-and-validation wall-clock time, paid for
   with more parameters and peak memory.
2. Which pipeline differences persist under cross-cohort transfer? The observed post-hoc E3−E2
   preprocessing contrast is larger than the evaluated E1−E0 and E2−E1 contrasts on this SHHS1 sample.
   This effect-size comparison does not establish general superiority of preprocessing over architecture,
   identify an individual causal operation or quantify the return on development resources.
3. Which errors remain despite an overall score improvement? N3→N2 recurs in E0 and E3 and has the
   greatest diagnostic leverage in the E3 oracle analysis, followed by N2→REM. These are candidate
   targets for follow-up studies, not a validated allocation rule for collecting target labels.

Per-class E6 metrics have now been recovered from the locked SHHS artifacts. E6 does not rescue N3:
recall is 0.2005 overall and 0.0721 near transitions, compared with 0.2582 and 0.0733 for E3. This result
shows that the evaluated record-wise z-scoring pipeline did not remedy N3 under-detection. E6 was
trained as its own pipeline, not applied as an amplitude-only intervention to the same frozen model.
It neither rules out amplitude-related contributors nor tests all label-free normalization methods.

Pooled E0 class metrics recovered from the locked run manifest show the same N3 failure: recall 0.2610
and 16,480/22,806 N3 epochs predicted as N2, versus 0.2582 and 16,674/22,806 for E3. Therefore E3's
overall advantage must not be described as an N3 improvement, and the 74.5% oracle value must remain
explicitly scoped to E3 predictions and the observed pooled-score difference between EDF out-of-fold
evaluation and SHHS ten-fold-model ensemble evaluation. This difference is not a pure causal effect
of a single domain change.

The predicted-to-true class ratio is a marginal emission diagnostic, not probability calibration.
Label-prior shift remains a plausible contributor, but prior correction was not fitted or validated.
Its usefulness depends on assumptions about the shift and cannot be inferred from class frequencies
alone; it is not claimed as either sufficient or irrelevant.

The post-hoc cross-cohort transition diagnostic uses reference labels to compare radius-one boundary
neighbourhoods with stable interiors. It shows that these neighbourhoods are harder for E0 and E3 on
both cohorts, but the persistent stable-region N3 deficit on SHHS1 prevents a boundary-only explanation.
This is an offline diagnostic, not a physiological-transition or deployment claim.

The context-group and silhouette analyses are supporting checks only. They should not appear in the
title, abstract contribution list, or conclusion. They and the detailed per-class, confusion, seed and
runtime tables are maintained in `supplement.tex`.

### Follow-up options, not submission prerequisites

The current empirical contribution does not require inventing a new architecture or validating an
adaptation method. First recover missing methodological detail and verify supporting artifacts already
available. Subject-level uncertainty, existing seed sensitivity or an independently evaluated cohort
can strengthen the same question without changing the paper's scope. Additional training or analysis
requires an explicit plan and author approval.

If a later study claims a remedy, class-specific calibration or limited fine-tuning is one option.
Fit and select it only on the designated adaptation/validation data, report both N3 sensitivity and
false positives, and label evaluation on the already inspected test cohort as exploratory. An unseen
holdout would be needed for independent confirmation of a hypothesis developed from these test errors.

## Pre-submission checklist

1. Verify the current BSPC Guide for Authors, including article type, abstract, word limits, review mode
   and source-file requirements. Access to the Guide returned HTTP 403 during the audit; a 5,000-word
   limit, mandatory template or anonymous-review mode has not been established here.
2. Recheck the recorded CRediT, ethics/data-use, funding, competing-interest and AI-use declarations
   against the authors' supporting records; both authors' approval is already confirmed.
3. Treat `highlights.txt` as the editable source of the four highlights. Check the required upload
   format; Elsevier's general highlights guidance specifies a Word document, so do not assume that
   this internal text file is the final upload format. Each current bullet is below 85 characters.
4. Check the applicable NSRR data-use agreement and acknowledgement requirements with the account
   holder, and retain the access confirmation. The manuscript does not include restricted data or IDs.
5. Run BibTeX and two full LaTeX passes; leave no undefined citations or references.
6. Render every page to an image and check tables, figures and page numbering.
7. Preserve the historical SHHS protocol snapshot `configs/shhs_zero_shot_v1.json`, whose SHA-256
   `165d7cdf...fe93` matches the run manifest. The current `configs/shhs_v1_protocol.json` hash
   `9541e233...fe9` is an expanded post-run audit record, not a replacement; see
   `Reports/SHHS_PROTOCOL_PROVENANCE.md`. On 5 September 2026 all 540 primary ensemble prediction
   hashes and their recomputed confusion matrices were checked against the original run manifest;
   no training, inference or bootstrap rerun was performed. Raw SHHS data and prediction artifacts
   remain outside the repository.
8. Update publication manifests only after the revised sources are built and every PDF page is checked.
   Build success and hash integrity do not certify scientific completeness or submission readiness.
