# Building the English manuscript

From `Reports/paper_en`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Suggested delivery path: `Reports/output/pdf/SleepTCN_Scientific_Article_EN.pdf`.

Figures are referenced as `../figures/*.png`, i.e. they are shared with the Vietnamese manuscript and
are **not** duplicated. Do not move this directory without updating those paths.

## Required LaTeX packages

Standard TeX Live / MiKTeX packages: `newtxtext`, `newtxmath`, `authblk`, `titlesec`, `abstract`,
`booktabs`, `tabularx`, `subcaption`, `placeins`, `microtype`, `xurl`, `fancyhdr`, `hyperref`.

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
(Background / Methods / Results / Conclusions), which is the norm for biomedical venues.

**Before submission** you must still: add ORCID identifiers if available; decide whether the supervising
author meets authorship criteria and should be added; confirm the CRediT contribution statement; and
replace the front matter with the target venue's template.

### Scientific scope (unchanged rules, inherited from `Reports/paper/BUILD.md`)

- Pre-registered Sleep-EDF comparisons: E1−E0, E2−E1, E3−E2, E3−E6.
- E3−E0 on Sleep-EDF and E3−E2 on SHHS1 must always be labelled post-hoc.
- The E1/E2 analysis on SHHS1 is secondary evidence, because the cohort was already opened for E0/E3/E6.
- Seed 42 is the primary campaign; seed 123 is a full post-protocol sensitivity repeat on the same
  split. The seeds are reported separately and their p-values are never pooled.
- Do not claim equivalence, non-inferiority, that P/N are useless, that ResNet is universally better,
  or that the model is parameter-efficient.
- E5 is an audit-only identifier: 153/153 E4--E5 input records were bitwise identical and the clipping
  fraction was zero. It was excluded before performance analysis and must never be described as a
  score-based negative result.

### Current scientific narrative

The manuscript answers three practical questions rather than claiming a new state-of-the-art model:

1. Do the TCN and ResNet-1D substitutions provide stable incremental value under a shared subject-wise
   protocol? They do not establish a stable predictive advantage; the supported benefit is operational:
   lower measured forward latency and lower observed training-and-validation wall-clock time, paid for
   with more parameters and peak memory.
2. Which development axis matters more under zero-shot transfer? In this experiment, preprocessing
   contrasts are larger than architectural contrasts, but the strongest such comparison is post-hoc and
   does not identify a single causal operation.
3. Where should target-domain adaptation focus? N3→N2 and N2→REM have the greatest diagnostic
   leverage, especially near stage transitions.

Per-class E6 metrics have now been recovered from the locked SHHS artifacts. E6 does not rescue N3:
recall is 0.2005 overall and 0.0721 near transitions, compared with 0.2582 and 0.0733 for E3. This result
rules out record-wise z-scoring as a sufficient stand-alone remedy. It does **not** identify z-scoring's
failure as a contribution by itself, and it does not establish a montage-dependent amplitude mechanism.

The predicted-to-true class ratio is a marginal emission diagnostic, not probability calibration.
Label-prior shift remains a plausible contributor: a prior correction would move N3 and N1 in the
required directions, but it was not fitted or validated and is not claimed as either sufficient or
irrelevant.

The context-group and silhouette analyses are supporting checks only. They should not appear in the
title, abstract contribution list, or conclusion.

### Highest-value follow-up

The next experiment, if one is performed, should use the held-out SHHS adaptation participants for a
small class-specific calibration or limited fine-tuning study. Its primary readout should be N3 recall
and false-positive rate near N2--N3 transitions, with N2--REM as the secondary target. More generic
hyperparameter tuning does not answer the deployment question developed in the paper.

## Pre-submission checklist

1. Swap the front matter for the target venue's template.
2. Add ORCID identifiers and author contribution statements if required.
3. Check the venue's ethics policy. The SHHS/NSRR acknowledgement text in the manuscript was verified
   against the official NSRR dataset page on 31 August 2026.
4. Run BibTeX and two full LaTeX passes; leave no undefined citations or references.
5. Render every page to an image and check tables, figures and page numbering.
6. Before releasing a reproducibility package, reconcile the SHHS run-manifest protocol hash
   `165d7cdf...fe93` with the currently checked-out protocol hash `9541e233...fe9`. The locked prediction
   hashes and recomputed metrics are verified, but the source-protocol package is not yet release-ready.
