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

**Before submission** you must still: insert e-mail addresses and ORCID identifiers; decide whether the
supervising author meets authorship criteria and should be added; and replace the front matter with the
target venue's template.

### Scientific scope (unchanged rules, inherited from `Reports/paper/BUILD.md`)

- Pre-registered Sleep-EDF comparisons: E1−E0, E2−E1, E3−E2, E3−E6.
- E3−E0 on Sleep-EDF and E3−E2 on SHHS1 must always be labelled post-hoc.
- The E1/E2 analysis on SHHS1 is secondary evidence, because the cohort was already opened for E0/E3/E6.
- Seed 42 is the primary campaign; seed 123 is a full post-protocol sensitivity repeat on the same
  split. The seeds are reported separately and their p-values are never pooled.
- Do not claim equivalence, non-inferiority, that P/N are useless, that ResNet is universally better,
  or that the model is parameter-efficient.

### New analyses added in the English version

These are derived from artefacts already in the repository; no new training was performed.

1. **Domain-gap decomposition** (§4.9, Table `tab:gap-decomposition`). Oracle correction of the
   N3→N2 channel alone recovers 74.5% of the Sleep-EDF→SHHS1 macro-F1 gap; N3→N2 together with
   N2→REM recovers 102.4%. Computed from the published confusion matrices.
2. **Predicted-to-true epoch ratio** as a marginal-calibration diagnostic (Table `tab:transfer-perclass`).
   Shows the SHHS miscalibration runs *opposite* to the label-prior shift, which excludes prior shift as
   the cause and rules out the standard Saerens correction as a remedy.
3. **Amplitude-threshold account of the N3 collapse** (§5.3), with four predictions, all verified
   against existing data — including the transition-region gradient (N3 recall 0.258 overall → 0.0733
   at stage boundaries).
4. **Receptive-field explanation of the Gate 8 null** (§5.2). The non-causal TCN spans 253 epochs
   (±63 min), so the ±1-epoch C/P/N groups are nested inside information the sequence model already has.
   The null was structurally predetermined and licenses no conclusion about temporal context.
5. **Sign-based effect size** Δ_sign = (W−L)/(W+L) with exact binomial sign tests, reported alongside
   every paired comparison as unadjusted supplementary robustness statistics.
6. Previously unused repository data now included: the transition-pair breakdown of the context
   ablation, the group-interaction index, SHHS transition-region per-class metrics, silhouette standard
   deviations, seed-123 SHHS replication, and absolute bootstrap intervals for SHHS.

### Proposed follow-up (stated in §5.7, not performed)

The highest-value next step is computing **per-class SHHS metrics for E6**. The amplitude account
predicts E6 should show substantially better N3 recall than E3 despite being worse overall. This
requires no retraining — only re-scoring existing predictions — and would confirm or refute the
central mechanistic claim of the paper. The 5 adaptation and 15 validation SHHS participants held out
and never used remain available for threshold recalibration.

## Pre-submission checklist

1. Swap the front matter for the target venue's template.
2. Add e-mail addresses, ORCID identifiers and author contribution statements if required.
3. Check the venue's ethics policy and its required SHHS/NSRR acknowledgement wording.
4. Run BibTeX and two full LaTeX passes; leave no undefined citations or references.
5. Render every page to an image and check tables, figures and page numbering.
