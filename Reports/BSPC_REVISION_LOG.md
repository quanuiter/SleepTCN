# BSPC revision log

Revision performed 05–06 September 2026 against baseline commit `1710c49fe529f7d498598ad25ff108163f3e6f06`. The pre-submission audit remains a historical input; this log and the final readiness report describe the resulting revision.

## Scope authorised and respected

The author requested corrections to the English manuscript and synchronised Vietnamese reports, with strong but evidence-supported writing. No new training, prediction inference, bootstrap, calibration fitting or adaptation experiment was commissioned. Existing numeric outputs were checked and previously available seed-sensitivity evidence was incorporated. Historical configurations, labels, predictions and numeric aggregate JSON were not edited.

Ngô Nhật Quân confirmed first/corresponding authorship. The authors confirmed SHHS/NSRR access, no ethics approval was required for this de-identified secondary analysis, approximately VND 6,000,000 in student support from the University of Information Technology/VNU-HCM, no competing interests, use of no other AI tool besides OpenAI Codex, and approval of the final manuscript. A role-based CRediT statement was added without percentages. Nguyen Ho Duy Tri is acknowledged for academic supervision, guidance in defining the research topic and constructive manuscript comments, but is not listed as an author or assigned a CRediT role. The GitHub URL was supplied, and the manuscript identifies experimental source/provenance commit `f7c22e7`. The final editorial revision must still be pushed before upload.

## Scientific revisions

| Area | Revision | Evidence |
|---|---|---|
| Positioning | Retained empirical model/preprocessing evaluation, computational trade-offs and cross-model transfer errors; removed a mandatory-adaptation narrative | Main questions, results and locked/secondary protocols |
| E1 | Sequence configuration plus training-recipe replacement; E0 encoder/cache reused | `configs/experiments_v2.json`, `src/sleeptcn/experiment.py` |
| E2 | 75-dimensional C/P/N to 128-dimensional current-epoch package, including encoder objective/optimisation/selection differences | Config, models/features/training code |
| Statistics | Corrected positive E3−E2 pooled CI versus non-significant subject-wise Wilcoxon; separated pooled/subject-mean metrics and Holm families | Gate5 seed42/123 JSON and statistics code |
| E6 | Whole-record, label-free transductive normalisation; tested pipeline failed to remedy N3, not an exclusion of amplitude or all label-free remedies | Preprocessing code and locked SHHS predictions |
| Transfer gap/oracle | Descriptive EDF OOF versus SHHS ensemble score difference; oracle remains label-using and non-achievable, non-additive | Confusion counts, ensemble code, oracle calculation |
| Central finding | Added E0/E3/E6 N3 table while preserving E3 aggregate gain, 3.76× forward speed-up, parameter/memory costs and observed training time | Primary SHHS results and Gate6 artifacts |
| Methods | Added actual layer/train tables, context and masks, source-only weight inference, target-record statistics, region definitions/supports | Implementation and protocol snapshots |
| Supporting evidence | Added existing paired SHHS extension tables and verified existing seed123 boundary diagnostics; no new analysis generated | Existing JSON/MD and source hashes |
| Clipping | Used SHHS-specific archived numeric audit rather than extrapolating EDF E5 identity to SHHS | `configs/shhs_v1_protocol.json` numeric audit |
| Rounding | Original SHHS transition E3−E6 delta .0182565623 rounds to .0183; joint oracle gap fraction rounds to 102.3% | Original analysis and unrounded confusion-based calculation |
| Literature | Added Davidson et al. 2025 N3 scoring context, removed unsupported direct amplitude inference from age-stage meta-analysis | Verified primary SLEEP paper, DOI 10.1093/sleep/zsaf063 |
| Provenance | Original manifest/snapshot identity and 540 ensemble files verified; current and historical N3 audit hashes distinguished | `Reports/SHHS_PROTOCOL_PROVENANCE.md`, `Reports/ARTIFACT_INDEX.md` |
| Presentation | English-only reproducible figure derivatives, S-table numbering/cross-references, visible keywords/PDF metadata and highlights synchronised | Source and rendered PDFs |

## Verification scope

- Primary SHHS: 540/540 ensemble hashes match; confusion matrices recomputed per recording and pooled counts match the archived manifest. Primary analysis and test-gate hashes also match.
- Existing seed123 boundary evidence: EDF campaign identity and 20 E0/E3 fold-prediction hashes; SHHS extension manifest and 360 E0/E3 ensemble hashes match recorded sources.
- No claim of verifying the complete raw-input/preprocessing/checkpoint chain or regenerating predictions. See the artifact index for exact boundaries.
- Full repository suite: 150 tests passed, 19 warnings; no tests were skipped to conceal a failure. Warnings are retained in the test output rather than treated as scientific findings.
- Figure regeneration only redraws pinned existing aggregate data; it does not refit, predict or resample.
- Final PDF rendering, delivery hashes and source-bundle validation are recorded in `BSPC_FINAL_READINESS_REPORT.md` after completion.

## Deliberately not done

No new baseline, held-out cohort, adaptation, calibration, demographic aggregation or extra uncertainty analysis was run. No repository was published, no license was invented, and no journal submission was made. No acceptance probability is asserted.

The remaining author-only steps are listed in `BSPC_AUTHOR_CONFIRMATIONS_VI.md`; they are not silently replaced by boilerplate declarations.
