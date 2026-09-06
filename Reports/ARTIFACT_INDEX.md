# SleepTCN artifact index for the BSPC revision

Index updated: **2026-09-06**. Paths below are relative to the repository root unless explicitly marked
external. The review baseline is commit `1710c49fe529f7d498598ad25ff108163f3e6f06`; manuscript edits are
working-tree revisions, not a newly created release. No protocol, prediction or checkpoint was changed
by the document revision.

This index separates **file integrity**, **recorded provenance** and **independent regeneration**.
Hash matches identify bytes; they do not establish public preregistration, clinical validity or that
an independent party can reproduce the complete experiment.

## Access and release status

- **Repository:** code, configuration snapshots and the listed aggregate results are available in this
  checkout. This describes local availability, not a published archive or a license grant.
- **External/restricted:** original SHHS records, labels, participant-level manifests, predictions and
  checkpoints are outside the repository. Access and redistribution must be checked against the
  applicable data-use terms. Do not publish participant identifiers or these artifacts by copying this
  internal working directory into a submission archive.
- **Repository URL confirmed:** `https://github.com/quanuiter/SleepTCN` is the public code URL supplied by
  the authors. The remote baseline is older than the current local manuscript edits; push and identify the
  exact submission commit/tag before upload. Do not promise a versioned archive or license that has not been
  created.
- **Journal files versus archive:** main manuscript, supplement and requested editorial files form the
  journal upload; internal reviews, notes and this full technical inventory are not automatically upload
  items. The reproducibility archive can carry an appropriately sanitized version of this index.

## Repository scientific inputs and aggregate artifacts

The SHA-256 values in this table were read directly from the current files on 2026-09-06. They identify
the current scientific snapshots, not all earlier versions cited in historical run reports.

| Artifact | Purpose and evidence scope | SHA-256 |
|---|---|---|
| `configs/shhs_zero_shot_v1.json` | Historical primary SHHS protocol snapshot; matches the original run-manifest protocol field | `165d7cdf614ff071da7bd5ca94eb4e52dd8bee1ce5eafb712c2c8a0d0550fe93` |
| `configs/shhs_v1_protocol.json` | Expanded post-run audit record, including preprocessing checks; not a substitute for the historical snapshot | `9541e2334cdae98b5d36b95a2656993cdaaffb35a3160dad70af11d14b653fe9` |
| `configs/experiments_v2.json` | EDF experiment/training/statistical specification; interpretation and seed-chronology qualifications are recorded in the provenance note | `1d812bbfb45e9ca90e2654b41311954fd6e66a56e1bbcdbfba48df8147d0ae1b` |
| `data/splits/sleepedf_sc_10fold_seed42_v2.json` | Shared subject-wise EDF outer partitions | `6bc7ad74c07ff05f1d880cb5e720eea12386824ef465b966507906fa248925de` |
| `runs/v2/analysis/gate5_paired_results_seed42.json` | Primary EDF pooled scores, class metrics and paired contrasts | `ee6302f5ebb9781406c3a77b45c631a17b7dab5889420e6730d5b50524767cb0` |
| `runs/v2/analysis/gate5_paired_results_seed123.json` | EDF seed-123 sensitivity results, reported separately from seed 42 | `14f77acd03f146e6c812fecaf59ba35a5089870e27e9c03190f00b8fe800c743` |
| `runs/v2/analysis/multiseed_sensitivity_seed42_seed123.json` | Two-seed comparison; not independent-cohort confirmation and not pooled p-values | `3c5ccde1f01a689a3c74e6212504f9484b5be86e86fba8f0030a93334e32f72f` |
| `runs/v2/analysis/gate6_latency_fold00_seed42.json` | V100 forward-pass timing/memory benchmark, not preprocessing, I/O or online decision latency | `e1cdba979d9f434bb19803ea5341052239f0e98c89f1bff95c21287e99d81cc7` |
| `runs/v2/gate8/analysis_seed42.json` | Context-group masking/retraining analysis, supporting rather than a causal mechanism claim | `25465db4b81d02d638aa8e974b7aa4f3ec51fcb9d4190287c27b88cb3e5380e1` |
| `runs/v2/analysis/transition_regions_edf_seed42_t2.json` | Post-hoc EDF reference-label-region diagnostic | `d323e30937be0b3a48a44b7333f5ccced47631eb22bfb62d09136474c9c397e3` |
| `runs/v2/analysis/transition_regions_shhs_t2.json` | Post-hoc SHHS reference-label-region diagnostic | `d91e45144e037f0b5aee7c3a9e4fd7ca353d5a218511793f847e9307ffbd25c3` |
| `Reports/SHHS_E0_E3_E6_N3_AUDIT.json` | Descriptive pooled N3 audit; underlying primary confusion counts reverified from ensemble predictions | `3333b5f12788e592323f05a3a1514fdd3fe724832fcc34e0419669c57223b0db` |
| `Reports/SHHS_E3_E2_PAIRED_AUDIT.json` | Post-hoc E3−E2 contrast on the already evaluated SHHS sample | `d654e4f47140ae3f2a35ae7737b98c5ba0ee4a2e5dc45242c5171de2bd9d938a` |
| `Reports/POSTHOC_E3_E0_AUDIT.json` | Post-hoc EDF full-pipeline contrast, outside the primary Holm family | `fb56c7de13f89daed6a86f44ea26751dd7a63d7f284d194b36255ba626e532bb` |

The EDF confidence intervals and most secondary diagnostics were checked against stored aggregate
outputs, not independently rerun from all original EDF predictions during this documentation task.
The seed-123 region summaries used in Supplementary Table S16 were separately checked during this
revision. Their existing source identities match the EDF campaign and 20 E0/E3 fold-prediction files,
and the SHHS extension manifest and 360 E0/E3 ensemble files. No diagnostic was rerun to create this
table. The local aggregate files are currently excluded by the repository's run-artifact ignore rules:

| Existing local summary | SHA-256 |
|---|---|
| `runs/v2/analysis/transition_regions_edf_seed123_t2.json` | `b45cf0c75a66315d3e8767c596c2d98c20f4d5527d4c02dc5d3c27c892b43b93` |
| `runs/v2/analysis/transition_regions_shhs_seed123_t2.json` | `f7565f8801dfe25fdf5005f6427043272401dd5274557a601a615969a3264b85` |

They are not included in the journal source bundle. Review their embedded input paths/identifiers and
data-use restrictions before distributing any aggregate source file; a sanitized reviewer export may
be appropriate. A manifest entry here records local identity, not permission to publish the contents.

## External primary SHHS artifacts directly checked

External campaign root: `E:/research/Dataset/SHHS_v1/zero_shot_v1/`.
The following read-only checks were completed during the revision on 2026-09-05. The earlier review's
statement that this drive was inaccessible describes that earlier review, not the later verification.

| External path relative to the campaign root | Direct check | SHA-256 |
|---|---|---|
| `test/run_manifest.json` | File hash and protocol linkage read; 540 ensemble records examined | `f9cd5ebbd20f26b188b5dc13ac6e417ff8ef0fa8dcae78760cfcb27940bf58cf` |
| `test/test_gate.json` | File hash checked; stored historical gate result, not a new full-gate rerun | `51828329b2ebb2d99e5d71d6b9c78fd5a3fad037162fa50855af52066e4d2646` |
| `analysis/zero_shot_analysis.json` | File hash and reported aggregate values checked; no bootstrap rerun | `83aa53fed3dc7be9b6f14cb63ddbd7417a7af256b9f308383500ee6e068943df` |
| 540 ensemble NPZ files listed by `ensemble_records` | Every individual SHA-256 matched; every confusion matrix recomputed from stored labels, predictions and masks | Individual hashes remain in the restricted original manifest; identifiers are not reproduced here |

Recomputed matrices used class order `W, N1, N2, N3, REM` and the valid-label mask. Per-record matrices
matched the manifest; their sums matched the pooled matrices. Aggregate-only verification readout:

| Model | Files/hash matches | Valid epochs | Reference N3 epochs | N3→N2 count | N3 recall |
|---|---:|---:|---:|---:|---:|
| E0 | 180/180 | 169,012 | 22,806 | 16,480 | 0.2610277997 |
| E3 | 180/180 | 169,012 | 22,806 | 16,674 | 0.2581776725 |
| E6 | 180/180 | 169,012 | 22,806 | 17,764 | 0.2005174077 |

This did **not** verify all 5,400 individual-fold prediction files, regenerate the ensemble from those
files, recheck the complete checkpoint/preprocessed-input/raw-recording chain, retrain models, rerun
inference or recompute bootstrap confidence intervals. A complete replay needs those additional inputs
and checks; the successful ensemble audit must not be reported as if it had performed them.

## Additional campaign sources

The following records guide retrieval of supporting evidence. Hashes referenced by these historical
reports are not automatically reverified here just because the report itself is present.

| Scope | Repository record / external location | Existing analysis entry point |
|---|---|---|
| SHHS E1/E2 secondary extension | `docs/SHHS_COMPONENT_EXTENSION_RESULTS.md`; external `SHHS_v1/zero_shot_components_v1/` | `scripts/analyze_shhs_component_extension.py` |
| SHHS E3−E2 post-hoc analysis | `Reports/SHHS_E3_E2_PAIRED_AUDIT.json`; `configs/shhs_e3_e2_paired_v1.json` | `scripts/analyze_shhs_e3_e2.py` |
| SHHS seed-123 E4 extension | `Reports/SHHS_SEED123_E4_EXTENSION.md`; external `SHHS_v1/zero_shot_e4_seed123_v1/` | `scripts/analyze_shhs_bandpass_extension.py` |
| E6 class-level descriptive reanalysis | `Reports/SHHS_E6_PER_CLASS_REANALYSIS.md`; primary ensemble files above | Recompute class counts from those files; keep the original prediction/manifest bytes unchanged |
| EDF region diagnostics | Region JSON files above and original EDF campaign predictions | `scripts/analyze_transition_regions_edf.py` |
| SHHS region diagnostics | Region JSON above and the relevant original run manifest/predictions | `scripts/analyze_transition_regions_complete.py` |
| EDF paired analysis and seed comparison | Gate-5 JSON files and complete EDF prediction workspace | `scripts/analyze_paired_results.py`, `scripts/analyze_seed_sensitivity.py` |
| Gate-8 context diagnostics | Gate-8 campaign predictions and `runs/v2/gate8/analysis_seed42.json` | `scripts/analyze_gate8_results.py` |

## Verification and replay instructions

Run from the repository root with the project's verified Python environment. Read each entry point's
`--help` before use. Do not run a training or analysis campaign merely because this index lists it;
the author requested notification before additional experiments or new analyses.

Safe read-only examples:

```powershell
Get-FileHash -Algorithm SHA256 configs/shhs_zero_shot_v1.json
Get-FileHash -Algorithm SHA256 Reports/SHHS_E0_E3_E6_N3_AUDIT.json
.venv/Scripts/python.exe scripts/validate_shhs_zero_shot.py --help
.venv/Scripts/python.exe scripts/analyze_shhs_zero_shot.py --help
.venv/Scripts/python.exe scripts/analyze_paired_results.py --help
```

For an authorized replay, create a new output directory and retain the old files. The relevant CLIs are:

- `validate_shhs_zero_shot.py --run-manifest <original-manifest> --output <new-gate.json>` checks
  fold and ensemble artifacts and alignment; it is broader than the ensemble-only check above.
- `analyze_shhs_zero_shot.py --run-manifest <original-manifest> --test-gate <validated-gate>
  --protocol <matching-historical-protocol> --output <new-analysis.json>` recalculates the locked
  analysis. It also writes a sidecar hash; neither output should replace the historical analysis.
- `analyze_paired_results.py --workspace <complete-EDF-workspace> --seed <42-or-123>
  --bootstrap-resamples 10000 --bootstrap-seed 2026 --output <new-analysis.json>` requires complete
  campaign artifacts and a clean, readable Git worktree. Do not bypass its clean-worktree guard or
  overwrite this revision to satisfy it; use a separate authorized replay workspace.
- Region scripts take explicit output JSON/CSV/Markdown paths. Use a versioned output set; record the
  chosen cohort, seed, masks and source hashes. A new analysis of an already inspected test cohort is
  post-hoc, even when its new script/config has a hash.

For exact historical reproduction, also preserve the effective run configuration, code revision,
dependency/environment lock, split and device/software details. Current source defaults alone do not
prove which settings produced a historical checkpoint.

## Version history and interpretation safeguards

- The two SHHS protocol hashes above refer to different roles, not a mismatch to be repaired by
  rewriting JSON. See `SHHS_PROTOCOL_PROVENANCE.md`.
- The earlier N3 audit hash `6b12447f27cf71f8a7b7c100919ab5438dc69fc3efd9f4d7a1439c3f29b6496b`
  is historical. Commit `1710c49` added provenance metadata and corrected the derived E3 N3→N2 fraction
  to `16674/22806 = 0.7311233885819521`; confusion counts did not change. This was not metadata-only.
- An older Gate-5 report hash in `docs/GATE5_STATISTICAL_RESULTS.md` identifies its reported historical
  version; the current aggregate hash is listed above. Do not relabel an old hash as a current-file
  check or silently replace a run's expected hash.
- E1 reuses the encoder/cache but changes sequence-model configuration and training recipe. E2 changes
  feature/context package and encoder recipe. Neither is architecture-only. Older shorthand labels in
  immutable configs do not override the implemented design.
- E6 estimates statistics from the full unlabelled target recording. It is label-free but transductive;
  its failure to remedy N3 in this pipeline does not exclude amplitude-related contributors or all
  other normalization/adaptation methods.
- EDF scores are out-of-fold and SHHS scores use a ten-model ensemble. The oracle refers to the
  observed pooled-score difference between these evaluations, not a pure causal domain-shift effect.

## Publication files

The 06 September 2026 source/PDF delivery is identified by `Reports/REPORT_MANIFEST.sha256` and
`Reports/paper_en/SUBMISSION_MANIFEST.sha256`. Build results, all-page visual checks and independent
source-bundle validation are recorded in `Reports/BSPC_FINAL_READINESS_REPORT.md`. The portable ZIP is
`Reports/output/source/SleepTCN_BSPC_Manuscript_Source.zip`. Manifest membership is a file list, not
proof of complete experimental replay. Update the publication manifests again after any subsequent
author changes; do not change expected hashes in historical scientific run records.

Before the authors submit: verify the current BSPC upload requirements, push/tag the manuscript revision
represented by the PDFs, approve the final PDFs, and read the final readiness report. No manuscript was
submitted by creating this index.
