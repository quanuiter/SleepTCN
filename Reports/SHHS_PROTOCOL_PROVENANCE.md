# SHHS protocol provenance reconciliation

Status (2026-09-05): protocol-snapshot identity reconciled; all 540 primary-campaign ensemble
prediction files checked against the original run manifest; no predictions were regenerated.

The SHHS run manifest records protocol SHA-256
`165d7cdf614ff071da7bd5ca94eb4e52dd8bee1ce5eafb712c2c8a0d0550fe93`. That hash matches the
repository file `configs/shhs_zero_shot_v1.json` byte-for-byte. This file is the historical locked
snapshot linked by the original E0/E3/E6 no-weight-update run manifest. The snapshot describes the
campaign to be locked before validation inference. Its matching hash establishes file identity and
the recorded linkage; a hash alone does not establish the time of locking or public preregistration.

The repository also contains `configs/shhs_v1_protocol.json`, whose SHA-256 is
`9541e2334cdae98b5d36b95a2656993cdaaffb35a3160dad70af11d14b653fe9`. It is a richer post-run audit
record containing technical checks, processed-data summaries and completed-test metadata. It is not a
replacement for the historical snapshot and must not be substituted for the run-manifest protocol hash.

## Direct read-only verification on 2026-09-05

The external files were accessible during the revision, although they were not accessible during the
earlier manuscript review. The original run manifest at
`E:/research/Dataset/SHHS_v1/zero_shot_v1/test/run_manifest.json` has SHA-256
`f9cd5ebbd20f26b188b5dc13ac6e417ff8ef0fa8dcae78760cfcb27940bf58cf` and records the historical protocol
hash above. The following checks were performed without altering any scientific input or output:

- All 540 listed ensemble NPZ files matched their recorded SHA-256: 180 each for E0, E3 and E6.
- Confusion matrices were recomputed from `y`, `prediction` and `valid_mask` in every file. Every
  per-record matrix matched the manifest, and their sums matched the three stored pooled matrices.
- Each model had 169,012 valid epochs and 22,806 reference N3 epochs. N3→N2 counts were 16,480,
  16,674 and 17,764 for E0, E3 and E6; N3 recalls were 0.2610277997, 0.2581776725 and 0.2005174077.
- The primary aggregate analysis matched SHA-256
  `83aa53fed3dc7be9b6f14cb63ddbd7417a7af256b9f308383500ee6e068943df`, and the original test gate
  matched `51828329b2ebb2d99e5d71d6b9c78fd5a3fad037162fa50855af52066e4d2646`.

This check did not regenerate predictions, rerun training, recompute bootstrap intervals, or verify the
5,400 individual-fold prediction files and the complete checkpoint/raw-input/preprocessing chain.
Those are separate reproducibility checks. The externally held recordings and predictions are not
included in the repository; any sharing must follow the applicable NSRR/data-use terms. A public or
reviewer-accessible archive URL, version and permitted contents still require author confirmation.

## N3 audit version history

`Reports/SHHS_E0_E3_E6_N3_AUDIT.json` currently has SHA-256
`3333b5f12788e592323f05a3a1514fdd3fe724832fcc34e0419669c57223b0db`. The earlier hash
`6b12447f27cf71f8a7b7c100919ab5438dc69fc3efd9f4d7a1439c3f29b6496b` identifies an earlier version,
not the current file. The change recorded in commit `1710c49` added provenance metadata **and**
corrected the derived E3 N3→N2 fraction from `0.7310356923616592` to
`0.7311233885819521`, equal to `16674 / 22806`. The underlying confusion counts were unchanged.
It would therefore be inaccurate to describe this as a metadata-only change. Neither version is a
replacement for the original prediction files or run manifest. Current inline references should use
the current hash; the earlier value is retained here as a historical identifier.

## Protocol interpretation and seed chronology

The historical experiment configuration lists seeds 42, 123 and 2025. The reported campaigns contain
42 and 123; current reports describe seed 123 as a sensitivity repeat performed after seed-42 results
were viewed. No seed-2025 result is reported. Listing a seed in a configuration does not establish that
it was run, or retrospectively turn a later analysis into independent confirmation. The authors still
need to confirm the decision chronology and any reason for not completing a listed seed. Historical
configuration files must not be rewritten to make this account appear simpler.

Likewise, labels such as `preregistered_before_gate8_training` in local configuration files describe
internal planning, not evidence of public registration. Current prose uses *pre-specified* for contrasts
defined before their corresponding analysis and explicitly labels later extensions as secondary or
post-hoc. The E1 contrast replaces both sequence-model configuration and training recipe; E2 replaces
the feature/context package and encoder-training recipe. Older shorthand labels in immutable
configuration files do not establish architecture-only effects.

## Scope of the reconciliation

File integrity, recorded provenance and independent regeneration are distinct. This reconciliation
resolves the apparent protocol-hash mismatch and corroborates the primary ensemble predictions and
confusion counts. It does not certify full end-to-end reproducibility, public preregistration, clinical
validity or a remedy for the N3 error. Preserve both protocol snapshots and this record in the
reproducibility archive; journal upload files should be assembled separately.
