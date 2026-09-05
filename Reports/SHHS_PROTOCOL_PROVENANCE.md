# SHHS protocol provenance reconciliation

Status: protocol-snapshot provenance reconciled; no predictions were regenerated.

The SHHS run manifest records protocol SHA-256
`165d7cdf614ff071da7bd5ca94eb4e52dd8bee1ce5eafb712c2c8a0d0550fe93`. That hash matches the
repository file `configs/shhs_zero_shot_v1.json` byte-for-byte. This file is the historical locked
snapshot used to define the E0/E3/E6 no-weight-update campaign before validation inference.

The repository also contains `configs/shhs_v1_protocol.json`, whose SHA-256 is
`9541e2334cdae98b5d36b95a2656993cdaaffb35a3160dad70af11d14b653fe9`. It is a richer post-run audit
record containing technical checks, processed-data summaries and completed-test metadata. It is not a
replacement for the historical snapshot and must not be substituted for the run-manifest protocol hash.

This reconciliation resolves the apparent protocol-hash mismatch at the snapshot level. It does not
claim that the locked SHHS predictions have been regenerated locally: the raw SHHS data and prediction
artifacts remain outside the repository under the NSRR/data-use constraints, and E6 remains a descriptive
reanalysis of locked predictions. The current source protocol and this record should be preserved with
the submission package.
