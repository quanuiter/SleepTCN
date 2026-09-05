# SHHS1 E6 per-class reanalysis

## Scope

This is a descriptive reanalysis of the locked SHHS1 no-weight-update ensemble artifacts
stored on the USB drive:

E6 computes its mean and standard deviation from the complete filtered and clipped
target recording without labels. It is therefore label-free target-record
normalisation with a transductive dependency, not purely inductive zero-shot
inference. “Zero-shot” names the locked no-weight-update campaign and does not
remove this qualification.

- `E:\research\Dataset\SHHS_v1\zero_shot_v1\test\run_manifest.json`
- `E:\research\Dataset\SHHS_v1\zero_shot_v1\test\test_gate.json`

The campaign is complete and the test gate is marked as passed. The analysis
contains 180 records/subjects and 169,012 valid epochs. All 180 E6 artifact
hashes matched the run manifest, and the per-record confusion matrices computed
from the NPZ predictions matched the manifest values.

## Important provenance note

The protocol hash in the USB run manifest is
`165d7cdf614ff071da7bd5ca94eb4e52dd8bee1ce5eafb712c2c8a0d0550fe93`, which matches the retained
historical snapshot `configs/shhs_zero_shot_v1.json`. The currently checked-out expanded audit record
`configs/shhs_v1_protocol.json` hashes to
`9541e2334cdae98b5d36b95a2656993cdaaffb35a3160dad70af11d14b653fe9` and is not a replacement for the
historical snapshot. The reconciliation is documented in `Reports/SHHS_PROTOCOL_PROVENANCE.md`.
This file remains a direct descriptive reanalysis of the locked predictions, not a newly regenerated
report.

## Overall pooled results

| Experiment | Accuracy | Macro-F1 | Cohen's kappa | N3 precision | N3 recall | N3 F1 | N3 predicted/true |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.6648 | 0.5761 | 0.5339 | 0.9007 | 0.2610 | 0.4048 | 0.2898 |
| E3 | 0.7016 | 0.6099 | 0.5801 | 0.9440 | 0.2582 | 0.4055 | 0.2735 |
| E6 | 0.6742 | 0.5732 | 0.5408 | 0.9052 | 0.2005 | 0.3283 | 0.2215 |
| E6 − E3 | −0.0274 | −0.0367 | −0.0392 | −0.0389 | −0.0577 | −0.0772 | −0.0520 |

E6 subject-level macro-F1 is 0.5407 (SD 0.1213), compared with 0.5680 (SD
0.1186) for E3. E6 wins on 55 of 180 subjects and loses on 125.

### E6 per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| W | 0.8584 | 0.8207 | 0.8392 | 36,983 |
| N1 | 0.2214 | 0.5181 | 0.3102 | 7,002 |
| N2 | 0.7145 | 0.7161 | 0.7153 | 76,287 |
| N3 | 0.9052 | 0.2005 | 0.3283 | 22,806 |
| REM | 0.5807 | 0.8007 | 0.6732 | 25,934 |

## Transition-region analysis

Using the existing radius-1 transition definition, 44,744 epochs were selected;
7,563 of them had N3 as the true label.

| Experiment | Transition Macro-F1 | N3 precision | N3 recall | N3 F1 |
|---|---:|---:|---:|---:|
| E3 | 0.4689 | 0.8306 | 0.0733 | 0.1346 |
| E6 | 0.4507 | 0.7591 | 0.0721 | 0.1316 |

## Interpretation for the manuscript

The E0 control shows that the N3 failure is not specific to E3: E0 and E3 have
near-identical N3 recall (0.2610 and 0.2582), and 16,480 (72.3%) versus 16,674
(73.1%) of 22,806 reference N3 epochs are predicted as N2. E3 improves pooled
performance overall but does not materially improve N3 detection over the
15-CNN--BiLSTM control.

The pre-specified follow-up hypothesis that per-record z-scoring would rescue
N3 detection on SHHS1 is not supported. E6 has lower overall N3 recall than E3
(0.2005 vs 0.2582), lower N3 F1 (0.3283 vs 0.4055), and remains strongly
under-emitting for N3 (predicted/true ratio 0.2215). In the transition region,
N3 recall is essentially unchanged (0.0721 vs 0.0733).

The defensible claim is consequently that per-record z-scoring does not resolve
the cross-model N3 failure observed on SHHS1. The result is useful as
a negative control/sensitivity analysis, but should not be presented as an
improvement or as proof that absolute amplitude scale alone explains the domain
gap.
