# Locked CUDA ADAST models: local evaluation

This evaluation does not train or select a checkpoint. It consumes the twenty
verified final CUDA checkpoints and the already locked five-adaptation/180-test
SHHS selection. The historical CPU fold-0 pilot is not imported into either
ten-fold ensemble.

The result archive has SHA-256
`1f9545cb8ad2790cf6e5f111e0124632c59582d8eaf50b9a72185fb0e6dda778`.
The CUDA training aggregate has SHA-256
`821f985f6337b5acb8a72d0aae0dd7594750cbe409c4f34d798e107110fcf9a9`.
Both training verifications passed, including exact reconstruction of the
Colab float32 uniform initialization and Linux int64 source permutation bytes.
No training code, input, checkpoint or scientific setting was changed to pass
verification.

## Evaluation rules

- Freeze all twenty checkpoint hashes and the executed code before inference.
- CPU float32, four threads, batch size 128; maximum 18,000 seconds for the
  attempt, including scoring and independent verification.
- Read complete target EEG recordings without annotations and verify raw-EDF
  hashes. No target-test signals or labels are sent to Colab.
- Apply source attention to source-only, target attention to ADAST; combine
  heads by maximum logits, softmax separately per fold, then average ten
  probability arrays in float64 and cast the mean to float32.
- Only after all target predictions are saved, open target references and apply
  the existing historical benchmark mask. Expected support: 183,528 inferred
  epochs, 169,012 scored valid epochs, 180 people.
- Source diagnostics use the held-out subjects of each outer fold, with source
  attention for both arms. Aggregate once across 78 people and 195,469 epochs;
  do not call this source result a ten-model ensemble.
- Use paired participant bootstrap (10,000 resamples, seed 2031); keep its
  inferential family separate from the historical primary tests. These
  intervals do not include training variability.
- Save all target fold probabilities and source OOF probabilities. A separate
  process verifies hashes, recomputes every confusion matrix and bootstrap,
  and replays 40 target model-record pairs plus 20 source model-batch pairs.
- Keep completed artifacts if stopped. Do not automatically resume/restart,
  train new models, extend the budget or change frozen code/configuration.

## Scope

The ADAST implementation uses pinned upstream model classes and the existing
harmonized preprocessing/protocol. It is a limited-budget, one-seed, post-hoc
comparison on the previously examined SHHS cohort, not an untouched holdout,
not a reproduction of the published ADAST dataset protocol, and not a general
ranking of all domain-adaptation methods. Each training epoch has 38 updates
(4,864 source epochs), not a full source-data pass.

The user authorized this evaluation after training verification. The runner is
`scripts/run_adast_10fold_local_evaluation.py`; private manifests, predictions,
logs and progress remain under ignored `runs`. Only verified cohort aggregates
may be exported to the report/manuscript. New CUDA timings do not replace the
paper's historical operational benchmark.
