"""Post-hoc diagnostics and label-independent signal preparation, without training."""
from __future__ import annotations

import numpy as np

from .preprocessing import preprocess_signal_variant
from .shhs_preprocessing import SHHSPreprocessConfig, resample_continuous_eeg


def confusion_diagnostics(matrices: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics for one or a batch of 5x5 confusion matrices (rows = reference).

    Undefined precision/recall/rates are NaN; F1 uses zero_division=0 for
    comparability with the existing five-stage macro-F1 definition.
    """
    cm = np.asarray(matrices, dtype=np.float64)
    if cm.shape[-2:] != (5, 5) or not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("expected finite nonnegative (...,5,5) confusion counts")
    support, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                         where=denominator > 0)

    return {
        "macro_f1": f1.mean(axis=-1), "n3_f1": f1[..., 3],
        "n3_recall": ratio(tp[..., 3], support[..., 3]),
        "n3_precision": ratio(tp[..., 3], predicted[..., 3]),
        "n2_recall": ratio(tp[..., 2], support[..., 2]),
        "n3_to_n2_rate": ratio(cm[..., 3, 2], support[..., 3]),
        "n2_to_n3_rate": ratio(cm[..., 2, 3], support[..., 2]),
    }


def paired_diagnostic_bootstrap(left: np.ndarray, right: np.ndarray, *,
                                resamples: int = 10000, seed: int = 2031) -> dict:
    """Paired subject-cluster percentile CIs for pooled diagnostic differences."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 3 or not len(left):
        raise ValueError("expected paired non-empty (subjects,5,5) arrays")
    confusion_diagnostics(left)
    confusion_diagnostics(right)
    if not np.array_equal(left.sum(-1), right.sum(-1)):
        raise ValueError("reference class support differs between paired subjects")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    observed_left = confusion_diagnostics(left.sum(0))
    observed_right = confusion_diagnostics(right.sum(0))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(left), size=(resamples, len(left)), dtype=np.int32)
    distributions = {key: [] for key in observed_left}
    for start in range(0, resamples, 256):
        selected = draws[start:start+256]
        a = confusion_diagnostics(left[selected].sum(1))
        b = confusion_diagnostics(right[selected].sum(1))
        for key in distributions:
            distributions[key].extend(a[key] - b[key])
    output = {}
    for key, values in distributions.items():
        values = np.asarray(values)
        defined = np.isfinite(values)
        if defined.sum():
            low, high = np.quantile(values[defined], [0.025, 0.975])
            interval = [float(low), float(high)]
        else:
            interval = None
        delta = observed_left[key] - observed_right[key]
        output[key] = {"left": float(observed_left[key]) if np.isfinite(observed_left[key]) else None,
                       "right": float(observed_right[key]) if np.isfinite(observed_right[key]) else None,
                       "difference": float(delta) if np.isfinite(delta) else None,
                       "ci95": interval, "defined_bootstrap_draws": int(defined.sum())}
    return {"unit": "subject", "subjects": len(left), "resamples": resamples,
            "seed": seed, "estimand": "difference_of_pooled_metrics",
            "scope": "exploratory_same_cohort_fixed_checkpoints_not_confirmatory",
            "metrics": output}


def label_free_signal_variants(continuous_uv: np.ndarray,
                               config: SHHSPreprocessConfig) -> dict:
    """All complete recording epochs, selected solely from signal length.

    No annotations, sleep boundaries or target valid-label masks are accepted.
    E3/E4 here use the full record: future inference baselines must use the same
    context, not substitute historical trimmed-window predictions.
    """
    resampled = resample_continuous_eeg(continuous_uv, config)
    result = {}
    for variant in ["filtered_v2", "bandpass_v2"]:
        signal, clip_fraction, metadata = preprocess_signal_variant(
            resampled, variant, config.sleepedf_compatible_config())
        result[variant] = (signal.reshape(-1, config.target_samples_per_epoch),
                           clip_fraction, metadata)
    return result
