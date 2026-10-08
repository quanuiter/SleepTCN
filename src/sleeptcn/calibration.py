"""CPU-only research utilities; never fit calibration or priors on target labels.

Temperature scaling is a deliberately simple baseline, not a reproduction of
Alexandari et al.'s bias-corrected calibration. EM assumes label shift and useful
source posteriors; convergence does not validate either assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

STAGES = ("W", "N1", "N2", "N3", "REM")
PROBABILITY_FLOOR = 1e-12


def probabilities(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5 or not len(p):
        raise ValueError("probabilities must have non-empty shape (epochs, 5)")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    p = np.maximum(p, PROBABILITY_FLOOR)
    return p / p.sum(axis=1, keepdims=True)


def apply_temperature(values: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scores = np.log(probabilities(values)) / temperature
    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


def fit_temperature(source_validation_p: np.ndarray, labels: np.ndarray) -> float:
    """Minimize epoch-weighted validation NLL, with fixed T bounds [0.05,20].

    Caller must supply the corresponding source fold's validation role only.
    Ignored/padded epochs must be removed explicitly before calling this function.
    """
    p = probabilities(source_validation_p)
    y = np.asarray(labels)
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be an integer vector aligned to probabilities")
    if not np.isin(y, np.arange(5)).all():
        raise ValueError("validation labels must be in 0..4; filter ignored epochs first")
    if len(np.unique(y)) != 5:
        raise ValueError("pilot requires all five stages in source validation")
    log_p = np.log(p)

    def nll(log_t: float) -> float:
        scores = log_p / np.exp(log_t)
        return float(np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(y)), y]))

    result = minimize_scalar(nll, bounds=(-np.log(20), np.log(20)), method="bounded",
                             options={"xatol": 1e-8, "maxiter": 500})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("temperature optimization did not converge")
    # Include the identity and both bounds; never worsen source-validation NLL.
    candidates = [0.0, float(result.x), -np.log(20), np.log(20)]
    return float(np.exp(min(candidates, key=nll)))


def _prior(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.shape != (5,) or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("prior must contain five finite, strictly positive values")
    if not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-6):
        raise ValueError("prior must sum to one")
    return p / p.sum()


def adjust_prior(values: np.ndarray, source_prior: np.ndarray,
                 target_prior: np.ndarray) -> np.ndarray:
    p = probabilities(values)
    source, target = _prior(source_prior), _prior(target_prior)
    scores = np.log(p) + np.log(target / source)
    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


@dataclass(frozen=True)
class EMResult:
    target_prior: np.ndarray
    iterations: int
    converged: bool
    log_likelihood: tuple[float, ...]


def estimate_target_prior(unlabelled_p: np.ndarray, source_prior: np.ndarray,
                          *, tolerance: float = 1e-8,
                          max_iterations: int = 1000) -> EMResult:
    """Epoch-weighted ML/EM on unlabelled adaptation predictions only.

    Initialization is the source prior. A 1e-12 numerical floor avoids log(0);
    no label-based smoothing, N3 multiplier, or true target prevalence is used.
    """
    p, source = probabilities(unlabelled_p), _prior(source_prior)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    log_ratio = np.log(p) - np.log(source)
    target = source.copy()
    history = [float(logsumexp(log_ratio + np.log(target), axis=1).mean())]
    for iteration in range(1, max_iterations + 1):
        updated = adjust_prior(p, source, target).mean(axis=0)
        updated = np.maximum(updated, PROBABILITY_FLOOR)
        updated /= updated.sum()
        history.append(float(logsumexp(log_ratio + np.log(updated), axis=1).mean()))
        delta = np.max(np.abs(updated - target))
        target = updated
        if delta < tolerance:
            return EMResult(target, iteration, True, tuple(history))
    return EMResult(target, max_iterations, False, tuple(history))


def source_class_weights(labels: np.ndarray) -> np.ndarray:
    """Candidate TCN loss weights N/(5*n_c), computed from training labels only.

    Preparation utility, not connected to the historical E0--E6 runner. A new
    campaign must record these weights and retrain a same-revision control.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("training labels must be a one-dimensional integer vector")
    if not np.isin(y, [-1, 0, 1, 2, 3, 4]).all():
        raise ValueError("training labels must be -1 or 0..4")
    counts = np.bincount(y[y >= 0], minlength=5)
    if (counts == 0).any():
        raise ValueError("all five training classes are required for inverse-frequency weights")
    return counts.sum() / (5.0 * counts)
