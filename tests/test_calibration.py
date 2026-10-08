import numpy as np
import pytest

from sleeptcn.calibration import (
    adjust_prior, apply_temperature, estimate_target_prior, fit_temperature,
    probabilities, source_class_weights,
)


def test_temperature_identity_and_invalid_inputs():
    p = np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])
    np.testing.assert_allclose(apply_temperature(p, 1), p)
    for invalid in [p * 2, np.full((1, 5), np.nan), np.empty((0, 5)), p[:, :4]]:
        with pytest.raises(ValueError):
            probabilities(invalid)
    for t in [0, -1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            apply_temperature(p, t)


def test_calibration_improves_nll_for_overconfident_synthetic_predictions():
    # Same prediction for each group of ten: 6/10 correct, 1/10 for each other class.
    y = np.concatenate([np.array([c] * 6 + [j for j in range(5) if j != c])
                        for c in range(5)])
    p = np.full((50, 5), 0.005)
    p[np.arange(50), np.repeat(np.arange(5), 10)] = 0.98
    temperature = fit_temperature(p, y)
    q = apply_temperature(p, temperature)
    assert temperature > 1
    assert -np.log(q[np.arange(50), y]).mean() < -np.log(p[np.arange(50), y]).mean()
    np.testing.assert_array_equal(q.argmax(1), p.argmax(1))
    with pytest.raises(ValueError, match="filter ignored"):
        fit_temperature(p, np.full(50, -1, dtype=int))
    with pytest.raises(ValueError, match="all five"):
        fit_temperature(p, np.zeros(50, dtype=int))


def test_em_recovers_known_shift_and_monotone_likelihood():
    # Finite observation space. Columns P(x|class) sum to one, unchanged across domains.
    conditional = np.full((5, 5), 0.05)
    np.fill_diagonal(conditional, 0.8)
    source = np.full(5, 0.2)
    target = np.array([0.1, 0.15, 0.3, 0.35, 0.1])
    source_joint = conditional * source
    posterior = source_joint / source_joint.sum(axis=1, keepdims=True)
    counts = np.rint(10000 * (conditional @ target)).astype(int)
    predictions = np.repeat(posterior, counts, axis=0)
    result = estimate_target_prior(predictions, source)
    assert result.converged
    np.testing.assert_allclose(result.target_prior, target, atol=1e-6)
    assert np.min(np.diff(result.log_likelihood)) >= -1e-12
    corrected = adjust_prior(predictions, source, result.target_prior)
    np.testing.assert_allclose(corrected.sum(axis=1), 1)


def test_em_no_shift_identity_and_non_convergence_is_explicit():
    p = np.full((5, 5), 0.05)
    np.fill_diagonal(p, 0.8)
    source = np.full(5, 0.2)
    result = estimate_target_prior(p, source)
    assert result.converged
    np.testing.assert_allclose(result.target_prior, source)
    np.testing.assert_allclose(adjust_prior(p, source, source), p)
    result = estimate_target_prior(p[:1], source, max_iterations=1)
    assert not result.converged
    for prior in [np.zeros(5), np.ones(5), [0.25, 0.25, 0.25, 0.25, 0]]:
        with pytest.raises(ValueError):
            estimate_target_prior(p, prior)


def test_weighted_loss_preparation_ignores_unknown_but_rejects_missing_classes():
    weights = source_class_weights(np.array([-1, 0, 1, 2, 3, 4, 2, 2]))
    np.testing.assert_allclose(weights, [1.4, 1.4, 7/15, 1.4, 1.4])
    with pytest.raises(ValueError):
        source_class_weights(np.array([0, 1, 2, 3]))


def test_exact_zero_probabilities_remain_finite():
    p = np.eye(5)
    q = apply_temperature(p, 0.05)
    assert np.isfinite(q).all()
    result = estimate_target_prior(p, np.full(5, 0.2))
    assert result.converged
