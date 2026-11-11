"""Synthetic checks of the Amendment A3 analysis functions (no study data)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a3_analyze", ROOT / "scripts/a3_analyze.py")
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3)


def synthetic(seed: int, subjects: int = 12, epochs: int = 400):
    rng = np.random.default_rng(seed)
    records, probabilities = {}, {}
    for s in range(subjects):
        for night in range(2):
            key = f"S{s:02d}{night}"
            y = rng.integers(-1, 5, size=epochs)
            sd = np.where(y == 3, rng.normal(22, 4, epochs), rng.normal(11, 3, epochs)).clip(2, None)
            p = rng.dirichlet(np.ones(5), size=epochs)
            records[key] = {"y": np.where(y < 0, 0, y), "valid": y >= 0, "index": np.arange(epochs),
                            "sd": sd, "subject": key[:3]}
            probabilities[key] = p
    return records, probabilities


def test_metrics_match_sklearn():
    records, probabilities = synthetic(1)
    predictions = {k: p.argmax(axis=1) for k, p in probabilities.items()}
    subjects, matrices = a3.subject_confusions(records, predictions)
    assert len(subjects) == 12
    y = np.concatenate([records[k]["y"][records[k]["valid"]] for k in records])
    pred = np.concatenate([predictions[k][records[k]["valid"]] for k in records])
    values = a3.all_values(matrices)
    assert np.isclose(values["pooled_macro_f1"], f1_score(y, pred, average="macro"))
    assert np.isclose(values["accuracy"], accuracy_score(y, pred))
    assert np.isclose(values["kappa"], cohen_kappa_score(y, pred))
    assert np.isclose(values["recall_N3"], recall_score(y, pred, average=None)[3])
    per_subject = []
    for subject in subjects:
        keys = [k for k in records if records[k]["subject"] == subject]
        ys = np.concatenate([records[k]["y"][records[k]["valid"]] for k in keys])
        ps = np.concatenate([predictions[k][records[k]["valid"]] for k in keys])
        per_subject.append(f1_score(ys, ps, average="macro", labels=range(5), zero_division=0))
    assert np.isclose(values["subject_mean_macro_f1"], np.mean(per_subject))


def test_bootstrap_matches_explicit_resampling():
    records, probabilities = synthetic(2)
    _, matrices = a3.subject_confusions(records, {k: p.argmax(axis=1) for k, p in probabilities.items()})
    weights = a3.bootstrap_weights(len(matrices))
    assert weights.shape == (a3.RESAMPLES, len(matrices)) and np.all(weights.sum(axis=1) == len(matrices))
    rng = np.random.default_rng(a3.SEED)
    sampled = rng.integers(0, len(matrices), size=(a3.RESAMPLES, len(matrices)))
    resampled = a3.all_values(matrices, weights)
    for row in (0, 7, 123):
        explicit = a3.all_values(matrices[sampled[row]])
        for name in ("subject_mean_macro_f1", "pooled_macro_f1", "recall_N3", "n3_to_n2_rate", "kappa"):
            assert np.isclose(resampled[name][row], explicit[name])


def test_rules():
    records, probabilities = synthetic(3)
    fit = a3.fit_source(records, sorted(records))
    assert 11 < fit["threshold_uv"] < 22 and fit["slope"] > 0
    key = sorted(records)[0]
    p, sd = probabilities[key], records[key]["sd"]
    base = p.argmax(axis=1)
    hard = a3.rule_hard(p, sd, fit)
    changed = hard != base
    assert np.all(base[changed] == 2) and np.all(hard[changed] == 3) and np.all(sd[changed] >= fit["threshold_uv"])
    assert np.all(hard[(base == 2) & (sd >= fit["threshold_uv"])] == 3)
    assert np.array_equal(p.argmax(axis=1), base)  # input probabilities are not modified in place
    soft = a3.rule_soft(p, sd, fit)
    # Redistributing the N2+N3 mass can change the winning class in either direction, but an epoch can
    # only become N2 or N3 if that mass exceeds every other class, and can only leave N2 or N3.
    moved = soft != base
    assert np.all(np.isin(soft[moved], (2, 3)) | np.isin(base[moved], (2, 3)))
    # a likelihood ratio of one leaves every prediction unchanged
    neutral = {**fit, "slope": 0.0, "intercept": fit["log_prior_odds"]}
    assert np.array_equal(a3.rule_soft(p, sd, neutral), base)


def test_difference_and_verdict():
    records, probabilities = synthetic(4)
    fit = a3.fit_source(records, sorted(records))
    result, matrices = a3.analyse_arm(records, probabilities, {k: fit for k in records}, {})
    assert result["subjects"] == 12
    delta = result["R1_hard_minus_unmodified"]
    assert delta["recall_N3"]["value"] > 0 and delta["recall_N3"]["ci95"][0] > 0
    assert result["R1_hard_verdict"] in {"improves", "trade-off", "harms", "no detectable change"}
    zero = a3.difference(matrices["unmodified"], matrices["unmodified"], a3.bootstrap_weights(12))
    assert all(v["value"] == 0 and v["ci95"] == [0.0, 0.0] for v in zero.values())
