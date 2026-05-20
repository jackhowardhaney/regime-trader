"""Mandatory look-ahead bias test for HMMEngine — Phase 2 requirement."""

import numpy as np
import pytest
from core.hmm_engine import HMMEngine


def _make_synthetic_data(n: int = 700, n_features: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, n_features))
    returns = rng.standard_normal(n) * 0.01
    return features, returns


def _train_and_get_engine(features, returns) -> HMMEngine:
    engine = HMMEngine(n_candidates=[3], n_init=3)
    engine.fit(features, returns)
    return engine


def test_no_look_ahead_bias():
    """Regime at T must be identical with data[0:T] vs data[0:T+100]."""
    features, returns = _make_synthetic_data(n=700)
    engine = _train_and_get_engine(features, returns)

    T = 400
    regimes_short = engine.predict_regime_filtered(features[0:T])
    regime_at_T = int(regimes_short[-1])

    engine._cached_alpha = None  # reset cache so second call is independent
    regimes_long = engine.predict_regime_filtered(features[0: T + 100])
    regime_at_T_in_long = int(regimes_long[T - 1])

    assert regime_at_T == regime_at_T_in_long, (
        f"LOOK-AHEAD BIAS DETECTED: "
        f"regime at T={T} with data[0:{T}] = {regime_at_T}, "
        f"but regime at T={T} with data[0:{T + 100}] = {regime_at_T_in_long}"
    )


def test_incremental_matches_batch():
    """Incremental forward updates must match the full-batch forward pass."""
    features, returns = _make_synthetic_data(n=600)
    engine = _train_and_get_engine(features, returns)

    batch_labels = engine.predict_regime_filtered(features)

    engine._cached_alpha = None
    incremental_labels = []
    for t in range(len(features)):
        label_idx, _ = engine.predict_regime_filtered_incremental(features[t])
        incremental_labels.append(label_idx)

    np.testing.assert_array_equal(
        batch_labels,
        incremental_labels,
        err_msg="Incremental forward pass diverges from batch forward pass.",
    )


def test_future_data_does_not_change_past_regime():
    """Adding 200 future bars must not alter any regime label in the original window."""
    features, returns = _make_synthetic_data(n=800)
    engine = _train_and_get_engine(features, returns)

    T = 500
    labels_short = engine.predict_regime_filtered(features[0:T])

    engine._cached_alpha = None
    labels_long = engine.predict_regime_filtered(features[0: T + 200])

    np.testing.assert_array_equal(
        labels_short,
        labels_long[:T],
        err_msg="Future data altered past regime labels — look-ahead bias present.",
    )
