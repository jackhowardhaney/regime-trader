"""Tests for HMMEngine regime detection."""

import pytest
import numpy as np
from core.hmm_engine import HMMEngine


class TestHMMEngine:
    """Unit tests for HMMEngine fit, predict, and regime labeling."""

    def test_fit_returns_self(self) -> None:
        """fit() should return the engine instance for chaining."""
        ...

    def test_predict_before_fit_raises(self) -> None:
        """predict() before fit() should raise RuntimeError."""
        ...

    def test_predict_shape_matches_input(self) -> None:
        """predict() output length should equal number of input timesteps."""
        ...

    def test_regime_labels_in_range(self) -> None:
        """All predicted regime labels must be in [0, n_regimes)."""
        ...

    def test_regime_probabilities_sum_to_one(self) -> None:
        """Posterior probabilities across regimes must sum to 1.0 per timestep."""
        ...

    def test_regime_name_unknown_label(self) -> None:
        """regime_name() should return 'unknown' for an out-of-range label."""
        ...
