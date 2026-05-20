"""Tests for RegimeStrategyManager allocation logic."""

import pytest
from core.regime_strategies import RegimeStrategyManager, RegimeAllocation


class TestRegimeStrategyManager:
    """Unit tests for weight computation and regime overrides."""

    def test_get_allocation_valid_regime(self) -> None:
        """get_allocation() should return a RegimeAllocation for known regimes."""
        ...

    def test_get_allocation_invalid_regime_raises(self) -> None:
        """get_allocation() should raise ValueError for unknown regime names."""
        ...

    def test_weights_sum_to_one(self) -> None:
        """Computed weights including CASH must sum to 1.0."""
        ...

    def test_weights_respect_cash_floor(self) -> None:
        """CASH weight must be >= the regime's cash_pct."""
        ...

    def test_inverse_vol_weighting(self) -> None:
        """Higher-vol asset should receive a lower weight than lower-vol asset."""
        ...

    def test_overrides_applied(self) -> None:
        """Custom overrides at construction time must replace default params."""
        ...
