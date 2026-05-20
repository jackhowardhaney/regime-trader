"""Tests for RiskManager position sizing and drawdown controls."""

import pytest
from core.risk_manager import RiskManager, RiskParams


class TestRiskManager:
    """Unit tests for leverage scaling, drawdown breach, and stop-loss logic."""

    def test_no_breach_below_threshold(self) -> None:
        """is_drawdown_breach() returns False when drawdown < max_portfolio_drawdown."""
        ...

    def test_breach_at_threshold(self) -> None:
        """is_drawdown_breach() returns True when drawdown >= max_portfolio_drawdown."""
        ...

    def test_drawdown_breach_returns_all_cash(self) -> None:
        """apply_risk_limits() returns 100% CASH when drawdown limit is breached."""
        ...

    def test_position_size_clamped(self) -> None:
        """No individual weight may exceed max_position_size after risk limits."""
        ...

    def test_volatility_scalar_reduces_exposure(self) -> None:
        """volatility_scalar() < 1 when realized vol > volatility_target."""
        ...

    def test_stop_loss_triggered(self) -> None:
        """stop_loss_triggered() returns True when loss >= stop_loss_pct."""
        ...

    def test_peak_updates_monotonically(self) -> None:
        """peak_value must never decrease after update_peak() calls."""
        ...
