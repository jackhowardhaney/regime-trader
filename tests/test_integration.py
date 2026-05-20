"""Phase 9 integration tests — five scenarios covering the full pipeline.

a. End-to-end dry run: OHLCV → features → HMM → strategy → risk → decisions
b. Look-ahead bias: forward algorithm identical with different end dates
c. Risk stress: extreme signals capped, rapid-fire blocked, no-stop rejected
d. Broker mock: bracket order → modify stop → cancel → verify clean state
e. Recovery: state_snapshot.json loaded, duplicate-entry protection active
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ======================================================================
# Shared helpers
# ======================================================================

def _make_ohlcv(n: int = 900, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with DatetimeIndex — enough for feature engineering."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n, tz="America/New_York")
    log_rets = rng.normal(0.0004, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    noise = abs(rng.normal(0, 0.005, n))
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_synthetic_features(n: int = 600, n_features: int = 14, seed: int = 42):
    """Synthetic feature array + returns for direct HMM training."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, n_features))
    returns = rng.standard_normal(n) * 0.01
    return features, returns


def _minimal_config() -> dict:
    return {
        "broker": {"symbols": ["SPY"], "timeframe": "1Day"},
        "hmm": {
            "n_candidates": [3], "n_init": 2,
            "stability_bars": 3, "flicker_window": 20,
            "flicker_threshold": 4, "min_confidence": 0.55,
        },
        "strategy": {"rebalance_threshold": 0.10},
        "risk": {
            "max_risk_per_trade": 0.01, "max_exposure": 0.80,
            "max_single_position": 0.15, "max_leverage": 1.25,
            "max_concurrent": 5, "max_daily_trades": 20,
            "daily_dd_reduce": 0.02, "daily_dd_halt": 0.03,
            "weekly_dd_reduce": 0.05, "weekly_dd_halt": 0.07,
            "max_dd_from_peak": 0.10,
        },
    }


def _train_hmm_on_ohlcv(ohlcv: pd.DataFrame):
    """Train a 3-regime HMM on OHLCV data. Returns (engine, features_df)."""
    from core.hmm_engine import HMMEngine
    from data.feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    features_df = fe.build_hmm_features(ohlcv)
    log_rets = np.log(ohlcv["close"] / ohlcv["close"].shift(1)).dropna()
    features_df, log_rets = features_df.align(log_rets, join="inner", axis=0)

    engine = HMMEngine(n_candidates=[3], n_init=2)
    engine.fit(features_df.values, log_rets.values)
    return engine, features_df


def _make_signal(
    symbol: str = "SPY",
    direction: str = "LONG",
    entry_price: float = 500.0,
    stop_loss: float = 490.0,
    position_size_pct: float = 0.14,
    leverage: float = 1.0,
):
    from core.regime_strategies import Signal

    return Signal(
        symbol=symbol,
        direction=direction,
        confidence=0.85,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=None,
        position_size_pct=position_size_pct,
        leverage=leverage,
        regime_id=0,
        regime_name="BULL",
        regime_probability=0.85,
        timestamp=datetime.utcnow(),
        reasoning="integration test",
        strategy_name="LowVolBullStrategy",
    )


def _make_portfolio(equity: float = 100_000.0, positions: dict = None):
    from core.risk_manager import PortfolioState

    return PortfolioState(
        equity=equity,
        cash=equity * 0.3,
        buying_power=equity * 0.6,
        positions=positions or {},
        daily_pnl=0.0,
        weekly_pnl=0.0,
        peak_equity=equity,
        drawdown=0.0,
        circuit_breaker_status="normal",
        flicker_rate=0,
    )


# ======================================================================
# a. End-to-end dry run
# ======================================================================

class TestEndToEndDryRun:
    """Full pipeline without touching the broker."""

    def test_pipeline_produces_decisions(self):
        """data → features → HMM → regime → signals → risk decisions."""
        from core.regime_strategies import StrategyOrchestrator
        from core.risk_manager import RiskManager

        ohlcv = _make_ohlcv(1200)
        engine, features_df = _train_hmm_on_ohlcv(ohlcv)
        config = _minimal_config()

        orchestrator = StrategyOrchestrator(config, engine.regime_info)
        risk_mgr = RiskManager(config)

        # Run inference on all features (no look-ahead)
        label_seq = engine.predict_regime_filtered(features_df.values)
        proba_matrix = engine.predict_regime_proba(features_df.values)
        label_idx = int(label_seq[-1])
        regime_state = engine.update_stability_filter(label_idx, proba_matrix[-1])

        signals = orchestrator.generate_signals(
            ["SPY"], {"SPY": ohlcv}, regime_state, engine.is_flickering()
        )
        assert len(signals) >= 1, "No signals generated"

        portfolio = _make_portfolio()
        for signal in signals:
            decision = risk_mgr.validate_signal(signal, portfolio)
            # Every decision must have the three required fields
            assert hasattr(decision, "approved")
            assert hasattr(decision, "modified_signal")
            assert hasattr(decision, "rejection_reason")
            # Approved signals must carry a modified (or identical) signal object
            if decision.approved:
                assert decision.modified_signal is not None

    def test_flat_signal_passes_all_risk_checks(self):
        """A FLAT signal must always be approved (reduces exposure, never rejected)."""
        from core.risk_manager import RiskManager

        risk_mgr = RiskManager(_minimal_config())
        flat = _make_signal(direction="FLAT")
        portfolio = _make_portfolio()
        decision = risk_mgr.validate_signal(flat, portfolio)
        assert decision.approved
        assert decision.modified_signal is flat

    def test_all_signals_have_stop_loss(self):
        """Every generated signal must have a positive stop_loss below entry."""
        ohlcv = _make_ohlcv(1200)
        engine, features_df = _train_hmm_on_ohlcv(ohlcv)

        from core.regime_strategies import StrategyOrchestrator

        orchestrator = StrategyOrchestrator(_minimal_config(), engine.regime_info)
        label_seq = engine.predict_regime_filtered(features_df.values)
        proba_matrix = engine.predict_regime_proba(features_df.values)
        regime_state = engine.update_stability_filter(
            int(label_seq[-1]), proba_matrix[-1]
        )

        signals = orchestrator.generate_signals(
            ["SPY"], {"SPY": ohlcv}, regime_state, engine.is_flickering()
        )
        for sig in signals:
            if sig.direction == "LONG":
                assert sig.stop_loss > 0, "stop_loss must be positive"
                assert sig.stop_loss < sig.entry_price, (
                    f"stop_loss {sig.stop_loss} must be below entry {sig.entry_price}"
                )


# ======================================================================
# b. Look-ahead bias — forward algorithm + backtest end-date invariance
# ======================================================================

class TestLookAheadBias:
    """Verify no future data influences past decisions at any level."""

    def test_forward_algorithm_invariant_to_future_bars(self):
        """Regime at index T must be the same whether we pass T or T+K bars."""
        features, returns = _make_synthetic_features(n=600)

        from core.hmm_engine import HMMEngine

        engine = HMMEngine(n_candidates=[3], n_init=2)
        engine.fit(features, returns)

        T = 400
        labels_short = engine.predict_regime_filtered(features[:T])
        engine._cached_alpha = None
        labels_long = engine.predict_regime_filtered(features[:T + 100])

        assert int(labels_short[-1]) == int(labels_long[T - 1]), (
            f"LOOK-AHEAD: regime at T={T} differs when future data included."
        )

    def test_adding_future_data_never_changes_past_labels(self):
        """All labels in [0, T) must be identical regardless of data after T."""
        features, returns = _make_synthetic_features(n=700)

        from core.hmm_engine import HMMEngine

        engine = HMMEngine(n_candidates=[3], n_init=2)
        engine.fit(features, returns)

        T = 500
        labels_short = engine.predict_regime_filtered(features[:T])
        engine._cached_alpha = None
        labels_long = engine.predict_regime_filtered(features[:T + 150])

        np.testing.assert_array_equal(
            labels_short,
            labels_long[:T],
            err_msg="Future data changed past regime labels — look-ahead bias.",
        )

    def test_incremental_matches_full_batch(self):
        """Incremental forward pass must produce identical results to batch pass."""
        features, returns = _make_synthetic_features(n=600)

        from core.hmm_engine import HMMEngine

        engine = HMMEngine(n_candidates=[3], n_init=2)
        engine.fit(features, returns)

        batch = engine.predict_regime_filtered(features)
        engine._cached_alpha = None
        incremental = []
        for obs in features:
            idx, _ = engine.predict_regime_filtered_incremental(obs)
            incremental.append(idx)

        np.testing.assert_array_equal(batch, incremental,
                                      err_msg="Incremental diverges from batch forward pass.")


# ======================================================================
# c. Risk stress tests
# ======================================================================

class TestRiskStress:
    """Extreme signals capped, rapid-fire blocked, no-stop rejected."""

    def setup_method(self):
        from core.risk_manager import RiskManager
        self.risk = RiskManager(_minimal_config())

    def test_extreme_position_size_capped_not_rejected(self):
        """A 50% position request must be capped to max_single_position (15%), not rejected."""
        signal = _make_signal(position_size_pct=0.50)
        portfolio = _make_portfolio()
        decision = self.risk.validate_signal(signal, portfolio)

        assert decision.approved, (
            f"Extreme signal should be capped, not rejected. "
            f"Reason: {decision.rejection_reason}"
        )
        assert decision.modified_signal.position_size_pct <= 0.15, (
            f"Expected cap ≤ 15%, got {decision.modified_signal.position_size_pct:.2%}"
        )

    def test_rapid_fire_duplicate_blocked(self):
        """Same symbol+direction within 60s must be rejected as duplicate."""
        signal = _make_signal()
        portfolio = _make_portfolio()
        portfolio.recent_orders = {
            f"{signal.symbol}:{signal.direction}": datetime.utcnow()
        }
        decision = self.risk.validate_signal(signal, portfolio)

        assert not decision.approved
        assert decision.rejection_reason is not None
        assert "duplicate" in decision.rejection_reason.lower()

    def test_signal_without_stop_loss_rejected(self):
        """Signal with stop_loss = 0 must be hard-rejected, not modified."""
        signal = _make_signal(stop_loss=0.0)
        portfolio = _make_portfolio()
        decision = self.risk.validate_signal(signal, portfolio)

        assert not decision.approved
        assert decision.rejection_reason is not None
        assert "stop" in decision.rejection_reason.lower()

    def test_signal_with_stop_above_entry_rejected(self):
        """Stop loss >= entry price is logically invalid and must be rejected."""
        signal = _make_signal(entry_price=500.0, stop_loss=510.0)
        portfolio = _make_portfolio()
        decision = self.risk.validate_signal(signal, portfolio)

        assert not decision.approved
        assert "stop" in decision.rejection_reason.lower()

    def test_halt_peak_circuit_breaker_blocks_all_signals(self):
        """When circuit breaker is halt_peak, every LONG signal must be blocked."""
        self.risk.circuit_breaker._status = "halt_peak"
        signal = _make_signal()
        portfolio = _make_portfolio()
        decision = self.risk.validate_signal(signal, portfolio)

        assert not decision.approved
        assert "halt" in decision.rejection_reason.lower()

    def test_daily_trade_limit_blocks_excess(self):
        """When daily_trade_count >= max_daily_trades, new signals are rejected."""
        signal = _make_signal()
        portfolio = _make_portfolio()
        portfolio.daily_trade_count = 20   # equals max_daily_trades
        decision = self.risk.validate_signal(signal, portfolio)

        assert not decision.approved
        assert "daily trade limit" in decision.rejection_reason.lower()

    def test_reduce_mode_halves_position_size(self):
        """reduce_daily CB status must halve the final position size."""
        self.risk.circuit_breaker._status = "reduce_daily"
        signal = _make_signal(position_size_pct=0.14)
        portfolio = _make_portfolio()
        decision = self.risk.validate_signal(signal, portfolio)

        if decision.approved and decision.modified_signal:
            assert decision.modified_signal.position_size_pct <= 0.14 * 0.5 + 1e-4


# ======================================================================
# d. Broker mock — bracket order lifecycle
# ======================================================================

class TestBrokerOrderLifecycle:
    """Mock-based tests for the Phase 6 order executor."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.get_account.return_value = {
            "equity": 100_000.0, "cash": 50_000.0,
            "buying_power": 200_000.0, "status": "ACTIVE",
            "trading_blocked": False,
        }
        mock_order = MagicMock()
        mock_order.id = "order-abc-123"
        mock_order.client_order_id = "rt-brk-12345678"
        client.trading.submit_order.return_value = mock_order
        return client

    @pytest.fixture
    def executor(self, mock_client):
        from broker.order_executor import OrderExecutor
        return OrderExecutor(mock_client)

    def test_bracket_order_creates_record_with_trade_id(self, executor, mock_client):
        """submit_bracket_order() must return an OrderRecord with a unique trade_id."""
        signal = _make_signal()
        record = executor.submit_bracket_order(signal)

        assert record is not None
        assert record.trade_id is not None
        assert len(record.trade_id) == 36   # UUID
        assert record.order_type == "bracket"
        assert record.order_id == "order-abc-123"
        mock_client.trading.submit_order.assert_called_once()

    def test_modify_stop_tighten_accepted(self, executor, mock_client):
        """modify_stop() with a higher stop (tighter for long) must return True."""
        mock_leg = MagicMock()
        mock_leg.order_type = "stop"
        mock_leg.stop_price = 490.0
        mock_leg.id = "stop-leg-456"
        mock_parent = MagicMock()
        mock_parent.legs = [mock_leg]
        mock_parent.side = "buy"
        mock_client.trading.get_order_by_id.return_value = mock_parent

        result = executor.modify_stop("SPY", "order-abc-123", new_stop=492.0)
        assert result is True
        mock_client.trading.replace_order_by_id.assert_called_once()

    def test_modify_stop_widen_rejected(self, executor, mock_client):
        """modify_stop() moving stop farther from entry (widen) must return False."""
        mock_leg = MagicMock()
        mock_leg.order_type = "stop"
        mock_leg.stop_price = 490.0
        mock_leg.id = "stop-leg-456"
        mock_parent = MagicMock()
        mock_parent.legs = [mock_leg]
        mock_parent.side = "buy"
        mock_client.trading.get_order_by_id.return_value = mock_parent

        result = executor.modify_stop("SPY", "order-abc-123", new_stop=485.0)
        assert result is False
        mock_client.trading.replace_order_by_id.assert_not_called()

    def test_cancel_order_calls_api(self, executor, mock_client):
        """cancel_order() must delegate to the Alpaca API."""
        executor.cancel_order("order-abc-123")
        mock_client.trading.cancel_order_by_id.assert_called_once_with("order-abc-123")

    def test_close_all_positions_calls_api(self, executor, mock_client):
        """close_all_positions() must call Alpaca close_all_positions with cancel_orders=True."""
        executor.close_all_positions()
        mock_client.trading.close_all_positions.assert_called_once_with(cancel_orders=True)

    def test_trade_id_unique_per_order(self, executor):
        """Each submit_bracket_order() call must generate a distinct trade_id."""
        s1 = _make_signal(symbol="SPY")
        s2 = _make_signal(symbol="SPY")
        r1 = executor.submit_bracket_order(s1)
        r2 = executor.submit_bracket_order(s2)
        assert r1.trade_id != r2.trade_id


# ======================================================================
# e. Recovery — state snapshot and no double-entry
# ======================================================================

class TestStateRecovery:
    """Verify restart recovery and duplicate-entry prevention."""

    def test_load_state_snapshot(self, tmp_path, monkeypatch):
        """LiveTrader._load_state() must restore regime and equity tracking from file."""
        import main as main_module

        state = {
            "version": 1,
            "session_start": "2026-05-20T08:00:00",
            "last_save": "2026-05-20T10:30:00",
            "last_regime": "BULL",
            "day_open_equity": 103_000.0,
            "week_open_equity": 101_000.0,
            "daily_trade_count": 5,
            "bar_count": 42,
            "peak_equity": 106_000.0,
            "circuit_breaker_status": "normal",
        }
        state_file = tmp_path / "state_snapshot.json"
        state_file.write_text(json.dumps(state))
        monkeypatch.setattr(main_module, "_STATE_FILE", state_file)

        from main import LiveTrader

        trader = LiveTrader(
            config=_minimal_config(),
            credentials={"alpaca_api_key": "", "alpaca_secret_key": "", "paper": True},
        )
        trader._load_state()

        assert trader._last_regime == "BULL"
        assert trader._day_open_equity == 103_000.0
        assert trader._week_open_equity == 101_000.0
        assert trader._daily_trade_count == 5

    def test_no_double_entry_via_duplicate_protection(self):
        """recent_orders tracking prevents re-entering a position after recovery."""
        from core.risk_manager import RiskManager

        risk_mgr = RiskManager(_minimal_config())
        signal = _make_signal("SPY", "LONG")
        portfolio = _make_portfolio(positions={"SPY": 94_000.0})
        # Simulate the order was sent just before the kill
        portfolio.recent_orders = {
            f"{signal.symbol}:{signal.direction}": datetime.utcnow()
        }

        decision = risk_mgr.validate_signal(signal, portfolio)
        assert not decision.approved, (
            "Duplicate-order protection failed — double-entry would have occurred."
        )

    def test_state_file_missing_is_handled_gracefully(self, tmp_path, monkeypatch):
        """_load_state() with no snapshot file must not raise — just use defaults."""
        import main as main_module

        monkeypatch.setattr(
            main_module, "_STATE_FILE", tmp_path / "nonexistent_state.json"
        )
        from main import LiveTrader

        trader = LiveTrader(
            config=_minimal_config(),
            credentials={"alpaca_api_key": "", "alpaca_secret_key": "", "paper": True},
        )
        try:
            trader._load_state()   # must not raise
        except Exception as exc:
            pytest.fail(f"_load_state() raised on missing file: {exc}")

        # Defaults should be in place
        assert trader._last_regime == "UNKNOWN"
        assert trader._daily_trade_count == 0
