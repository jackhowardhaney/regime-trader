"""Risk Management Layer — Phase 5.

Operates INDEPENDENTLY of the HMM. Even if the HMM fails completely,
circuit breakers catch drawdowns based on actual P&L.
The RiskManager has ABSOLUTE VETO POWER over any signal.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("regime_trader")

LOCK_FILE = Path("trading_halted.lock")


# ------------------------------------------------------------------
# PortfolioState
# ------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Snapshot of current portfolio passed into every risk check."""
    equity: float
    cash: float
    buying_power: float
    positions: Dict[str, float]          # symbol → current market value
    daily_pnl: float
    weekly_pnl: float
    peak_equity: float
    drawdown: float                       # current drawdown from peak (positive = loss)
    circuit_breaker_status: str          # "normal" | "reduce_daily" | "halt_daily" | ...
    flicker_rate: int                     # confirmed regime changes in last flicker_window
    open_position_count: int = 0
    daily_trade_count: int = 0
    recent_orders: Dict[str, datetime] = field(default_factory=dict)  # "SYM:DIR" → timestamp


# ------------------------------------------------------------------
# RiskDecision
# ------------------------------------------------------------------

@dataclass
class RiskDecision:
    """Returned by RiskManager.validate_signal()."""
    approved: bool
    modified_signal: Optional[object]    # Signal | None
    rejection_reason: Optional[str]
    modifications: List[str]             # human-readable list of changes applied


# ------------------------------------------------------------------
# CircuitBreaker
# ------------------------------------------------------------------

class CircuitBreaker:
    """
    Fires on actual P&L — completely independent of regime classification.

    Levels (from settings.yaml risk section):
      daily_dd_reduce  (default 0.02) → reduce all sizes 50% rest of day
      daily_dd_halt    (default 0.03) → close ALL positions, halt rest of day
      weekly_dd_reduce (default 0.05) → reduce all sizes 50% rest of week
      weekly_dd_halt   (default 0.07) → close ALL, halt rest of week
      max_dd_from_peak (default 0.10) → halt ALL, write trading_halted.lock
    """

    def __init__(self, config: dict) -> None:
        risk = config.get("risk", {})
        self.daily_dd_reduce: float = risk.get("daily_dd_reduce", 0.02)
        self.daily_dd_halt: float = risk.get("daily_dd_halt", 0.03)
        self.weekly_dd_reduce: float = risk.get("weekly_dd_reduce", 0.05)
        self.weekly_dd_halt: float = risk.get("weekly_dd_halt", 0.07)
        self.max_dd_from_peak: float = risk.get("max_dd_from_peak", 0.10)

        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._status: str = "normal"
        self._history: List[dict] = []

    @property
    def status(self) -> str:
        return self._status

    def update(
        self,
        equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        regime_name: str = "unknown",
    ) -> str:
        """Evaluate current P&L against thresholds and update status."""
        if self._peak_equity == 0.0:
            self._peak_equity = equity
        elif equity > self._peak_equity:
            self._peak_equity = equity

        self._daily_pnl = daily_pnl
        self._weekly_pnl = weekly_pnl

        base = self._peak_equity if self._peak_equity > 0 else 1.0
        peak_dd = (self._peak_equity - equity) / base
        daily_dd = abs(daily_pnl) / base if daily_pnl < 0 else 0.0
        weekly_dd = abs(weekly_pnl) / base if weekly_pnl < 0 else 0.0

        old_status = self._status

        # Priority order: peak > weekly halt > daily halt > weekly reduce > daily reduce
        if peak_dd >= self.max_dd_from_peak:
            self._status = "halt_peak"
            self._write_lock_file(peak_dd, equity, regime_name)
        elif weekly_dd >= self.weekly_dd_halt:
            self._status = "halt_weekly"
        elif daily_dd >= self.daily_dd_halt:
            self._status = "halt_daily"
        elif weekly_dd >= self.weekly_dd_reduce:
            self._status = "reduce_weekly"
        elif daily_dd >= self.daily_dd_reduce:
            self._status = "reduce_daily"
        else:
            self._status = "normal"

        if self._status != old_status:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "old_status": old_status,
                "new_status": self._status,
                "daily_dd_pct": round(daily_dd * 100, 3),
                "weekly_dd_pct": round(weekly_dd * 100, 3),
                "peak_dd_pct": round(peak_dd * 100, 3),
                "equity": round(equity, 2),
                "regime_at_trigger": regime_name,
            }
            self._history.append(record)
            logger.warning(
                "CIRCUIT BREAKER %s → %s | daily=%.2f%% weekly=%.2f%% peak=%.2f%% equity=$%.2f regime=%s",
                old_status, self._status,
                daily_dd * 100, weekly_dd * 100, peak_dd * 100,
                equity, regime_name,
            )

        return self._status

    def check(self) -> str:
        return self._status

    def is_halted(self) -> bool:
        return self._status in ("halt_daily", "halt_weekly", "halt_peak")

    def should_reduce(self) -> bool:
        return self._status in ("reduce_daily", "reduce_weekly")

    def size_multiplier(self) -> float:
        if self.is_halted():
            return 0.0
        if self.should_reduce():
            return 0.5
        return 1.0

    def reset_daily(self) -> None:
        """Call at the start of each trading day."""
        if self._status in ("halt_daily", "reduce_daily"):
            logger.info("Circuit breaker daily reset: %s → normal", self._status)
            self._status = "normal"
        self._daily_pnl = 0.0

    def reset_weekly(self) -> None:
        """Call at the start of each trading week."""
        if self._status in ("halt_weekly", "reduce_weekly"):
            logger.info("Circuit breaker weekly reset: %s → normal", self._status)
            self._status = "normal"
        self._weekly_pnl = 0.0

    def get_history(self) -> List[dict]:
        return list(self._history)

    def _write_lock_file(self, drawdown: float, equity: float, regime: str) -> None:
        """Write trading_halted.lock — requires manual deletion to resume trading."""
        content = (
            f"TRADING HALTED — manual intervention required\n"
            f"Timestamp  : {datetime.utcnow().isoformat()}\n"
            f"Peak DD    : {drawdown * 100:.2f}%\n"
            f"Equity     : ${equity:,.2f}\n"
            f"Regime     : {regime}\n"
            f"Threshold  : {self.max_dd_from_peak * 100:.1f}%\n\n"
            f"Delete this file to resume trading.\n"
        )
        try:
            LOCK_FILE.write_text(content)
            logger.critical(
                "PEAK DD EXCEEDED %.1f%% — trading halted. Delete %s to resume.",
                self.max_dd_from_peak * 100, LOCK_FILE,
            )
        except OSError as exc:
            logger.error("Failed to write lock file: %s", exc)


# ------------------------------------------------------------------
# RiskManager
# ------------------------------------------------------------------

class RiskManager:
    """
    Validates and modifies every signal before execution.
    Has absolute veto power — operates independently of HMM output.

    Usage:
        decision = risk_manager.validate_signal(signal, portfolio_state)
        if decision.approved:
            execute(decision.modified_signal)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        risk = config.get("risk", {})
        hmm_cfg = config.get("hmm", {})

        # Portfolio-level limits (from settings.yaml)
        self.max_exposure: float = risk.get("max_exposure", 0.80)
        self.max_single_position: float = risk.get("max_single_position", 0.15)
        self.max_leverage: float = risk.get("max_leverage", 1.25)
        self.max_concurrent: int = risk.get("max_concurrent", 5)
        self.max_daily_trades: int = risk.get("max_daily_trades", 20)
        self.max_risk_per_trade: float = risk.get("max_risk_per_trade", 0.01)

        # Order validation
        self.min_position_value: float = 100.0
        self.duplicate_window_seconds: int = 60
        self.max_spread_pct: float = 0.005          # 0.5%

        # Correlation
        self.corr_reduce_threshold: float = 0.70
        self.corr_reject_threshold: float = 0.85
        self.corr_window: int = 60

        # HMM confidence threshold (for leverage rules)
        self.min_confidence: float = hmm_cfg.get("min_confidence", 0.55)
        self.flicker_threshold: int = hmm_cfg.get("flicker_threshold", 4)

        self.circuit_breaker = CircuitBreaker(config)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def validate_signal(
        self, signal, portfolio_state: PortfolioState
    ) -> RiskDecision:
        """
        Run all risk checks in priority order.
        Returns a RiskDecision with approved=True/False and any modifications.
        """
        from core.regime_strategies import Signal as RegimeSignal

        modifications: List[str] = []

        # ── 0. Hard halt checks ──────────────────────────────────────
        if LOCK_FILE.exists() or self.circuit_breaker.status == "halt_peak":
            return self._reject("HALT: trading_halted.lock — manual intervention required")

        if self.circuit_breaker.is_halted():
            return self._reject(
                f"HALT: circuit breaker active ({self.circuit_breaker.status})"
            )

        # ── 1. Stop loss required ────────────────────────────────────
        if not hasattr(signal, "stop_loss") or signal.stop_loss <= 0:
            return self._reject("REJECTED: missing or invalid stop loss")
        if signal.stop_loss >= signal.entry_price:
            return self._reject(
                f"REJECTED: stop loss {signal.stop_loss} >= entry {signal.entry_price}"
            )

        # ── 2. Daily trade limit ─────────────────────────────────────
        if portfolio_state.daily_trade_count >= self.max_daily_trades:
            return self._reject(
                f"REJECTED: daily trade limit ({self.max_daily_trades}) reached"
            )

        # ── 3. Concurrent position limit ────────────────────────────
        if (portfolio_state.open_position_count >= self.max_concurrent
                and signal.symbol not in portfolio_state.positions):
            return self._reject(
                f"REJECTED: max concurrent positions ({self.max_concurrent}) reached"
            )

        # ── 4. Duplicate order block ─────────────────────────────────
        order_key = f"{signal.symbol}:{signal.direction}"
        last_sent = portfolio_state.recent_orders.get(order_key)
        if last_sent:
            elapsed = (datetime.utcnow() - last_sent).total_seconds()
            if elapsed < self.duplicate_window_seconds:
                return self._reject(
                    f"REJECTED: duplicate {order_key} within {self.duplicate_window_seconds}s"
                )

        # ── 5. FLAT passes after halt checks ────────────────────────
        if signal.direction == "FLAT":
            return RiskDecision(
                approved=True,
                modified_signal=signal,
                rejection_reason=None,
                modifications=[],
            )

        # ── 6. Position sizing ───────────────────────────────────────
        signal = self._size_position(signal, portfolio_state, modifications)
        if signal is None:
            return self._reject(
                "REJECTED: position too small (< $100 minimum)",
                modifications=modifications,
            )

        # ── 7. Leverage rules ────────────────────────────────────────
        signal = self._apply_leverage_rules(signal, portfolio_state, modifications)

        # ── 8. Circuit breaker size multiplier ──────────────────────
        cb_mult = self.circuit_breaker.size_multiplier()
        if cb_mult < 1.0:
            old = signal.position_size_pct
            signal.position_size_pct = round(old * cb_mult, 4)
            modifications.append(
                f"CB reduce ({self.circuit_breaker.status}): {old:.2%} → {signal.position_size_pct:.2%}"
            )

        # ── 9. Total portfolio exposure cap (80%) ────────────────────
        current_exp = (
            sum(portfolio_state.positions.values()) / portfolio_state.equity
            if portfolio_state.equity > 0 else 0.0
        )
        headroom = self.max_exposure - current_exp
        if headroom <= 0.01:
            return self._reject(
                f"REJECTED: portfolio at max exposure ({self.max_exposure:.0%})",
                modifications=modifications,
            )
        if signal.position_size_pct > headroom:
            modifications.append(
                f"Exposure cap: {signal.position_size_pct:.2%} → {headroom:.2%}"
            )
            signal.position_size_pct = round(headroom, 4)

        # ── 10. Single position cap (15%) ────────────────────────────
        if signal.position_size_pct > self.max_single_position:
            modifications.append(
                f"Position cap: {signal.position_size_pct:.2%} → {self.max_single_position:.2%}"
            )
            signal.position_size_pct = self.max_single_position

        # ── 11. Flicker → force 1.0x leverage ───────────────────────
        if portfolio_state.flicker_rate > self.flicker_threshold and signal.leverage > 1.0:
            modifications.append(
                f"Flicker ({portfolio_state.flicker_rate} changes): leverage forced 1.0x"
            )
            signal.leverage = 1.0

        if modifications:
            logger.info("Signal %s modified: %s", signal.symbol, "; ".join(modifications))

        return RiskDecision(
            approved=True,
            modified_signal=signal,
            rejection_reason=None,
            modifications=modifications,
        )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _size_position(
        self,
        signal,
        state: PortfolioState,
        modifications: List[str],
    ):
        """
        size = (equity * 0.01) / abs(entry - stop_loss)
        GAP RISK: overnight_size = min(normal, size_where_3x_gap = 2%_portfolio)
        Cap at regime max (signal.position_size_pct) then portfolio max (15%).
        Minimum position value: $100.
        """
        stop_dist = abs(signal.entry_price - signal.stop_loss)
        if stop_dist <= 0:
            return None

        risk_dollars = state.equity * self.max_risk_per_trade
        shares_normal = risk_dollars / stop_dist
        normal_pct = (shares_normal * signal.entry_price) / state.equity

        # GAP RISK: 3x gap-through; size so 3x gap costs only 2% of portfolio
        gap_risk_dollars = state.equity * 0.02
        gap_dist = stop_dist * 3.0
        shares_gap = gap_risk_dollars / gap_dist if gap_dist > 0 else shares_normal
        gap_pct = (shares_gap * signal.entry_price) / state.equity

        sized_pct = min(normal_pct, gap_pct)
        if gap_pct < normal_pct:
            modifications.append(
                f"Gap risk: {normal_pct:.2%} → {gap_pct:.2%} (3x overnight gap = 2% portfolio)"
            )

        # Cap at regime max, then hard portfolio max
        final_pct = min(sized_pct, signal.position_size_pct, self.max_single_position)

        if state.equity * final_pct < self.min_position_value:
            return None

        if abs(final_pct - signal.position_size_pct) > 0.001:
            modifications.append(
                f"Risk sizing: {signal.position_size_pct:.2%} → {final_pct:.2%}"
            )

        signal.position_size_pct = round(final_pct, 4)
        return signal

    # ------------------------------------------------------------------
    # Leverage rules
    # ------------------------------------------------------------------

    def _apply_leverage_rules(
        self, signal, state: PortfolioState, modifications: List[str]
    ):
        """
        Force 1.0x if: uncertain, circuit breaker active, 3+ positions, high flicker.
        Only low-vol strategies may request 1.25x.
        Hard cap at max_leverage (1.25x).
        """
        force_reasons: List[str] = []

        if signal.regime_probability < self.min_confidence:
            force_reasons.append("regime uncertain")
        if self.circuit_breaker.status != "normal":
            force_reasons.append(f"CB={self.circuit_breaker.status}")
        if state.open_position_count >= 3:
            force_reasons.append(f"{state.open_position_count} open positions")
        if state.flicker_rate > self.flicker_threshold:
            force_reasons.append("high flicker")

        if force_reasons and signal.leverage > 1.0:
            modifications.append(
                f"Leverage 1.0x forced ({', '.join(force_reasons)})"
            )
            signal.leverage = 1.0

        if signal.leverage > self.max_leverage:
            modifications.append(
                f"Leverage capped {signal.leverage}x → {self.max_leverage}x"
            )
            signal.leverage = self.max_leverage

        return signal

    # ------------------------------------------------------------------
    # Correlation check (call before validate_signal if price history available)
    # ------------------------------------------------------------------

    def check_correlation(
        self,
        signal,
        returns_history: Dict[str, pd.Series],
        modifications: List[str],
    ):
        """
        60-day rolling correlation with existing positions.
        > 0.85 → reject (return None)
        > 0.70 → reduce size 50%
        """
        if signal.symbol not in returns_history:
            return signal

        sig_ret = returns_history[signal.symbol].tail(self.corr_window)

        for sym, hist_ret in returns_history.items():
            if sym == signal.symbol:
                continue
            paired = sig_ret.align(hist_ret.tail(self.corr_window), join="inner")
            s1, s2 = paired[0], paired[1]
            if len(s1) < 10:
                continue
            corr = float(s1.corr(s2))
            if np.isnan(corr):
                continue

            if corr > self.corr_reject_threshold:
                logger.warning(
                    "Correlation reject: %s vs %s corr=%.2f > %.2f",
                    signal.symbol, sym, corr, self.corr_reject_threshold,
                )
                return None

            if corr > self.corr_reduce_threshold:
                old = signal.position_size_pct
                signal.position_size_pct = round(old * 0.5, 4)
                modifications.append(
                    f"Corr reduce ({signal.symbol}/{sym}={corr:.2f}): {old:.2%} → {signal.position_size_pct:.2%}"
                )

        return signal

    # ------------------------------------------------------------------
    # Order validation helpers
    # ------------------------------------------------------------------

    def validate_buying_power(self, signal, state: PortfolioState) -> bool:
        """Check account has enough buying power for the order."""
        notional = state.equity * signal.position_size_pct * signal.leverage
        return state.buying_power >= notional

    def validate_spread(self, bid: float, ask: float) -> bool:
        """Return False if bid-ask spread exceeds 0.5%."""
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return False
        return (ask - bid) / mid < self.max_spread_pct

    # ------------------------------------------------------------------
    # Circuit breaker lifecycle
    # ------------------------------------------------------------------

    def update_circuit_breakers(
        self,
        equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        regime_name: str = "unknown",
    ) -> str:
        """Update circuit breakers with latest P&L. Call once per bar."""
        return self.circuit_breaker.update(equity, daily_pnl, weekly_pnl, regime_name)

    @staticmethod
    def is_trading_halted() -> bool:
        """Return True if the lock file exists."""
        return LOCK_FILE.exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject(
        reason: str, modifications: Optional[List[str]] = None
    ) -> RiskDecision:
        logger.warning("Risk rejection: %s", reason)
        return RiskDecision(
            approved=False,
            modified_signal=None,
            rejection_reason=reason,
            modifications=modifications or [],
        )
