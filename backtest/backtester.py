"""Walk-forward allocation backtester — Phase 4.

ALLOCATION-BASED: sets a target portfolio allocation each bar based on the
detected volatility regime, rebalancing only when the allocation changes
meaningfully (>10%). Does NOT track individual trade entries/exits.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger("regime_trader")


@dataclass
class TradeRecord:
    bar_index: int
    timestamp: object
    symbol: str
    old_allocation: float
    new_allocation: float
    shares_delta: int
    fill_price: float
    slippage_cost: float
    regime_name: str
    regime_probability: float
    strategy_name: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    regime_history: pd.DataFrame
    config: dict


class WalkForwardBacktester:
    """
    Walk-forward allocation backtester.

    Each fold:
      a. Train HMM on In-Sample window (252 bars, BIC model selection)
      b. Build StrategyOrchestrator from trained regime_infos (vol-rank sort)
      c. Walk Out-of-Sample bar by bar — forward algorithm only, no look-ahead
      d. Rebalance when target allocation drifts >10% from current
      e. Record equity, regime, and trade log
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        bt = config.get("backtest", {})
        self.train_window: int = bt.get("train_window", 252)
        self.test_window: int = bt.get("test_window", 126)
        self.step_size: int = bt.get("step_size", 126)
        self.slippage_pct: float = bt.get("slippage_pct", 0.0005)
        self.initial_capital: float = bt.get("initial_capital", 100_000)
        self.rebalance_threshold: float = (
            config.get("strategy", {}).get("rebalance_threshold", 0.10)
        )
        self._feature_engineer = FeatureEngineer()
        self._risk_manager = RiskManager(config)

    def run(
        self,
        price_data: Dict[str, pd.DataFrame],
        symbols: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Execute the full walk-forward backtest.

        Args:
            price_data: {symbol: OHLCV DataFrame with DatetimeIndex}
            symbols:    subset to trade; defaults to all keys in price_data
        """
        if symbols is None:
            symbols = list(price_data.keys())

        primary = symbols[0]
        primary_bars = price_data[primary].rename(columns=str.lower).copy()

        features_df = self._feature_engineer.build_hmm_features(primary_bars)
        log_rets = np.log(
            primary_bars["close"] / primary_bars["close"].shift(1)
        ).dropna()
        features_df, log_rets = features_df.align(log_rets, join="inner")

        n = len(features_df)
        if n < self.train_window + self.test_window:
            raise ValueError(
                f"Need at least {self.train_window + self.test_window} bars, have {n}."
            )

        equity_records: List[tuple] = []
        trade_records: List[TradeRecord] = []
        regime_records: List[dict] = []

        cash = self.initial_capital
        shares: Dict[str, int] = {sym: 0 for sym in symbols}
        current_alloc: Dict[str, float] = {sym: 0.0 for sym in symbols}

        # Daily / weekly P&L tracking for circuit breakers
        day_open_equity: float = self.initial_capital
        week_open_equity: float = self.initial_capital
        prev_day: Optional[tuple] = None
        prev_week: Optional[tuple] = None

        oos_idx = self.train_window
        fold = 0

        while oos_idx < n:
            is_start = max(0, oos_idx - self.train_window)
            oos_end = min(oos_idx + self.test_window, n)
            fold += 1
            logger.info(
                "Fold %d: IS=[%d:%d]  OOS=[%d:%d]",
                fold, is_start, oos_idx, oos_idx, oos_end,
            )

            # --- a. Train HMM on IS window ---
            is_features = features_df.iloc[is_start:oos_idx].values
            is_returns = log_rets.iloc[is_start:oos_idx].values

            hmm_cfg = self.config.get("hmm", {})
            engine = HMMEngine(
                n_candidates=hmm_cfg.get("n_candidates", [3, 4, 5, 6, 7]),
                n_init=hmm_cfg.get("n_init", 5),
                covariance_type=hmm_cfg.get("covariance_type", "full"),
                stability_bars=hmm_cfg.get("stability_bars", 3),
                flicker_window=hmm_cfg.get("flicker_window", 20),
                flicker_threshold=hmm_cfg.get("flicker_threshold", 4),
                min_confidence=hmm_cfg.get("min_confidence", 0.55),
            )
            try:
                engine.fit(is_features, is_returns)
            except Exception as exc:
                logger.warning("HMM fit failed for fold %d: %s. Skipping.", fold, exc)
                oos_idx += self.step_size
                continue

            # --- b. Build orchestrator from trained regime_infos ---
            orchestrator = StrategyOrchestrator(self.config, engine.regime_info)

            # --- c. Walk OOS bar by bar ---
            for bar_idx in range(oos_idx, oos_end):
                # Use ALL data up to current bar — strict no look-ahead
                hist_features = features_df.iloc[: bar_idx + 1].values
                label_sequence = engine.predict_regime_filtered(hist_features)
                proba_matrix = engine.predict_regime_proba(hist_features)
                label_idx = int(label_sequence[-1])
                proba = proba_matrix[-1]

                regime_state = engine.update_stability_filter(label_idx, proba)

                hist_bars = {
                    sym: price_data[sym].rename(columns=str.lower).iloc[: bar_idx + 1]
                    for sym in symbols
                }

                signals = orchestrator.generate_signals(
                    symbols, hist_bars, regime_state, engine.is_flickering()
                )

                current_prices = {
                    sym: float(price_data[sym].rename(columns=str.lower)["close"].iloc[bar_idx])
                    for sym in symbols
                }

                # --- Day / week boundary resets for circuit breakers ---
                bar_ts = features_df.index[bar_idx]
                bar_day = (bar_ts.year, bar_ts.month, bar_ts.day)
                bar_week = (bar_ts.year, bar_ts.isocalendar()[1])

                if prev_day is not None and bar_day != prev_day:
                    day_open_equity = cash + sum(
                        shares[s] * current_prices[s] for s in symbols
                    )
                    self._risk_manager.circuit_breaker.reset_daily()
                if prev_week is not None and bar_week != prev_week:
                    week_open_equity = cash + sum(
                        shares[s] * current_prices[s] for s in symbols
                    )
                    self._risk_manager.circuit_breaker.reset_weekly()
                prev_day = bar_day
                prev_week = bar_week

                cb_halted = self._risk_manager.circuit_breaker.is_halted()

                # --- Fill delay: execute at bar N+1 open ---
                fill_bar = bar_idx + 1
                if fill_bar < n and not cb_halted:
                    for signal in signals:
                        sym = signal.symbol
                        target_alloc = signal.position_size_pct * signal.leverage
                        old_alloc = current_alloc.get(sym, 0.0)

                        if abs(target_alloc - old_alloc) <= self.rebalance_threshold:
                            continue

                        sym_df = price_data[sym].rename(columns=str.lower)
                        fill_col = "open" if "open" in sym_df.columns else "close"
                        fill_price = float(sym_df[fill_col].iloc[fill_bar])
                        slipped = fill_price * (1.0 + self.slippage_pct)

                        # ALLOCATION MATH (exactly as specified)
                        equity = cash + sum(
                            shares[s] * current_prices[s] for s in symbols
                        )
                        target_shares = int(equity * target_alloc / slipped)
                        delta = target_shares - shares[sym]
                        cash -= delta * slipped
                        shares[sym] = target_shares
                        current_alloc[sym] = target_alloc

                        trade_records.append(TradeRecord(
                            bar_index=fill_bar,
                            timestamp=features_df.index[fill_bar],
                            symbol=sym,
                            old_allocation=old_alloc,
                            new_allocation=target_alloc,
                            shares_delta=delta,
                            fill_price=slipped,
                            slippage_cost=abs(delta) * fill_price * self.slippage_pct,
                            regime_name=regime_state.label,
                            regime_probability=regime_state.probability,
                            strategy_name=signal.strategy_name,
                        ))

                # --- d. Mark to market ---
                equity = cash + sum(shares[sym] * current_prices[sym] for sym in symbols)
                equity_records.append((features_df.index[bar_idx], equity))

                # --- e. Update circuit breakers (independent of HMM) ---
                daily_pnl = equity - day_open_equity
                weekly_pnl = equity - week_open_equity
                self._risk_manager.update_circuit_breakers(
                    equity, daily_pnl, weekly_pnl, regime_state.label
                )

                # --- Record regime ---
                regime_records.append({
                    "timestamp": features_df.index[bar_idx],
                    "regime": regime_state.label,
                    "probability": regime_state.probability,
                    "is_confirmed": regime_state.is_confirmed,
                    "is_flickering": engine.is_flickering(),
                })

            oos_idx += self.step_size

        equity_curve = pd.Series(
            [v for _, v in equity_records],
            index=[t for t, _ in equity_records],
            name="equity",
        )
        trade_log = pd.DataFrame([
            {
                "bar_index": t.bar_index,
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "old_allocation": t.old_allocation,
                "new_allocation": t.new_allocation,
                "shares_delta": t.shares_delta,
                "fill_price": t.fill_price,
                "slippage_cost": t.slippage_cost,
                "regime": t.regime_name,
                "regime_probability": t.regime_probability,
                "strategy": t.strategy_name,
            }
            for t in trade_records
        ])
        regime_history = (
            pd.DataFrame(regime_records).set_index("timestamp")
            if regime_records
            else pd.DataFrame()
        )

        final_equity = equity_curve.iloc[-1] if len(equity_curve) else 0.0
        logger.info(
            "Backtest complete: %d OOS bars, %d rebalances, final equity=%.2f",
            len(equity_records), len(trade_records), final_equity,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_log,
            regime_history=regime_history,
            config=self.config,
        )
