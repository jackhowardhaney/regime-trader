"""Entry point for the regime-trader bot — Phase 7: Main Loop & Orchestration.

CLI usage:
  python main.py live --paper                  # paper trade
  python main.py live --paper --dry-run        # full pipeline, no orders
  python main.py live --paper --train-only     # train HMM and exit
  python main.py live --dashboard              # display running instance dashboard
  python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
  python main.py backtest --compare
  python main.py backtest --stress-test
"""

import argparse
import json
import logging
import queue
import signal as signal_module
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("regime_trader")

_STATE_FILE = Path("state_snapshot.json")
_LIVE_STATE_FILE = Path("live_state.json")
_MODELS_DIR = Path("models")
_MODEL_MAX_AGE_DAYS = 7
_FEED_TIMEOUT_SEC = 120   # seconds before treating WebSocket as dropped


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="regime-trader")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- backtest subcommand ---
    bt = sub.add_parser("backtest", help="Walk-forward backtest or stress test")
    bt.add_argument("--symbols", nargs="+", default=["SPY"])
    bt.add_argument("--start", default="2019-01-01")
    bt.add_argument("--end", default="2024-12-31")
    bt.add_argument("--compare", action="store_true")
    bt.add_argument("--stress-test", action="store_true")
    bt.add_argument("--output-dir", default="results")
    bt.add_argument("--config", default="config/settings.yaml")

    # --- live subcommand ---
    live = sub.add_parser("live", help="Live / paper trading loop")
    live.add_argument("--paper", action="store_true", help="Use paper trading account")
    live.add_argument("--dry-run", action="store_true",
                      help="Full pipeline, no orders submitted")
    live.add_argument("--train-only", action="store_true",
                      help="Train HMM model and exit")
    live.add_argument("--dashboard", action="store_true",
                      help="Show dashboard for a running live instance")
    live.add_argument("--config", default="config/settings.yaml")

    return parser.parse_args()


# ======================================================================
# Config / credentials
# ======================================================================

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_credentials() -> dict:
    import os
    return {
        "alpaca_api_key": os.getenv("ALPACA_API_KEY", ""),
        "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
        "paper": os.getenv("ALPACA_PAPER", "true").lower() == "true",
    }


# ======================================================================
# Shared data fetch helper (used by backtest + live startup)
# ======================================================================

def _fetch_price_data(
    symbols: list,
    start: str,
    end: str,
    credentials: dict,
    config: dict,
) -> Dict:
    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketDataFetcher

    client = AlpacaClient(
        api_key=credentials["alpaca_api_key"],
        secret_key=credentials["alpaca_secret_key"],
        paper=credentials["paper"],
    )
    client.connect()
    fetcher = MarketDataFetcher(client)
    timeframe = config["broker"].get("timeframe", "1Day")
    return fetcher.fetch_historical(symbols, timeframe, start, end)


# ======================================================================
# Backtest / stress-test runners (Phase 4 — unchanged)
# ======================================================================

def run_backtest(args: argparse.Namespace, config: dict, credentials: dict) -> None:
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer

    price_data = _fetch_price_data(
        args.symbols, args.start, args.end, credentials, config
    )
    backtester = WalkForwardBacktester(config)
    result = backtester.run(price_data, symbols=args.symbols)

    analyzer = PerformanceAnalyzer(
        risk_free_rate=config.get("backtest", {}).get("risk_free_rate", 0.045)
    )
    primary_bars = price_data[args.symbols[0]].rename(columns=str.lower)
    analyzer.print_report(result, price_series=primary_bars["close"], compare=args.compare)
    analyzer.save_csvs(result, Path(args.output_dir))


def run_stress_test(args: argparse.Namespace, config: dict, credentials: dict) -> None:
    from backtest.backtester import WalkForwardBacktester
    from backtest.stress_test import StressTest
    from rich.console import Console
    from rich.table import Table

    console = Console()
    price_data = _fetch_price_data(
        args.symbols, "2015-01-01", args.end, credentials, config
    )
    backtester = WalkForwardBacktester(config)
    stress = StressTest(config)
    results = stress.run_all(price_data, backtester)

    t = Table(title="Stress Test Results", header_style="bold red")
    t.add_column("Scenario", style="cyan")
    t.add_column("Mean Max Loss")
    t.add_column("Worst Case")
    t.add_column("Circuit Breaker %")
    t.add_column("Notes")

    for scenario, r in results.items():
        notes = "; ".join(f"{k}={v}" for k, v in r.details.items())
        t.add_row(
            scenario,
            f"{r.mean_max_loss * 100:.2f}%",
            f"{r.worst_case_loss * 100:.2f}%",
            f"{r.circuit_breaker_pct * 100:.1f}%",
            notes,
        )
    console.print(t)


# ======================================================================
# Train-only runner
# ======================================================================

def run_train_only(args: argparse.Namespace, config: dict, credentials: dict) -> None:
    """Fetch historical data, train HMM, save model, print results, exit."""
    from core.hmm_engine import HMMEngine
    from data.feature_engineering import FeatureEngineer

    _MODELS_DIR.mkdir(exist_ok=True)
    symbols = config["broker"].get("symbols", ["SPY"])
    primary = symbols[0]
    timeframe = config["broker"].get("timeframe", "1Day")

    logger.info("Training HMM for %s (%s)…", primary, timeframe)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    price_data = _fetch_price_data([primary], start, end, credentials, config)
    bars = price_data[primary].rename(columns=str.lower)

    fe = FeatureEngineer()
    features_df = fe.build_hmm_features(bars)
    log_rets = np.log(bars["close"] / bars["close"].shift(1)).dropna()
    features_df, log_rets = features_df.align(log_rets, join="inner")

    hmm_cfg = config.get("hmm", {})
    engine = HMMEngine(
        n_candidates=hmm_cfg.get("n_candidates", [3, 4, 5, 6, 7]),
        n_init=hmm_cfg.get("n_init", 5),
        stability_bars=hmm_cfg.get("stability_bars", 3),
        flicker_window=hmm_cfg.get("flicker_window", 20),
        flicker_threshold=hmm_cfg.get("flicker_threshold", 4),
        min_confidence=hmm_cfg.get("min_confidence", 0.55),
    )
    engine.fit(features_df.values, log_rets.values)

    model_path = _MODELS_DIR / f"hmm_{primary}_{timeframe}.pkl"
    engine.save(model_path)

    print(f"\nHMM trained and saved to {model_path}")
    print(f"  n_regimes: {engine.n_regimes}  BIC: {engine.bic:.2f}")
    print(f"  Labels: {engine.regime_labels}")
    for rid, info in engine.regime_info.items():
        print(
            f"  [{rid}] {info.regime_name:15s}  "
            f"ret={info.expected_return*100:.3f}%  "
            f"vol={info.expected_volatility*100:.1f}%  "
            f"strategy={info.recommended_strategy_type}"
        )


# ======================================================================
# Dashboard viewer (reads live_state.json written by the live loop)
# ======================================================================

def run_dashboard(args: argparse.Namespace, config: dict) -> None:
    from rich.console import Console
    from rich.live import Live

    console = Console()
    if not _LIVE_STATE_FILE.exists():
        console.print("[red]No running instance found.[/red] "
                      f"({_LIVE_STATE_FILE} does not exist)")
        return

    def _render():
        try:
            state = json.loads(_LIVE_STATE_FILE.read_text())
        except Exception:
            return "[red]Cannot read live_state.json[/red]"
        return _build_dashboard_from_state(state)

    with Live(_render(), refresh_per_second=1, screen=True) as live:
        try:
            while True:
                time.sleep(1)
                live.update(_render())
        except KeyboardInterrupt:
            pass


# ======================================================================
# LiveTrader — Phase 7 main loop
# ======================================================================

class LiveTrader:
    """
    Orchestrates startup, the bar-by-bar main loop, and clean shutdown.

    Startup sequence (8 steps):
      1. Connect to Alpaca, verify account
      2. Check market hours
      3. Load or train HMM (retrain if > 7 days old)
      4. Initialize risk manager from live portfolio
      5. Initialize position tracker, sync positions
      6. Load state_snapshot.json if present
      7. Start WebSocket data feeds
      8. Log "System online"

    Main loop (each bar):
      1-7: features → HMM → regime → signals → risk validate → orders
      8: trailing stop updates
      9: circuit breaker check
      10: dashboard refresh
      11: weekly HMM retrain if needed
    """

    def __init__(self, config: dict, credentials: dict, dry_run: bool = False) -> None:
        self.config = config
        self.credentials = credentials
        self.dry_run = dry_run

        self._shutdown_event = __import__("threading").Event()
        self._bar_queue: queue.Queue = queue.Queue()

        # Components — initialized in startup()
        self._client = None
        self._fetcher = None
        self._engine = None
        self._orchestrator = None
        self._risk_manager = None
        self._position_tracker = None
        self._order_executor = None
        self._feature_engineer = None

        # Rolling bar history: {symbol: DataFrame}
        self._bar_history: Dict[str, pd.DataFrame] = {}
        self._features_df: Optional[pd.DataFrame] = None
        self._first_inference: bool = True

        # P&L tracking
        self._session_start_equity: float = 0.0
        self._day_open_equity: float = 0.0
        self._week_open_equity: float = 0.0
        self._prev_day: Optional[tuple] = None
        self._prev_week: Optional[tuple] = None

        # Order tracking: symbol → OrderRecord (for stop modification)
        self._open_order_records: Dict[str, object] = {}

        # Session metadata
        self._session_start: datetime = datetime.utcnow()
        self._bar_count: int = 0
        self._order_count: int = 0
        self._last_regime: str = "UNKNOWN"
        self._daily_trade_count: int = 0
        self._recent_signals: List[dict] = []

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> None:
        _MODELS_DIR.mkdir(exist_ok=True)
        from broker.alpaca_client import AlpacaClient
        from broker.order_executor import OrderExecutor
        from broker.position_tracker import PositionTracker
        from core.hmm_engine import HMMEngine
        from core.regime_strategies import StrategyOrchestrator
        from core.risk_manager import RiskManager
        from data.feature_engineering import FeatureEngineer
        from data.market_data import MarketDataFetcher

        symbols = self.config["broker"].get("symbols", ["SPY"])
        timeframe = self.config["broker"].get("timeframe", "1Day")
        self._symbols = symbols
        self._primary = symbols[0]
        self._timeframe = timeframe
        self._feature_engineer = FeatureEngineer()

        # 1. Connect to Alpaca
        logger.info("Step 1: Connecting to Alpaca…")
        self._client = AlpacaClient(
            api_key=self.credentials["alpaca_api_key"],
            secret_key=self.credentials["alpaca_secret_key"],
            paper=self.credentials["paper"],
        )
        self._client.connect()
        acct = self._client.get_account()
        logger.info("Account verified: equity=$%s status=%s", acct["equity"], acct["status"])

        # 2. Check market hours
        logger.info("Step 2: Checking market hours…")
        clock = self._client.get_clock()
        if not clock["is_open"]:
            logger.info(
                "Market closed. Next open: %s. Continuing in paper/dry-run mode.",
                clock["next_open"],
            )

        # 3. Load or train HMM
        logger.info("Step 3: Loading or training HMM…")
        self._fetcher = MarketDataFetcher(self._client)
        self._engine = self._load_or_train_hmm()
        self._orchestrator = StrategyOrchestrator(self.config, self._engine.regime_info)

        # 4. Initialize risk manager
        logger.info("Step 4: Initializing risk manager…")
        self._risk_manager = RiskManager(self.config)

        # 5. Initialize position tracker, sync
        logger.info("Step 5: Syncing positions…")
        self._position_tracker = PositionTracker(self._client)
        self._order_executor = OrderExecutor(self._client)
        current_regime = self._engine.regime_labels[0] if self._engine.regime_labels else "UNKNOWN"
        self._position_tracker.sync_with_alpaca(current_regime=current_regime)

        # 6. Recover from state_snapshot if present
        logger.info("Step 6: Checking for state snapshot…")
        self._load_state()

        # Seed bar history with enough historical data for feature computation
        end = datetime.now().strftime("%Y-%m-%d")
        lookback_days = max(365 * 3, self._engine.MIN_TRAIN_BARS + 300)
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        logger.info("Loading historical bars for feature window…")
        for sym in symbols:
            df = self._fetcher.get_historical_bars(sym, timeframe, start, end)
            if not df.empty:
                self._bar_history[sym] = df
                logger.info("  %s: %d bars loaded", sym, len(df))

        # Compute initial features and run full forward pass
        self._rebuild_features()
        if self._features_df is not None and len(self._features_df) > 0:
            self._run_full_inference()

        # Seed equity tracking
        equity = float(self._client.get_account()["equity"])
        self._session_start_equity = equity
        self._day_open_equity = equity
        self._week_open_equity = equity

        # 7. Start WebSocket data feeds
        logger.info("Step 7: Starting WebSocket data feeds…")
        self._fetcher.subscribe_bars(symbols, timeframe, self._on_bar)
        self._position_tracker.start_stream()
        self._position_tracker.register_fill_callback(self._on_fill)

        # Register shutdown handlers
        signal_module.signal(signal_module.SIGINT, self._shutdown_handler)
        signal_module.signal(signal_module.SIGTERM, self._shutdown_handler)

        # 8. System online
        logger.info("Step 8: System online. dry_run=%s", self.dry_run)
        self._save_live_state()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        from rich.live import Live

        with Live(self._build_dashboard(), refresh_per_second=0.5) as live:
            while not self._shutdown_event.is_set():
                try:
                    bar_dict = self._bar_queue.get(timeout=_FEED_TIMEOUT_SEC)
                except queue.Empty:
                    logger.warning(
                        "Data feed timeout (%ds) — pausing signals, stops remain active.",
                        _FEED_TIMEOUT_SEC,
                    )
                    live.update(self._build_dashboard())
                    continue

                try:
                    self._process_bar(bar_dict)
                except Exception as exc:
                    logger.error("Unhandled error in process_bar: %s", exc)
                    logger.debug(traceback.format_exc())
                    self._save_state()
                    # Continue — don't crash the loop on a single bad bar

                live.update(self._build_dashboard())

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        logger.info("Shutting down…")

        # Close WebSocket connections
        try:
            if self._fetcher:
                pass  # Streams are daemon threads — they exit with the process
        except Exception:
            pass

        # Do NOT close positions — stops remain in place
        logger.info("Positions left open with stops in place.")

        # Save state snapshot
        self._save_state()

        # Print session summary
        self._print_session_summary()

        # Clean up live state file
        if _LIVE_STATE_FILE.exists():
            try:
                _LIVE_STATE_FILE.unlink()
            except OSError:
                pass

    def _shutdown_handler(self, signum, frame) -> None:
        logger.info("Signal %s received — initiating shutdown.", signum)
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Bar processing (main loop body)
    # ------------------------------------------------------------------

    def _on_bar(self, bar_dict: dict) -> None:
        """WebSocket callback — enqueue bar for processing in main thread."""
        self._bar_queue.put(bar_dict)

    def _on_fill(self, symbol: str, fill_price: float, fill_qty: float, side: str) -> None:
        """Position tracker fill callback — update equity tracking."""
        logger.info("Fill callback: %s %s x%.2f @ $%.2f", side, symbol, fill_qty, fill_price)
        self._daily_trade_count += 1
        self._order_count += 1

    def _process_bar(self, bar_dict: dict) -> None:
        sym = bar_dict.get("symbol", self._primary)
        bar_ts = pd.Timestamp(bar_dict["timestamp"])

        self._bar_count += 1

        # Append bar to history
        new_row = pd.DataFrame(
            [{
                "open": bar_dict["open"],
                "high": bar_dict["high"],
                "low": bar_dict["low"],
                "close": bar_dict["close"],
                "volume": bar_dict["volume"],
            }],
            index=[bar_ts],
        )
        if sym in self._bar_history:
            self._bar_history[sym] = pd.concat(
                [self._bar_history[sym], new_row]
            ).tail(3000)  # cap memory
        else:
            self._bar_history[sym] = new_row

        # Day / week boundary resets
        self._check_day_week_boundaries(bar_ts)

        # --- 2. Compute features (rolling window, no future data) ---
        self._rebuild_features()
        if self._features_df is None or len(self._features_df) < 30:
            return

        # --- 3 & 4. Filtered HMM prediction + stability filter ---
        try:
            if self._first_inference:
                self._run_full_inference()
                self._first_inference = False
                label_idx = self._last_label_idx
                proba = self._last_proba
            else:
                new_obs = self._features_df.iloc[-1].values
                label_idx, proba = self._engine.predict_regime_filtered_incremental(new_obs)
            regime_state = self._engine.update_stability_filter(label_idx, proba)
        except Exception as exc:
            logger.warning("HMM error — holding current regime: %s", exc)
            return

        self._last_regime = regime_state.label

        # --- 5. Flicker rate ---
        is_flickering = self._engine.is_flickering()

        # --- 6. Strategy: target allocation per symbol ---
        signals = self._orchestrator.generate_signals(
            self._symbols, self._bar_history, regime_state, is_flickering
        )

        # --- 7. Risk validation and order execution ---
        acct = self._with_retry(self._client.get_account)
        equity = float(acct["equity"])
        portfolio_state = self._build_portfolio_state(equity, regime_state)

        self._risk_manager.update_circuit_breakers(
            equity,
            equity - self._day_open_equity,
            equity - self._week_open_equity,
            regime_state.label,
        )

        for signal in signals:
            decision = self._risk_manager.validate_signal(signal, portfolio_state)
            sig_log = {
                "ts": bar_ts.isoformat(),
                "symbol": signal.symbol,
                "direction": signal.direction,
                "regime": regime_state.label,
                "approved": decision.approved,
                "reason": decision.rejection_reason or "",
                "mods": decision.modifications,
            }
            self._recent_signals = (self._recent_signals + [sig_log])[-20:]

            if decision.approved and decision.modified_signal is not None:
                final = decision.modified_signal
                if decision.modifications:
                    logger.info("Signal modified: %s", "; ".join(decision.modifications))

                if final.direction == "FLAT":
                    pos = self._position_tracker.get_position(final.symbol)
                    if pos and not self.dry_run:
                        self._order_executor.close_position(final.symbol)
                else:
                    self._execute_long(final)
            else:
                if decision.rejection_reason:
                    logger.info("Signal rejected: %s", decision.rejection_reason)

        # --- 8. Update trailing stops ---
        self._update_trailing_stops(regime_state.label)

        # --- 9. Circuit breaker check ---
        cb_status = self._risk_manager.circuit_breaker.status
        if cb_status in ("halt_daily", "halt_weekly", "halt_peak"):
            logger.warning("CIRCUIT BREAKER HALT: %s", cb_status)

        # --- 10. Save live state for dashboard ---
        self._save_live_state()

        # --- 11. Weekly HMM retrain ---
        self._check_weekly_retrain()

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute_long(self, signal) -> None:
        """Submit a bracket order for a LONG signal, respecting rebalance threshold."""
        pos = self._position_tracker.get_position(signal.symbol)
        acct_info = self._with_retry(self._client.get_account)
        equity = float(acct_info["equity"])
        target_alloc = signal.position_size_pct * signal.leverage

        if pos is not None:
            current_alloc = (pos.qty * pos.current_price) / equity if equity > 0 else 0.0
            if abs(target_alloc - current_alloc) < self.config.get("strategy", {}).get(
                "rebalance_threshold", 0.10
            ):
                return  # drift within threshold, no trade needed

        if self.dry_run:
            logger.info(
                "[DRY-RUN] Would submit bracket: %s LONG %.0f%% @ $%.2f stop=$%.2f",
                signal.symbol,
                signal.position_size_pct * 100,
                signal.entry_price,
                signal.stop_loss,
            )
            return

        try:
            record = self._order_executor.submit_bracket_order(signal)
            self._open_order_records[signal.symbol] = record
            self._position_tracker.set_entry_regime(signal.symbol, signal.regime_name)
            self._position_tracker.set_stop(signal.symbol, signal.stop_loss)
            logger.info(
                "Bracket order submitted: %s [trade_id=%s]",
                signal.symbol, record.trade_id,
            )
        except Exception as exc:
            logger.error("submit_bracket_order failed for %s: %s", signal.symbol, exc)

    # ------------------------------------------------------------------
    # Trailing stop updates
    # ------------------------------------------------------------------

    def _update_trailing_stops(self, regime_name: str) -> None:
        """
        Compute ATR-based trailing stops per regime and tighten if better.
          LowVolBull:         2.0 × ATR(14)
          MidVolCautious:     1.5 × ATR(14)
          HighVolDefensive:   1.0 × ATR(14)
        """
        bars = self._bar_history.get(self._primary)
        if bars is None or len(bars) < 15:
            return

        tr = pd.concat([
            bars["high"] - bars["low"],
            (bars["high"] - bars["close"].shift()).abs(),
            (bars["low"] - bars["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

        if "BULL" in regime_name or "EUPHORIA" in regime_name:
            multiplier = 2.0
        elif "BEAR" in regime_name or "CRASH" in regime_name:
            multiplier = 1.0
        else:
            multiplier = 1.5

        trail_dist = atr * multiplier

        for sym, pos in self._position_tracker.get_all_positions().items():
            new_stop = round(pos.current_price - trail_dist, 2)
            if new_stop <= 0:
                continue

            record = self._open_order_records.get(sym)
            if record and not self.dry_run:
                tightened = self._order_executor.modify_stop(
                    sym, record.order_id, new_stop
                )
                if tightened:
                    self._position_tracker.set_stop(sym, new_stop)

    # ------------------------------------------------------------------
    # HMM training helpers
    # ------------------------------------------------------------------

    def _load_or_train_hmm(self):
        from core.hmm_engine import HMMEngine

        model_path = _MODELS_DIR / f"hmm_{self._primary}_{self._timeframe}.pkl"

        if model_path.exists():
            try:
                engine = HMMEngine.load(model_path)
                age_days = (datetime.utcnow() - engine.training_date).days
                if age_days <= _MODEL_MAX_AGE_DAYS:
                    logger.info(
                        "Loaded HMM from %s (age=%d days, n_regimes=%d)",
                        model_path, age_days, engine.n_regimes,
                    )
                    return engine
                logger.info(
                    "HMM model is %d days old (> %d) — retraining.",
                    age_days, _MODEL_MAX_AGE_DAYS,
                )
            except Exception as exc:
                logger.warning("Failed to load HMM: %s — retraining.", exc)

        return self._retrain_hmm()

    def _retrain_hmm(self):
        from core.hmm_engine import HMMEngine
        from data.feature_engineering import FeatureEngineer

        logger.info("Retraining HMM for %s…", self._primary)
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
        bars = self._fetcher.get_historical_bars(self._primary, self._timeframe, start, end)

        if bars.empty:
            raise RuntimeError("No historical bars available for HMM training.")

        fe = FeatureEngineer()
        features_df = fe.build_hmm_features(bars)
        log_rets = np.log(bars["close"] / bars["close"].shift(1)).dropna()
        features_df, log_rets = features_df.align(log_rets, join="inner")

        hmm_cfg = self.config.get("hmm", {})
        engine = HMMEngine(
            n_candidates=hmm_cfg.get("n_candidates", [3, 4, 5, 6, 7]),
            n_init=hmm_cfg.get("n_init", 5),
            stability_bars=hmm_cfg.get("stability_bars", 3),
            flicker_window=hmm_cfg.get("flicker_window", 20),
            flicker_threshold=hmm_cfg.get("flicker_threshold", 4),
            min_confidence=hmm_cfg.get("min_confidence", 0.55),
        )
        engine.fit(features_df.values, log_rets.values)

        model_path = _MODELS_DIR / f"hmm_{self._primary}_{self._timeframe}.pkl"
        engine.save(model_path)
        logger.info("HMM retrained and saved to %s (n_regimes=%d)", model_path, engine.n_regimes)
        return engine

    def _rebuild_features(self) -> None:
        """Recompute features from current bar_history (no look-ahead)."""
        bars = self._bar_history.get(self._primary)
        if bars is None or len(bars) < 30:
            return
        try:
            self._features_df = self._feature_engineer.build_hmm_features(bars)
        except Exception as exc:
            logger.warning("Feature rebuild failed: %s", exc)

    def _run_full_inference(self) -> None:
        """Full forward pass — used on startup and after HMM retrain."""
        if self._features_df is None or len(self._features_df) == 0:
            return
        features = self._features_df.values
        label_sequence = self._engine.predict_regime_filtered(features)
        proba_matrix = self._engine.predict_regime_proba(features)
        self._last_label_idx = int(label_sequence[-1])
        self._last_proba = proba_matrix[-1]

    def _check_weekly_retrain(self) -> None:
        """Retrain HMM if >7 days have elapsed since last training."""
        if self._engine is None or self._engine.training_date is None:
            return
        age_days = (datetime.utcnow() - self._engine.training_date).days
        if age_days >= _MODEL_MAX_AGE_DAYS:
            logger.info("Weekly HMM retrain triggered (age=%d days).", age_days)
            try:
                self._engine = self._retrain_hmm()
                self._orchestrator.update_regime_infos(self._engine.regime_info)
                self._first_inference = True  # force full forward pass next bar
            except Exception as exc:
                logger.error("Weekly retrain failed: %s — continuing with old model.", exc)

    # ------------------------------------------------------------------
    # Portfolio state assembly
    # ------------------------------------------------------------------

    def _build_portfolio_state(self, equity: float, regime_state) -> object:
        from core.risk_manager import PortfolioState

        positions_mv = {
            sym: pos.qty * pos.current_price
            for sym, pos in self._position_tracker.get_all_positions().items()
        }
        cash = equity - sum(positions_mv.values())

        return PortfolioState(
            equity=equity,
            cash=cash,
            buying_power=float(self._client.get_available_margin()),
            positions=positions_mv,
            daily_pnl=equity - self._day_open_equity,
            weekly_pnl=equity - self._week_open_equity,
            peak_equity=self._risk_manager.circuit_breaker._peak_equity or equity,
            drawdown=self._risk_manager.circuit_breaker._peak_equity and
                     (self._risk_manager.circuit_breaker._peak_equity - equity)
                     / self._risk_manager.circuit_breaker._peak_equity or 0.0,
            circuit_breaker_status=self._risk_manager.circuit_breaker.status,
            flicker_rate=self._engine.get_regime_flicker_rate(),
            open_position_count=self._position_tracker.open_position_count(),
            daily_trade_count=self._daily_trade_count,
        )

    # ------------------------------------------------------------------
    # Day / week boundary tracking
    # ------------------------------------------------------------------

    def _check_day_week_boundaries(self, bar_ts: pd.Timestamp) -> None:
        bar_day = (bar_ts.year, bar_ts.month, bar_ts.day)
        try:
            iso = bar_ts.isocalendar()
            bar_week = (bar_ts.year, iso[1])
        except Exception:
            bar_week = self._prev_week

        if self._prev_day is not None and bar_day != self._prev_day:
            acct = self._with_retry(self._client.get_account)
            equity = float(acct["equity"])
            self._day_open_equity = equity
            self._daily_trade_count = 0
            self._risk_manager.circuit_breaker.reset_daily()

        if self._prev_week is not None and bar_week != self._prev_week:
            acct = self._with_retry(self._client.get_account)
            equity = float(acct["equity"])
            self._week_open_equity = equity
            self._risk_manager.circuit_breaker.reset_weekly()

        self._prev_day = bar_day
        self._prev_week = bar_week

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        state = {
            "version": 1,
            "session_start": self._session_start.isoformat(),
            "last_save": datetime.utcnow().isoformat(),
            "peak_equity": self._risk_manager.circuit_breaker._peak_equity
                           if self._risk_manager else 0.0,
            "day_open_equity": self._day_open_equity,
            "week_open_equity": self._week_open_equity,
            "last_regime": self._last_regime,
            "circuit_breaker_status": (
                self._risk_manager.circuit_breaker.status
                if self._risk_manager else "normal"
            ),
            "daily_trade_count": self._daily_trade_count,
            "bar_count": self._bar_count,
        }
        try:
            _STATE_FILE.write_text(json.dumps(state, indent=2))
            logger.debug("State snapshot saved to %s", _STATE_FILE)
        except OSError as exc:
            logger.warning("Could not save state snapshot: %s", exc)

    def _load_state(self) -> None:
        if not _STATE_FILE.exists():
            return
        try:
            state = json.loads(_STATE_FILE.read_text())
            self._last_regime = state.get("last_regime", "UNKNOWN")
            self._day_open_equity = state.get("day_open_equity", 0.0)
            self._week_open_equity = state.get("week_open_equity", 0.0)
            self._daily_trade_count = state.get("daily_trade_count", 0)
            logger.info(
                "Recovered from state snapshot: regime=%s daily_trades=%d",
                self._last_regime, self._daily_trade_count,
            )
        except Exception as exc:
            logger.warning("Could not load state snapshot: %s", exc)

    def _save_live_state(self) -> None:
        """Write dashboard state for --dashboard viewer."""
        positions = {
            sym: {
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "stop_level": pos.stop_level,
                "regime_at_entry": pos.regime_at_entry,
                "regime_current": pos.regime_current,
                "holding_period_minutes": pos.holding_period_minutes,
            }
            for sym, pos in self._position_tracker.get_all_positions().items()
        } if self._position_tracker else {}

        state = {
            "last_update": datetime.utcnow().isoformat(),
            "regime": self._last_regime,
            "circuit_breaker": (
                self._risk_manager.circuit_breaker.status
                if self._risk_manager else "unknown"
            ),
            "dry_run": self.dry_run,
            "bar_count": self._bar_count,
            "positions": positions,
            "recent_signals": self._recent_signals[-10:],
        }
        try:
            _LIVE_STATE_FILE.write_text(json.dumps(state, indent=2))
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Dashboard rendering
    # ------------------------------------------------------------------

    def _build_dashboard(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.columns import Columns

        acct_info: dict = {}
        try:
            if self._client:
                acct_info = self._client.get_account()
        except Exception:
            pass

        equity = acct_info.get("equity", 0.0)
        daily_pnl = equity - self._day_open_equity
        weekly_pnl = equity - self._week_open_equity
        session_pnl = equity - self._session_start_equity
        cb_status = (
            self._risk_manager.circuit_breaker.status
            if self._risk_manager else "unknown"
        )

        # --- Regime panel ---
        regime_color = (
            "red" if "BEAR" in self._last_regime or "CRASH" in self._last_regime
            else "green" if "BULL" in self._last_regime or "EUPHORIA" in self._last_regime
            else "yellow"
        )
        cb_color = "green" if cb_status == "normal" else "red"
        regime_text = Text()
        regime_text.append(f"  Regime: ", style="bold")
        regime_text.append(f"{self._last_regime}\n", style=f"bold {regime_color}")
        regime_text.append(f"  Circuit Breaker: ", style="bold")
        regime_text.append(f"{cb_status}\n", style=f"bold {cb_color}")
        regime_text.append(f"  Bars processed: {self._bar_count}\n")
        regime_text.append(f"  Dry-run: {self.dry_run}\n")
        regime_panel = Panel(regime_text, title="[bold cyan]Regime[/]", expand=False)

        # --- Portfolio panel ---
        portfolio_text = Text()
        portfolio_text.append(f"  Equity:       ${equity:>12,.2f}\n")
        portfolio_text.append(f"  Daily P&L:    ")
        portfolio_text.append(
            f"${daily_pnl:>+,.2f}\n",
            style="green" if daily_pnl >= 0 else "red",
        )
        portfolio_text.append(f"  Weekly P&L:   ")
        portfolio_text.append(
            f"${weekly_pnl:>+,.2f}\n",
            style="green" if weekly_pnl >= 0 else "red",
        )
        portfolio_text.append(f"  Session P&L:  ")
        portfolio_text.append(
            f"${session_pnl:>+,.2f}\n",
            style="green" if session_pnl >= 0 else "red",
        )
        portfolio_text.append(f"  Orders today: {self._daily_trade_count}\n")
        portfolio_panel = Panel(portfolio_text, title="[bold cyan]Portfolio[/]", expand=False)

        # --- Positions table ---
        pos_table = Table(show_header=True, header_style="bold blue")
        pos_table.add_column("Symbol")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Current", justify="right")
        pos_table.add_column("P&L", justify="right")
        pos_table.add_column("Stop", justify="right")
        pos_table.add_column("Regime@Entry")

        if self._position_tracker:
            for sym, pos in self._position_tracker.get_all_positions().items():
                pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
                pos_table.add_row(
                    sym,
                    f"{pos.qty:.1f}",
                    f"${pos.entry_price:.2f}",
                    f"${pos.current_price:.2f}",
                    Text(f"${pos.unrealized_pnl:+,.2f}", style=pnl_style),
                    f"${pos.stop_level:.2f}",
                    pos.regime_at_entry,
                )
        positions_panel = Panel(pos_table, title="[bold cyan]Positions[/]")

        from rich.console import Group
        return Group(
            Columns([regime_panel, portfolio_panel], equal=True),
            positions_panel,
        )

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _with_retry(self, fn, *args, max_retries: int = 3, **kwargs):
        """Call fn with up to max_retries attempts and exponential backoff."""
        delay = 1.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    logger.warning(
                        "Retry %d/%d for %s (%.1fs): %s",
                        attempt + 1, max_retries, fn.__name__, delay, exc,
                    )
                    time.sleep(delay)
                    delay *= 2
        raise last_exc

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def _print_session_summary(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        duration = datetime.utcnow() - self._session_start
        hours, rem = divmod(int(duration.total_seconds()), 3600)
        mins = rem // 60

        equity = 0.0
        try:
            if self._client:
                equity = float(self._client.get_account()["equity"])
        except Exception:
            pass

        session_pnl = equity - self._session_start_equity
        pnl_pct = session_pnl / self._session_start_equity * 100 if self._session_start_equity else 0

        t = Table(title="Session Summary", header_style="bold")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Duration", f"{hours}h {mins}m")
        t.add_row("Starting equity", f"${self._session_start_equity:,.2f}")
        t.add_row("Ending equity", f"${equity:,.2f}")
        t.add_row("Session P&L", f"${session_pnl:+,.2f} ({pnl_pct:+.2f}%)")
        t.add_row("Bars processed", str(self._bar_count))
        t.add_row("Orders submitted", str(self._order_count))
        t.add_row("Open positions", str(
            self._position_tracker.open_position_count()
            if self._position_tracker else 0
        ))
        t.add_row("CB status", (
            self._risk_manager.circuit_breaker.status
            if self._risk_manager else "unknown"
        ))
        console.print(t)


# ======================================================================
# Dashboard state renderer (used by --dashboard mode)
# ======================================================================

def _build_dashboard_from_state(state: dict):
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    last_update = state.get("last_update", "unknown")
    regime = state.get("regime", "UNKNOWN")
    cb = state.get("circuit_breaker", "unknown")
    bars = state.get("bar_count", 0)
    dry_run = state.get("dry_run", False)

    regime_color = (
        "red" if "BEAR" in regime or "CRASH" in regime
        else "green" if "BULL" in regime or "EUPHORIA" in regime
        else "yellow"
    )
    info = Text()
    info.append(f"  Last update: {last_update}\n")
    info.append(f"  Regime: ", style="bold")
    info.append(f"{regime}\n", style=f"bold {regime_color}")
    info.append(f"  Circuit Breaker: {cb}\n")
    info.append(f"  Bars: {bars}   Dry-run: {dry_run}\n")
    info_panel = Panel(info, title="[bold cyan]Live Instance[/]")

    pos_table = Table(show_header=True, header_style="bold blue")
    pos_table.add_column("Symbol")
    pos_table.add_column("Qty", justify="right")
    pos_table.add_column("Entry", justify="right")
    pos_table.add_column("Current", justify="right")
    pos_table.add_column("Unreal P&L", justify="right")
    pos_table.add_column("Stop", justify="right")

    for sym, pos in state.get("positions", {}).items():
        pnl = pos.get("unrealized_pnl", 0.0)
        pos_table.add_row(
            sym,
            f"{pos.get('qty', 0):.1f}",
            f"${pos.get('entry_price', 0):.2f}",
            f"${pos.get('current_price', 0):.2f}",
            Text(f"${pnl:+,.2f}", style="green" if pnl >= 0 else "red"),
            f"${pos.get('stop_level', 0):.2f}",
        )

    from rich.console import Group
    return Group(
        info_panel,
        Panel(pos_table, title="[bold cyan]Positions[/]"),
    )


# ======================================================================
# Live runner (entry point for `python main.py live`)
# ======================================================================

def run_live(args: argparse.Namespace, config: dict, credentials: dict) -> None:
    """Instantiate and run the LiveTrader."""
    trader = LiveTrader(config, credentials, dry_run=getattr(args, "dry_run", False))
    try:
        trader.startup()
        trader.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.critical("Fatal error in live loop: %s", exc)
        logger.debug(traceback.format_exc())
    finally:
        trader.shutdown()


# ======================================================================
# Main entry point
# ======================================================================

def main() -> None:
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    config_path = Path(getattr(args, "config", "config/settings.yaml"))

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    credentials = load_credentials()

    if args.command == "backtest":
        if args.stress_test:
            run_stress_test(args, config, credentials)
        else:
            run_backtest(args, config, credentials)

    elif args.command == "live":
        if args.dashboard:
            run_dashboard(args, config)
        elif args.train_only:
            run_train_only(args, config, credentials)
        else:
            run_live(args, config, credentials)


if __name__ == "__main__":
    main()
