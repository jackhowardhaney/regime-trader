"""Autonomous swing trader — finds setups, enters plays, manages exits on its own.

The bot runs continuously. Every morning at market open it scans the full symbol
universe for swing setups (EMA bounce, MACD cross, RSI reset, golden cross,
momentum breakout, BB breakout, volume surge). Any symbol scoring ≥65 with an
active signal gets a bracket order submitted automatically. Stops are set at
1.5× ATR(14) below entry; targets at 3.75× ATR(14) above entry (2.5:1 R:R).
No regime logic, no manual triggers.

CLI:
  python main.py live --paper                   # paper trade
  python main.py live --paper --dry-run         # full pipeline, no orders
  python main.py live --paper --scan-only       # run one scan and print results
  python main.py live --dashboard               # display running instance dashboard
  python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
  python main.py backtest --compare
  python main.py backtest --stress-test
"""

import argparse
import json
import logging
import os
import queue
import signal as signal_module
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request as _urllib_request

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("swing_trader")

_STATE_FILE = Path("state_snapshot.json")
_LIVE_STATE_FILE = Path("live_state.json")
_MODELS_DIR = Path("models")
_FEED_TIMEOUT_SEC = 120

# Swing trading parameters
MIN_SCORE = 65          # minimum opportunity score to enter a play
MIN_SCORE_FORCED = 52   # lower threshold when daily minimum not yet hit
MAX_POSITIONS = 15      # hard cap on concurrent open plays
RISK_PER_TRADE = 0.05   # 5% of equity risked per trade ($500 risk on $10k)
ATR_STOP_MULT = 1.5     # stop = entry - ATR * multiplier (scanner logic)
ATR_TARGET_MULT = 3.75  # target = entry + ATR * multiplier (scanner, 2.5:1 R:R)
ATR_PERIOD = 14

# Watchlist entry logic — tighter stops, slightly higher risk (conviction premium)
WATCHLIST_MIN_BLEND  = 70
WATCHLIST_ATR_STOP   = 1.2   # tighter stop — conviction earns precision
WATCHLIST_ATR_TARGET = 3.0   # adjusted to maintain 2.5:1 R:R with tighter stop
WATCHLIST_RISK_PCT   = 0.055

# Adaptive ratio — shifts toward better-performing source over time
MIN_TRADES_TO_ADJUST = 4
RATIO_STEP  = 0.05
RATIO_MIN   = 0.25
RATIO_MAX   = 0.75

WEBAPP_URL = os.getenv("WEBAPP_URL", "").rstrip("/")

# Short selling — margin-aware, capped at 25% of equity notional
MAX_SHORT_POSITIONS    = 4
MAX_SHORT_EXPOSURE_PCT = 0.25   # total short notional ≤ 25% of equity
SHORT_MIN_SCORE        = 60
SHORT_ATR_STOP         = 1.5    # stop = entry + ATR×mult (above entry)
SHORT_ATR_TARGET       = 3.0    # target = entry − ATR×mult (below entry)
SHORT_RISK_PCT         = 0.04

# Daily trading discipline — always active
MIN_DAILY_ENTRIES = 2       # enter at least this many plays per trading day
MIN_DEPLOYED_PCT  = 0.25    # keep at least 25% of equity in open positions at all times


# ── Webapp API helpers (blend ratio + play recording) ──────────────────────────

def _api_get(path: str) -> dict:
    if not WEBAPP_URL:
        return {}
    req = _urllib_request.Request(f"{WEBAPP_URL}{path}", method="GET")
    with _urllib_request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())


def _api_put(path: str, body: dict) -> dict:
    if not WEBAPP_URL:
        return {}
    payload = json.dumps(body).encode()
    req = _urllib_request.Request(
        f"{WEBAPP_URL}{path}", data=payload,
        headers={"Content-Type": "application/json"}, method="PUT",
    )
    with _urllib_request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())


def _api_post(path: str, body: dict) -> dict:
    if not WEBAPP_URL:
        return {}
    payload = json.dumps(body).encode()
    req = _urllib_request.Request(
        f"{WEBAPP_URL}{path}", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with _urllib_request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())


def _fetch_blend_ratio() -> tuple:
    try:
        d = _api_get("/api/blend-ratio")
        return float(d.get("scanner_pct", 0.60)), float(d.get("watchlist_pct", 0.40))
    except Exception:
        return 0.60, 0.40


def _compute_quality(perf: dict) -> float:
    wr  = perf.get("win_rate", 0) / 100.0
    avg = perf.get("avg_pnl", 0)
    return wr * avg if avg > 0 else 0.0


def _update_blend_ratio(scanner_pct: float, watchlist_pct: float) -> None:
    try:
        data = _api_get("/api/blend-analysis")
        perf = data.get("source_performance", {})
        sn   = perf.get("scanner", {}).get("count", 0)
        wn   = perf.get("watchlist", {}).get("count", 0)
        if sn < MIN_TRADES_TO_ADJUST or wn < MIN_TRADES_TO_ADJUST:
            reason = (f"Not enough data (scanner={sn}, watchlist={wn}). "
                      f"Need {MIN_TRADES_TO_ADJUST} each. Keeping {scanner_pct:.0%}/{watchlist_pct:.0%}.")
            _api_put("/api/blend-ratio", {"scanner_pct": scanner_pct, "reason": reason})
            logger.info("Blend ratio unchanged: %s", reason)
            return
        sq = _compute_quality(perf.get("scanner", {}))
        wq = _compute_quality(perf.get("watchlist", {}))
        new_sp = scanner_pct
        if sq <= 0 and wq <= 0:
            new_sp = 0.50
            reason = f"Both sources losing (sq={sq:.1f}, wq={wq:.1f}). Reset 50/50."
        elif sq > 0 and wq <= 0:
            new_sp = min(RATIO_MAX, scanner_pct + RATIO_STEP)
            reason = f"Scanner winning (q={sq:.2f}) / watchlist losing. +{RATIO_STEP:.0%} to scanner."
        elif wq > 0 and sq <= 0:
            new_sp = max(RATIO_MIN, scanner_pct - RATIO_STEP)
            reason = f"Watchlist winning (q={wq:.2f}) / scanner losing. +{RATIO_STEP:.0%} to watchlist."
        else:
            total_q = sq + wq
            ideal   = sq / total_q if total_q else 0.5
            gap     = abs(sq - wq) / max(sq, wq)
            if gap > 0.20:
                new_sp = max(RATIO_MIN, min(RATIO_MAX,
                    scanner_pct + RATIO_STEP * (1 if ideal > scanner_pct else -1)
                ))
                winner = "scanner" if sq > wq else "watchlist"
                reason = f"{winner.title()} leading by {gap:.0%}. Adjusting → scanner {new_sp:.0%}."
            else:
                reason = f"Both positive, gap {gap:.0%} <20%. Holding {scanner_pct:.0%}/{watchlist_pct:.0%}."
        _api_put("/api/blend-ratio", {"scanner_pct": new_sp, "reason": reason})
        logger.info("Blend ratio → scanner=%.0f%% watchlist=%.0f%% | %s",
                    new_sp * 100, (1 - new_sp) * 100, reason)
    except Exception as exc:
        logger.warning("Blend ratio update failed: %s", exc)


def _record_to_dashboard(symbol, direction, entry, stop, target, shares, signal, notes, source):
    try:
        _api_post("/api/plays", {
            "symbol": symbol, "direction": direction,
            "entry_price": entry, "stop_loss": stop, "take_profit": target,
            "shares": shares, "signal": signal, "notes": notes,
            "submit_order": False, "source": source,
        })
    except Exception as exc:
        logger.debug("Dashboard record failed for %s: %s", symbol, exc)


try:
    from data.sector_symbols import ALL_HANDPICK_SYMBOLS as SCAN_SYMBOLS_DEFAULT
except Exception:
    SCAN_SYMBOLS_DEFAULT = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
        "JPM", "GS", "BAC", "MS", "V", "MA",
        "XOM", "CVX", "SLB", "COP",
        "EQIX", "AMT", "PLD",
        "NUE", "CAT", "URI", "HON",
        "TXN", "AVGO", "QCOM",
        "UNH", "LLY", "JNJ", "ABBV",
        "COIN", "IPGP", "PLTR", "MSTR",
        "SPY", "QQQ", "IWM",
    ]


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swing-trader")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Walk-forward backtest or stress test")
    bt.add_argument("--symbols", nargs="+", default=["SPY"])
    bt.add_argument("--start", default="2019-01-01")
    bt.add_argument("--end", default="2024-12-31")
    bt.add_argument("--compare", action="store_true")
    bt.add_argument("--stress-test", action="store_true")
    bt.add_argument("--output-dir", default="results")
    bt.add_argument("--config", default="config/settings.yaml")

    live = sub.add_parser("live", help="Autonomous swing trading loop")
    live.add_argument("--paper", action="store_true")
    live.add_argument("--dry-run", action="store_true",
                      help="Full pipeline, no orders submitted")
    live.add_argument("--scan-only", action="store_true",
                      help="Run one scan, print results, exit")
    live.add_argument("--dashboard", action="store_true")
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
# Backtest runners (unchanged from prior version)
# ======================================================================

def _fetch_price_data(symbols, start, end, credentials, config) -> Dict:
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


def run_backtest(args, config, credentials) -> None:
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer

    price_data = _fetch_price_data(args.symbols, args.start, args.end, credentials, config)
    backtester = WalkForwardBacktester(config)
    result = backtester.run(price_data, symbols=args.symbols)

    analyzer = PerformanceAnalyzer(
        risk_free_rate=config.get("backtest", {}).get("risk_free_rate", 0.045)
    )
    primary_bars = price_data[args.symbols[0]].rename(columns=str.lower)
    analyzer.print_report(result, price_series=primary_bars["close"], compare=args.compare)
    analyzer.save_csvs(result, Path(args.output_dir))


def run_stress_test(args, config, credentials) -> None:
    from backtest.backtester import WalkForwardBacktester
    from backtest.stress_test import StressTest
    from rich.console import Console
    from rich.table import Table

    console = Console()
    price_data = _fetch_price_data(args.symbols, "2015-01-01", args.end, credentials, config)
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
# Dashboard viewer
# ======================================================================

def run_dashboard(args, config) -> None:
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
# ATR helper
# ======================================================================

def _compute_atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - bars["close"].shift()).abs(),
        (bars["low"] - bars["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


# ======================================================================
# SwingTrader — autonomous main loop
# ======================================================================

class SwingTrader:
    """
    Autonomous swing trader. Scans for technical setups every trading day
    at open, enters qualifying plays with bracket orders, and lets stops
    and targets handle all exits. No regime detection, no manual triggers.

    Entry logic (run once per trading day):
      1. Fetch 90-day OHLCV for every symbol in the universe
      2. Score each symbol with the opportunity scanner (0–100)
      3. For symbols scoring ≥65 with an active signal:
         - Compute ATR(14)
         - stop  = current_price - ATR * 1.5
         - target = current_price + ATR * 3.75  (2.5:1 R:R)
         - shares = floor((equity * 0.02) / (ATR * 1.5))
         - Submit bracket order
      4. Skip if already in position or max plays reached (15)

    Exit logic (continuous, via bracket orders placed at entry):
      - Alpaca handles stop and target fills automatically
      - Trailing stop tightened on each bar if ATR-based level improves
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
        self._risk_manager = None
        self._position_tracker = None
        self._order_executor = None

        # Bar history: {symbol: DataFrame}
        self._bar_history: Dict[str, pd.DataFrame] = {}

        # P&L tracking
        self._session_start_equity: float = 0.0
        self._day_open_equity: float = 0.0
        self._week_open_equity: float = 0.0
        self._prev_day: Optional[tuple] = None
        self._prev_week: Optional[tuple] = None

        # Order tracking: symbol → OrderRecord
        self._open_order_records: Dict[str, object] = {}

        # Scale-out tracking: symbols where 90% has already been sold at +7%
        self._scaled_out_symbols: set = set()

        # Session metadata
        self._session_start: datetime = datetime.utcnow()
        self._bar_count: int = 0
        self._order_count: int = 0
        self._daily_trade_count: int = 0
        self._daily_entries: int = 0        # new entries opened today
        self._last_scan_day: Optional[tuple] = None
        self._last_deployment_check: Optional[tuple] = None  # (day, hour)
        self._scan_results: List[dict] = []
        self._recent_entries: List[dict] = []

        self._symbols = config.get("swing", {}).get("universe", SCAN_SYMBOLS_DEFAULT)
        self._timeframe = config["broker"].get("timeframe", "1Day")

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> None:
        from broker.alpaca_client import AlpacaClient
        from broker.order_executor import OrderExecutor
        from broker.position_tracker import PositionTracker
        from core.risk_manager import RiskManager
        from data.market_data import MarketDataFetcher

        logger.info("SwingTrader starting up — autonomous mode")

        logger.info("Step 1: Connecting to Alpaca…")
        self._client = AlpacaClient(
            api_key=self.credentials["alpaca_api_key"],
            secret_key=self.credentials["alpaca_secret_key"],
            paper=self.credentials["paper"],
        )
        self._client.connect()
        acct = self._client.get_account()
        logger.info("Account: equity=$%s status=%s", acct["equity"], acct["status"])

        logger.info("Step 2: Checking market hours…")
        clock = self._client.get_clock()
        if not clock["is_open"]:
            logger.info("Market closed. Next open: %s", clock["next_open"])

        logger.info("Step 3: Initializing components…")
        self._fetcher = MarketDataFetcher(self._client)
        self._risk_manager = RiskManager(self.config)
        self._position_tracker = PositionTracker(self._client)
        self._order_executor = OrderExecutor(self._client)
        self._position_tracker.sync_with_alpaca(current_regime="SWING")
        self._position_tracker.start_stream()
        self._position_tracker.register_fill_callback(self._on_fill)

        logger.info("Step 4: Loading bar history for universe…")
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
        for sym in self._symbols:
            try:
                df = self._fetcher.get_historical_bars(sym, self._timeframe, start, end)
                if not df.empty:
                    self._bar_history[sym] = df
            except Exception as exc:
                logger.warning("Could not load history for %s: %s", sym, exc)
        logger.info("Loaded history for %d/%d symbols", len(self._bar_history), len(self._symbols))

        equity = float(self._client.get_account()["equity"])
        self._session_start_equity = equity
        self._day_open_equity = equity
        self._week_open_equity = equity

        logger.info("Step 5: Starting WebSocket data feeds…")
        self._fetcher.subscribe_bars(self._symbols, self._timeframe, self._on_bar)

        signal_module.signal(signal_module.SIGINT, self._shutdown_handler)
        signal_module.signal(signal_module.SIGTERM, self._shutdown_handler)

        logger.info("Step 6: Running initial scan…")
        self._scan_and_enter()

        logger.info("System online — bot is making plays autonomously. dry_run=%s", self.dry_run)
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
                    logger.warning("Data feed timeout (%ds) — waiting for next bar.", _FEED_TIMEOUT_SEC)
                    live.update(self._build_dashboard())
                    continue

                try:
                    self._process_bar(bar_dict)
                except Exception as exc:
                    logger.error("Unhandled error in process_bar: %s", exc)
                    logger.debug(traceback.format_exc())
                    self._save_state()

                live.update(self._build_dashboard())

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        logger.info("Shutting down — positions left open with stops in place.")
        self._save_state()
        self._print_session_summary()
        if _LIVE_STATE_FILE.exists():
            try:
                _LIVE_STATE_FILE.unlink()
            except OSError:
                pass

    def _shutdown_handler(self, signum, frame) -> None:
        logger.info("Signal %s received — shutting down.", signum)
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Bar ingestion
    # ------------------------------------------------------------------

    def _on_bar(self, bar_dict: dict) -> None:
        self._bar_queue.put(bar_dict)

    def _on_fill(self, symbol: str, fill_price: float, fill_qty: float, side: str) -> None:
        logger.info("Fill: %s %s x%.2f @ $%.2f", side, symbol, fill_qty, fill_price)
        self._daily_trade_count += 1
        self._order_count += 1

    def _process_bar(self, bar_dict: dict) -> None:
        sym = bar_dict.get("symbol", self._symbols[0])
        bar_ts = pd.Timestamp(bar_dict["timestamp"])
        self._bar_count += 1

        # Append bar to rolling history
        new_row = pd.DataFrame(
            [{"open": bar_dict["open"], "high": bar_dict["high"],
              "low": bar_dict["low"], "close": bar_dict["close"],
              "volume": bar_dict["volume"]}],
            index=[bar_ts],
        )
        if sym in self._bar_history:
            self._bar_history[sym] = pd.concat([self._bar_history[sym], new_row]).tail(500)
        else:
            self._bar_history[sym] = new_row

        self._check_day_week_boundaries(bar_ts)

        # Once per trading day: scan for new swing setups and enter plays
        bar_day = (bar_ts.year, bar_ts.month, bar_ts.day)
        if self._last_scan_day != bar_day:
            self._last_scan_day = bar_day
            logger.info("New trading day (%s-%s-%s) — running swing scan.", *bar_day)
            self._scan_and_enter()

        # Update trailing stops and check scale-out triggers on every bar
        self._update_trailing_stops()
        self._check_scale_out()

        # Deployment check — if < 25% of equity deployed, scan immediately
        check_key = (bar_ts.year, bar_ts.month, bar_ts.day, bar_ts.hour)
        if self._last_deployment_check != check_key:
            self._last_deployment_check = check_key
            try:
                acct_tmp = self._with_retry(self._client.get_account)
                eq_tmp   = float(acct_tmp["equity"])
                bp_tmp   = float(acct_tmp.get("buying_power", eq_tmp))
                # buying_power on a margin account ≈ 2× undeployed cash
                # deployed ≈ (equity - cash) / equity
                cash_tmp    = float(acct_tmp.get("cash", 0))
                deployed_pct = max(0.0, (eq_tmp - max(0, cash_tmp)) / eq_tmp) if eq_tmp > 0 else 0
                n_pos = len(self._position_tracker.get_all_positions())
                if deployed_pct < MIN_DEPLOYED_PCT and n_pos < MAX_POSITIONS:
                    logger.info(
                        "Deployed %.1f%% < %.0f%% target — triggering extra scan.",
                        deployed_pct * 100, MIN_DEPLOYED_PCT * 100,
                    )
                    self._scan_and_enter()
            except Exception as exc:
                logger.debug("Deployment check failed: %s", exc)

        # Circuit breaker
        acct = self._with_retry(self._client.get_account)
        equity = float(acct["equity"])
        self._risk_manager.update_circuit_breakers(
            equity,
            equity - self._day_open_equity,
            equity - self._week_open_equity,
            "SWING",
        )
        cb_status = self._risk_manager.circuit_breaker.status
        if cb_status not in ("normal", "ok"):
            logger.warning("CIRCUIT BREAKER: %s — skipping new entries.", cb_status)

        self._save_live_state()

    # ------------------------------------------------------------------
    # Core: scan and autonomously enter plays
    # ------------------------------------------------------------------

    def _scan_and_enter(self) -> None:
        """
        Score the full universe with adaptive two-logic blend (scanner + watchlist),
        rank by composite score, and submit bracket orders for qualifying setups.
        Also scores bearish setups for short selling (capped at 25% exposure).
        """
        from core.opportunity_scanner import score_symbol, score_symbol_short

        cb_status = (
            self._risk_manager.circuit_breaker.status
            if self._risk_manager else "normal"
        )
        if cb_status not in ("normal", "ok"):
            logger.info("Circuit breaker active (%s) — skipping scan.", cb_status)
            return

        acct = self._with_retry(self._client.get_account)
        equity = float(acct["equity"])
        open_positions = set(self._position_tracker.get_all_positions().keys())
        n_open = len(open_positions)

        if n_open >= MAX_POSITIONS:
            logger.info("Max positions reached (%d/%d) — running ratio update only.", n_open, MAX_POSITIONS)
            scanner_pct, watchlist_pct = _fetch_blend_ratio()
            _update_blend_ratio(scanner_pct, watchlist_pct)
            return

        # Tally existing short positions for exposure cap
        all_alpaca_positions = []
        try:
            all_alpaca_positions = self._client.get_positions()
        except Exception:
            pass
        existing_shorts = [p for p in all_alpaca_positions if p.get("side") == "short"]
        n_existing_shorts = len(existing_shorts)
        current_short_exp = sum(abs(float(p.get("market_value", 0))) for p in existing_shorts)

        acct_fresh = self._with_retry(self._client.get_account)
        equity = float(acct_fresh["equity"])
        max_short_exposure = equity * MAX_SHORT_EXPOSURE_PCT
        short_slots = max(0, MAX_SHORT_POSITIONS - n_existing_shorts)

        # Adaptive ratio — split remaining long slots between scanner and watchlist
        scanner_pct, watchlist_pct = _fetch_blend_ratio()
        long_slots_total = MAX_POSITIONS - n_open - short_slots
        long_slots_total = max(0, long_slots_total)
        slots_total = long_slots_total
        scanner_slots   = max(1, round(slots_total * scanner_pct)) if slots_total > 0 else 0
        watchlist_slots = max(0, slots_total - scanner_slots)
        if slots_total == 1:
            scanner_slots   = 1 if scanner_pct >= watchlist_pct else 0
            watchlist_slots = 1 - scanner_slots

        # Decide score threshold — lower it if we haven't hit daily minimum by midday
        now_et_hour = (datetime.utcnow().hour - 4) % 24
        threshold = (MIN_SCORE_FORCED
                     if self._daily_entries < MIN_DAILY_ENTRIES and now_et_hour >= 12
                     else MIN_SCORE)

        logger.info(
            "Scan: slots=%d (scanner=%d, watchlist=%d)  ratio=%.0f%%/%.0f%%  threshold=%d",
            slots_total, scanner_slots, watchlist_slots,
            scanner_pct * 100, watchlist_pct * 100, threshold,
        )

        # ── SCANNER LOGIC (long + short in one pass) ──────────────────────────
        scored: List[dict] = []
        scored_short: List[dict] = []
        for sym, bars in self._bar_history.items():
            if len(bars) < 50:
                continue
            try:
                result = score_symbol(bars, sym)
                if result:
                    scored.append(result)
                rs = score_symbol_short(bars, sym)
                if rs:
                    scored_short.append(rs)
            except Exception as exc:
                logger.debug("score_symbol failed for %s: %s", sym, exc)

        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        self._scan_results = scored[:20]

        scanner_candidates = [
            r for r in scored
            if r.get("score", 0) >= threshold
            and r.get("firing_signals")
            and r["symbol"] not in open_positions
        ]

        scanner_entered = 0
        for cand in scanner_candidates:
            if scanner_entered >= scanner_slots:
                break
            if self._enter_play(cand, equity, source="scanner", direction="LONG"):
                scanner_entered += 1
                open_positions.add(cand["symbol"])

        # ── WATCHLIST LOGIC ───────────────────────────────────────────────────
        watchlist_entered = 0
        if watchlist_slots > 0:
            try:
                blend_data = _api_get("/api/blend-analysis")
                wl_candidates = [
                    c for c in blend_data.get("candidates", [])
                    if c.get("blend_score", 0) >= WATCHLIST_MIN_BLEND
                    and c.get("bsh") == "BUY"
                    and c["symbol"] not in open_positions
                ]
                logger.info("Watchlist: %d candidates (blend ≥ %d)", len(wl_candidates), WATCHLIST_MIN_BLEND)
                end_str   = datetime.now().strftime("%Y-%m-%d")
                start_str = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
                for cand in wl_candidates:
                    if watchlist_entered >= watchlist_slots:
                        break
                    wl_sym = cand["symbol"]
                    # Fetch bars if this symbol isn't in the scanner universe
                    if wl_sym not in self._bar_history and self._fetcher:
                        try:
                            df = self._fetcher.get_historical_bars(
                                wl_sym, self._timeframe, start_str, end_str
                            )
                            if df is not None and not df.empty:
                                self._bar_history[wl_sym] = df
                            else:
                                logger.debug("No bars for watchlist symbol %s — skipping", wl_sym)
                                continue
                        except Exception as exc:
                            logger.debug("Bar fetch failed for watchlist %s: %s", wl_sym, exc)
                            continue
                    wl_entry = {
                        "symbol": wl_sym,
                        "score": int(cand.get("blend_score", 0)),
                        "firing_signals": ["watchlist_blend"],
                    }
                    if self._enter_play(
                        wl_entry, equity,
                        atr_stop_mult=WATCHLIST_ATR_STOP,
                        atr_target_mult=WATCHLIST_ATR_TARGET,
                        risk_pct=WATCHLIST_RISK_PCT,
                        source="watchlist",
                    ):
                        watchlist_entered += 1
                        open_positions.add(wl_sym)
            except Exception as exc:
                logger.warning("Watchlist scan failed: %s", exc)

        # ── SHORTS ────────────────────────────────────────────────────────────
        short_entered = 0
        if short_slots > 0 and current_short_exp < max_short_exposure:
            # Merge scanner shorts + watchlist SELL picks
            short_candidates: List[dict] = [
                r for r in scored_short
                if r.get("short_score", 0) >= SHORT_MIN_SCORE
                and r["symbol"] not in open_positions
                and r["symbol"] not in {p.get("symbol") for p in existing_shorts}
            ]
            # Watchlist SELL picks get +10% conviction bonus
            try:
                wl_sell_data = _api_get("/api/blend-analysis")
                wl_sells = [
                    c["symbol"] for c in wl_sell_data.get("candidates", [])
                    if c.get("bsh") == "SELL"
                ]
                for sc in short_candidates:
                    if sc["symbol"] in wl_sells:
                        sc["short_score"] = min(100.0, sc["short_score"] * 1.10)
            except Exception:
                pass
            # Also add watchlist SELL picks not already in scored_short
            scored_short_syms = {r["symbol"] for r in scored_short}
            try:
                end_str   = datetime.now().strftime("%Y-%m-%d")
                start_str = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
                for wl_sym in (wl_sells if 'wl_sells' in dir() else []):
                    if wl_sym in scored_short_syms or wl_sym in open_positions:
                        continue
                    if wl_sym not in self._bar_history and self._fetcher:
                        try:
                            df = self._fetcher.get_historical_bars(
                                wl_sym, self._timeframe, start_str, end_str
                            )
                            if df is not None and not df.empty:
                                self._bar_history[wl_sym] = df
                        except Exception:
                            continue
                    wl_bars = self._bar_history.get(wl_sym)
                    if wl_bars is None or len(wl_bars) < 50:
                        continue
                    rs = score_symbol_short(wl_bars, wl_sym)
                    if rs and rs.get("short_score", 0) >= SHORT_MIN_SCORE:
                        rs["short_score"] = min(100.0, rs["short_score"] * 1.10)
                        short_candidates.append(rs)
            except Exception as exc:
                logger.debug("Watchlist SELL short scoring failed: %s", exc)

            short_candidates.sort(key=lambda x: x.get("short_score", 0), reverse=True)
            short_exp_ref = [current_short_exp]

            for sc in short_candidates:
                if short_entered >= short_slots:
                    break
                if short_exp_ref[0] >= max_short_exposure:
                    break
                sym = sc["symbol"]
                bars = self._bar_history.get(sym)
                if bars is None or len(bars) < ATR_PERIOD + 2:
                    continue
                atr = _compute_atr(bars)
                if atr <= 0:
                    continue
                price = float(bars["close"].iloc[-1])
                risk_per_share = atr * SHORT_ATR_STOP
                shares_by_risk = int((equity * SHORT_RISK_PCT) / risk_per_share)
                remaining_budget = max_short_exposure - short_exp_ref[0]
                shares_by_budget = int(remaining_budget / price) if price > 0 else 0
                shares = min(shares_by_risk, shares_by_budget)
                if shares < 1:
                    continue
                # Margin check: short requires 50% margin, keep within 90% of buying power
                bp = float(acct_fresh.get("buying_power", equity))
                cost = price * shares
                if cost * 0.50 > bp * 0.90:
                    logger.debug("Short %s margin check failed (cost=%.0f bp=%.0f) — skipping", sym, cost, bp)
                    continue
                short_cand = {
                    "symbol": sym,
                    "score": int(sc.get("short_score", 0)),
                    "short_score": sc.get("short_score", 0),
                    "firing_signals": sc.get("firing_signals", ["short_scan"]),
                }
                if self._enter_play(
                    short_cand, equity,
                    atr_stop_mult=SHORT_ATR_STOP,
                    atr_target_mult=SHORT_ATR_TARGET,
                    risk_pct=SHORT_RISK_PCT,
                    source="scanner",
                    direction="SHORT",
                ):
                    short_entered += 1
                    short_exp_ref[0] += cost
                    open_positions.add(sym)

        logger.info(
            "Scan done: entered %d (scanner=%d, watchlist=%d, short=%d)",
            scanner_entered + watchlist_entered + short_entered,
            scanner_entered, watchlist_entered, short_entered,
        )

        # ── UPDATE ADAPTIVE RATIO ─────────────────────────────────────────────
        _update_blend_ratio(scanner_pct, watchlist_pct)

    def _enter_play(
        self, candidate: dict, equity: float,
        atr_stop_mult: float = ATR_STOP_MULT,
        atr_target_mult: float = ATR_TARGET_MULT,
        risk_pct: float = RISK_PER_TRADE,
        source: str = "scanner",
        direction: str = "LONG",
    ) -> bool:
        """Submit a bracket order for a qualifying swing setup. Returns True on success."""
        sym = candidate["symbol"]
        bars = self._bar_history.get(sym)
        if bars is None or len(bars) < ATR_PERIOD + 2:
            logger.warning("Not enough bars for %s — skipping.", sym)
            return False

        price = float(bars["close"].iloc[-1])
        atr = _compute_atr(bars)
        if atr <= 0:
            return False

        if direction == "SHORT":
            stop = round(price + atr * atr_stop_mult, 2)
            target = round(price - atr * atr_target_mult, 2)
            if target <= 0:
                return False
            risk_per_share = stop - price
        else:
            stop = round(price - atr * atr_stop_mult, 2)
            target = max(round(price + atr * atr_target_mult, 2), round(price * 1.07, 2))
            risk_per_share = price - stop

        if risk_per_share <= 0:
            return False

        shares = int((equity * risk_pct) / risk_per_share)
        if shares < 1:
            logger.info("Position too small for %s — skipping.", sym)
            return False

        signals = candidate.get("firing_signals", [])
        score = candidate.get("score", candidate.get("short_score", 0))
        strategy = signals[0] if signals else "composite"

        log_entry = {
            "ts": datetime.utcnow().isoformat(), "symbol": sym, "score": score,
            "signals": signals, "entry": price, "stop": stop, "target": target,
            "shares": shares, "atr": round(atr, 4), "source": source,
            "direction": direction,
        }

        if self.dry_run:
            logger.info(
                "[DRY-RUN] %s %s  %s: score=%d  %dsh @ $%.2f  stop=$%.2f  target=$%.2f",
                direction, source.upper(), sym, score, shares, price, stop, target,
            )
            self._recent_entries = (self._recent_entries + [log_entry])[-20:]
            return True

        try:
            from core.regime_strategies import Signal
            signal = Signal(
                symbol=sym, direction=direction, confidence=score / 100.0,
                entry_price=price, stop_loss=stop, take_profit=target,
                position_size_pct=shares * price / equity, leverage=1.0,
                regime_id=0, regime_name="SWING", regime_probability=1.0,
                timestamp=datetime.utcnow(),
                reasoning=f"{source} {direction} score={score} signals={signals}",
                strategy_name=strategy,
            )
            record = self._order_executor.submit_bracket_order(signal)
            self._open_order_records[sym] = record
            self._position_tracker.set_entry_regime(sym, "SWING")
            self._position_tracker.set_stop(sym, stop)
            self._recent_entries = (self._recent_entries + [log_entry])[-20:]
            self._daily_entries += 1
            rr = abs(target - price) / risk_per_share if risk_per_share > 0 else 0
            _record_to_dashboard(
                sym, direction, price, stop, target, shares, strategy,
                f"{source} {direction} score={score} signals={','.join(signals)}", source,
            )
            logger.info(
                "[%s %s] %s: score=%d  %dsh @ $%.2f  stop=$%.2f  target=$%.2f  R:R=%.1f:1",
                direction, source.upper(), sym, score, shares, price, stop, target, rr,
            )
            return True
        except Exception as exc:
            logger.error("Failed to enter %s %s: %s", direction, sym, exc)
            return False

    # ------------------------------------------------------------------
    # Trailing stop management
    # ------------------------------------------------------------------

    def _update_trailing_stops(self) -> None:
        """Tighten ATR-based trailing stops for all open positions."""
        for sym, pos in self._position_tracker.get_all_positions().items():
            bars = self._bar_history.get(sym)
            if bars is None or len(bars) < ATR_PERIOD + 2:
                continue
            try:
                atr = _compute_atr(bars)
                new_stop = round(pos.current_price - atr * ATR_STOP_MULT, 2)
                if new_stop <= 0:
                    continue
                record = self._open_order_records.get(sym)
                if record and not self.dry_run:
                    tightened = self._order_executor.modify_stop(sym, record.order_id, new_stop)
                    if tightened:
                        self._position_tracker.set_stop(sym, new_stop)
            except Exception as exc:
                logger.debug("Trailing stop update failed for %s: %s", sym, exc)

    def _check_scale_out(self) -> None:
        """Sell 90% of any position that has reached +7% gain, leaving a trailing runner."""
        for sym, pos in self._position_tracker.get_all_positions().items():
            if sym in self._scaled_out_symbols:
                continue
            entry = pos.entry_price
            if entry <= 0:
                continue
            gain_pct = (pos.current_price - entry) / entry
            if gain_pct < 0.07:
                continue

            total_qty = int(pos.qty)
            if total_qty < 2:
                self._scaled_out_symbols.add(sym)
                continue

            sell_qty = max(1, int(total_qty * 0.90))
            keep_qty = total_qty - sell_qty
            if keep_qty < 1:
                sell_qty = total_qty - 1
                keep_qty = 1

            logger.info(
                "SCALE-OUT %s: +%.1f%% — selling %d/%d shares @ ~$%.2f, keeping %d as runner",
                sym, gain_pct * 100, sell_qty, total_qty, pos.current_price, keep_qty,
            )

            if self.dry_run:
                self._scaled_out_symbols.add(sym)
                continue

            try:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.enums import OrderSide, TimeInForce
                from alpaca.trading.requests import MarketOrderRequest
                tc = TradingClient(
                    self.credentials["alpaca_api_key"],
                    self.credentials["alpaca_secret_key"],
                    paper=self.credentials["paper"],
                )
                order_req = MarketOrderRequest(
                    symbol=sym,
                    qty=sell_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                tc.submit_order(order_req)
                self._scaled_out_symbols.add(sym)
            except Exception as exc:
                logger.error("Scale-out order failed for %s: %s", sym, exc)

    # ------------------------------------------------------------------
    # Portfolio / state helpers
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
            self._daily_entries = 0
            self._last_deployment_check = None
            self._risk_manager.circuit_breaker.reset_daily()

        if self._prev_week is not None and bar_week != self._prev_week:
            acct = self._with_retry(self._client.get_account)
            equity = float(acct["equity"])
            self._week_open_equity = equity
            self._risk_manager.circuit_breaker.reset_weekly()

        self._prev_day = bar_day
        self._prev_week = bar_week

    def _save_state(self) -> None:
        state = {
            "version": 2,
            "session_start": self._session_start.isoformat(),
            "last_save": datetime.utcnow().isoformat(),
            "day_open_equity": self._day_open_equity,
            "week_open_equity": self._week_open_equity,
            "circuit_breaker_status": (
                self._risk_manager.circuit_breaker.status
                if self._risk_manager else "normal"
            ),
            "daily_trade_count": self._daily_trade_count,
            "bar_count": self._bar_count,
        }
        try:
            _STATE_FILE.write_text(json.dumps(state, indent=2))
        except OSError as exc:
            logger.warning("Could not save state: %s", exc)

    def _save_live_state(self) -> None:
        positions = {
            sym: {
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "stop_level": pos.stop_level,
                "regime_at_entry": pos.regime_at_entry,
                "holding_period_minutes": pos.holding_period_minutes,
            }
            for sym, pos in self._position_tracker.get_all_positions().items()
        } if self._position_tracker else {}

        state = {
            "last_update": datetime.utcnow().isoformat(),
            "mode": "SWING_AUTO",
            "circuit_breaker": (
                self._risk_manager.circuit_breaker.status
                if self._risk_manager else "unknown"
            ),
            "dry_run": self.dry_run,
            "bar_count": self._bar_count,
            "open_positions": len(positions),
            "max_positions": MAX_POSITIONS,
            "positions": positions,
            "top_scan": self._scan_results[:10],
            "recent_entries": self._recent_entries[-10:],
        }
        try:
            _LIVE_STATE_FILE.write_text(json.dumps(state, indent=2))
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Dashboard (Rich terminal UI)
    # ------------------------------------------------------------------

    def _build_dashboard(self):
        from rich.columns import Columns
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        acct_info: dict = {}
        try:
            if self._client:
                acct_info = self._client.get_account()
        except Exception:
            pass

        equity = float(acct_info.get("equity", 0.0))
        daily_pnl = equity - self._day_open_equity
        session_pnl = equity - self._session_start_equity
        cb_status = (
            self._risk_manager.circuit_breaker.status
            if self._risk_manager else "unknown"
        )
        n_open = self._position_tracker.open_position_count() if self._position_tracker else 0
        cb_color = "green" if cb_status in ("normal", "ok") else "red"

        status_text = Text()
        status_text.append("  Mode: ", style="bold")
        status_text.append("SWING AUTO\n", style="bold cyan")
        status_text.append("  Positions: ", style="bold")
        status_text.append(f"{n_open}/{MAX_POSITIONS}\n",
                           style="yellow" if n_open >= MAX_POSITIONS else "green")
        status_text.append("  Circuit Breaker: ", style="bold")
        status_text.append(f"{cb_status}\n", style=f"bold {cb_color}")
        status_text.append(f"  Bars: {self._bar_count}  Dry-run: {self.dry_run}\n")
        status_panel = Panel(status_text, title="[bold cyan]Status[/]", expand=False)

        portfolio_text = Text()
        portfolio_text.append(f"  Equity:       ${equity:>12,.2f}\n")
        portfolio_text.append("  Daily P&L:    ")
        portfolio_text.append(f"${daily_pnl:>+,.2f}\n", style="green" if daily_pnl >= 0 else "red")
        portfolio_text.append("  Session P&L:  ")
        portfolio_text.append(f"${session_pnl:>+,.2f}\n", style="green" if session_pnl >= 0 else "red")
        portfolio_text.append(f"  Orders today: {self._daily_trade_count}\n")
        portfolio_panel = Panel(portfolio_text, title="[bold cyan]Portfolio[/]", expand=False)

        pos_table = Table(show_header=True, header_style="bold blue")
        pos_table.add_column("Symbol")
        pos_table.add_column("Shares", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Current", justify="right")
        pos_table.add_column("P&L", justify="right")
        pos_table.add_column("Stop", justify="right")
        pos_table.add_column("Target", justify="right")

        if self._position_tracker:
            for sym, pos in self._position_tracker.get_all_positions().items():
                pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
                record = self._open_order_records.get(sym)
                target_str = (
                    f"${record.take_profit:.2f}" if record and hasattr(record, "take_profit")
                    else "—"
                )
                pos_table.add_row(
                    sym,
                    f"{pos.qty:.0f}",
                    f"${pos.entry_price:.2f}",
                    f"${pos.current_price:.2f}",
                    Text(f"${pos.unrealized_pnl:+,.2f}", style=pnl_style),
                    f"${pos.stop_level:.2f}",
                    target_str,
                )

        scan_table = Table(show_header=True, header_style="bold magenta")
        scan_table.add_column("Symbol")
        scan_table.add_column("Score", justify="right")
        scan_table.add_column("Signals")
        for r in self._scan_results[:8]:
            score = r.get("score", 0)
            score_color = "green" if score >= MIN_SCORE else "yellow"
            scan_table.add_row(
                r.get("symbol", ""),
                Text(str(score), style=score_color),
                ", ".join(r.get("firing_signals", [])),
            )

        return Group(
            Columns([status_panel, portfolio_panel], equal=True),
            Panel(pos_table, title="[bold cyan]Open Plays[/]"),
            Panel(scan_table, title="[bold cyan]Last Scan (top 8)[/]"),
        )

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _with_retry(self, fn, *args, max_retries: int = 3, **kwargs):
        delay = 1.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    logger.warning("Retry %d/%d for %s: %s", attempt + 1, max_retries,
                                   fn.__name__, exc)
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
        t.add_row("Mode", "SWING AUTO")
        t.add_row("Duration", f"{hours}h {rem // 60}m")
        t.add_row("Starting equity", f"${self._session_start_equity:,.2f}")
        t.add_row("Ending equity", f"${equity:,.2f}")
        t.add_row("Session P&L", f"${session_pnl:+,.2f} ({pnl_pct:+.2f}%)")
        t.add_row("Bars processed", str(self._bar_count))
        t.add_row("Orders submitted", str(self._order_count))
        t.add_row("Open positions", str(
            self._position_tracker.open_position_count()
            if self._position_tracker else 0
        ))
        console.print(t)


# ======================================================================
# Scan-only runner
# ======================================================================

def run_scan_only(config: dict, credentials: dict) -> None:
    """Fetch bars, score the universe, and print results. Then exit."""
    from broker.alpaca_client import AlpacaClient
    from core.opportunity_scanner import score_symbol
    from data.market_data import MarketDataFetcher
    from rich.console import Console
    from rich.table import Table

    console = Console()
    symbols = config.get("swing", {}).get("universe", SCAN_SYMBOLS_DEFAULT)

    client = AlpacaClient(
        api_key=credentials["alpaca_api_key"],
        secret_key=credentials["alpaca_secret_key"],
        paper=credentials["paper"],
    )
    client.connect()
    fetcher = MarketDataFetcher(client)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")

    console.print(f"[bold]Scanning {len(symbols)} symbols…[/]")
    results = []
    for sym in symbols:
        try:
            df = fetcher.get_historical_bars(sym, "1Day", start, end)
            if df is not None and len(df) >= 50:
                r = score_symbol(df, sym)
                if r:
                    results.append(r)
        except Exception:
            pass

    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    t = Table(title="Swing Scan Results", header_style="bold magenta")
    t.add_column("Symbol")
    t.add_column("Score", justify="right")
    t.add_column("Price", justify="right")
    t.add_column("Signals")
    t.add_column("Qualifies?")

    for r in results:
        score = r.get("score", 0)
        qualifies = score >= MIN_SCORE and bool(r.get("firing_signals"))
        t.add_row(
            r.get("symbol", ""),
            f"[green]{score}[/]" if score >= MIN_SCORE else str(score),
            f"${r.get('price', 0):.2f}",
            ", ".join(r.get("firing_signals", [])),
            "[green]YES[/]" if qualifies else "[dim]no[/]",
        )

    console.print(t)
    console.print(f"\n[bold]Qualifying plays (score ≥ {MIN_SCORE} with signal):[/] "
                  f"{sum(1 for r in results if r.get('score', 0) >= MIN_SCORE and r.get('firing_signals'))}")


# ======================================================================
# Dashboard state renderer
# ======================================================================

def _build_dashboard_from_state(state: dict):
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    last_update = state.get("last_update", "unknown")
    mode = state.get("mode", "SWING_AUTO")
    cb = state.get("circuit_breaker", "unknown")
    n_open = state.get("open_positions", 0)
    max_pos = state.get("max_positions", MAX_POSITIONS)
    dry_run = state.get("dry_run", False)

    info = Text()
    info.append(f"  Mode: {mode}\n")
    info.append(f"  Last update: {last_update}\n")
    info.append(f"  Positions: {n_open}/{max_pos}    CB: {cb}    dry_run: {dry_run}\n")
    info_panel = Panel(info, title="[bold cyan]Live Instance[/]")

    pos_table = Table(show_header=True, header_style="bold blue")
    pos_table.add_column("Symbol")
    pos_table.add_column("Shares", justify="right")
    pos_table.add_column("Entry", justify="right")
    pos_table.add_column("Current", justify="right")
    pos_table.add_column("P&L", justify="right")
    pos_table.add_column("Stop", justify="right")

    for sym, pos in state.get("positions", {}).items():
        pnl = pos.get("unrealized_pnl", 0.0)
        pos_table.add_row(
            sym,
            f"{pos.get('qty', 0):.0f}",
            f"${pos.get('entry_price', 0):.2f}",
            f"${pos.get('current_price', 0):.2f}",
            Text(f"${pnl:+,.2f}", style="green" if pnl >= 0 else "red"),
            f"${pos.get('stop_level', 0):.2f}",
        )

    scan_table = Table(show_header=True, header_style="bold magenta")
    scan_table.add_column("Symbol")
    scan_table.add_column("Score", justify="right")
    scan_table.add_column("Signals")
    for r in state.get("top_scan", [])[:8]:
        score = r.get("score", 0)
        scan_table.add_row(
            r.get("symbol", ""),
            f"[green]{score}[/]" if score >= MIN_SCORE else str(score),
            ", ".join(r.get("firing_signals", [])),
        )

    return Group(
        info_panel,
        Panel(pos_table, title="[bold cyan]Open Plays[/]"),
        Panel(scan_table, title="[bold cyan]Last Scan[/]"),
    )


# ======================================================================
# Live runner
# ======================================================================

def run_live(args, config, credentials) -> None:
    trader = SwingTrader(config, credentials, dry_run=getattr(args, "dry_run", False))
    try:
        trader.startup()
        trader.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.critical("Fatal error: %s", exc)
        logger.debug(traceback.format_exc())
    finally:
        trader.shutdown()


# ======================================================================
# Entry point
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
        elif args.scan_only:
            run_scan_only(config, credentials)
        else:
            run_live(args, config, credentials)


if __name__ == "__main__":
    main()
