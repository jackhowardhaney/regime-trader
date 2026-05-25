"""Regime Trader Web App — FastAPI backend."""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STATIC_DIR = Path(__file__).parent / "static"
# DB_PATH can be overridden via env var so Railway volumes work
_db_env = os.getenv("DB_PATH", "")
DB_PATH = Path(_db_env) if _db_env else Path(__file__).parent / "trader.db"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(ROOT / "results")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="Regime Trader")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Database ───────────────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                added_at TEXT DEFAULT (datetime('now')),
                notes TEXT,
                score INTEGER DEFAULT 50,
                bsh TEXT DEFAULT 'HOLD',
                auto_generated INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS plays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                shares REAL,
                entry_date TEXT,
                exit_price REAL,
                exit_date TEXT,
                pnl REAL,
                pnl_pct REAL,
                signal TEXT,
                notes TEXT,
                alpaca_order_id TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbols TEXT,
                start_date TEXT,
                end_date TEXT,
                total_return REAL,
                cagr REAL,
                sharpe REAL,
                max_drawdown REAL,
                n_trades INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS symbol_backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                period_days INTEGER DEFAULT 90,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_return REAL DEFAULT 0,
                best_trade REAL DEFAULT 0,
                worst_trade REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                qualified INTEGER DEFAULT 0,
                fail_reason TEXT,
                run_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_sbt_symbol ON symbol_backtests(symbol, run_at DESC);
        """)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS play_backtests (
                play_id     INTEGER PRIMARY KEY,
                symbol      TEXT NOT NULL,
                status      TEXT,
                entry_date  TEXT,
                end_date    TEXT,
                play_ret    REAL,
                spy_ret     REAL,
                peer_symbol TEXT,
                peer_ret    REAL,
                beats_spy   INTEGER,
                computed_at TEXT DEFAULT (datetime('now'))
            );
        """)
        # Migrate existing watchlist table — add columns if missing
        existing = [r[1] for r in conn.execute("PRAGMA table_info(watchlist)").fetchall()]
        if "score" not in existing:
            conn.execute("ALTER TABLE watchlist ADD COLUMN score INTEGER DEFAULT 50")
        if "bsh" not in existing:
            conn.execute("ALTER TABLE watchlist ADD COLUMN bsh TEXT DEFAULT 'HOLD'")
        if "auto_generated" not in existing:
            conn.execute("ALTER TABLE watchlist ADD COLUMN auto_generated INTEGER DEFAULT 0")


_daily_scan_lock = threading.Lock()

def run_daily_watchlist_scan(force: bool = False) -> int:
    """
    Score all 116 sector symbols, pick top 20 BUY + bottom 10 SELL (excluding
    current plays). Replace auto-generated watchlist entries. Returns count added.
    """
    with _daily_scan_lock:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with get_db() as conn:
            last = conn.execute(
                "SELECT value FROM app_settings WHERE key='watchlist_scan_date'"
            ).fetchone()
            if not force and last and last["value"] == today:
                return 0  # already ran today

        try:
            from data.sector_symbols import SYMBOL_SECTORS
            from core.opportunity_scanner import score_symbol as _score_sym
        except Exception:
            return 0

        # Get symbols currently in active/pending plays — exclude them
        with get_db() as conn:
            play_rows = conn.execute(
                "SELECT symbol FROM plays WHERE status IN ('ACTIVE','PENDING')"
            ).fetchall()
        excluded = {r["symbol"] for r in play_rows}

        universe = [s for s in SYMBOL_SECTORS.keys() if s not in excluded]

        scored = []
        end_dt   = datetime.utcnow()
        start_dt = end_dt - timedelta(days=250)

        for sym in universe:
            try:
                df = _load_bars(sym, days=250)
                if df is None or len(df) < 50:
                    continue
                r = _score_sym(df, sym)
                if r:
                    scored.append(r)
            except Exception:
                continue

        if not scored:
            return 0

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Top 20 BUY (score ≥ 55) + bottom 10 SELL (score ≤ 40)
        buys  = [r for r in scored if r["score"] >= 55][:20]
        sells = [r for r in sorted(scored, key=lambda x: x["score"])
                 if r["score"] <= 40][:10]
        picks = buys + sells

        with get_db() as conn:
            # Remove all auto-generated entries
            conn.execute("DELETE FROM watchlist WHERE auto_generated=1")
            for r in picks:
                sym   = r["symbol"]
                sc    = r["score"]
                bsh   = "BUY" if sc >= 55 else "SELL"
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO watchlist
                           (symbol, added_at, score, bsh, auto_generated)
                           VALUES (?,datetime('now'),?,?,1)""",
                        (sym, sc, bsh),
                    )
                except Exception:
                    pass
            conn.execute(
                "INSERT OR REPLACE INTO app_settings(key,value) VALUES('watchlist_scan_date',?)",
                (today,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO app_settings(key,value) VALUES('watchlist_scan_ts',?)",
                (datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),),
            )
        return len(picks)


def _background_daily_scan():
    """Run on startup in a thread; re-runs if date changes."""
    while True:
        try:
            run_daily_watchlist_scan()
        except Exception:
            pass
        time.sleep(3600)  # check every hour; scan gate prevents re-runs same day


# ── Pydantic models ────────────────────────────────────────────────
class WatchlistAdd(BaseModel):
    symbol: str
    notes: Optional[str] = None


class PlayCreate(BaseModel):
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    shares: float
    signal: Optional[str] = None
    notes: Optional[str] = None
    submit_order: bool = False


class PlayUpdate(BaseModel):
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    alpaca_order_id: Optional[str] = None


class PlayClose(BaseModel):
    exit_price: Optional[float] = None


class BacktestRequest(BaseModel):
    symbols: List[str] = ["SPY", "QQQ"]
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"


# ── Price / Signal helpers ─────────────────────────────────────────
_cache: Dict[str, tuple] = {}
CACHE_TTL = 60

# ── Alpaca data client (singleton) ─────────────────────────────────
_alpaca_data_lock = threading.Lock()
_alpaca_data_client = None


def _get_alpaca_data():
    global _alpaca_data_client
    if _alpaca_data_client is not None:
        return _alpaca_data_client
    with _alpaca_data_lock:
        if _alpaca_data_client is not None:
            return _alpaca_data_client
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                return None
            from alpaca.data.historical import StockHistoricalDataClient
            _alpaca_data_client = StockHistoricalDataClient(
                api_key=api_key, secret_key=secret_key
            )
        except Exception:
            return None
    return _alpaca_data_client


def _load_bars(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch daily bars from Alpaca. Returns DataFrame with lowercase columns
    (close, open, high, low, volume). Falls back to yfinance if Alpaca
    is unavailable or returns insufficient data.
    """
    client = _get_alpaca_data()
    if client is not None:
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.utcnow() - timedelta(days=days + 60),
            )
            raw = client.get_stock_bars(req)
            df = raw.df
            if df is not None and not df.empty:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.loc[symbol]
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
                df.columns = [c.lower() for c in df.columns]
                df = df.tail(days)
                if len(df) >= 20:
                    return df
        except Exception:
            pass

    if not _HAS_YFINANCE:
        return None
    try:
        hist = yf.Ticker(symbol).history(period="1y", interval="1d")
        if hist.empty:
            return None
        hist.columns = [c.lower() for c in hist.columns]
        return hist.tail(days)
    except Exception:
        return None


def get_quote(symbol: str) -> Optional[Dict]:
    now = time.time()
    key = f"q:{symbol}"
    if key in _cache and now - _cache[key][1] < CACHE_TTL:
        return _cache[key][0]
    try:
        df = _load_bars(symbol, days=5)
        if df is None or df.empty:
            return None
        price = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else price
        change_pct = (price - prev) / prev * 100
        data = {
            "symbol": symbol,
            "price": round(price, 2),
            "change": round(price - prev, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(df["volume"].iloc[-1]),
        }
        _cache[key] = (data, now)
        return data
    except Exception:
        return None


def get_signals(symbol: str) -> Dict:
    now = time.time()
    key = f"sig:{symbol}"
    if key in _cache and now - _cache[key][1] < 300:
        return _cache[key][0]
    try:
        hist = _load_bars(symbol, days=400)
        if hist is None or len(hist) < 50:
            return {"error": "Insufficient data"}

        close = hist["close"]
        volume = hist["volume"]
        price = float(close.iloc[-1])

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - 100 / (1 + gain / loss.replace(0, 1e-10))).iloc[-1])

        # MAs
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        sig_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd_line - sig_line).iloc[-1])

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = float((bb_mid + 2 * bb_std).iloc[-1])
        bb_lower = float((bb_mid - 2 * bb_std).iloc[-1])
        bb_pct = (price - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50

        # Volume
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 1

        # Score signals
        score = 0
        signals = []

        if rsi < 30:
            score += 2
            signals.append({"name": "RSI Oversold", "value": f"{rsi:.0f}", "bias": "bullish", "strength": "strong"})
        elif rsi < 40:
            score += 1
            signals.append({"name": "RSI Low", "value": f"{rsi:.0f}", "bias": "bullish", "strength": "weak"})
        elif rsi > 70:
            score -= 2
            signals.append({"name": "RSI Overbought", "value": f"{rsi:.0f}", "bias": "bearish", "strength": "strong"})
        elif rsi > 60:
            score -= 1
            signals.append({"name": "RSI High", "value": f"{rsi:.0f}", "bias": "bearish", "strength": "weak"})
        else:
            signals.append({"name": "RSI Neutral", "value": f"{rsi:.0f}", "bias": "neutral", "strength": "neutral"})

        if price > ma50:
            score += 1
            if ma200 and price > ma200:
                score += 1
                signals.append({"name": "Above MA50 & MA200", "value": f"${ma200:.2f}", "bias": "bullish", "strength": "strong"})
            else:
                signals.append({"name": "Above MA50", "value": f"${ma50:.2f}", "bias": "bullish", "strength": "weak"})
        else:
            score -= 1
            if ma200 and price < ma200:
                score -= 1
                signals.append({"name": "Below MA50 & MA200", "value": f"${ma200:.2f}", "bias": "bearish", "strength": "strong"})
            else:
                signals.append({"name": "Below MA50", "value": f"${ma50:.2f}", "bias": "bearish", "strength": "weak"})

        if macd_hist > 0:
            score += 1
            signals.append({"name": "MACD Bullish", "value": f"{macd_hist:+.4f}", "bias": "bullish", "strength": "weak"})
        else:
            score -= 1
            signals.append({"name": "MACD Bearish", "value": f"{macd_hist:+.4f}", "bias": "bearish", "strength": "weak"})

        if bb_pct < 20:
            score += 1
            signals.append({"name": "Near BB Lower Band", "value": f"{bb_pct:.0f}%", "bias": "bullish", "strength": "weak"})
        elif bb_pct > 80:
            score -= 1
            signals.append({"name": "Near BB Upper Band", "value": f"{bb_pct:.0f}%", "bias": "bearish", "strength": "weak"})

        if vol_ratio > 1.5:
            signals.append({"name": "High Volume", "value": f"{vol_ratio:.1f}x avg", "bias": "neutral", "strength": "info"})

        # Use opportunity_scanner composite score (0-100) for granular scoring
        try:
            from core.opportunity_scanner import score_symbol as _score_symbol
            _scored = _score_symbol(hist, symbol)
            if _scored:
                buy_pct = int(_scored["score"])
            else:
                buy_pct = max(0, min(100, int(((score + 5) / 10) * 100)))
        except Exception:
            buy_pct = max(0, min(100, int(((score + 5) / 10) * 100)))
        overall = "BULLISH" if buy_pct >= 60 else "BEARISH" if buy_pct <= 40 else "NEUTRAL"
        bsh = "BUY" if buy_pct >= 60 else "SELL" if buy_pct <= 40 else "HOLD"

        # ATR-based stop/target (replaces flat %)
        tr = pd.concat([
            hist["high"] - hist["low"],
            (hist["high"] - close.shift()).abs(),
            (hist["low"] - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        stop_dist = max(atr * 1.5, price * 0.02)   # min 2%
        target_dist = stop_dist * 2.5               # 2.5:1 R:R

        result = {
            "symbol": symbol,
            "price": round(price, 2),
            "rsi": round(rsi, 1),
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "ma200": round(ma200, 2) if ma200 else None,
            "macd_hist": round(macd_hist, 4),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "bb_pct": round(bb_pct, 1),
            "vol_ratio": round(vol_ratio, 2),
            "atr": round(atr, 3),
            "score": score,
            "buy_pct": buy_pct,
            "overall": overall,
            "bsh": bsh,
            "signals": signals,
            "suggested_entry": round(price, 2),
            "suggested_stop": round(price - stop_dist, 2),
            "suggested_target": round(price + target_dist, 2),
            "stop_method": f"ATR×1.5 ({atr:.2f})",
        }
        _cache[key] = (result, now)
        return result
    except Exception as e:
        return {"error": str(e)}


def _load_equity_csv(path: Path) -> pd.DataFrame:
    ec = pd.read_csv(path, index_col=0)
    ec.index = pd.to_datetime(ec.index, utc=True, errors="coerce").tz_convert(None)
    ec.columns = ["equity"]
    return ec


def get_alpaca_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        if not api_key or not secret_key:
            return None
        from broker.alpaca_client import AlpacaClient
        client = AlpacaClient(api_key=api_key, secret_key=secret_key, paper=paper)
        client.connect()
        return client
    except Exception:
        return None


# ── Backtest state ─────────────────────────────────────────────────
_bt: Dict[str, Any] = {"running": False, "progress": 0, "result": None, "error": None}

# ── Scanner state ───────────────────────────────────────────────────
_scan: Dict[str, Any] = {"running": False, "progress": 0, "result": None, "error": None}
_sim:  Dict[str, Any] = {"running": False, "progress": 0, "result": None, "error": None}
_qbt:  Dict[str, Any] = {"running": False, "progress": 0, "completed": 0, "total": 0, "result": None, "error": None}


def _run_backtest(symbols: List[str], start: str, end: str):
    """
    Fast in-process signal-based backtest. Works with as few as 10 trading days.
    Uses Alpaca bars via _load_bars; no subprocess, no HMM training required.
    Entry: any signal fires → enter at next open.
    Exit: +5% target / -3% stop / 10-bar hold.
    Portfolio: 10% allocation per trade, equal-weight across symbols.
    """
    global _bt
    _bt = {"running": True, "progress": 5, "result": None, "error": None}
    try:
        from core.opportunity_scanner import score_symbol

        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
        req_days  = max(10, (end_dt - start_dt).days)
        fetch_days = req_days + 150  # extra warmup for indicators

        _bt["progress"] = 10

        # ── 1. Pre-fetch all bars (Alpaca first, yfinance fallback) ──────
        bars_full: Dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols):
            _bt["progress"] = 10 + int(25 * i / max(len(symbols), 1))
            df = _load_bars(sym, days=fetch_days)
            if df is not None and len(df) >= 10:
                bars_full[sym] = df

        if not bars_full:
            _bt.update({"running": False,
                        "error": f"No price data returned for: {', '.join(symbols)}"})
            return

        _bt["progress"] = 35

        # ── 2. Signal simulation per symbol ─────────────────────────────
        all_trades: List[Dict] = []
        for sym, full_df in bars_full.items():
            n = len(full_df)
            # Find the index where the requested period starts
            period_start_idx = next(
                (i for i, ts in enumerate(full_df.index) if ts >= start_dt), None
            )
            if period_start_idx is None:
                period_start_idx = max(50, n - req_days)
            period_start_idx = max(period_start_idx, 15)  # need warmup rows

            for i in range(period_start_idx, n - 2):
                if full_df.index[i] > end_dt:
                    break
                scored = score_symbol(full_df.iloc[: i + 1], sym)
                if not scored or not scored["firing_signals"]:
                    continue
                entry = float(full_df["open"].iloc[i + 1])
                if entry <= 0:
                    continue
                target_px = entry * 1.05
                stop_px   = entry * 0.97
                exit_px   = float(full_df["close"].iloc[min(i + 10, n - 1)])
                for j in range(i + 1, min(i + 11, n)):
                    if float(full_df["high"].iloc[j]) >= target_px:
                        exit_px = target_px
                        break
                    if float(full_df["low"].iloc[j]) <= stop_px:
                        exit_px = stop_px
                        break
                all_trades.append({
                    "date":   full_df.index[i + 1],
                    "symbol": sym,
                    "ret":    (exit_px / entry - 1),
                })

        _bt["progress"] = 70

        # ── 3. Build portfolio equity curve ─────────────────────────────
        # Use primary symbol's date range as the x-axis
        primary_sym = list(bars_full.keys())[0]
        primary_df  = bars_full[primary_sym]
        period_df   = primary_df[primary_df.index >= start_dt]
        if period_df.empty:
            period_df = primary_df.tail(req_days)

        alloc_per_trade = 0.10  # 10% of portfolio per signal
        equity          = 100_000.0
        trade_by_date: Dict[pd.Timestamp, List[float]] = {}
        for t in all_trades:
            trade_by_date.setdefault(t["date"], []).append(t["ret"])

        curve_dates:  List[str]   = []
        curve_values: List[float] = []
        for ts in period_df.index:
            if ts in trade_by_date:
                rets = trade_by_date[ts]
                port_ret = sum(r * alloc_per_trade for r in rets)
                equity *= (1 + port_ret)
            curve_dates.append(ts.strftime("%Y-%m-%d"))
            curve_values.append(round(equity, 2))

        if len(curve_values) < 2:
            curve_dates  = [start, end]
            curve_values = [100_000.0, 100_000.0]

        # ── 4. Stats ─────────────────────────────────────────────────────
        s_eq, e_eq = curve_values[0], curve_values[-1]
        ret  = (e_eq / s_eq - 1) * 100
        actual_days = max((pd.Timestamp(curve_dates[-1]) - pd.Timestamp(curve_dates[0])).days, 1)
        years = actual_days / 365.25
        cagr  = ((e_eq / s_eq) ** (1 / years) - 1) * 100 if years >= (1/12) else ret

        eq_s   = pd.Series(curve_values)
        rets_s = eq_s.pct_change().dropna()
        excess = rets_s - 0.045 / 252
        sharpe = float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() > 0 else 0.0

        roll_max = eq_s.cummax()
        dd_vals  = ((eq_s - roll_max) / roll_max * 100).round(2).tolist()
        max_dd   = float(min(dd_vals))

        wins   = [t["ret"] for t in all_trades if t["ret"] > 0]
        losses = [t["ret"] for t in all_trades if t["ret"] <= 0]
        win_rate = round(len(wins) / len(all_trades) * 100, 1) if all_trades else 0.0

        summary = {
            "total_return":   round(ret,    2),
            "cagr":           round(cagr,   2),
            "sharpe":         round(sharpe, 3),
            "max_drawdown":   round(max_dd, 2),
            "n_trades":       len(all_trades),
            "win_rate":       win_rate,
            "equity_dates":   curve_dates,
            "equity_values":  curve_values,
            "drawdown_values": dd_vals,
            "regime_breakdown": [],
            "symbols":        symbols,
            "start_date":     start,
            "end_date":       end,
        }

        with get_db() as conn:
            conn.execute(
                "INSERT INTO backtest_runs "
                "(symbols,start_date,end_date,total_return,cagr,sharpe,max_drawdown,n_trades) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (json.dumps(symbols), start, end, ret, cagr, sharpe, max_dd, len(all_trades)),
            )

        _bt.update({"running": False, "progress": 100, "result": summary})

    except Exception as e:
        import traceback as _tb
        _bt.update({"running": False, "error": str(e) + "\n" + _tb.format_exc()[-800:]})


# ── Watchlist routes ───────────────────────────────────────────────
@app.get("/api/watchlist")
def list_watchlist():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM watchlist ORDER BY added_at").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/watchlist")
def add_watchlist(body: WatchlistAdd):
    sym = body.symbol.upper().strip()
    with get_db() as conn:
        try:
            conn.execute("INSERT INTO watchlist (symbol, notes) VALUES (?,?)", (sym, body.notes))
        except sqlite3.IntegrityError:
            raise HTTPException(400, f"{sym} already in watchlist")
    return {"symbol": sym}


@app.delete("/api/watchlist/{symbol}")
def del_watchlist(symbol: str):
    with get_db() as conn:
        conn.execute("DELETE FROM watchlist WHERE symbol=?", (symbol.upper(),))
    return {"removed": True}


@app.get("/api/watchlist/quotes")
def watchlist_quotes():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT symbol, score, bsh, auto_generated FROM watchlist"
        ).fetchall()
        scan_ts = conn.execute(
            "SELECT value FROM app_settings WHERE key='watchlist_scan_ts'"
        ).fetchone()
    last_refresh = scan_ts["value"] if scan_ts else None
    results = []
    for row in rows:
        sym = row["symbol"]
        q = get_quote(sym)
        if not q:
            continue
        stored_score = row["score"] or 50
        stored_bsh   = row["bsh"] or "HOLD"
        # Use stored score for auto-generated entries; re-score manual entries live
        if row["auto_generated"]:
            buy_pct = stored_score
            bsh     = stored_bsh
            sig     = get_signals(sym)
            rsi     = sig.get("rsi", 50)
        else:
            sig     = get_signals(sym)
            buy_pct = sig.get("buy_pct", 50)
            bsh     = sig.get("bsh", "HOLD")
            rsi     = sig.get("rsi", 50)
        results.append({**q,
                        "overall": "BULLISH" if buy_pct >= 60 else "BEARISH" if buy_pct <= 40 else "NEUTRAL",
                        "bsh": bsh,
                        "rsi": rsi,
                        "buy_pct": buy_pct,
                        "auto_generated": bool(row["auto_generated"])})
    # BUY first sorted by score desc, then SELL sorted by score asc
    buys  = sorted([r for r in results if r["bsh"] == "BUY"],  key=lambda x: -x["buy_pct"])
    sells = sorted([r for r in results if r["bsh"] == "SELL"], key=lambda x: x["buy_pct"])
    holds = sorted([r for r in results if r["bsh"] == "HOLD"], key=lambda x: -x["buy_pct"])
    return {"quotes": buys + sells + holds, "last_refresh": last_refresh}


@app.post("/api/watchlist/scan")
def trigger_watchlist_scan():
    count = run_daily_watchlist_scan(force=True)
    return {"added": count}


@app.get("/api/signals/{symbol}")
def symbol_signals(symbol: str):
    return get_signals(symbol.upper())


_NEWS_BULLISH = {"beat","surge","growth","upgrade","buy","strong","record","rally","gain","rise",
                  "soar","jump","high","profit","outperform","positive","boost","expand","upside"}
_NEWS_BEARISH = {"miss","drop","concern","risk","cut","loss","warning","downgrade","sell","decline",
                  "fall","weak","slump","disappoint","below","fear","pressure","lawsuit","investigate"}

@app.get("/api/news/{symbol}")
def symbol_news(symbol: str):
    sym = symbol.upper()
    cache_key = f"news:{sym}"
    now = time.time()
    cached = _cache.get(cache_key)
    if cached and now - cached[1] < 1800:   # 30-min cache
        return cached[0]
    try:
        import requests as _req
        api_key    = os.getenv("ALPACA_API_KEY", "")
        api_secret = os.getenv("ALPACA_SECRET_KEY", "")
        resp = _req.get(
            "https://data.alpaca.markets/v1beta1/news",
            params={"symbols": sym, "limit": 3, "sort": "desc"},
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
            timeout=8,
        )
        articles = resp.json().get("news", [])
    except Exception:
        articles = []

    result = []
    for a in articles[:3]:
        text = ((a.get("headline") or "") + " " + (a.get("summary") or "")).lower()
        words = set(text.split())
        bull = len(words & _NEWS_BULLISH)
        bear = len(words & _NEWS_BEARISH)
        if bull > bear:
            sentiment, color = "BUY", "#16a34a"
        elif bear > bull:
            sentiment, color = "SELL", "#dc2626"
        else:
            sentiment, color = "HOLD", "#d97706"
        summary = a.get("summary") or a.get("headline") or ""
        if len(summary) > 180:
            summary = summary[:177] + "…"
        result.append({
            "headline": a.get("headline", ""),
            "source":   a.get("source", ""),
            "url":      a.get("url", ""),
            "summary":  summary,
            "sentiment": sentiment,
            "color":    color,
            "published": (a.get("created_at") or "")[:10],
        })

    _cache[cache_key] = (result, now)
    return result


# ── Plays routes ───────────────────────────────────────────────────
@app.get("/api/plays")
def list_plays(status: Optional[str] = None):
    with get_db() as conn:
        if status:
            rows = conn.execute("SELECT * FROM plays WHERE status=? ORDER BY created_at DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM plays ORDER BY created_at DESC").fetchall()
    plays = [dict(r) for r in rows]
    # Enrich active plays with current price
    for p in plays:
        if p["status"] in ("ACTIVE", "PENDING"):
            q = get_quote(p["symbol"])
            if q:
                cur = q["price"]
                if p["direction"] == "LONG":
                    p["current_pnl"] = round((cur - (p["entry_price"] or cur)) * (p["shares"] or 0), 2)
                else:
                    p["current_pnl"] = round(((p["entry_price"] or cur) - cur) * (p["shares"] or 0), 2)
                p["current_price"] = cur
                p["day_change"] = q.get("change")
                p["day_change_pct"] = q.get("change_pct")
    return plays


@app.post("/api/plays")
def create_play(body: PlayCreate):
    sym = body.symbol.upper()
    risk = abs(body.entry_price - body.stop_loss)
    reward = abs(body.take_profit - body.entry_price)
    rr = round(reward / risk, 2) if risk > 0 else 0

    alpaca_order_id = None
    status = "PENDING"
    if body.submit_order:
        try:
            client = get_alpaca_client()
            if client:
                from broker.order_executor import OrderExecutor
                executor = OrderExecutor(client)
                side = "buy" if body.direction == "LONG" else "sell"
                # Use the trading client directly
                from alpaca.trading.requests import LimitOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce
                order_req = LimitOrderRequest(
                    symbol=sym,
                    qty=int(body.shares),
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=body.entry_price,
                )
                order = client._trading.submit_order(order_req)
                alpaca_order_id = str(order.id)
                status = "ACTIVE"
        except Exception:
            status = "PENDING"

    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO plays (symbol,direction,status,entry_price,stop_loss,take_profit,
               shares,entry_date,signal,notes,alpaca_order_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sym, body.direction, status, body.entry_price, body.stop_loss,
             body.take_profit, body.shares, datetime.utcnow().isoformat(),
             body.signal, body.notes, alpaca_order_id),
        )
        pid = cur.lastrowid

    return {"id": pid, "rr_ratio": rr, "alpaca_order_id": alpaca_order_id, "status": status}


@app.put("/api/plays/{pid}")
def update_play(pid: int, body: PlayUpdate):
    sets, vals = [], []
    if body.stop_loss is not None:
        sets.append("stop_loss=?"); vals.append(body.stop_loss)
    if body.take_profit is not None:
        sets.append("take_profit=?"); vals.append(body.take_profit)
    if body.notes is not None:
        sets.append("notes=?"); vals.append(body.notes)
    if body.status is not None:
        sets.append("status=?"); vals.append(body.status)
    if body.alpaca_order_id is not None:
        sets.append("alpaca_order_id=?"); vals.append(body.alpaca_order_id)
    if not sets:
        raise HTTPException(400, "Nothing to update")
    vals.append(pid)
    with get_db() as conn:
        conn.execute(f"UPDATE plays SET {','.join(sets)} WHERE id=?", vals)
    return {"updated": True}


@app.post("/api/plays/{pid}/close")
def close_play(pid: int, body: PlayClose):
    with get_db() as conn:
        play = conn.execute("SELECT * FROM plays WHERE id=?", (pid,)).fetchone()
    if not play:
        raise HTTPException(404, "Play not found")
    play = dict(play)

    exit_price = body.exit_price
    if not exit_price:
        q = get_quote(play["symbol"])
        exit_price = q["price"] if q else play["entry_price"]

    if play["direction"] == "LONG":
        pnl = (exit_price - play["entry_price"]) * play["shares"]
    else:
        pnl = (play["entry_price"] - exit_price) * play["shares"]
    pnl_pct = pnl / (play["entry_price"] * play["shares"]) * 100 if play["entry_price"] else 0

    with get_db() as conn:
        conn.execute(
            "UPDATE plays SET status='CLOSED',exit_price=?,exit_date=?,pnl=?,pnl_pct=? WHERE id=?",
            (exit_price, datetime.utcnow().isoformat(), round(pnl, 2), round(pnl_pct, 2), pid),
        )
    return {"pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2)}


@app.delete("/api/plays/{pid}")
def delete_play(pid: int):
    with get_db() as conn:
        conn.execute("DELETE FROM plays WHERE id=?", (pid,))
    return {"deleted": True}


# ── Backtest routes ────────────────────────────────────────────────
@app.post("/api/backtest/run")
def start_backtest(body: BacktestRequest):
    if _bt["running"]:
        raise HTTPException(409, "Backtest already running")
    threading.Thread(target=_run_backtest, args=(body.symbols, body.start_date, body.end_date), daemon=True).start()
    return {"started": True}


@app.get("/api/backtest/status")
def bt_status():
    return _bt


@app.get("/api/backtest/latest")
def bt_latest():
    ec_f = RESULTS_DIR / "equity_curve.csv"
    if not ec_f.exists():
        return None
    try:
        ec = _load_equity_csv(ec_f)
        s_eq, e_eq = float(ec["equity"].iloc[0]), float(ec["equity"].iloc[-1])
        days = max((ec.index[-1] - ec.index[0]).days, 1)
        years = days / 365.25
        ret = (e_eq / s_eq - 1) * 100
        cagr = ((e_eq / s_eq) ** (1 / years) - 1) * 100
        rets = ec["equity"].pct_change().dropna()
        excess = rets - 0.045 / 252
        sharpe = float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() > 0 else 0
        roll_max = ec["equity"].cummax()
        max_dd = float(((ec["equity"] - roll_max) / roll_max).min() * 100)
        dd_series = ((ec["equity"] - roll_max) / roll_max * 100).round(2).tolist()
        n_trades = 0
        tl_f = RESULTS_DIR / "trade_log.csv"
        if tl_f.exists():
            n_trades = len(pd.read_csv(tl_f))
        rh_f = RESULTS_DIR / "regime_history.csv"
        regime_breakdown = []
        if rh_f.exists():
            rh = pd.read_csv(rh_f, index_col=0)
            if "regime" in rh.columns:
                counts = rh["regime"].value_counts()
                regime_breakdown = [{"regime": r, "pct": round(c / len(rh) * 100, 1)} for r, c in counts.items()]
        return {
            "total_return": round(ret, 2), "cagr": round(cagr, 2),
            "sharpe": round(sharpe, 3), "max_drawdown": round(max_dd, 2),
            "n_trades": n_trades,
            "equity_dates": ec.index.strftime("%Y-%m-%d").tolist(),
            "equity_values": ec["equity"].round(2).tolist(),
            "drawdown_values": dd_series,
            "regime_breakdown": regime_breakdown,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/backtest/history")
def bt_history():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,symbols,start_date,end_date,total_return,cagr,sharpe,max_drawdown,n_trades,created_at FROM backtest_runs ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
    return [dict(r) for r in rows]


# ── History routes ─────────────────────────────────────────────────
@app.get("/api/history")
def trade_history(symbol: Optional[str] = None, days: Optional[int] = None):
    q = "SELECT * FROM plays WHERE status='CLOSED'"
    p = []
    if symbol:
        q += " AND symbol=?"; p.append(symbol.upper())
    if days:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        q += " AND exit_date>=?"; p.append(cutoff)
    q += " ORDER BY exit_date DESC"
    with get_db() as conn:
        rows = conn.execute(q, p).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/history/metrics")
def history_metrics():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT pnl, pnl_pct, exit_date FROM plays WHERE status='CLOSED' ORDER BY exit_date"
        ).fetchall()
    if not rows:
        return {"total_pnl": 0, "win_rate": 0, "profit_factor": 0, "n_trades": 0, "monthly": [], "avg_win": 0, "avg_loss": 0}
    pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    monthly = {}
    for r in rows:
        if r["exit_date"] and r["pnl"] is not None:
            m = r["exit_date"][:7]
            monthly[m] = monthly.get(m, 0) + r["pnl"]
    return {
        "total_pnl": round(sum(pnls), 2),
        "n_trades": len(pnls),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else 0,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
        "monthly": [{"month": k, "pnl": round(v, 2)} for k, v in sorted(monthly.items())],
    }


@app.get("/api/history/export")
def export_history():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM plays WHERE status='CLOSED' ORDER BY exit_date DESC").fetchall()
    if not rows:
        raise HTTPException(404, "No closed trades")
    df = pd.DataFrame([dict(r) for r in rows])
    out = StringIO()
    df.to_csv(out, index=False)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trade_history.csv"},
    )


# ── Portfolio route ────────────────────────────────────────────────
@app.get("/api/portfolio")
def get_portfolio():
    try:
        client = get_alpaca_client()
        if not client:
            return {"connected": False, "message": "Alpaca not configured — check .env"}
        acct = client.get_account()
        equity = float(acct.get("equity", 0))
        last_equity = float(acct.get("last_equity", equity))

        # Compute invested capital and unrealized PNL from active plays
        invested = 0.0
        unrealized_pnl = 0.0
        open_count = 0
        with get_db() as conn:
            rows = conn.execute(
                "SELECT symbol, entry_price, shares, direction FROM plays WHERE status='ACTIVE'"
            ).fetchall()
        for row in rows:
            cost = (row["entry_price"] or 0) * (row["shares"] or 0)
            invested += cost
            open_count += 1
            q = get_quote(row["symbol"])
            if q:
                cur = q["price"]
                if row["direction"] == "LONG":
                    unrealized_pnl += (cur - (row["entry_price"] or cur)) * (row["shares"] or 0)
                else:
                    unrealized_pnl += ((row["entry_price"] or cur) - cur) * (row["shares"] or 0)

        return {
            "connected": True,
            "equity": equity,
            "cash": float(acct.get("cash", 0)),
            "buying_power": float(acct.get("buying_power", 0)),
            "day_pl": round(equity - last_equity, 2),
            "day_pl_pct": round((equity - last_equity) / last_equity * 100, 2) if last_equity else 0,
            "invested": round(invested, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "open_positions": open_count,
            "paper": True,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


def _compute_return_over_period(df: "pd.DataFrame", start: str, end: str) -> Optional[float]:
    """Return pct return of close prices between start and end dates (inclusive)."""
    try:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        sub = df[(df.index >= s) & (df.index <= e)]
        if len(sub) < 2:
            # If start is before earliest bar, use earliest available
            sub = df[df.index <= e]
            if len(sub) < 2:
                return None
        return float((sub["close"].iloc[-1] - sub["close"].iloc[0]) / sub["close"].iloc[0] * 100)
    except Exception:
        return None


def _find_peer(symbol: str, start: str, end: str, play_df: "pd.DataFrame") -> tuple:
    """Find the most correlated peer in the same sector over the play period."""
    try:
        from data.sector_symbols import SYMBOL_SECTORS, SECTOR_SYMBOLS
        sectors = SYMBOL_SECTORS.get(symbol, [])
        if not sectors:
            return None, None
        candidates = []
        for sec in sectors[:2]:
            candidates.extend(SECTOR_SYMBOLS[sec]["symbols"])
        candidates = [s for s in set(candidates) if s != symbol][:25]
        s_ts = pd.Timestamp(start)
        e_ts = pd.Timestamp(end)
        play_sub = play_df[(play_df.index >= s_ts) & (play_df.index <= e_ts)]
        if len(play_sub) < 5:
            play_sub = play_df[play_df.index <= e_ts].tail(max(5, len(play_df)))
        play_ret = play_sub["close"].pct_change().dropna()
        best_sym, best_corr, best_ret = None, -2.0, None
        for peer in candidates:
            pdf = _load_bars(peer, days=400)
            if pdf is None or pdf.empty:
                continue
            peer_sub = pdf[(pdf.index >= s_ts) & (pdf.index <= e_ts)]
            if len(peer_sub) < 5:
                peer_sub = pdf[pdf.index <= e_ts].tail(max(5, len(pdf)))
            peer_ret_s = peer_sub["close"].pct_change().dropna()
            aligned = play_ret.align(peer_ret_s, join="inner")[0]
            if len(aligned) < 5:
                continue
            corr = float(play_ret.align(peer_ret_s, join="inner")[0].corr(
                play_ret.align(peer_ret_s, join="inner")[1]
            ))
            if corr > best_corr:
                peer_total = _compute_return_over_period(pdf, start, end)
                best_sym, best_corr, best_ret = peer, corr, peer_total
        return best_sym, best_ret
    except Exception:
        return None, None


def _run_play_backtests():
    """Compute per-play comparison vs SPY and closest peer. Stores in play_backtests."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    spy_df = _load_bars("SPY", days=400)
    with get_db() as conn:
        plays = conn.execute(
            "SELECT id, symbol, status, entry_date, exit_date, created_at FROM plays"
        ).fetchall()
    results = []
    for p in plays:
        play_id = p["id"]
        sym = p["symbol"]
        status = p["status"]
        start = p["entry_date"] or (p["created_at"] or today)[:10]
        end = p["exit_date"][:10] if p["exit_date"] else today
        play_df = _load_bars(sym, days=400)
        if play_df is None:
            continue
        play_ret = _compute_return_over_period(play_df, start, end)
        spy_ret = _compute_return_over_period(spy_df, start, end) if spy_df is not None else None
        peer_sym, peer_ret = _find_peer(sym, start, end, play_df)
        beats_spy = 1 if (play_ret is not None and spy_ret is not None and play_ret > spy_ret) else 0
        results.append((play_id, sym, status, start, end,
                         round(play_ret, 2) if play_ret is not None else None,
                         round(spy_ret, 2) if spy_ret is not None else None,
                         peer_sym,
                         round(peer_ret, 2) if peer_ret is not None else None,
                         beats_spy))
    with get_db() as conn:
        conn.execute("DELETE FROM play_backtests")
        conn.executemany(
            "INSERT INTO play_backtests (play_id,symbol,status,entry_date,end_date,"
            "play_ret,spy_ret,peer_symbol,peer_ret,beats_spy) VALUES (?,?,?,?,?,?,?,?,?,?)",
            results
        )


_play_bt_lock = threading.Lock()
_play_bt_last: float = 0.0


@app.get("/api/plays/backtests")
def get_play_backtests(force: bool = False):
    global _play_bt_last
    now = time.time()
    stale = (now - _play_bt_last) > 3600
    if (stale or force) and _play_bt_lock.acquire(blocking=False):
        try:
            _play_bt_last = now
            threading.Thread(target=_compute_play_backtests_bg, daemon=True).start()
        finally:
            _play_bt_lock.release()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT pb.*, p.signal, p.direction FROM play_backtests pb "
            "LEFT JOIN plays p ON p.id = pb.play_id ORDER BY pb.entry_date DESC"
        ).fetchall()
    return {"results": [dict(r) for r in rows], "computing": stale or force}


def _compute_play_backtests_bg():
    try:
        _run_play_backtests()
    except Exception:
        pass


@app.post("/api/plays/backtests/refresh")
def refresh_play_backtests():
    threading.Thread(target=_compute_play_backtests_bg, daemon=True).start()
    return {"status": "computing"}


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ── Scanner routes ─────────────────────────────────────────────────
@app.get("/api/scanner/sectors")
def scanner_sectors():
    from data.sector_symbols import SECTOR_SYMBOLS
    return [
        {"key": k, "name": v["name"], "color": v["color"], "count": len(v["symbols"])}
        for k, v in SECTOR_SYMBOLS.items()
    ]


@app.post("/api/scanner/start")
def scanner_start():
    global _scan
    if _scan["running"]:
        raise HTTPException(409, "Scan already running")
    from data.sector_symbols import ALL_HANDPICK_SYMBOLS
    threading.Thread(target=_run_scan, args=(ALL_HANDPICK_SYMBOLS,), daemon=True).start()
    return {"started": True}


@app.get("/api/scanner/status")
def scanner_status():
    return _scan


@app.post("/api/scanner/sim/start")
def scanner_sim_start():
    global _sim
    if _sim["running"]:
        raise HTTPException(409, "Simulation already running")
    if not _scan.get("result"):
        raise HTTPException(400, "Run a scan first")
    syms = [r["symbol"] for r in _scan["result"]["results"]]
    threading.Thread(target=_run_sim, args=(syms,), daemon=True).start()
    return {"started": True}


@app.get("/api/scanner/sim_status")
def scanner_sim_status():
    return _sim


@app.post("/api/scanner/backtest/start")
def scanner_bt_start(body: dict = {}):
    global _qbt
    if _qbt["running"]:
        raise HTTPException(409, "Backtest run already in progress")
    symbols = body.get("symbols") if body else None
    if not symbols:
        from data.sector_symbols import ALL_HANDPICK_SYMBOLS
        symbols = ALL_HANDPICK_SYMBOLS
    threading.Thread(target=_run_qual_backtests, args=(symbols,), daemon=True).start()
    return {"started": True, "total": len(symbols)}


@app.get("/api/scanner/backtest/status")
def scanner_bt_status():
    return _qbt


@app.get("/api/scanner/backtest/results")
def scanner_bt_results():
    return _latest_bt_results()


@app.get("/api/scanner/backtest/{symbol}")
def scanner_bt_symbol(symbol: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM symbol_backtests WHERE symbol=? ORDER BY run_at DESC LIMIT 1",
            (symbol.upper(),),
        ).fetchone()
    if not row:
        return {"symbol": symbol.upper(), "tested": False}
    return {**dict(row), "tested": True}


@app.post("/api/scanner/push")
def scanner_push(body: dict):
    symbols = body.get("symbols", [])
    added = []
    for sym in symbols:
        sym = sym.upper().strip()
        try:
            with get_db() as conn:
                conn.execute("INSERT INTO watchlist (symbol) VALUES (?)", (sym,))
            added.append(sym)
        except Exception:
            pass
    return {"added": added}


def _run_qual_backtests(symbols: List[str]) -> None:
    """Run per-symbol 90-day qualification backtests and persist results."""
    global _qbt
    _qbt = {"running": True, "progress": 0, "completed": 0, "total": len(symbols), "result": None, "error": None}
    try:
        from core.opportunity_scanner import backtest_symbol
        lock = threading.Lock()
        completed = [0]

        def _worker(sym: str) -> None:
            r = backtest_symbol(_load_bars, sym)
            if r:
                with get_db() as conn:
                    conn.execute(
                        """INSERT INTO symbol_backtests
                           (symbol,period_days,total_trades,winning_trades,win_rate,avg_return,
                            best_trade,worst_trade,profit_factor,max_drawdown,qualified,fail_reason)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (r["symbol"], r["period_days"], r["total_trades"], r["winning_trades"],
                         r["win_rate"], r["avg_return"], r["best_trade"], r["worst_trade"],
                         r["profit_factor"], r["max_drawdown"], 1 if r["qualified"] else 0,
                         r.get("fail_reason")),
                    )
            with lock:
                completed[0] += 1
                pct = int(completed[0] / len(symbols) * 100)
                _qbt["completed"] = completed[0]
                _qbt["progress"] = pct

        with ThreadPoolExecutor(max_workers=8) as exe:
            list(exe.map(_worker, symbols))

        _qbt.update({"running": False, "progress": 100, "result": _latest_bt_results()})
    except Exception as e:
        _qbt.update({"running": False, "error": str(e)})


def _latest_bt_results() -> List[Dict]:
    """Return most-recent backtest per symbol, sorted by win_rate desc."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT b.* FROM symbol_backtests b
            INNER JOIN (
                SELECT symbol, MAX(run_at) as mr FROM symbol_backtests GROUP BY symbol
            ) l ON b.symbol = l.symbol AND b.run_at = l.mr
            ORDER BY b.qualified DESC, b.win_rate DESC
        """).fetchall()
    return [dict(r) for r in rows]


def _run_scan(symbols: List[str]) -> None:
    global _scan
    _scan = {"running": True, "progress": 5, "result": None, "error": None}
    try:
        from core.opportunity_scanner import run_scan, SIGNAL_META, SCORE_COMPONENTS
        _scan["progress"] = 10

        results = run_scan(_load_bars, symbols, max_workers=8)
        _scan["progress"] = 95

        _scan.update({
            "running": False,
            "progress": 100,
            "result": {
                "results": results,
                "total_scanned": len(results),
                "signal_meta": SIGNAL_META,
                "score_components": SCORE_COMPONENTS,
            },
        })
    except Exception as e:
        _scan.update({"running": False, "error": str(e)})


def _run_sim(symbols: List[str]) -> None:
    global _sim
    _sim = {"running": True, "progress": 5, "result": None, "error": None}
    try:
        from core.opportunity_scanner import run_90day_sim
        _sim["progress"] = 10
        result = run_90day_sim(_load_bars, symbols, max_workers=6)
        _sim.update({"running": False, "progress": 100, "result": result})
    except Exception as e:
        _sim.update({"running": False, "error": str(e)})


# ── Serve frontend ─────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.on_event("startup")
def startup():
    init_db()
    t = threading.Thread(target=_background_daily_scan, daemon=True)
    t.start()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
