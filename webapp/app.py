"""Regime Trader Web App — FastAPI backend."""

import hashlib
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
DEPLOY_TIME = datetime.utcnow()

app = FastAPI(title="Regime Trader")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Auth ────────────────────────────────────────────────────────────
_DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "65Alannah")
_SESSION_SECRET     = os.getenv("SESSION_SECRET", "regime-trader-2024")
_BOT_SECRET         = os.getenv("BOT_SECRET", "")
_VALID_TOKEN = hashlib.sha256(
    f"{_SESSION_SECRET}:{_DASHBOARD_PASSWORD}".encode()
).hexdigest()
_AUTH_EXCLUDED = {"/api/auth/login", "/api/auth/logout", "/api/auth/status"}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if (request.method == "OPTIONS"
            or not path.startswith("/api/")
            or path in _AUTH_EXCLUDED):
        return await call_next(request)
    # Session cookie (browser)
    if request.cookies.get("rt_session") == _VALID_TOKEN:
        return await call_next(request)
    # Bot-secret header (scanner service)
    if _BOT_SECRET and request.headers.get("X-Bot-Secret") == _BOT_SECRET:
        return await call_next(request)
    return JSONResponse({"error": "unauthorized"}, status_code=401)

@app.post("/api/auth/login")
async def auth_login(request: Request):
    body = await request.json()
    if body.get("password") != _DASHBOARD_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    resp = JSONResponse({"ok": True})
    resp.set_cookie("rt_session", _VALID_TOKEN, httponly=True, samesite="lax", max_age=86400 * 30)
    return resp

@app.post("/api/auth/logout")
async def auth_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("rt_session")
    return resp

@app.get("/api/auth/status")
async def auth_status(request: Request):
    token = request.cookies.get("rt_session", "")
    return {"authenticated": token == _VALID_TOKEN}


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
        # Migrate plays table — add source column if missing
        play_cols = [r[1] for r in conn.execute("PRAGMA table_info(plays)").fetchall()]
        if "source" not in play_cols:
            conn.execute("ALTER TABLE plays ADD COLUMN source TEXT DEFAULT 'scanner'")
        if "exit_reason" not in play_cols:
            conn.execute("ALTER TABLE plays ADD COLUMN exit_reason TEXT")
        if "bt_win_rate" not in play_cols:
            conn.execute("ALTER TABLE plays ADD COLUMN bt_win_rate REAL")
        if "bt_qualified" not in play_cols:
            conn.execute("ALTER TABLE plays ADD COLUMN bt_qualified INTEGER DEFAULT 0")
        if "reentry_flag" not in play_cols:
            conn.execute("ALTER TABLE plays ADD COLUMN reentry_flag INTEGER DEFAULT 0")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                signal TEXT NOT NULL,
                direction TEXT NOT NULL DEFAULT 'LONG',
                n_trades INTEGER DEFAULT 0,
                n_wins INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (signal, direction)
            );
        """)


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


def _find_alpaca_exit_price(client, symbol: str, entry_date: str, direction: str = "LONG") -> Optional[float]:
    """Look up the most recent filled exit order for symbol.
    LONG exits = SELL orders. SHORT exits = BUY orders (covering the short).

    Uses get_order_history (limit=200) and filters client-side to avoid
    potential SDK issues with the GetOrdersRequest 'symbols' parameter.
    """
    try:
        exit_side = "buy" if direction == "SHORT" else "sell"

        after_ts: Optional[str] = None
        if entry_date:
            try:
                after_ts = datetime.fromisoformat(entry_date.replace("Z", "")).isoformat()
            except Exception:
                pass

        orders = client.get_order_history(limit=200)
        for order in orders:
            if order.get("symbol", "").upper() != symbol.upper():
                continue
            raw_side   = order.get("side",   "")
            raw_status = order.get("status", "")
            # Handle both plain value ("sell") and enum-string ("OrderSide.SELL")
            side   = raw_side.split(".")[-1].lower()
            status = raw_status.split(".")[-1].lower()
            if side != exit_side or status != "filled":
                continue
            if after_ts and order.get("filled_at") and order["filled_at"] < after_ts:
                continue
            price = order.get("filled_avg_price")
            if price:
                return float(price)
    except Exception:
        pass
    return None


def _sync_alpaca_plays():
    """Background thread: sync Alpaca positions/orders → DB every 10 minutes.

    - Discovers new Alpaca positions not yet in DB and creates ACTIVE records
    - PENDING → ACTIVE  when Alpaca confirms the entry fill
    - ACTIVE  → CLOSED  when Alpaca no longer holds the position (stop/target hit)
    """
    time.sleep(30)  # let webapp fully start before first sync
    while True:
        try:
            client = get_alpaca_client()
            if client:
                alpaca_positions = {p["symbol"]: p for p in client.get_positions()}
                with get_db() as conn:
                    open_plays = conn.execute(
                        "SELECT * FROM plays WHERE status IN ('PENDING','ACTIVE')"
                    ).fetchall()
                    tracked_syms = {play["symbol"] for play in open_plays}

                    # Discover new Alpaca positions not yet in DB
                    for sym, pos in alpaca_positions.items():
                        if sym not in tracked_syms:
                            avg_entry = float(pos.get("avg_entry_price") or 0)
                            qty = float(pos.get("qty") or 0)
                            direction = "SHORT" if pos.get("side") == "short" else "LONG"
                            if avg_entry > 0 and qty > 0:
                                conn.execute(
                                    """INSERT INTO plays
                                       (symbol, direction, status, entry_price, shares,
                                        entry_date, signal, notes, source)
                                       VALUES (?,?,?,?,?,?,?,?,?)""",
                                    (sym, direction, "ACTIVE", avg_entry, qty,
                                     datetime.utcnow().isoformat()[:10],
                                     "auto-sync",
                                     f"Auto-discovered from Alpaca position sync ({direction})",
                                     "scanner"),
                                )

                    # Update existing plays
                    for play in open_plays:
                        sym = play["symbol"]
                        if sym in alpaca_positions:
                            if play["status"] == "PENDING":
                                avg_entry = float(
                                    alpaca_positions[sym].get("avg_entry_price", 0)
                                    or play["entry_price"] or 0
                                )
                                conn.execute(
                                    "UPDATE plays SET status='ACTIVE', entry_price=? WHERE id=?",
                                    (avg_entry or play["entry_price"], play["id"]),
                                )
                        else:
                            # Position gone — find exit price from closed orders
                            direction = play["direction"] or "LONG"
                            exit_price = _find_alpaca_exit_price(
                                client, sym, play["entry_date"] or "", direction
                            )
                            entry_p = float(play["entry_price"] or 0)
                            shares  = float(play["shares"] or 0)
                            if exit_price and entry_p and shares:
                                if direction == "SHORT":
                                    pnl = round((entry_p - exit_price) * shares, 2)
                                else:
                                    pnl = round((exit_price - entry_p) * shares, 2)
                                pnl_pct = round(pnl / (entry_p * shares) * 100, 2)
                            else:
                                pnl = pnl_pct = None
                            # Determine exit reason
                            exit_reason = "unknown"
                            if exit_price is not None:
                                stop_l  = float(play["stop_loss"]  or 0)
                                take_p  = float(play["take_profit"] or 0)
                                if stop_l > 0 and exit_price <= stop_l * 1.02:
                                    exit_reason = "stop_hit"
                                elif take_p > 0 and exit_price >= take_p * 0.98:
                                    exit_reason = "target_hit"
                            reentry_flag = 1 if exit_reason == "target_hit" and pnl and pnl > 0 else 0
                            conn.execute(
                                """UPDATE plays SET status='CLOSED', exit_price=?,
                                   exit_date=?, pnl=?, pnl_pct=?, exit_reason=?, reentry_flag=? WHERE id=?""",
                                (exit_price,
                                 datetime.utcnow().isoformat()[:10],
                                 pnl, pnl_pct, exit_reason, reentry_flag, play["id"]),
                            )
                            if pnl is not None:
                                _update_signal_performance(conn, play["signal"] or "", play["direction"] or "LONG", pnl)

                    # Backfill: retry CLOSED plays that have no exit_price yet (last 60 days)
                    cutoff = (datetime.utcnow() - timedelta(days=60)).isoformat()[:10]
                    null_exits = conn.execute(
                        """SELECT * FROM plays WHERE status='CLOSED'
                           AND exit_price IS NULL AND pnl IS NULL AND entry_date >= ?""",
                        (cutoff,)
                    ).fetchall()
                    if null_exits:
                        bg_orders = client.get_order_history(limit=500)
                        def _norm_bg(raw: str) -> str:
                            return raw.split(".")[-1].lower() if raw else ""
                        for play in null_exits:
                            sym       = play["symbol"]
                            direction = play["direction"] or "LONG"
                            entry_p   = float(play["entry_price"] or 0)
                            shares    = float(play["shares"] or 0)
                            entry_side_val = "sell" if direction == "SHORT" else "buy"
                            exit_side_val  = "buy"  if direction == "SHORT" else "sell"
                            sym_orders = [o for o in bg_orders if o.get("symbol","").upper() == sym.upper()]
                            entry_ok = any(
                                _norm_bg(o.get("side","")) == entry_side_val
                                and _norm_bg(o.get("status","")) == "filled"
                                and float(o.get("filled_avg_price") or 0) > 0
                                for o in sym_orders
                            )
                            if not entry_ok:
                                conn.execute(
                                    "UPDATE plays SET pnl=0.0, pnl_pct=0.0, exit_reason='entry_expired' WHERE id=?",
                                    (play["id"],)
                                )
                                continue
                            exit_price = None
                            for o in sym_orders:
                                if _norm_bg(o.get("side","")) == exit_side_val and _norm_bg(o.get("status","")) == "filled":
                                    p = float(o.get("filled_avg_price") or 0)
                                    if p > 0:
                                        exit_price = p
                                        break
                            if not exit_price:
                                continue
                            mult = -1 if direction == "SHORT" else 1
                            pnl     = round((exit_price - entry_p) * shares * mult, 2) if entry_p and shares else None
                            pnl_pct = round(pnl / (entry_p * shares) * 100, 2) if pnl is not None else None
                            stop_l = float(play["stop_loss"] or 0)
                            take_p = float(play["take_profit"] or 0)
                            exit_reason = "unknown"
                            if stop_l > 0 and exit_price <= stop_l * 1.02:
                                exit_reason = "stop_hit"
                            elif take_p > 0 and exit_price >= take_p * 0.98:
                                exit_reason = "target_hit"
                            reentry_flag = 1 if exit_reason == "target_hit" and pnl and pnl > 0 else 0
                            conn.execute(
                                """UPDATE plays SET exit_price=?, pnl=?, pnl_pct=?,
                                   exit_reason=?, reentry_flag=? WHERE id=?""",
                                (exit_price, pnl, pnl_pct, exit_reason, reentry_flag, play["id"]),
                            )
                            if pnl is not None:
                                _update_signal_performance(conn, play["signal"] or "", direction, pnl)
        except Exception:
            pass
        time.sleep(600)  # run every 10 minutes


def _update_signal_performance(conn, signal: str, direction: str, pnl: float) -> None:
    if not signal or signal == "auto-sync":
        return
    is_win = 1 if pnl > 0 else 0
    conn.execute("""
        INSERT INTO signal_performance (signal, direction, n_trades, n_wins, total_pnl, updated_at)
        VALUES (?, ?, 1, ?, ?, datetime('now'))
        ON CONFLICT(signal, direction) DO UPDATE SET
            n_trades = n_trades + 1,
            n_wins = n_wins + ?,
            total_pnl = total_pnl + ?,
            updated_at = datetime('now')
    """, (signal, direction, is_win, pnl, is_win, pnl))


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
    source: Optional[str] = "scanner"
    bt_win_rate: Optional[float] = None
    bt_qualified: Optional[int] = None


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
        equity          = 10_000.0
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
            curve_values = [10_000.0, 10_000.0]

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


# ── Build info ────────────────────────────────────────────────────
@app.get("/api/build_info")
def build_info():
    return {"deployed_at": DEPLOY_TIME.strftime("%b %d %Y, %-I:%M %p") + " UTC"}


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
def watchlist_quotes(fresh: bool = False):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT symbol, score, bsh, auto_generated FROM watchlist"
        ).fetchall()
        scan_ts = conn.execute(
            "SELECT value FROM app_settings WHERE key='watchlist_scan_ts'"
        ).fetchone()
    last_refresh = scan_ts["value"] if scan_ts else None
    if fresh:
        for row in rows:
            sym = row["symbol"]
            _cache.pop(f"q:{sym}", None)
            _cache.pop(f"sig:{sym}", None)
    results = []
    for row in rows:
        sym = row["symbol"]
        q = get_quote(sym)
        if not q:
            continue
        stored_score = row["score"] or 50
        stored_bsh   = row["bsh"] or "HOLD"
        sig = get_signals(sym)
        if row["auto_generated"]:
            # Use stored score/bsh for auto entries; only fetch live RSI
            buy_pct = stored_score
            bsh     = stored_bsh
        else:
            buy_pct = sig.get("buy_pct", 50)
            bsh     = sig.get("bsh", "HOLD")
        rsi = sig.get("rsi", 50)
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


_NEWS_BULLISH = {
    "beat","surge","growth","upgrade","buy","strong","record","rally","gain","rise","soar","jump",
    "high","profit","outperform","positive","boost","expand","upside","bull","breakout","momentum",
    "revenue","earnings","guidance","raised","exceed","top","win","agreement","deal","partner",
    "launch","approve","fda","cleared","awarded","contract","dividend","buyback","acquisition",
    "beat estimates","above expectations","new high","all-time","accelerat","innovate","robust",
}
_NEWS_BEARISH = {
    "miss","drop","concern","risk","cut","loss","warning","downgrade","sell","decline","fall",
    "weak","slump","disappoint","below","fear","pressure","lawsuit","investigate","fraud","probe",
    "recall","layoff","restructur","debt","bankrupt","default","fine","penalt","regulat","halt",
    "suspend","delay","recall","shortage","tariff","sanction","litigation","class action","short",
    "missed estimates","below expectations","new low","all-time low","violat","seize","crash",
}

def _news_sentiment(text: str) -> tuple:
    words = set(text.lower().split())
    bull = sum(1 for w in _NEWS_BULLISH if w in text.lower())
    bear = sum(1 for w in _NEWS_BEARISH if w in text.lower())
    if bull > bear:
        return "BUY", "#16a34a", bull, bear
    elif bear > bull:
        return "SELL", "#dc2626", bull, bear
    return "HOLD", "#d97706", bull, bear

def _fetch_alpaca_news(sym: str, limit: int = 6) -> list:
    try:
        import requests as _req
        resp = _req.get(
            "https://data.alpaca.markets/v1beta1/news",
            params={"symbols": sym, "limit": limit, "sort": "desc"},
            headers={
                "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
                "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
            },
            timeout=8,
        )
        out = []
        for a in resp.json().get("news", []):
            summary = a.get("summary") or a.get("headline") or ""
            if len(summary) > 200: summary = summary[:197] + "…"
            out.append({
                "headline":  a.get("headline", ""),
                "source":    a.get("source", "Alpaca News"),
                "source_tag": "alpaca",
                "url":       a.get("url", ""),
                "summary":   summary,
                "published": (a.get("created_at") or "")[:10],
            })
        return out
    except Exception:
        return []

def _fetch_yahoo_rss(sym: str, limit: int = 5) -> list:
    try:
        import requests as _req
        import xml.etree.ElementTree as ET
        resp = _req.get(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
        root = ET.fromstring(resp.text)
        out = []
        for item in root.findall(".//item")[:limit]:
            title = item.findtext("title") or ""
            link  = item.findtext("link") or ""
            desc  = item.findtext("description") or ""
            pub   = item.findtext("pubDate") or ""
            # pubDate like "Mon, 25 May 2026 12:00:00 +0000" → "2026-05-25"
            try:
                from email.utils import parsedate
                import calendar
                pd_t = parsedate(pub)
                pub_str = f"{pd_t[0]}-{pd_t[1]:02d}-{pd_t[2]:02d}" if pd_t else pub[:10]
            except Exception:
                pub_str = pub[:10]
            if len(desc) > 200: desc = desc[:197] + "…"
            out.append({
                "headline":   title,
                "source":     "Yahoo Finance",
                "source_tag": "yahoo",
                "url":        link,
                "summary":    desc,
                "published":  pub_str,
            })
        return out
    except Exception:
        return []

def _fetch_google_rss(sym: str, limit: int = 5) -> list:
    try:
        import requests as _req
        import xml.etree.ElementTree as ET
        query = f"{sym} stock"
        resp = _req.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
        root = ET.fromstring(resp.text)
        out = []
        for item in root.findall(".//item")[:limit]:
            title = item.findtext("title") or ""
            link  = item.findtext("link") or ""
            pub   = item.findtext("pubDate") or ""
            source_el = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            source_name = "Google News"
            # Extract publisher from title if formatted as "Headline - Publisher"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title, source_name = parts[0].strip(), parts[1].strip()
            try:
                from email.utils import parsedate
                pd_t = parsedate(pub)
                pub_str = f"{pd_t[0]}-{pd_t[1]:02d}-{pd_t[2]:02d}" if pd_t else pub[:10]
            except Exception:
                pub_str = pub[:10]
            out.append({
                "headline":   title,
                "source":     source_name,
                "source_tag": "google",
                "url":        link,
                "summary":    "",
                "published":  pub_str,
            })
        return out
    except Exception:
        return []

def _deduplicate(articles: list) -> list:
    seen, out = [], []
    for a in articles:
        h = a["headline"].lower()
        words = set(h.split())
        duplicate = False
        for s in seen:
            overlap = len(words & s) / max(len(words | s), 1)
            if overlap > 0.6:
                duplicate = True
                break
        if not duplicate:
            seen.append(words)
            out.append(a)
    return out

@app.get("/api/news/{symbol}")
def symbol_news(symbol: str):
    sym = symbol.upper()
    cache_key = f"news2:{sym}"
    now = time.time()
    cached = _cache.get(cache_key)
    if cached and now - cached[1] < 1800:
        return cached[0]

    # Gather from all three sources in parallel
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=3) as ex:
        f_alp = ex.submit(_fetch_alpaca_news, sym, 6)
        f_yah = ex.submit(_fetch_yahoo_rss,   sym, 5)
        f_goo = ex.submit(_fetch_google_rss,  sym, 5)
        raw = f_alp.result() + f_yah.result() + f_goo.result()

    articles = _deduplicate(raw)[:10]

    total_bull, total_bear = 0, 0
    result = []
    for a in articles:
        text = a["headline"] + " " + a.get("summary", "")
        sentiment, color, bull, bear = _news_sentiment(text)
        total_bull += bull
        total_bear += bear
        result.append({**a, "sentiment": sentiment, "color": color})

    # Aggregate news score 0–100 (50 = neutral)
    net = total_bull - total_bear
    news_score = max(0, min(100, 50 + net * 4))
    if news_score >= 60:
        score_label, score_color = "Bullish", "#16a34a"
    elif news_score <= 40:
        score_label, score_color = "Bearish", "#dc2626"
    else:
        score_label, score_color = "Neutral", "#d97706"

    payload = {
        "articles":    result,
        "news_score":  news_score,
        "score_label": score_label,
        "score_color": score_color,
    }
    _cache[cache_key] = (payload, now)
    return payload


# ── Blend Analysis ─────────────────────────────────────────────────
def _blend_score(scanner_score: float, in_watchlist: bool, bt: Optional[dict]) -> float:
    """Compute blended conviction score for a symbol candidate."""
    conviction_mult = 1.25 if in_watchlist else 1.0
    base = float(scanner_score or 50) * conviction_mult
    bt_bonus = 0.0
    if bt and bt.get("qualified"):
        pf = float(bt.get("profit_factor") or 1.0)
        # win_rate stored as percentage (0-100); guard against 0 triggering `or`
        wr = float(bt["win_rate"] if bt.get("win_rate") is not None else 50.0) / 100.0
        bt_bonus = min(15.0, (pf - 1.0) * 8.0 + max(0.0, wr - 0.50) * 20.0)
    return min(100.0, base + bt_bonus)


def _pos_size_from_blend(blend: float) -> Optional[float]:
    if blend >= 90: return 6.5
    if blend >= 80: return 5.5
    if blend >= 70: return 5.0
    if blend >= 60: return 3.5
    return None


_DEFAULT_RATIO = {"scanner_pct": 0.60, "watchlist_pct": 0.40}


@app.get("/api/blend-ratio")
def get_blend_ratio():
    """Return the current adaptive scanner/watchlist slot ratio."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key='blend_ratio'"
        ).fetchone()
    if row:
        try:
            return json.loads(row["value"])
        except Exception:
            pass
    return _DEFAULT_RATIO


@app.put("/api/blend-ratio")
def set_blend_ratio(body: dict):
    """Bot calls this after each scan to persist the updated ratio."""
    scanner_pct  = max(0.25, min(0.75, float(body.get("scanner_pct",  0.60))))
    watchlist_pct = round(1.0 - scanner_pct, 4)
    payload = json.dumps({
        "scanner_pct":  round(scanner_pct, 4),
        "watchlist_pct": watchlist_pct,
        "updated_at": datetime.utcnow().isoformat()[:16],
        "reason": body.get("reason", ""),
    })
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO app_settings (key, value) VALUES ('blend_ratio', ?)",
            (payload,),
        )
    return {"updated": True, "scanner_pct": scanner_pct, "watchlist_pct": watchlist_pct}


@app.get("/api/blend-analysis")
def blend_analysis():
    with get_db() as conn:
        src_rows = conn.execute(
            "SELECT source, pnl, pnl_pct FROM plays WHERE status='CLOSED' AND pnl IS NOT NULL"
        ).fetchall()
        wl_rows = conn.execute(
            "SELECT symbol, score, bsh, notes FROM watchlist ORDER BY score DESC NULLS LAST"
        ).fetchall()
        bt_rows = conn.execute(
            """SELECT s.symbol, s.win_rate, s.avg_return, s.profit_factor, s.qualified
               FROM symbol_backtests s
               INNER JOIN (
                   SELECT symbol, MAX(run_at) AS latest FROM symbol_backtests GROUP BY symbol
               ) m ON s.symbol=m.symbol AND s.run_at=m.latest"""
        ).fetchall()
        ratio_row = conn.execute(
            "SELECT value FROM app_settings WHERE key='blend_ratio'"
        ).fetchone()

    # Source performance breakdown
    src_perf: Dict[str, Any] = {}
    for r in src_rows:
        src = r["source"] or "scanner"
        if src not in src_perf:
            src_perf[src] = {"count": 0, "total_pnl": 0.0, "wins": 0}
        src_perf[src]["count"] += 1
        src_perf[src]["total_pnl"] += r["pnl"] or 0
        if (r["pnl"] or 0) > 0:
            src_perf[src]["wins"] += 1
    for src, d in src_perf.items():
        n = d["count"] or 1
        d["avg_pnl"] = round(d["total_pnl"] / n, 2)
        d["win_rate"] = round(d["wins"] / n * 100, 1)
        d["total_pnl"] = round(d["total_pnl"], 2)

    bt_map = {r["symbol"]: dict(r) for r in bt_rows}

    # Score each watchlist symbol
    candidates = []
    for row in wl_rows:
        sym = row["symbol"]
        scanner_score = float(row["score"] or 50)
        bt = bt_map.get(sym)
        blend = _blend_score(scanner_score, True, bt)
        pos_pct = _pos_size_from_blend(blend)
        candidates.append({
            "symbol": sym,
            "scanner_score": round(scanner_score, 1),
            "bsh": row["bsh"] or "HOLD",
            "bt_qualified": bool(bt and bt.get("qualified")),
            # win_rate already stored as percent (0-100) — do NOT multiply again
            "win_rate": round(bt.get("win_rate") or 0, 1) if bt else None,
            "profit_factor": round(bt.get("profit_factor") or 1.0, 2) if bt else None,
            "blend_score": round(blend, 1),
            "pos_size_pct": pos_pct,
        })

    candidates.sort(key=lambda x: x["blend_score"], reverse=True)
    current_ratio = json.loads(ratio_row["value"]) if ratio_row else _DEFAULT_RATIO

    return {
        "source_performance": src_perf,
        "candidates": candidates[:25],
        "current_ratio": current_ratio,
        "formula": {
            "conviction_mult": "×1.25 for watchlist symbols (human monitoring = 25% score boost)",
            "bt_bonus": "Up to +15 pts: (profit_factor−1)×8 + (win_rate−50%)×20 — only if 90-day BT passed",
            "pos_sizing": "≥90→6.5% | ≥80→5.5% | ≥70→5.0% | ≥60→3.5% | <60→skip",
            "scanner_entry": "Score ≥65, ATR×1.5 stop, ATR×3.75 target, 5% risk",
            "watchlist_entry": "Blend ≥70, ATR×1.2 stop (tighter), ATR×3.0 target, 5.5% risk",
        },
    }


# ── Plays routes ───────────────────────────────────────────────────
@app.get("/api/plays/equity-curve")
def plays_equity_curve():
    """Account equity over time from Alpaca portfolio history (daily snapshots)."""
    client = get_alpaca_client()
    if not client:
        return {"dates": [], "values": [], "starting": 10000.0}
    try:
        from alpaca.trading.requests import PortfolioHistoryRequest
        history = client.trading.get_portfolio_history(
            filter=PortfolioHistoryRequest(period="1M", timeframe="1D")
        )
        timestamps = history.timestamp or []
        equities   = history.equity   or []
        pairs = [
            (datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"), round(float(e), 2))
            for t, e in zip(timestamps, equities)
            if e is not None and float(e) > 0
        ]
        if not pairs:
            return {"dates": [], "values": [], "starting": 10000.0}
        dates, values = zip(*pairs)
        return {"dates": list(dates), "values": list(values), "starting": 10000.0}
    except Exception:
        return {"dates": [], "values": [], "starting": 10000.0}


@app.get("/api/plays")
def list_plays(status: Optional[str] = None, fresh: bool = False):
    with get_db() as conn:
        base = """
            SELECT p.*,
                   CASE WHEN w.symbol IS NOT NULL THEN 1 ELSE 0 END AS in_watchlist
            FROM plays p
            LEFT JOIN watchlist w ON p.symbol = w.symbol
        """
        if status:
            rows = conn.execute(base + "WHERE p.status=? ORDER BY p.created_at DESC", (status,)).fetchall()
        else:
            rows = conn.execute(base + "ORDER BY p.created_at DESC").fetchall()
    plays = [dict(r) for r in rows]
    if fresh:
        for p in plays:
            if p["status"] in ("ACTIVE", "PENDING"):
                _cache.pop(f"q:{p['symbol']}", None)
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
                order = client.trading.submit_order(order_req)
                alpaca_order_id = str(order.id)
                status = "ACTIVE"
        except Exception:
            status = "PENDING"

    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO plays (symbol,direction,status,entry_price,stop_loss,take_profit,
               shares,entry_date,signal,notes,alpaca_order_id,source,bt_win_rate,bt_qualified)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sym, body.direction, status, body.entry_price, body.stop_loss,
             body.take_profit, body.shares, datetime.utcnow().isoformat(),
             body.signal, body.notes, alpaca_order_id, body.source or "scanner",
             body.bt_win_rate, body.bt_qualified),
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
_REAL_TRADE = "status='CLOSED' AND exit_price IS NOT NULL"

@app.get("/api/history")
def trade_history(symbol: Optional[str] = None, days: Optional[int] = None):
    q = f"SELECT * FROM plays WHERE {_REAL_TRADE}"
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
            f"SELECT pnl, pnl_pct, exit_date FROM plays WHERE {_REAL_TRADE} ORDER BY exit_date"
        ).fetchall()
    if not rows:
        return {"total_pnl": 0, "win_rate": 0, "profit_factor": 0, "n_trades": 0, "monthly": [], "avg_win": 0, "avg_loss": 0}
    n_total = len(rows)
    pnls   = [r["pnl"] for r in rows if r["pnl"] is not None]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]   # strictly negative only
    monthly = {}
    for r in rows:
        if r["exit_date"] and r["pnl"] is not None and r["pnl"] != 0:
            m = r["exit_date"][:7]
            monthly[m] = monthly.get(m, 0) + r["pnl"]
    loss_sum = abs(sum(losses)) if losses else 0
    return {
        "total_pnl":     round(sum(pnls), 2),
        "n_trades":      n_total,
        "win_rate":      round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        "profit_factor": round(sum(wins) / loss_sum, 2) if losses and loss_sum > 0 else (999.0 if wins else 0),
        "avg_win":       round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss":      round(sum(losses) / len(losses), 2) if losses else 0,
        "monthly":       [{"month": k, "pnl": round(v, 2)} for k, v in sorted(monthly.items())],
    }


@app.get("/api/signal-performance")
def signal_performance():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT signal, direction, n_trades, n_wins, total_pnl,
                   CASE WHEN n_trades > 0 THEN ROUND(n_wins * 100.0 / n_trades, 1) ELSE 0 END as win_rate,
                   CASE WHEN n_trades > 0 THEN ROUND(total_pnl / n_trades, 2) ELSE 0 END as avg_pnl,
                   updated_at
            FROM signal_performance ORDER BY n_trades DESC
        """).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/reentry-candidates")
def reentry_candidates():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT symbol, direction, signal, exit_price, pnl, exit_date
            FROM plays WHERE reentry_flag=1 AND status='CLOSED'
            ORDER BY exit_date DESC LIMIT 20
        """).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/debug/orders")
def debug_orders():
    """Return raw order history from Alpaca for debugging (dev use)."""
    client = get_alpaca_client()
    if not client:
        raise HTTPException(503, "Alpaca not connected")
    with get_db() as conn:
        null_syms = [r["symbol"] for r in conn.execute(
            "SELECT DISTINCT symbol FROM plays WHERE status='CLOSED' AND exit_price IS NULL"
        ).fetchall()]
    orders = client.get_order_history(limit=200)
    relevant = [o for o in orders if o.get("symbol", "").upper() in [s.upper() for s in null_syms]]
    return {"null_exit_symbols": null_syms, "relevant_orders": relevant, "total_orders_fetched": len(orders)}


@app.post("/api/history/fix-null-pnl")
def fix_null_pnl():
    """Direct fix: set pnl=0 for CLOSED plays whose entries never executed.
    Also handles the duplicate MET play created by the sync after bracket expiry."""
    client = get_alpaca_client()
    if not client:
        raise HTTPException(503, "Alpaca not connected")
    all_orders = client.get_order_history(limit=500)

    def _norm(raw: str) -> str:
        return raw.split(".")[-1].lower() if raw else ""

    # Build set of symbols that have a filled entry order in Alpaca
    filled_entry_syms: set = set()
    for o in all_orders:
        status = _norm(o.get("status", ""))
        side   = _norm(o.get("side", ""))
        filled_price = float(o.get("filled_avg_price") or 0)
        if status == "filled" and filled_price > 0:
            # Could be buy (LONG entry) or sell (SHORT entry) — just track the symbol
            filled_entry_syms.add(o.get("symbol", "").upper())

    fixed = []
    with get_db() as conn:
        null_plays = conn.execute(
            "SELECT * FROM plays WHERE status='CLOSED' AND pnl IS NULL"
        ).fetchall()
        for play in null_plays:
            sym = play["symbol"].upper()
            pid = play["id"]
            direction = play["direction"] or "LONG"

            # Check if a duplicate ACTIVE play exists for this symbol+direction
            dup = conn.execute(
                "SELECT id FROM plays WHERE symbol=? AND direction=? AND status='ACTIVE'",
                (sym, direction)
            ).fetchone()

            if sym not in filled_entry_syms:
                # Entry never filled on Alpaca — phantom trade
                conn.execute(
                    "UPDATE plays SET pnl=0.0, pnl_pct=0.0, exit_reason='entry_expired' WHERE id=?",
                    (pid,)
                )
                fixed.append({"symbol": sym, "id": pid, "action": "entry_expired pnl=0"})
            elif dup:
                # Entry did fill but sync duplicated the play — original CLOSED is superseded
                conn.execute(
                    "UPDATE plays SET pnl=0.0, pnl_pct=0.0, exit_reason='superseded_by_sync' WHERE id=?",
                    (pid,)
                )
                fixed.append({"symbol": sym, "id": pid, "action": "superseded_by_sync pnl=0"})
            else:
                fixed.append({"symbol": sym, "id": pid, "action": "skipped_needs_exit_price"})
    return {"fixed": len([f for f in fixed if "skipped" not in f["action"]]), "details": fixed}


@app.post("/api/history/backfill-pnl")
def backfill_pnl():
    """Force-retry exit price lookup for CLOSED plays with null pnl (last 60 days).

    Two cases:
    - Entry order expired/cancelled (never filled) → pnl=0, exit_reason='entry_expired'
    - Entry filled, exit order found → compute pnl normally
    """
    client = get_alpaca_client()
    if not client:
        raise HTTPException(503, "Alpaca not connected")
    cutoff = (datetime.utcnow() - timedelta(days=60)).isoformat()[:10]
    # Fetch all orders once to avoid repeated API calls
    all_orders = client.get_order_history(limit=500)
    updated = 0
    with get_db() as conn:
        null_exits = conn.execute(
            "SELECT * FROM plays WHERE status='CLOSED' AND exit_price IS NULL AND entry_date >= ?",
            (cutoff,)
        ).fetchall()
        for play in null_exits:
            sym       = play["symbol"]
            direction = play["direction"] or "LONG"
            entry_p   = float(play["entry_price"] or 0)
            shares    = float(play["shares"] or 0)
            entry_side_val = "sell" if direction == "SHORT" else "buy"
            exit_side_val  = "buy"  if direction == "SHORT" else "sell"

            # Filter orders for this symbol
            sym_orders = [o for o in all_orders if o.get("symbol", "").upper() == sym.upper()]

            def _norm(raw: str) -> str:
                return raw.split(".")[-1].lower() if raw else ""

            # Check if entry was ever filled
            entry_filled_price = None
            for o in sym_orders:
                if _norm(o.get("side", "")) == entry_side_val and _norm(o.get("status", "")) == "filled":
                    p = float(o.get("filled_avg_price") or 0)
                    if p > 0:
                        entry_filled_price = p
                        break

            if entry_filled_price is None:
                # Entry expired/cancelled — trade was never actually placed
                conn.execute(
                    "UPDATE plays SET pnl=0.0, pnl_pct=0.0, exit_reason='entry_expired' WHERE id=?",
                    (play["id"],)
                )
                updated += 1
                continue

            # Entry filled — look for a filled exit order
            exit_price = None
            for o in sym_orders:
                if _norm(o.get("side", "")) == exit_side_val and _norm(o.get("status", "")) == "filled":
                    p = float(o.get("filled_avg_price") or 0)
                    if p > 0:
                        exit_price = p
                        break

            if exit_price is None:
                continue  # exit not yet known; background sync will catch it later

            if entry_p and shares:
                mult = -1 if direction == "SHORT" else 1
                pnl     = round((exit_price - entry_p) * shares * mult, 2)
                pnl_pct = round(pnl / (entry_p * shares) * 100, 2)
            else:
                pnl = pnl_pct = None

            stop_l = float(play["stop_loss"] or 0)
            take_p = float(play["take_profit"] or 0)
            exit_reason = "unknown"
            if stop_l > 0 and exit_price <= stop_l * 1.02:
                exit_reason = "stop_hit"
            elif take_p > 0 and exit_price >= take_p * 0.98:
                exit_reason = "target_hit"
            reentry_flag = 1 if exit_reason == "target_hit" and pnl and pnl > 0 else 0
            conn.execute(
                """UPDATE plays SET exit_price=?, pnl=?, pnl_pct=?,
                   exit_reason=?, reentry_flag=? WHERE id=?""",
                (exit_price, pnl, pnl_pct, exit_reason, reentry_flag, play["id"]),
            )
            if pnl is not None:
                _update_signal_performance(conn, play["signal"] or "", direction, pnl)
            updated += 1
    return {"backfilled": updated, "checked": len(null_exits)}


@app.get("/api/history/export")
def export_history():
    with get_db() as conn:
        rows = conn.execute(f"SELECT * FROM plays WHERE {_REAL_TRADE} ORDER BY exit_date DESC").fetchall()
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
    """Return pct return of close prices between start and end dates (inclusive).
    Returns None if the window contains fewer than 2 trading bars — no fallback to avoid
    inflating returns from multi-year data when the play is too recent."""
    try:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        # Never measure a future end date
        today_ts = pd.Timestamp(datetime.utcnow().strftime("%Y-%m-%d"))
        if e > today_ts:
            e = today_ts
        # Need at least 1 bar before or on start to anchor the return
        anchor = df[df.index <= s]
        after  = df[(df.index > s) & (df.index <= e)]
        if anchor.empty or after.empty:
            return None
        open_price = float(anchor["close"].iloc[-1])
        close_price = float(after["close"].iloc[-1])
        return round((close_price - open_price) / open_price * 100, 2)
    except Exception:
        return None


def _find_peer(symbol: str, start: str, end: str, play_df: "pd.DataFrame") -> tuple:
    """Find the most correlated peer in the same sector over the play period.
    Requires at least 3 actual bars in the period — no multi-year fallback."""
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
        today_ts = pd.Timestamp(datetime.utcnow().strftime("%Y-%m-%d"))
        if e_ts > today_ts:
            e_ts = today_ts
        anchor = play_df[play_df.index <= s_ts]
        after  = play_df[(play_df.index > s_ts) & (play_df.index <= e_ts)]
        if anchor.empty or len(after) < 2:
            return None, None
        play_rets = after["close"].pct_change().dropna()
        best_sym, best_corr, best_ret = None, -2.0, None
        for peer in candidates:
            pdf = _load_bars(peer, days=400)
            if pdf is None or pdf.empty:
                continue
            peer_after = pdf[(pdf.index > s_ts) & (pdf.index <= e_ts)]
            if len(peer_after) < 2:
                continue
            peer_rets = peer_after["close"].pct_change().dropna()
            aligned_p, aligned_q = play_rets.align(peer_rets, join="inner")
            if len(aligned_p) < 2:
                continue
            corr = float(aligned_p.corr(aligned_q))
            if corr > best_corr:
                peer_anchor = pdf[pdf.index <= s_ts]
                if peer_anchor.empty or peer_after.empty:
                    continue
                peer_total = round(
                    (float(peer_after["close"].iloc[-1]) - float(peer_anchor["close"].iloc[-1]))
                    / float(peer_anchor["close"].iloc[-1]) * 100, 2
                )
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
        # Prefer explicit entry_date; fall back to created_at date only (not datetime)
        raw_start = p["entry_date"] or (p["created_at"] or today)
        start = raw_start[:10]
        end = (p["exit_date"] or today)[:10]
        # Skip plays with no meaningful holding period (same-day or future)
        if start >= today and status != "CLOSED":
            results.append((play_id, sym, status, start, end, None, None, None, None, 0))
            continue
        play_df = _load_bars(sym, days=400)
        if play_df is None:
            continue
        play_ret = _compute_return_over_period(play_df, start, end)
        spy_ret  = _compute_return_over_period(spy_df, start, end) if spy_df is not None else None
        peer_sym, peer_ret = _find_peer(sym, start, end, play_df)
        beats_spy = 1 if (play_ret is not None and spy_ret is not None and play_ret > spy_ret) else 0
        results.append((play_id, sym, status, start, end,
                         play_ret,
                         spy_ret,
                         peer_sym,
                         peer_ret,
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


# ── 5-Year Strategy Backtest ───────────────────────────────────────
_S5Y_STATE: Dict = {"running": False, "progress": 0, "result": None, "error": None}

# Exact same universe the live bot scans — keeps backtest consistent with live behavior
_S5Y_SYMBOLS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD",
    "JPM","GS","BAC","MS","V","MA",
    "XOM","CVX","SLB","COP",
    "EQIX","AMT","PLD",
    "NUE","CAT","URI","HON",
    "TXN","AVGO","QCOM",
    "UNH","LLY","JNJ","ABBV",
    "COIN","IPGP","PLTR","MSTR",
    "SPY","QQQ","IWM",
]

def _run_5y_backtest():
    import numpy as np
    try:
        from core.opportunity_scanner import score_symbol as _score_sym
    except Exception as e:
        _S5Y_STATE["error"] = f"Import failed: {e}"
        _S5Y_STATE["running"] = False
        return

    _S5Y_STATE.update({"running": True, "progress": 0, "result": None, "error": None})
    DAYS = 1825
    SCORE_THRESHOLD  = 65
    ATR_STOP_MULT    = 1.5    # current bot: stop = entry - ATR×1.5
    ATR_TARGET_MULT  = 3.75   # current bot: target = ATR×3.75 (floored at +7%)
    SCALE_TRIGGER    = 1.07   # sell 90% when gain hits +7%
    SCALE_PCT        = 0.90
    HOLD_DAYS        = 10
    POSITION_PCT     = 0.10

    all_trades: list = []  # list of (date_idx, ret_pct)
    date_index = None      # will use SPY dates as the master calendar

    # Load SPY first for benchmark
    spy_df = _load_bars("SPY", days=DAYS + 60)
    if spy_df is None or spy_df.empty:
        _S5Y_STATE.update({"error": "SPY data unavailable", "running": False})
        return

    total_syms = len(_S5Y_SYMBOLS)
    for si, sym in enumerate(_S5Y_SYMBOLS):
        _S5Y_STATE["progress"] = int(si / total_syms * 80)
        try:
            df = _load_bars(sym, days=DAYS + 60)
            if df is None or len(df) < 110:
                continue
            open_ = df["open"].astype(float)
            high_ = df["high"].astype(float)
            low_  = df["low"].astype(float)
            close_ = df["close"].astype(float)
            n = len(df)
            high_arr = high_.values
            low_arr  = low_.values
            window_start = max(55, n - DAYS)
            for i in range(window_start, n - HOLD_DAYS - 1):
                scored = _score_sym(df.iloc[:i + 1], sym)
                if not scored or scored["score"] < SCORE_THRESHOLD or not scored["firing_signals"]:
                    continue
                entry = float(open_.iloc[i + 1])
                if entry <= 0:
                    continue
                # True Range ATR (matches live bot: accounts for gaps)
                s = max(1, i - 14)
                prev_closes = close_.values[s - 1: i]
                hl = high_arr[s:i + 1] - low_arr[s:i + 1]
                hpc = np.abs(high_arr[s:i + 1] - prev_closes)
                lpc = np.abs(low_arr[s:i + 1] - prev_closes)
                tr14 = np.maximum(hl, np.maximum(hpc, lpc))
                atr14 = float(np.mean(tr14)) if len(tr14) > 0 else 0.0
                if atr14 <= 0:
                    continue
                init_stop     = max(entry - atr14 * ATR_STOP_MULT, entry * 0.95)
                scaleout_px   = entry * SCALE_TRIGGER  # +7% trigger
                hold_end      = float(close_.iloc[min(i + HOLD_DAYS, n - 1)])
                partial_triggered = False
                trail_stop    = init_stop
                trail_high    = entry
                full_exit     = hold_end
                runner_exit   = hold_end
                for j in range(i + 1, min(i + HOLD_DAYS + 1, n)):
                    h = float(high_arr[j]) if j < n else 0.0
                    l = float(low_arr[j])  if j < n else 1e9
                    if not partial_triggered:
                        if h >= scaleout_px:
                            partial_triggered = True
                            trail_high = scaleout_px
                            trail_stop = max(trail_high - atr14 * ATR_STOP_MULT, entry)
                            if l <= trail_stop:
                                runner_exit = trail_stop; break
                        elif l <= init_stop:
                            full_exit = init_stop; break
                    else:
                        if h > trail_high:
                            trail_high = h
                            trail_stop = max(trail_stop, trail_high - atr14 * ATR_STOP_MULT)
                        if l <= trail_stop:
                            runner_exit = trail_stop; break
                if partial_triggered:
                    ret = (SCALE_PCT * (scaleout_px / entry - 1)
                           + (1 - SCALE_PCT) * (runner_exit / entry - 1)) * 100
                else:
                    ret = (full_exit / entry - 1) * 100
                # Map bar index to calendar date
                try:
                    bar_date = df.index[i + 1]
                except Exception:
                    bar_date = None
                all_trades.append({"sym": sym, "ret": ret, "date": str(bar_date)[:10] if bar_date else ""})
        except Exception:
            continue

    if not all_trades:
        _S5Y_STATE.update({"error": "No trades fired in 5-year window", "running": False})
        return

    _S5Y_STATE["progress"] = 85

    # ── Equity curve ──────────────────────────────────────────────────
    # Sort trades by date, simulate equity with 10% position sizing
    sorted_trades = sorted(all_trades, key=lambda x: x["date"])
    STARTING_EQUITY = 10_000.0
    equity = STARTING_EQUITY
    equity_curve = [equity]
    daily_rets: list = []
    for t in sorted_trades:
        pos_size = equity * POSITION_PCT
        pnl = pos_size * t["ret"] / 100
        equity += pnl
        equity_curve.append(round(equity, 2))
        daily_rets.append(t["ret"] / 100)

    # ── SPY buy-and-hold curve ────────────────────────────────────────
    spy_close = spy_df["close"].astype(float)
    spy_ret_total = float((spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0] * 100)
    spy_final = round(STARTING_EQUITY * (1 + spy_ret_total / 100), 2)
    spy_daily = spy_close.pct_change().dropna().values.tolist()

    # ── Key stats ─────────────────────────────────────────────────────
    rets_arr = np.array(daily_rets)
    total_return = (equity_curve[-1] - STARTING_EQUITY) / STARTING_EQUITY * 100
    n_trades = len(sorted_trades)
    wins = [r for r in daily_rets if r > 0]
    losses = [r for r in daily_rets if r <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades else 0
    avg_win  = float(np.mean(wins)) * 100 if wins else 0
    avg_loss = float(np.mean(losses)) * 100 if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else 99.0
    # Max drawdown
    peak, max_dd = STARTING_EQUITY, 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)
    # Sharpe (annualized, rf=0)
    if len(rets_arr) > 1 and np.std(rets_arr) > 0:
        sharpe = float(np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252))
    else:
        sharpe = 0.0

    _S5Y_STATE["progress"] = 90

    # ── OLS regression (strategy vs SPY) ─────────────────────────────
    try:
        from scipy import stats as _stats
        spy_rets_arr = np.array(spy_daily)
        # Align lengths by taking the shorter
        min_len = min(len(rets_arr), len(spy_rets_arr))
        s_rets = rets_arr[-min_len:]
        m_rets = spy_rets_arr[-min_len:]
        slope, intercept, r_value, p_value, std_err = _stats.linregress(m_rets, s_rets)
        alpha_annualized = float(intercept * 252 * 100)
        beta = float(slope)
        r_squared = float(r_value ** 2)
        p_val = float(p_value)
    except Exception:
        alpha_annualized, beta, r_squared, p_val = 0.0, 1.0, 0.0, 1.0

    # ── Linear regression on equity curve (trend line) ────────────────
    try:
        x = np.arange(len(equity_curve), dtype=float)
        trend_slope, trend_intercept = np.polyfit(x, equity_curve, 1)
        trend_slope_per_trade = float(trend_slope)
    except Exception:
        trend_slope_per_trade = 0.0

    _S5Y_STATE["progress"] = 95

    # ── Monte Carlo (10,000 simulations) ─────────────────────────────
    try:
        np.random.seed(42)
        N_SIM = 10_000
        sim_finals = []
        for _ in range(N_SIM):
            sampled = np.random.choice(rets_arr, size=n_trades, replace=True)
            eq = STARTING_EQUITY
            for r in sampled:
                eq += eq * POSITION_PCT * r
            sim_finals.append(eq)
        sim_arr = np.array(sim_finals)
        mc_p5  = round(float(np.percentile(sim_arr, 5)),  2)
        mc_p25 = round(float(np.percentile(sim_arr, 25)), 2)
        mc_p50 = round(float(np.percentile(sim_arr, 50)), 2)
        mc_p75 = round(float(np.percentile(sim_arr, 75)), 2)
        mc_p95 = round(float(np.percentile(sim_arr, 95)), 2)
        mc_prob_profit = round(float(np.mean(sim_arr > STARTING_EQUITY) * 100), 1)
        mc_prob_2x     = round(float(np.mean(sim_arr > STARTING_EQUITY * 2) * 100), 1)
    except Exception:
        mc_p5 = mc_p25 = mc_p50 = mc_p75 = mc_p95 = STARTING_EQUITY
        mc_prob_profit = mc_prob_2x = 0.0

    # ── Optimisation suggestions ───────────────────────────────────────
    suggestions = []
    if win_rate < 52:
        suggestions.append("Win rate below 52% — tighten score threshold to ≥70 to filter marginal setups")
    if avg_loss < -4.5:
        suggestions.append("Average loss approaching stop (-3%) — consider tightening stop to -2.5% to reduce drawdown")
    if avg_win < 3:
        suggestions.append("Average win (+%.1f%%) smaller than expected — consider raising target to +6%% for better reward:risk" % avg_win)
    if beta > 1.2:
        suggestions.append(f"Beta {beta:.2f} > 1.2 — strategy is amplifying market moves; add volatility filter (avoid entries when VIX >25)")
    if alpha_annualized < 2:
        suggestions.append(f"Alpha {alpha_annualized:.1f}% is low — strategy is not consistently beating the market; review signal weights")
    if max_dd > 25:
        suggestions.append(f"Max drawdown {max_dd:.1f}% is high — reduce position size to 7% or add portfolio-level stop (close all if equity drops 15%)")
    if profit_factor < 1.5:
        suggestions.append(f"Profit factor {profit_factor:.2f} — targeting ≥1.8 for robust edge; eliminate weakest-performing signal type")
    if not suggestions:
        suggestions.append("Strategy metrics look solid. Continue running until 100+ closed trades for statistical significance.")

    _S5Y_STATE["progress"] = 100
    _S5Y_STATE["result"] = {
        "computed_at": datetime.utcnow().isoformat(),
        "config_desc": "ATR×1.5 stop · ATR×3.75 target (floor +7%) · scale-out 90% at +7% · score ≥65",
        "symbols_tested": len(_S5Y_SYMBOLS),
        "starting_equity": STARTING_EQUITY,
        "n_trades": n_trades,
        "total_return_pct": round(total_return, 2),
        "final_equity": round(equity_curve[-1], 2),
        "spy_final": spy_final,
        "spy_ret_pct": round(spy_ret_total, 2),
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(min(profit_factor, 99.0), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "alpha_annualized": round(alpha_annualized, 2),
        "beta": round(beta, 3),
        "r_squared": round(r_squared, 3),
        "p_value": round(p_val, 4),
        "trend_slope": round(trend_slope_per_trade, 2),
        "mc_p5":  mc_p5,  "mc_p25": mc_p25, "mc_p50": mc_p50,
        "mc_p75": mc_p75, "mc_p95": mc_p95,
        "mc_prob_profit": mc_prob_profit,
        "mc_prob_2x": mc_prob_2x,
        "suggestions": suggestions,
        # Sampled equity curve (every Nth point to keep payload small)
        "equity_curve": equity_curve[::max(1, len(equity_curve)//200)],
        "trade_dates": [t["date"] for t in sorted_trades][::max(1, n_trades//200)],
    }
    _S5Y_STATE["running"] = False


@app.post("/api/strategy/backtest5y")
def start_5y_backtest():
    if _S5Y_STATE["running"]:
        return {"status": "already_running", "progress": _S5Y_STATE["progress"]}
    _S5Y_STATE.update({"running": True, "progress": 0, "result": None, "error": None})
    threading.Thread(target=_run_5y_backtest, daemon=True).start()
    return {"status": "started"}


@app.get("/api/strategy/backtest5y/status")
def get_5y_status():
    return {
        "running":  _S5Y_STATE["running"],
        "progress": _S5Y_STATE["progress"],
        "result":   _S5Y_STATE["result"],
        "error":    _S5Y_STATE["error"],
    }


@app.get("/api/strategy/research")
def get_strategy_research():
    """Return optimizer results + scale-out comparison + current config summary."""
    opt_path = ROOT / "bt_optimize_results.json"
    optimizer_results, spy_5yr_pct = [], None
    if opt_path.exists():
        try:
            data = json.loads(opt_path.read_text())
            optimizer_results = data.get("results", [])
            spy_5yr_pct = data.get("spy_5yr_pct")
        except Exception:
            pass

    # Scale-out test results (from offline optimize_scaleout.py run — 5yr, 38 symbols)
    scaleout_results = [
        {"name": "Hard +7% (baseline)", "partial_pct": 100, "sharpe": 2.01, "max_dd": 12.6,
         "total_ret": 680.6, "win_rate": 43.0, "pct_ran_past_7": 0.0, "avg_runner_ret": 0.0, "current": False},
        {"name": "Scale 50% at +7%",    "partial_pct":  50, "sharpe": 1.11, "max_dd": 12.1,
         "total_ret": 398.2, "win_rate": 43.0, "pct_ran_past_7": 23.4, "avg_runner_ret": 10.2, "current": False},
        {"name": "Scale 75% at +7%",    "partial_pct":  75, "sharpe": 1.56, "max_dd": 12.3,
         "total_ret": 545.8, "win_rate": 43.0, "pct_ran_past_7": 23.4, "avg_runner_ret":  9.8, "current": False},
        {"name": "Scale 90% at +7%",    "partial_pct":  90, "sharpe": 1.93, "max_dd": 12.4,
         "total_ret": 648.3, "win_rate": 43.0, "pct_ran_past_7": 23.4, "avg_runner_ret":  9.6, "current": True},
    ]

    return {
        "optimizer_results": optimizer_results,
        "spy_5yr_pct": spy_5yr_pct,
        "scaleout_results": scaleout_results,
        "current_config": {
            "score_threshold": 65,
            "atr_stop_mult":   1.5,
            "atr_target_mult": 3.75,
            "target_floor_pct": 7.0,
            "scale_out_pct":   90,
            "hold_days":       10,
            "risk_per_trade_pct": 2.0,
        },
    }


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
    threading.Thread(target=_background_daily_scan, daemon=True).start()
    threading.Thread(target=_sync_alpaca_plays, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
