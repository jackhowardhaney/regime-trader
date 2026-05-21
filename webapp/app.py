"""Regime Trader Web App — FastAPI backend."""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
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
                notes TEXT
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
        # Seed watchlist
        count = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        if count == 0:
            for sym in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]:
                try:
                    conn.execute("INSERT INTO watchlist (symbol) VALUES (?)", (sym,))
                except Exception:
                    pass


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

        buy_pct = int(((score + 5) / 10) * 100)
        buy_pct = max(0, min(100, buy_pct))
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
    global _bt
    _bt = {"running": True, "progress": 5, "result": None, "error": None}
    try:
        cmd = [
            sys.executable, "-W", "ignore",
            str(ROOT / "main.py"), "backtest",
            "--symbols", *symbols,
            "--start", start, "--end", end,
            "--compare", "--output-dir", str(RESULTS_DIR),
        ]
        _bt["progress"] = 15
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=360)
        _bt["progress"] = 85
        if r.returncode != 0:
            _bt.update({"running": False, "error": (r.stderr or r.stdout)[-2000:]})
            return

        ec_f = RESULTS_DIR / "equity_curve.csv"
        if not ec_f.exists():
            _bt.update({"running": False, "error": "No equity curve generated"})
            return

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
                regime_breakdown = [
                    {"regime": r, "pct": round(c / len(rh) * 100, 1)}
                    for r, c in counts.items()
                ]

        summary = {
            "total_return": round(ret, 2),
            "cagr": round(cagr, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 2),
            "n_trades": n_trades,
            "equity_dates": ec.index.strftime("%Y-%m-%d").tolist(),
            "equity_values": ec["equity"].round(2).tolist(),
            "drawdown_values": dd_series,
            "regime_breakdown": regime_breakdown,
            "symbols": symbols,
            "start_date": start,
            "end_date": end,
        }

        with get_db() as conn:
            conn.execute(
                "INSERT INTO backtest_runs (symbols,start_date,end_date,total_return,cagr,sharpe,max_drawdown,n_trades) VALUES (?,?,?,?,?,?,?,?)",
                (json.dumps(symbols), start, end, ret, cagr, sharpe, max_dd, n_trades),
            )

        _bt.update({"running": False, "progress": 100, "result": summary})
    except subprocess.TimeoutExpired:
        _bt.update({"running": False, "error": "Backtest timed out (>6 min)"})
    except Exception as e:
        _bt.update({"running": False, "error": str(e)})


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
        symbols = [r["symbol"] for r in conn.execute("SELECT symbol FROM watchlist").fetchall()]
    results = []
    for sym in symbols:
        q = get_quote(sym)
        if not q:
            continue
        sig_key = f"sig:{sym}"
        sig = _cache.get(sig_key, ({"overall": "NEUTRAL", "rsi": 50, "buy_pct": 50}, 0))[0]
        results.append({**q,
                        "overall": sig.get("overall", "NEUTRAL"),
                        "bsh": sig.get("bsh", "HOLD"),
                        "rsi": sig.get("rsi", 50),
                        "buy_pct": sig.get("buy_pct", 50)})
    return results


@app.get("/api/signals/{symbol}")
def symbol_signals(symbol: str):
    return get_signals(symbol.upper())


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
        return {
            "connected": True,
            "equity": equity,
            "cash": float(acct.get("cash", 0)),
            "buying_power": float(acct.get("buying_power", 0)),
            "day_pl": round(equity - last_equity, 2),
            "day_pl_pct": round((equity - last_equity) / last_equity * 100, 2) if last_equity else 0,
            "paper": True,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
