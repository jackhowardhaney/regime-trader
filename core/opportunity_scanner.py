"""
Opportunity scanner: composite score, signal detection, 90-day walk-forward simulation.

score_symbol()  — pure function, takes OHLCV DataFrame → scored dict
run_scan()      — parallel fetch + score across a symbol list
run_90day_sim() — walk-forward backtest of signal quality
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import pandas as pd


SIGNAL_META: Dict[str, Dict] = {
    "momentum_breakout": {
        "label": "Momentum Breakout",
        "desc": "Price broke 20-day high on volume ≥1.5× average",
        "hex": "#4ade80",
    },
    "macd_cross": {
        "label": "MACD Cross",
        "desc": "MACD crossed above signal line (bullish, last 3 bars)",
        "hex": "#60a5fa",
    },
    "ema_bounce": {
        "label": "EMA Bounce",
        "desc": "Price recovered above 21 EMA after recent dip below",
        "hex": "#fbbf24",
    },
    "golden_cross": {
        "label": "Golden Cross",
        "desc": "50 SMA crossed above 200 SMA within last 10 bars",
        "hex": "#f59e0b",
    },
    "rsi_reset": {
        "label": "RSI Reset",
        "desc": "RSI was below 40 (oversold), now recovering above 45",
        "hex": "#a78bfa",
    },
    "volume_surge": {
        "label": "Volume Surge",
        "desc": "Today's volume exceeds 2× the 20-day average",
        "hex": "#fb923c",
    },
    "bb_breakout": {
        "label": "BB Breakout",
        "desc": "Price above upper Bollinger Band with expanding width",
        "hex": "#22d3ee",
    },
}

SCORE_COMPONENTS: Dict[str, Dict] = {
    "trend":    {"weight": 25, "desc": "EMA21 (+8), EMA50 (+9), SMA200 (+8)"},
    "momentum": {"weight": 25, "desc": "MACD positive (+10), ROC10 >3% (+8) or >0% (+4), breakout (+7)"},
    "rsi":      {"weight": 20, "desc": "Optimal 40–65 (+20). Partial 35–70. Penalty at extremes."},
    "volume":   {"weight": 15, "desc": "2× avg (+15), 1.5× avg (+10), above avg (+5)"},
    "pattern":  {"weight": 15, "desc": "Golden Cross (+15), EMA Bounce (+10), BB Breakout (+8)"},
}


def score_symbol(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    Compute composite score and detect firing signals from OHLCV data.
    Requires at minimum 50 bars. Returns None on insufficient data.
    """
    if df is None or len(df) < 50:
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)
    price = float(close.iloc[-1])

    # ── RSI ──────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi_s = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))
    rsi = float(rsi_s.iloc[-1])

    # ── MACD ─────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - macd_sig

    # ── EMAs / SMA200 ─────────────────────────────────────────────────
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    sma200 = close.rolling(200).mean() if n >= 200 else None

    # ── Volume ────────────────────────────────────────────────────────
    vol20 = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / (vol20 + 1e-10)

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_std = close.rolling(20).std()
    bb_upper = close.rolling(20).mean() + 2.0 * bb_std
    bb_width_now = float((4.0 * bb_std).iloc[-1])
    bb_width_avg = float((4.0 * bb_std).iloc[-6:-1].mean()) if n >= 6 else bb_width_now

    # ── 20-day high (prior bars only, no look-ahead) ──────────────────
    high20 = float(high.shift(1).rolling(20).max().iloc[-1]) if n > 21 else float(high.max())

    # ── Rate of change ────────────────────────────────────────────────
    roc10 = float((price / float(close.iloc[-11]) - 1) * 100) if n >= 11 else 0.0
    roc20 = float((price / float(close.iloc[-21]) - 1) * 100) if n >= 21 else 0.0

    # ── Signal detection ──────────────────────────────────────────────
    signals: List[str] = []

    if price > high20 and vol_ratio >= 1.5:
        signals.append("momentum_breakout")

    for i in range(1, min(4, n)):
        if float(hist.iloc[-i]) > 0 and float(hist.iloc[-i - 1]) <= 0:
            signals.append("macd_cross")
            break

    if n >= 5:
        below_recently = any(
            float(close.iloc[-j]) < float(ema21.iloc[-j]) for j in range(2, 5)
        )
        if below_recently and price > float(ema21.iloc[-1]) and roc10 > 0:
            signals.append("ema_bounce")

    if sma200 is not None and n >= 205:
        e50 = ema50.values
        s200 = sma200.values
        for i in range(1, min(11, n - 200)):
            if e50[-i] > s200[-i] and e50[-i - 1] <= s200[-i - 1]:
                signals.append("golden_cross")
                break

    if n >= 10 and float(rsi_s.iloc[-10:-1].min()) < 40 and rsi > 45:
        signals.append("rsi_reset")

    if vol_ratio >= 2.0:
        signals.append("volume_surge")

    if price > float(bb_upper.iloc[-1]) and bb_width_now > bb_width_avg * 1.15:
        signals.append("bb_breakout")

    # ── Composite score ───────────────────────────────────────────────
    above_ema21 = price > float(ema21.iloc[-1])
    above_ema50 = price > float(ema50.iloc[-1])
    above_sma200 = sma200 is not None and price > float(sma200.iloc[-1])

    trend = (8 if above_ema21 else 0) + (9 if above_ema50 else 0) + (8 if above_sma200 else 0)

    momentum = 0
    if float(hist.iloc[-1]) > 0:
        momentum += 10
    momentum += 8 if roc10 > 3 else (4 if roc10 > 0 else 0)
    if "momentum_breakout" in signals:
        momentum += 7
    momentum = min(momentum, 25)

    if 40 <= rsi <= 65:
        rsi_score = 20
    elif (35 <= rsi < 40) or (65 < rsi <= 70):
        rsi_score = 12
    elif (30 <= rsi < 35) or (70 < rsi <= 75):
        rsi_score = 6
    else:
        rsi_score = 0

    vol_score = 15 if vol_ratio >= 2.0 else (10 if vol_ratio >= 1.5 else (5 if vol_ratio >= 1.0 else 0))

    if "golden_cross" in signals:
        pat_score = 15
    elif "ema_bounce" in signals:
        pat_score = 10
    elif "bb_breakout" in signals:
        pat_score = 8
    else:
        pat_score = 0

    composite = trend + momentum + rsi_score + vol_score + pat_score

    grade = "A" if composite >= 80 else "B" if composite >= 65 else "C" if composite >= 50 else "D" if composite >= 35 else "F"

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "score": composite,
        "grade": grade,
        "breakdown": {
            "trend": trend,
            "momentum": momentum,
            "rsi": rsi_score,
            "volume": vol_score,
            "pattern": pat_score,
        },
        "rsi": round(rsi, 1),
        "roc10": round(roc10, 1),
        "roc20": round(roc20, 1),
        "vol_ratio": round(vol_ratio, 2),
        "macd_hist": round(float(hist.iloc[-1]), 5),
        "above_ema21": above_ema21,
        "above_ema50": above_ema50,
        "above_sma200": above_sma200,
        "firing_signals": signals,
    }


def run_scan(
    load_bars_fn: Callable,
    symbols: List[str],
    max_workers: int = 8,
) -> List[Dict]:
    """Fetch data and score all symbols in parallel. Returns list sorted by score desc."""
    from data.sector_symbols import SYMBOL_SECTORS

    results: List[Dict] = []
    lock = threading.Lock()

    def _worker(sym: str) -> None:
        try:
            df = load_bars_fn(sym, days=250)
            r = score_symbol(df, sym)
            if r:
                r["sectors"] = SYMBOL_SECTORS.get(sym, [])
                with lock:
                    results.append(r)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        list(exe.map(_worker, symbols))

    return sorted(results, key=lambda x: -x["score"])


def run_90day_sim(
    load_bars_fn: Callable,
    symbols: List[str],
    max_workers: int = 6,
) -> List[Dict]:
    """
    Walk-forward simulation over the last 90 trading days.
    Entry at next-bar open, exit at +5% target / -3% stop / 10 days — first hit wins.
    Returns per-signal aggregated stats across all symbols (capped at first 30).
    """
    agg: Dict[str, Dict] = {s: {"trades": [], "wins": 0} for s in SIGNAL_META}
    lock = threading.Lock()

    def _sim_sym(sym: str) -> None:
        try:
            df = load_bars_fn(sym, days=250)
        except Exception:
            return
        if df is None or len(df) < 110:
            return

        open_ = df["open"].astype(float)
        high_ = df["high"].astype(float)
        low_ = df["low"].astype(float)
        close_ = df["close"].astype(float)
        n = len(df)
        window_start = max(55, n - 90)

        local: Dict[str, Dict] = {s: {"trades": [], "wins": 0} for s in SIGNAL_META}

        for i in range(window_start, n - 11):
            scored = score_symbol(df.iloc[: i + 1], sym)
            if not scored:
                continue
            for sig in scored["firing_signals"]:
                if sig not in local:
                    continue
                entry = float(open_.iloc[i + 1])
                if entry <= 0:
                    continue
                target = entry * 1.05
                stop = entry * 0.97
                exit_px = float(close_.iloc[min(i + 10, n - 1)])
                for j in range(i + 1, min(i + 11, n)):
                    if float(high_.iloc[j]) >= target:
                        exit_px = target
                        break
                    if float(low_.iloc[j]) <= stop:
                        exit_px = stop
                        break
                ret = (exit_px / entry - 1) * 100
                local[sig]["trades"].append(ret)
                if ret > 0:
                    local[sig]["wins"] += 1

        with lock:
            for sig, data in local.items():
                agg[sig]["trades"].extend(data["trades"])
                agg[sig]["wins"] += data["wins"]

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        list(exe.map(_sim_sym, symbols[:30]))

    output: List[Dict] = []
    for sig, data in agg.items():
        trades = data["trades"]
        if not trades:
            continue
        output.append({
            "signal": sig,
            "label": SIGNAL_META[sig]["label"],
            "count": len(trades),
            "win_rate": round(data["wins"] / len(trades) * 100, 1),
            "avg_return": round(sum(trades) / len(trades), 2),
            "best": round(max(trades), 2),
            "worst": round(min(trades), 2),
            "hex": SIGNAL_META[sig]["hex"],
        })

    return sorted(output, key=lambda x: -x["win_rate"])
