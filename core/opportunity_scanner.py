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
    bsh = "BUY" if composite >= 60 else "SELL" if composite < 40 else "HOLD"

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "score": composite,
        "grade": grade,
        "bsh": bsh,
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


SIGNAL_META_SHORT: Dict[str, Dict] = {
    "momentum_breakdown": {
        "label": "Momentum Breakdown",
        "desc": "Price broke 20-day low on volume ≥1.5× average",
        "hex": "#f87171",
    },
    "macd_cross_bearish": {
        "label": "MACD Cross Bearish",
        "desc": "MACD crossed below signal line (last 3 bars)",
        "hex": "#fb7185",
    },
    "death_cross": {
        "label": "Death Cross",
        "desc": "50 SMA crossed below 200 SMA within last 10 bars",
        "hex": "#f43f5e",
    },
    "rsi_overbought_reset": {
        "label": "RSI Overbought Reset",
        "desc": "RSI was above 70, now falling below 65",
        "hex": "#e879f9",
    },
    "bb_breakdown": {
        "label": "BB Breakdown",
        "desc": "Price below lower Bollinger Band with expanding width",
        "hex": "#a78bfa",
    },
}


def score_symbol_short(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    Score bearish (short-side) setups. Returns a dict with short_score (0-100)
    and firing_signals. A high short_score means the stock looks like a strong
    short candidate — structurally weak, momentum rolling over.

    Avoids oversold stocks (RSI < 35) — they have limited downside from here.
    """
    if df is None or len(df) < 50:
        return None

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n      = len(close)
    price  = float(close.iloc[-1])

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi_s = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))
    rsi   = float(rsi_s.iloc[-1])

    # Bail early: oversold stocks have limited downside
    if rsi < 35:
        return None

    # MACD
    ema12    = close.ewm(span=12, adjust=False).mean()
    ema26    = close.ewm(span=26, adjust=False).mean()
    macd     = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    hist     = macd - macd_sig

    # MAs
    ema21  = close.ewm(span=21, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if n >= 200 else None

    # Volume / Bollinger
    vol20         = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio     = float(volume.iloc[-1]) / (vol20 + 1e-10)
    bb_std        = close.rolling(20).std()
    bb_lower      = close.rolling(20).mean() - 2.0 * bb_std
    bb_width_now  = float((4.0 * bb_std).iloc[-1])
    bb_width_avg  = float((4.0 * bb_std).iloc[-6:-1].mean()) if n >= 6 else bb_width_now
    low20         = float(low.shift(1).rolling(20).min().iloc[-1]) if n > 21 else float(low.min())
    roc10         = float((price / float(close.iloc[-11]) - 1) * 100) if n >= 11 else 0.0
    is_down_day   = float(close.iloc[-1]) < float(close.iloc[-2]) if n >= 2 else True

    # ── Signal detection ───────────────────────────────────────────────
    signals: List[str] = []

    if price < low20 and vol_ratio >= 1.5:
        signals.append("momentum_breakdown")

    for i in range(1, min(4, n)):
        if float(hist.iloc[-i]) < 0 and float(hist.iloc[-i - 1]) >= 0:
            signals.append("macd_cross_bearish")
            break

    if sma200 is not None and n >= 205:
        s50  = sma50.values
        s200 = sma200.values
        for i in range(1, min(11, n - 200)):
            if s50[-i] < s200[-i] and s50[-i - 1] >= s200[-i - 1]:
                signals.append("death_cross")
                break

    if n >= 10 and float(rsi_s.iloc[-10:-1].max()) > 70 and rsi < 65:
        signals.append("rsi_overbought_reset")

    if price < float(bb_lower.iloc[-1]) and bb_width_now > bb_width_avg * 1.15:
        signals.append("bb_breakdown")

    # Require at least one signal to be a viable short
    if not signals:
        return None

    # ── Composite short score (0-100) ─────────────────────────────────
    below_ema21  = price < float(ema21.iloc[-1])
    below_ema50  = price < float(ema50.iloc[-1])
    below_sma200 = sma200 is not None and price < float(sma200.iloc[-1])

    # Trend (25 pts) — being below all MAs is bearish structure
    trend = (8 if below_ema21 else 0) + (9 if below_ema50 else 0) + (8 if below_sma200 else 0)

    # Momentum (25 pts) — negative MACD + negative ROC + breakdown
    momentum = 0
    if float(hist.iloc[-1]) < 0:
        momentum += 10
    momentum += 8 if roc10 < -3 else (4 if roc10 < 0 else 0)
    if "momentum_breakdown" in signals:
        momentum += 7
    momentum = min(momentum, 25)

    # RSI (20 pts) — ideal short RSI: 40-65 (not yet oversold, room to fall)
    if 40 <= rsi <= 65:
        rsi_score = 20
    elif 65 < rsi <= 72:
        rsi_score = 12   # getting overbought — good reversal setup
    elif rsi > 72:
        rsi_score = 4    # very overbought — can keep running; risky short
    else:
        rsi_score = 8    # 35-40: falling into oversold range; caution

    # Volume (15 pts) — high volume on a down day adds conviction
    if is_down_day and vol_ratio >= 2.0:
        vol_score = 15
    elif is_down_day and vol_ratio >= 1.5:
        vol_score = 10
    elif vol_ratio >= 1.0:
        vol_score = 5
    else:
        vol_score = 0

    # Pattern (15 pts)
    if "death_cross" in signals:
        pat_score = 15
    elif "bb_breakdown" in signals:
        pat_score = 10
    elif "macd_cross_bearish" in signals:
        pat_score = 8
    else:
        pat_score = 0

    composite = trend + momentum + rsi_score + vol_score + pat_score

    return {
        "symbol":       symbol,
        "price":        round(price, 2),
        "short_score":  composite,
        "grade":        "A" if composite >= 80 else "B" if composite >= 65 else "C" if composite >= 50 else "D",
        "rsi":          round(rsi, 1),
        "roc10":        round(roc10, 1),
        "vol_ratio":    round(vol_ratio, 2),
        "macd_hist":    round(float(hist.iloc[-1]), 5),
        "below_ema21":  below_ema21,
        "below_ema50":  below_ema50,
        "below_sma200": below_sma200,
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


def backtest_symbol(
    load_bars_fn: Callable,
    symbol: str,
    days: int = 90,
) -> Optional[Dict]:
    """
    90-day walk-forward backtest for a single symbol.
    Any bar where at least one signal fires → enter at next open,
    exit at +5% target / -3% stop / 10-day hold.
    Returns detailed stats + qualified flag (used for real-money gate).
    """
    try:
        df = load_bars_fn(symbol, days=250)
    except Exception:
        return None
    if df is None or len(df) < 110:
        return None

    open_ = df["open"].astype(float)
    high_ = df["high"].astype(float)
    low_ = df["low"].astype(float)
    close_ = df["close"].astype(float)
    n = len(df)
    window_start = max(55, n - days)

    trades: List[float] = []
    signals_per_trade: List[List[str]] = []
    seen: set = set()

    for i in range(window_start, n - 11):
        scored = score_symbol(df.iloc[: i + 1], symbol)
        if not scored or not scored["firing_signals"] or i in seen:
            continue
        seen.add(i)
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
        trades.append((exit_px / entry - 1) * 100)
        signals_per_trade.append(scored["firing_signals"])

    if not trades:
        return {
            "symbol": symbol,
            "period_days": days,
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "qualified": False,
            "fail_reason": "No signals fired in 90-day window",
        }

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / len(trades) * 100
    avg_return = sum(trades) / len(trades)
    profit_factor = sum(wins) / abs(sum(losses)) if losses else 99.0

    equity, peak, max_dd = 100.0, 100.0, 0.0
    for t in trades:
        equity *= 1 + t / 100
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)

    # Qualification thresholds for real-money eligibility
    fail_reasons = []
    if win_rate < 50:
        fail_reasons.append(f"win rate {win_rate:.0f}% < 50%")
    if avg_return < 0.5:
        fail_reasons.append(f"avg return {avg_return:.2f}% < 0.5%")
    if profit_factor < 1.2:
        fail_reasons.append(f"profit factor {profit_factor:.2f} < 1.2")
    if len(trades) < 3:
        fail_reasons.append(f"only {len(trades)} trades (need ≥3)")
    qualified = len(fail_reasons) == 0

    return {
        "symbol": symbol,
        "period_days": days,
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 2),
        "best_trade": round(max(trades), 2),
        "worst_trade": round(min(trades), 2),
        "profit_factor": round(min(profit_factor, 99.0), 2),
        "max_drawdown": round(max_dd, 2),
        "qualified": qualified,
        "fail_reason": "; ".join(fail_reasons) if fail_reasons else None,
    }


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
