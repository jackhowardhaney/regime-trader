"""
Adaptive blend scanner — runs once per trading day (Railway Cron or manual).

Two distinct entry logics:
  SCANNER  — 430-symbol universe, score ≥ 65, ATR×1.5 stop, ATR×3.75 target, 5% risk
  WATCHLIST — curated ~30 symbols, blend_score ≥ 70, ATR×1.2 stop (tighter),
               ATR×3.0 target, 5.5% risk

Slot allocation between the two is adaptive: the bot tracks closed-play P&L by
source and shifts slots toward whichever side is performing better. Adjustments
are gradual (±5% per day) with floors at 25% / ceilings at 75% for either side.

Set WEBAPP_URL env var to the Railway webapp URL.
"""
import os, sys, json, logging
from datetime import datetime, timedelta
from pathlib import Path
from urllib import request as _urllib_request

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor
from core.opportunity_scanner import score_symbol, score_symbol_short
from core.regime_strategies import Signal
from data.market_data import MarketDataFetcher

# ── Parameters ────────────────────────────────────────────────────────────────

# Scanner long logic
SCANNER_MIN_SCORE    = 65
SCANNER_ATR_STOP     = 1.5
SCANNER_ATR_TARGET   = 3.75
SCANNER_RISK_PCT     = 0.05

# Watchlist long logic — tighter stops (conviction = precision)
WATCHLIST_MIN_BLEND  = 70
WATCHLIST_ATR_STOP   = 1.2
WATCHLIST_ATR_TARGET = 3.0
WATCHLIST_RISK_PCT   = 0.055

# Short logic — inverted stops, margin-aware
# stop = entry + ATR×1.5 (above entry), target = entry - ATR×3.0 (below)
SHORT_MIN_SCORE        = 60
SHORT_ATR_STOP         = 1.5
SHORT_ATR_TARGET       = 3.0
SHORT_RISK_PCT         = 0.04    # 4% risk per short (slightly tighter than longs)
MAX_SHORT_POSITIONS    = 4       # 3-5 short positions
MAX_SHORT_EXPOSURE_PCT = 0.25    # total short notional ≤ 25% of equity

MAX_POSITION_PCT     = 0.15
ATR_PERIOD           = 14
MAX_POSITIONS        = 15
MIN_TRADES_TO_ADJUST = 4
RATIO_STEP           = 0.05
RATIO_MIN            = 0.25
RATIO_MAX            = 0.75

WEBAPP_URL = os.getenv("WEBAPP_URL", "").rstrip("/")

try:
    from data.sector_symbols import ALL_HANDPICK_SYMBOLS as SCAN_UNIVERSE
except Exception:
    SCAN_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
                     "JPM", "GS", "BAC", "MS", "V", "MA", "XOM", "CVX", "UNH", "LLY",
                     "SPY", "QQQ", "IWM"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - bars["close"].shift()).abs(),
        (bars["low"]  - bars["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


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


# ── Adaptive ratio ────────────────────────────────────────────────────────────

def _fetch_ratio() -> tuple[float, float]:
    """Fetch current scanner/watchlist slot ratio from the webapp DB."""
    try:
        d = _api_get("/api/blend-ratio")
        sp = float(d.get("scanner_pct", 0.60))
        wp = float(d.get("watchlist_pct", 0.40))
        return sp, wp
    except Exception:
        return 0.60, 0.40


def _compute_quality(perf: dict) -> float:
    """Quality metric: win_rate × avg_pnl. Rewards consistency + dollar return."""
    wr = perf.get("win_rate", 0) / 100.0
    avg = perf.get("avg_pnl", 0)
    return wr * avg if avg > 0 else 0.0


def _update_ratio(scanner_pct: float, watchlist_pct: float) -> None:
    """
    Pull closed-play performance from API, compute new ratio, push update.
    Logic:
      - Need MIN_TRADES_TO_ADJUST per source before adjusting anything
      - quality = win_rate × avg_pnl (both must be positive to count)
      - If scanner quality > watchlist quality by >20%: scanner gets +RATIO_STEP
      - If watchlist quality > scanner quality by >20%: watchlist gets +RATIO_STEP
      - Otherwise: no change
    """
    try:
        data = _api_get("/api/blend-analysis")
        perf = data.get("source_performance", {})

        scanner_p  = perf.get("scanner", {})
        watchlist_p = perf.get("watchlist", {})

        sn  = scanner_p.get("count", 0)
        wn  = watchlist_p.get("count", 0)

        if sn < MIN_TRADES_TO_ADJUST or wn < MIN_TRADES_TO_ADJUST:
            reason = (f"Not enough data: scanner={sn} watchlist={wn} closed plays "
                      f"(need {MIN_TRADES_TO_ADJUST} each). Keeping ratio "
                      f"{scanner_pct:.0%} / {watchlist_pct:.0%}.")
            print(f"  Ratio: {reason}")
            _api_put("/api/blend-ratio", {
                "scanner_pct": scanner_pct, "reason": reason
            })
            return

        sq = _compute_quality(scanner_p)
        wq = _compute_quality(watchlist_p)

        new_scanner_pct = scanner_pct
        reason = ""

        if sq <= 0 and wq <= 0:
            # Both losing — split evenly, let data accumulate
            new_scanner_pct = 0.50
            reason = f"Both sources losing (scanner q={sq:.1f}, watchlist q={wq:.1f}). Reset to 50/50."
        elif sq > 0 and wq <= 0:
            # Scanner winning, watchlist losing → push toward scanner
            new_scanner_pct = min(RATIO_MAX, scanner_pct + RATIO_STEP)
            reason = (f"Scanner winning (q={sq:.2f}) / watchlist losing (q={wq:.2f}). "
                      f"Shifting +{RATIO_STEP:.0%} to scanner.")
        elif wq > 0 and sq <= 0:
            # Watchlist winning, scanner losing → push toward watchlist
            new_scanner_pct = max(RATIO_MIN, scanner_pct - RATIO_STEP)
            reason = (f"Watchlist winning (q={wq:.2f}) / scanner losing (q={sq:.2f}). "
                      f"Shifting +{RATIO_STEP:.0%} to watchlist.")
        else:
            # Both positive — shift toward the better-performing one if gap > 20%
            total_q = sq + wq
            scanner_ideal = sq / total_q if total_q > 0 else 0.5
            gap = abs(sq - wq) / max(sq, wq)
            if gap > 0.20:
                new_scanner_pct = max(RATIO_MIN, min(RATIO_MAX,
                    scanner_pct + RATIO_STEP * (1 if scanner_ideal > scanner_pct else -1)
                ))
                winner = "scanner" if sq > wq else "watchlist"
                reason = (f"{winner.title()} outperforming by {gap:.0%} "
                          f"(scanner q={sq:.2f}, watchlist q={wq:.2f}). "
                          f"Adjusting ratio → scanner {new_scanner_pct:.0%}.")
            else:
                reason = (f"Both sources positive, gap {gap:.0%} <20% — holding ratio "
                          f"{scanner_pct:.0%} / {watchlist_pct:.0%}.")

        _api_put("/api/blend-ratio", {"scanner_pct": new_scanner_pct, "reason": reason})
        print(f"  Ratio updated: scanner={new_scanner_pct:.0%} watchlist={1-new_scanner_pct:.0%}")
        print(f"  Reason: {reason}")
    except Exception as e:
        print(f"  Warning: ratio update failed: {e}")


# ── Play recording ────────────────────────────────────────────────────────────

def _record_play(symbol, direction, entry, stop, target, shares, signal, notes, source):
    if not WEBAPP_URL:
        return
    try:
        _api_post("/api/plays", {
            "symbol": symbol, "direction": direction,
            "entry_price": entry, "stop_loss": stop, "take_profit": target,
            "shares": shares, "signal": signal, "notes": notes,
            "submit_order": False, "source": source,
        })
    except Exception as e:
        print(f"    Warning: dashboard record failed: {e}")


# ── Entry functions ───────────────────────────────────────────────────────────

def _enter_scanner_play(candidate, executor, client, equity, buying_power_ref, excluded):
    """Scanner logic: broad universe, looser stops, standard risk."""
    sym   = candidate["symbol"]
    bars  = candidate["_bars"]
    price = float(bars["close"].iloc[-1])
    a     = _atr(bars)
    if a <= 0:
        return False

    stop   = round(price - a * SCANNER_ATR_STOP, 2)
    target = max(round(price + a * SCANNER_ATR_TARGET, 2), round(price * 1.07, 2))
    rps    = price - stop
    if rps <= 0:
        return False

    shares = int((equity * SCANNER_RISK_PCT) / rps)
    cost   = shares * price
    if cost > equity * MAX_POSITION_PCT:
        shares = max(1, int((equity * MAX_POSITION_PCT) / price))
        cost = shares * price

    try:
        bp = float(client.get_account()["buying_power"])
        buying_power_ref[0] = bp
    except Exception:
        bp = buying_power_ref[0]

    if shares < 1 or cost > bp * 0.95:
        print(f"  {sym} [SCANNER]: insufficient buying power — skip")
        return False

    signals = candidate.get("firing_signals", [])
    score   = candidate.get("score", 0)

    try:
        sig = Signal(
            symbol=sym, direction="LONG", confidence=score / 100.0,
            entry_price=price, stop_loss=stop, take_profit=target,
            position_size_pct=shares * price / equity, leverage=1.0,
            regime_id=0, regime_name="SWING", regime_probability=1.0,
            timestamp=datetime.utcnow(),
            reasoning=f"scanner score={score}",
            strategy_name=signals[0] if signals else "composite",
        )
        executor.submit_bracket_order(sig)
        _record_play(sym, "LONG", price, stop, target, shares,
                     signals[0] if signals else "composite",
                     f"scanner score={score} signals={','.join(signals)}", "scanner")
        print(f"  ✓ SCANNER  LONG {sym:<8} score={score}  {shares}sh @ ${price:.2f}"
              f"  stop=${stop:.2f}  target=${target:.2f}  R:R={round((target-price)/(price-stop),2)}:1")
        excluded.add(sym)
        return True
    except Exception as e:
        print(f"  ✗ SCANNER  {sym}: {e}")
        return False


def _enter_watchlist_play(wl_candidate, executor, client, equity, buying_power_ref, excluded, fetcher, start, end):
    """
    Watchlist logic: curated symbols, tighter stops (ATR×1.2), slightly higher
    risk (5.5%) — conviction earns precision over breathing room.
    """
    sym          = wl_candidate["symbol"]
    blend_score  = wl_candidate["blend_score"]
    scanner_score = wl_candidate["scanner_score"]

    if sym in excluded:
        return False

    # Fetch fresh bars
    try:
        bars = fetcher.get_historical_bars(sym, "1Day", start, end)
        if bars is None or len(bars) < 50:
            return False
    except Exception:
        return False

    price = float(bars["close"].iloc[-1])
    a     = _atr(bars)
    if a <= 0:
        return False

    stop   = round(price - a * WATCHLIST_ATR_STOP, 2)           # tighter
    target = max(round(price + a * WATCHLIST_ATR_TARGET, 2), round(price * 1.07, 2))
    rps    = price - stop
    if rps <= 0:
        return False

    shares = int((equity * WATCHLIST_RISK_PCT) / rps)
    cost   = shares * price
    if cost > equity * MAX_POSITION_PCT:
        shares = max(1, int((equity * MAX_POSITION_PCT) / price))
        cost = shares * price

    try:
        bp = float(client.get_account()["buying_power"])
        buying_power_ref[0] = bp
    except Exception:
        bp = buying_power_ref[0]

    if shares < 1 or cost > bp * 0.95:
        print(f"  {sym} [WATCHLIST]: insufficient buying power — skip")
        return False

    signal_name = wl_candidate.get("bsh", "BUY").lower() + "_watchlist"

    try:
        sig = Signal(
            symbol=sym, direction="LONG", confidence=min(1.0, blend_score / 100.0),
            entry_price=price, stop_loss=stop, take_profit=target,
            position_size_pct=shares * price / equity, leverage=1.0,
            regime_id=0, regime_name="SWING", regime_probability=1.0,
            timestamp=datetime.utcnow(),
            reasoning=f"watchlist blend={blend_score:.1f} scanner={scanner_score}",
            strategy_name="watchlist_blend",
        )
        executor.submit_bracket_order(sig)
        _record_play(sym, "LONG", price, stop, target, shares,
                     "watchlist_blend",
                     f"watchlist blend={blend_score:.1f} scanner={scanner_score} tight_stop=ATR×{WATCHLIST_ATR_STOP}",
                     "watchlist")
        rr = round((target - price) / (price - stop), 2)
        print(f"  ✓ WATCHLIST LONG {sym:<8} blend={blend_score:.0f} scan={scanner_score}"
              f"  {shares}sh @ ${price:.2f}  stop=${stop:.2f}  target=${target:.2f}  R:R={rr}:1")
        excluded.add(sym)
        return True
    except Exception as e:
        print(f"  ✗ WATCHLIST {sym}: {e}")
        return False


def _enter_short_play(candidate, executor, client, equity, buying_power_ref, excluded,
                       short_exposure_ref, max_short_exposure):
    """
    Short entry: stop ABOVE entry, target BELOW entry.
    Total short notional capped at MAX_SHORT_EXPOSURE_PCT of equity.
    Margin check: shorting requires ~50% collateral (Reg T).
    """
    sym   = candidate["symbol"]
    bars  = candidate["_bars"]
    price = float(bars["close"].iloc[-1])
    a     = _atr(bars)
    if a <= 0:
        return False

    stop   = round(price + a * SHORT_ATR_STOP, 2)    # ABOVE entry (buy-to-cover at loss)
    target = round(price - a * SHORT_ATR_TARGET, 2)  # BELOW entry (buy-to-cover at profit)
    if target <= 0:
        return False

    rps = stop - price    # risk per share (always positive for shorts)
    if rps <= 0:
        return False

    shares = int((equity * SHORT_RISK_PCT) / rps)
    cost   = shares * price   # notional short exposure

    # Cap to stay within 25% of equity total short book
    remaining = max_short_exposure - short_exposure_ref[0]
    if cost > remaining:
        shares = max(0, int(remaining / price))
        cost   = shares * price
    if shares < 1:
        print(f"  {sym} [SHORT]: exposure cap reached — skip")
        return False

    # Margin check (Reg T: need ~50% of notional as collateral)
    try:
        bp = float(client.get_account()["buying_power"])
        buying_power_ref[0] = bp
    except Exception:
        bp = buying_power_ref[0]
    if cost * 0.50 > bp * 0.90:
        print(f"  {sym} [SHORT]: insufficient margin — skip")
        return False

    score   = candidate.get("short_score", 0)
    signals = candidate.get("firing_signals", [])
    source  = candidate.get("source", "scanner")

    try:
        sig = Signal(
            symbol=sym, direction="SHORT", confidence=min(1.0, score / 100.0),
            entry_price=price, stop_loss=stop, take_profit=target,
            position_size_pct=cost / equity, leverage=1.0,
            regime_id=0, regime_name="SWING", regime_probability=1.0,
            timestamp=datetime.utcnow(),
            reasoning=f"{source} short_score={score:.0f} signals={','.join(signals)}",
            strategy_name=signals[0] if signals else "short_composite",
        )
        executor.submit_bracket_order(sig)
        _record_play(sym, "SHORT", price, stop, target, shares,
                     signals[0] if signals else "short_composite",
                     f"{source} short_score={score:.0f} stop=ATR×{SHORT_ATR_STOP}",
                     source)
        short_exposure_ref[0] += cost
        rr = round((price - target) / (stop - price), 2)
        print(f"  ✓ SHORT  [{source}] {sym:<8} score={score:.0f}  {shares}sh @ ${price:.2f}"
              f"  stop=${stop:.2f}  target=${target:.2f}  R:R={rr}:1")
        excluded.add(sym)
        return True
    except Exception as e:
        print(f"  ✗ SHORT  {sym}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = AlpacaClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )
    client.connect()

    # PDT check to EXIT-only so entry orders are never blocked
    try:
        from alpaca.trading.models import AccountConfiguration
        from alpaca.trading.enums import PDTCheck
        current = client._trading.get_account_configurations()
        client._trading.set_account_configurations(AccountConfiguration(
            dtbp_check=current.dtbp_check,
            fractional_trading=current.fractional_trading,
            max_margin_multiplier=current.max_margin_multiplier,
            no_shorting=current.no_shorting,
            pdt_check=PDTCheck.EXIT,
            suspend_trade=current.suspend_trade,
            trade_confirm_email=current.trade_confirm_email,
            ptp_no_exception_entry=current.ptp_no_exception_entry,
        ))
    except Exception:
        pass

    clock = client.get_clock()
    if not clock.get("is_open", False):
        print(f"Market closed. Next open: {clock.get('next_open')}. Exiting.")
        return

    acct         = client.get_account()
    equity       = float(acct["equity"])
    buying_power = float(acct["buying_power"])
    bp_ref       = [buying_power]   # mutable ref passed into entry functions

    all_positions = client.get_positions()
    held    = {p["symbol"] for p in all_positions}
    pending = client.get_open_order_symbols()
    excluded = held | pending
    slots   = MAX_POSITIONS - len(held)

    # ── Short book state ─────────────────────────────────────────────
    short_positions = [p for p in all_positions if p.get("side") == "short"]
    n_existing_shorts   = len(short_positions)
    current_short_exp   = sum(abs(float(p.get("market_value", 0))) for p in short_positions)
    max_short_exposure  = equity * MAX_SHORT_EXPOSURE_PCT
    short_exposure_ref  = [current_short_exp]
    short_slots         = max(0, MAX_SHORT_POSITIONS - n_existing_shorts)

    # ── Fetch adaptive ratio for long slots ──────────────────────────
    scanner_pct, watchlist_pct = _fetch_ratio()
    long_slots      = max(0, slots - short_slots)
    scanner_slots   = max(1, round(long_slots * scanner_pct)) if long_slots > 0 else 0
    watchlist_slots = max(0, long_slots - scanner_slots)
    if long_slots == 1:
        scanner_slots   = 1 if scanner_pct >= watchlist_pct else 0
        watchlist_slots = 1 - scanner_slots

    print(f"\n{'='*62}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M ET')}  |  Adaptive Blend Scanner")
    print(f"  Equity: ${equity:,.2f}  |  Buying power: ${buying_power:,.2f}")
    print(f"  Slots: {slots} total  →  long={long_slots} (scanner={scanner_slots}, watchlist={watchlist_slots})  short={short_slots}")
    print(f"  Long ratio:  scanner {scanner_pct:.0%}  /  watchlist {watchlist_pct:.0%}")
    print(f"  Short book:  {n_existing_shorts}/{MAX_SHORT_POSITIONS} positions  "
          f"${current_short_exp:,.0f}/${max_short_exposure:,.0f} ({current_short_exp/equity*100:.1f}%/25%)")
    print(f"{'='*62}\n")

    if slots <= 0 and short_slots <= 0:
        print("Max positions reached — running ratio update and exiting.")
        _update_ratio(scanner_pct, watchlist_pct)
        return

    fetcher  = MarketDataFetcher(client)
    end      = datetime.now().strftime("%Y-%m-%d")
    start    = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    executor = OrderExecutor(client)

    # ══════════════════════════════════════════════════════════════════
    # SCANNER — one pass scores longs AND shorts (no double-fetch)
    # ══════════════════════════════════════════════════════════════════
    scanner_entered     = 0
    scored_long         = []
    scored_short_scan   = []   # short candidates found in scanner pass

    need_scanner = scanner_slots > 0 or (short_slots > 0 and current_short_exp < max_short_exposure)
    if need_scanner:
        print(f"── SCANNER PASS ({len(SCAN_UNIVERSE)} symbols — scoring long + short) ───────")
        for sym in SCAN_UNIVERSE:
            if sym in excluded:
                continue
            try:
                df = fetcher.get_historical_bars(sym, "1Day", start, end)
                if df is None or len(df) < 50:
                    continue
                # Long scoring
                r = score_symbol(df, sym)
                if r:
                    r["_bars"] = df
                    scored_long.append(r)
                # Short scoring (same bars — no extra API call)
                rs = score_symbol_short(df, sym)
                if rs:
                    rs["_bars"]  = df
                    rs["source"] = "scanner"
                    scored_short_scan.append(rs)
            except Exception:
                pass

        # ── Long entries from scanner ─────────────────────────────────
        if scanner_slots > 0:
            scored_long.sort(key=lambda x: x.get("score", 0), reverse=True)
            long_cands = [
                r for r in scored_long
                if r.get("score", 0) >= SCANNER_MIN_SCORE and r.get("firing_signals")
            ]
            print(f"   {len(long_cands)} long candidates (score ≥ {SCANNER_MIN_SCORE}). Entering up to {scanner_slots}:\n")
            for cand in long_cands:
                if scanner_entered >= scanner_slots:
                    break
                if _enter_scanner_play(cand, executor, client, equity, bp_ref, excluded):
                    scanner_entered += 1

    # ══════════════════════════════════════════════════════════════════
    # WATCHLIST — curated long picks (blend ≥ 70)
    # ══════════════════════════════════════════════════════════════════
    watchlist_entered = 0
    if watchlist_slots > 0:
        print(f"\n── WATCHLIST LONG ({watchlist_slots} slots, blend ≥ {WATCHLIST_MIN_BLEND}) ───────────────")
        try:
            blend_data = _api_get("/api/blend-analysis")
            wl_candidates = [
                c for c in blend_data.get("candidates", [])
                if c.get("blend_score", 0) >= WATCHLIST_MIN_BLEND
                and c.get("bsh") == "BUY"
                and c["symbol"] not in excluded
            ]
            print(f"   {len(wl_candidates)} watchlist candidates:\n")
            for c in wl_candidates:
                print(f"     {c['symbol']:<8} blend={c['blend_score']:.0f}  scanner={c['scanner_score']}  bsh={c['bsh']}")
            print()
            for cand in wl_candidates:
                if watchlist_entered >= watchlist_slots:
                    break
                if _enter_watchlist_play(cand, executor, client, equity, bp_ref, excluded, fetcher, start, end):
                    watchlist_entered += 1
        except Exception as e:
            print(f"   Watchlist scan failed: {e}")

    # ══════════════════════════════════════════════════════════════════
    # SHORTS — scanner low-scorers + watchlist SELL picks
    # ══════════════════════════════════════════════════════════════════
    shorts_entered = 0
    if short_slots > 0 and current_short_exp < max_short_exposure:
        print(f"\n── SHORTS ({short_slots} slots, score ≥ {SHORT_MIN_SCORE}, "
              f"25% cap=${max_short_exposure:,.0f}) ────────────────────")

        # Merge scanner short candidates with watchlist SELL picks
        short_candidates = [r for r in scored_short_scan
                            if r.get("short_score", 0) >= SHORT_MIN_SCORE and r["symbol"] not in excluded]

        try:
            blend_data = _api_get("/api/blend-analysis") if not watchlist_slots else blend_data
            for c in blend_data.get("candidates", []):
                if c.get("bsh") != "SELL" or c["symbol"] in excluded:
                    continue
                sym = c["symbol"]
                # Fetch bars for watchlist SELL picks (not in scanner universe)
                try:
                    df = fetcher.get_historical_bars(sym, "1Day", start, end)
                    if df is None or len(df) < 50:
                        continue
                    rs = score_symbol_short(df, sym)
                    if rs and rs["short_score"] >= SHORT_MIN_SCORE:
                        rs["_bars"]        = df
                        rs["source"]       = "watchlist"
                        rs["short_score"] *= 1.10    # +10% watchlist conviction bonus
                        short_candidates.append(rs)
                except Exception:
                    pass
        except Exception:
            pass

        # Deduplicate + rank by short_score
        seen: set = set()
        ranked: list = []
        for r in sorted(short_candidates, key=lambda x: x["short_score"], reverse=True):
            if r["symbol"] not in seen:
                seen.add(r["symbol"])
                ranked.append(r)

        print(f"   {len(ranked)} short candidates. Entering up to {short_slots}:\n")
        for cand in ranked:
            if shorts_entered >= short_slots:
                break
            if short_exposure_ref[0] >= max_short_exposure:
                break
            if _enter_short_play(cand, executor, client, equity, bp_ref, excluded,
                                  short_exposure_ref, max_short_exposure):
                shorts_entered += 1

    # ══════════════════════════════════════════════════════════════════
    # Update adaptive ratio based on live performance
    # ══════════════════════════════════════════════════════════════════
    print(f"\n── RATIO UPDATE ──────────────────────────────────────────────")
    _update_ratio(scanner_pct, watchlist_pct)

    total = scanner_entered + watchlist_entered + shorts_entered
    print(f"\n{'='*62}")
    print(f"  Done. Entered {total} plays  "
          f"(scanner long={scanner_entered}, watchlist long={watchlist_entered}, short={shorts_entered})")
    print(f"  Short book: ${short_exposure_ref[0]:,.0f} / ${max_short_exposure:,.0f} ({short_exposure_ref[0]/equity*100:.1f}%/25%)")
    print(f"  Remaining buying power: ${bp_ref[0]:,.2f}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
