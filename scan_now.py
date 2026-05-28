"""
One-shot: scan universe, score every symbol not already held, submit bracket
orders for top picks using remaining buying power.

Designed to run as a Railway Cron job at 9:35am ET Mon-Fri.
Set WEBAPP_URL env var to your Railway webapp URL so plays appear on the dashboard.
"""
import os, sys, logging, json
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
from core.opportunity_scanner import score_symbol
from core.regime_strategies import Signal
from data.market_data import MarketDataFetcher

# ── Parameters ────────────────────────────────────────────────────────
MIN_SCORE             = 65    # long entry threshold
WATCHLIST_MIN_SCORE   = 60    # long entry threshold for watchlist BUY symbols
MAX_SHORT_SCORE       = 35    # short entry threshold (bearish)
WATCHLIST_MAX_SHORT   = 40    # short entry threshold for watchlist SELL symbols
RISK_PCT         = 0.05    # 5% of equity risked per trade (max loss per play)
MAX_POSITION_PCT = 0.15    # never spend more than 15% of equity on one position
ATR_STOP_MULT    = 1.5
ATR_TARGET_MULT  = 3.75
ATR_PERIOD       = 14
MAX_POSITIONS    = 15
WEBAPP_URL       = os.getenv("WEBAPP_URL", "").rstrip("/")

try:
    from data.sector_symbols import ALL_HANDPICK_SYMBOLS as SCAN_UNIVERSE
except Exception:
    SCAN_UNIVERSE = [
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD",
        "JPM","GS","BAC","MS","V","MA",
        "CVX","COP","AMT","PLD",
        "AVGO","QCOM",
        "UNH","LLY","JNJ","ABBV",
        "PLTR","MSTR","RBLX","SNAP","UBER",
        "F","GM","RIVN",
        "DIS","NFLX","SPOT",
        "SPY","QQQ","IWM",
    ]

def atr(bars, period=ATR_PERIOD):
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - bars["close"].shift()).abs(),
        (bars["low"]  - bars["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _record_play(symbol, direction, entry_price, stop_loss, take_profit, shares, signal, notes):
    """POST new play to the dashboard webapp API."""
    if not WEBAPP_URL:
        return
    payload = json.dumps({
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "shares": shares,
        "signal": signal,
        "notes": notes,
        "submit_order": False,
        "source": "scanner",
    }).encode()
    try:
        req = _urllib_request.Request(
            f"{WEBAPP_URL}/api/plays",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urllib_request.urlopen(req, timeout=10) as resp:
            pass
    except Exception as e:
        print(f"    Warning: dashboard record failed: {e}")


def main():
    client = AlpacaClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )
    client.connect()

    # ── Set PDT check to EXIT-only so entry orders are never blocked ───
    # Alpaca rejects bracket orders preemptively if the account *could*
    # become PDT-flagged. EXIT mode only checks on sell orders, leaving
    # swing-trade entries unblocked. Safe for paper; swing trades aren't
    # same-day round trips so exits won't trigger it either.
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

    # ── Market hours check — exit cleanly if market is closed ──
    clock = client.get_clock()
    if not clock.get("is_open", False):
        next_open = clock.get("next_open", "unknown")
        print(f"Market is closed. Next open: {next_open}")
        print("Nothing to do — exiting.")
        return

    acct = client.get_account()
    equity        = float(acct["equity"])
    buying_power  = float(acct["buying_power"])
    print(f"\n{'='*55}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"  Account equity:    ${equity:>12,.2f}")
    print(f"  Buying power:      ${buying_power:>12,.2f}")
    print(f"  Dashboard:         {WEBAPP_URL or '(not set)'}")

    held      = {p["symbol"] for p in client.get_positions()}
    pending   = client.get_open_order_symbols()
    excluded  = held | pending
    slots     = MAX_POSITIONS - len(held)
    print(f"  Open positions:    {len(held)}  ({', '.join(sorted(held))})")
    if pending - held:
        print(f"  Pending orders:    {len(pending - held)}  ({', '.join(sorted(pending - held))})")
    print(f"  Slots available:   {slots}")
    print(f"{'='*55}\n")

    if slots <= 0:
        print("Max positions reached — nothing to do.")
        return

    fetcher = MarketDataFetcher(client)
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")

    print(f"Scanning {len(SCAN_UNIVERSE)} symbols…")
    scored = []
    for sym in SCAN_UNIVERSE:
        if sym in excluded:
            continue
        try:
            df = fetcher.get_historical_bars(sym, "1Day", start, end)
            if df is None or len(df) < 50:
                continue
            result = score_symbol(df, sym)
            if result:
                result["_bars"] = df
                scored.append(result)
        except Exception:
            pass

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Fetch BUY + SELL watchlist from dashboard
    watchlist_buys  = set()
    watchlist_sells = set()
    if WEBAPP_URL:
        try:
            req = _urllib_request.Request(f"{WEBAPP_URL}/api/watchlist", method="GET")
            with _urllib_request.urlopen(req, timeout=10) as resp:
                wl_data = json.loads(resp.read())
                watchlist_buys  = {e["symbol"] for e in wl_data if e.get("bsh") == "BUY"}
                watchlist_sells = {e["symbol"] for e in wl_data if e.get("bsh") == "SELL"}
            print(f"Watchlist BUY  ({len(watchlist_buys)}):  {', '.join(sorted(watchlist_buys))  or 'none'}")
            print(f"Watchlist SELL ({len(watchlist_sells)}): {', '.join(sorted(watchlist_sells)) or 'none'}")
        except Exception as e:
            print(f"  Warning: could not fetch watchlist: {e}")

    # ── LONG candidates ────────────────────────────────────────────────
    long_candidates = [
        r for r in scored
        if r.get("firing_signals") and (
            r.get("score", 0) >= MIN_SCORE
            or (r["symbol"] in watchlist_buys and r.get("score", 0) >= WATCHLIST_MIN_SCORE)
        )
    ]
    long_candidates.sort(key=lambda x: (0 if x["symbol"] in watchlist_buys else 1, -x.get("score", 0)))

    # ── SHORT candidates (inverted: low score = bearish) ───────────────
    short_candidates = [
        r for r in scored
        if r["symbol"] not in excluded and (
            r.get("score", 0) <= MAX_SHORT_SCORE
            or (r["symbol"] in watchlist_sells and r.get("score", 0) <= WATCHLIST_MAX_SHORT)
        )
    ]
    # Most bearish (lowest score) first; SELL watchlist symbols go first within each tier
    short_candidates.sort(key=lambda x: (0 if x["symbol"] in watchlist_sells else 1, x.get("score", 0)))

    print(f"\nTop scoring symbols (LONG):\n{'─'*50}")
    for r in scored[:10]:
        sym   = r["symbol"]
        score = r.get("score", 0)
        on_wl = sym in watchlist_buys
        qualifies = r.get("firing_signals") and (score >= MIN_SCORE or (on_wl and score >= WATCHLIST_MIN_SCORE))
        mark  = "✓" if qualifies else " "
        wl_tag = " [WL]" if on_wl else ""
        print(f"  {mark} {sym:<8} score={score:>3}{wl_tag}  signals={r.get('firing_signals')}")
    print(f"\nBottom scoring symbols (SHORT):\n{'─'*50}")
    for r in scored[-10:][::-1]:
        sym   = r["symbol"]
        score = r.get("score", 0)
        on_wl = sym in watchlist_sells
        qualifies = score <= MAX_SHORT_SCORE or (on_wl and score <= WATCHLIST_MAX_SHORT)
        mark  = "✓" if qualifies else " "
        wl_tag = " [WL]" if on_wl else ""
        print(f"  {mark} {sym:<8} score={score:>3}{wl_tag}")
    print(f"\nLONG  qualifying (score ≥ {MIN_SCORE}, WL ≥ {WATCHLIST_MIN_SCORE}): {len(long_candidates)}")
    print(f"SHORT qualifying (score ≤ {MAX_SHORT_SCORE}, WL ≤ {WATCHLIST_MAX_SHORT}): {len(short_candidates)}")
    print(f"Will enter up to {min(slots, len(long_candidates) + len(short_candidates))} plays\n")

    executor = OrderExecutor(client)
    entered  = 0

    def _enter_play(candidate, direction):
        nonlocal entered, buying_power
        sym  = candidate["symbol"]
        bars = candidate["_bars"]
        price = float(bars["close"].iloc[-1])
        a    = atr(bars)
        if a <= 0:
            return

        if direction == "LONG":
            stop    = round(price - a * ATR_STOP_MULT, 2)
            target  = max(round(price + a * ATR_TARGET_MULT, 2), round(price * 1.07, 2))
            risk_ps = price - stop
        else:
            stop    = round(price + a * ATR_STOP_MULT, 2)
            target  = min(round(price - a * ATR_TARGET_MULT, 2), round(price * 0.93, 2))
            risk_ps = stop - price

        if risk_ps <= 0:
            return

        try:
            buying_power = float(client.get_account()["buying_power"])
        except Exception:
            pass

        shares = int((equity * RISK_PCT) / risk_ps)
        cost   = shares * price
        max_cost = equity * MAX_POSITION_PCT
        if cost > max_cost:
            shares = max(1, int(max_cost / price))
            cost   = shares * price

        if shares < 1 or cost > buying_power * 0.95:
            print(f"  {sym}: insufficient buying power (BP=${buying_power:,.0f}) — skip")
            return

        signals = candidate.get("firing_signals", [])
        score   = candidate.get("score", 0)
        on_wl   = sym in (watchlist_buys if direction == "LONG" else watchlist_sells)

        try:
            signal = Signal(
                symbol=sym,
                direction=direction,
                confidence=(score / 100.0) if direction == "LONG" else ((100 - score) / 100.0),
                entry_price=price,
                stop_loss=stop,
                take_profit=target,
                position_size_pct=shares * price / equity,
                leverage=1.0,
                regime_id=0,
                regime_name="SWING",
                regime_probability=1.0,
                timestamp=datetime.utcnow(),
                reasoning=f"swing scan score={score} dir={direction}",
                strategy_name=signals[0] if signals else "bearish_composite" if direction == "SHORT" else "composite",
            )
            executor.submit_bracket_order(signal)
            entered += 1
            wl_label = " [WATCHLIST]" if on_wl else ""
            dir_label = "LONG " if direction == "LONG" else "SHORT"
            print(f"  ✓ ENTERED {dir_label} {sym}{wl_label}")
            print(f"    score={score}  signals={signals}")
            print(f"    {shares} sh @ ${price:.2f}  stop=${stop:.2f}  target=${target:.2f}")
            print(f"    cost=${cost:,.2f}  BP left=${buying_power:,.2f}")

            notes = f"score={score} cron=scan_now dir={direction} signals={','.join(signals)}"
            if WEBAPP_URL:
                _record_play(
                    symbol=sym, direction=direction,
                    entry_price=price, stop_loss=stop, take_profit=target,
                    shares=shares,
                    signal=signals[0] if signals else ("bearish_composite" if direction == "SHORT" else "composite"),
                    notes=notes,
                )
                print(f"    Dashboard: recorded ✓\n")
            else:
                print(f"    Dashboard: skipped (WEBAPP_URL not set — sync will discover)\n")
            excluded.add(sym)
        except Exception as e:
            print(f"  ✗ {sym} failed: {e}\n")

    # Enter longs first, then fill remaining slots with shorts
    for candidate in long_candidates:
        if entered >= slots:
            break
        _enter_play(candidate, "LONG")

    for candidate in short_candidates:
        if entered >= slots:
            break
        if candidate["symbol"] in excluded:
            continue
        _enter_play(candidate, "SHORT")

    print(f"{'='*55}")
    print(f"  Done. Entered {entered} new play(s).")
    print(f"  Remaining buying power: ${buying_power:,.2f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
