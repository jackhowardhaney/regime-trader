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
MIN_SCORE        = 65
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

    held = {p["symbol"] for p in client.get_positions()}
    slots = MAX_POSITIONS - len(held)
    print(f"  Open positions:    {len(held)}  ({', '.join(sorted(held))})")
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
        if sym in held:
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
    candidates = [r for r in scored if r.get("score", 0) >= MIN_SCORE and r.get("firing_signals")]

    print(f"\nTop scoring symbols:\n{'─'*50}")
    for r in scored[:15]:
        mark = "✓" if r.get("score",0) >= MIN_SCORE and r.get("firing_signals") else " "
        print(f"  {mark} {r['symbol']:<8} score={r.get('score',0):>3}  signals={r.get('firing_signals')}")
    print(f"\nQualifying (score ≥ {MIN_SCORE} with signal): {len(candidates)}")
    print(f"Will enter up to {min(slots, len(candidates))} plays\n")

    executor = OrderExecutor(client)
    entered  = 0

    for candidate in candidates[:slots]:
        sym  = candidate["symbol"]
        bars = candidate["_bars"]
        price = float(bars["close"].iloc[-1])
        a    = atr(bars)
        if a <= 0:
            continue

        stop   = round(price - a * ATR_STOP_MULT, 2)
        target = max(round(price + a * ATR_TARGET_MULT, 2), round(price * 1.07, 2))
        risk_ps = price - stop
        if risk_ps <= 0:
            continue

        # Refresh buying power from Alpaca before each order
        try:
            buying_power = float(client.get_account()["buying_power"])
        except Exception:
            pass

        # Risk-based sizing: risk exactly RISK_PCT of equity
        shares = int((equity * RISK_PCT) / risk_ps)
        cost   = shares * price

        # Hard cap: never put more than MAX_POSITION_PCT of equity into one play.
        # 20% cap = room for 5+ concurrent positions on any account size.
        max_cost = equity * MAX_POSITION_PCT
        if cost > max_cost:
            shares = max(1, int(max_cost / price))
            cost   = shares * price

        # Can't exceed available buying power
        if shares < 1 or cost > buying_power * 0.95:
            print(f"  {sym}: insufficient buying power (BP=${buying_power:,.0f}) — skip")
            continue

        signals = candidate.get("firing_signals", [])
        score   = candidate.get("score", 0)

        try:
            signal = Signal(
                symbol=sym,
                direction="LONG",
                confidence=score / 100.0,
                entry_price=price,
                stop_loss=stop,
                take_profit=target,
                position_size_pct=shares * price / equity,
                leverage=1.0,
                regime_id=0,
                regime_name="SWING",
                regime_probability=1.0,
                timestamp=datetime.utcnow(),
                reasoning=f"swing scan score={score} signals={signals}",
                strategy_name=signals[0] if signals else "composite",
            )
            record = executor.submit_bracket_order(signal)
            entered += 1
            print(f"  ✓ ENTERED {sym}")
            print(f"    score={score}  signals={signals}")
            print(f"    {shares} sh @ ${price:.2f}  stop=${stop:.2f}  target=${target:.2f}")
            print(f"    cost=${cost:,.2f}  BP left=${buying_power:,.2f}")

            # Record play in dashboard DB (only when WEBAPP_URL is configured)
            notes = f"score={score} cron=scan_now signals={','.join(signals)}"
            if WEBAPP_URL:
                _record_play(
                    symbol=sym, direction="LONG",
                    entry_price=price, stop_loss=stop, take_profit=target,
                    shares=shares, signal=signals[0] if signals else "composite",
                    notes=notes,
                )
                print(f"    Dashboard: recorded ✓\n")
            else:
                print(f"    Dashboard: skipped (WEBAPP_URL not set — sync will discover)\n")
        except Exception as e:
            print(f"  ✗ {sym} failed: {e}\n")

    print(f"{'='*55}")
    print(f"  Done. Entered {entered} new play(s).")
    print(f"  Remaining buying power: ${buying_power:,.2f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
