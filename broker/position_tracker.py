"""Real-time position tracking with WebSocket fill notifications — Phase 6."""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .alpaca_client import AlpacaClient

logger = logging.getLogger("regime_trader")


@dataclass
class TrackedPosition:
    """Full per-position state tracked across fills and regime changes."""
    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_level: float
    take_profit: Optional[float]
    regime_at_entry: str
    regime_current: str
    trade_id: str

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def holding_period_minutes(self) -> int:
        return max(0, int((datetime.utcnow() - self.entry_time).total_seconds() / 60))


class PositionTracker:
    """
    Tracks open positions with per-fill granularity via TradingStream.

    On every fill:
      - Updates the internal TrackedPosition state
      - Calls registered callbacks (used to update PortfolioState and CircuitBreaker)

    On startup:
      - Reconciles tracked state against actual Alpaca positions
        so the bot can resume after a restart without losing state.
    """

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client
        self._positions: Dict[str, TrackedPosition] = {}
        self._fill_callbacks: List[Callable] = []
        self._stream_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def sync_with_alpaca(self, current_regime: str = "UNKNOWN") -> None:
        """
        Reconcile tracked positions against the live Alpaca account.

        - Stale tracked positions (Alpaca says flat) are removed.
        - Untracked Alpaca positions are added as synthetic entries
          (entry_price = current_price, trade_id = "reconciled").
        """
        alpaca_positions = self.client.get_positions()
        alpaca_symbols = {p["symbol"] for p in alpaca_positions}

        with self._lock:
            # Remove positions Alpaca no longer holds
            for sym in list(self._positions.keys()):
                if sym not in alpaca_symbols:
                    logger.info("Sync: removing stale tracked position %s", sym)
                    del self._positions[sym]

            # Add untracked positions from Alpaca
            for pos in alpaca_positions:
                sym = pos["symbol"]
                if sym not in self._positions:
                    logger.info(
                        "Sync: adding untracked Alpaca position %s qty=%s @ $%s",
                        sym, pos["qty"], pos["avg_entry_price"],
                    )
                    self._positions[sym] = TrackedPosition(
                        symbol=sym,
                        qty=float(pos["qty"]),
                        entry_price=float(pos["avg_entry_price"]),
                        entry_time=datetime.utcnow(),
                        current_price=float(pos["current_price"]),
                        stop_level=0.0,
                        take_profit=None,
                        regime_at_entry=current_regime,
                        regime_current=current_regime,
                        trade_id="reconciled",
                    )

        logger.info(
            "Position sync complete: %d tracked position(s).", len(self._positions)
        )

    # ------------------------------------------------------------------
    # WebSocket fill stream
    # ------------------------------------------------------------------

    def start_stream(self) -> None:
        """Subscribe to TradingStream for fill notifications (background daemon thread)."""
        self._stream_thread = threading.Thread(
            target=self._run_stream,
            daemon=True,
            name="TradingStream",
        )
        self._stream_thread.start()
        logger.info("TradingStream started.")

    def _run_stream(self) -> None:
        from alpaca.trading.stream import TradingStream

        stream = TradingStream(
            api_key=self.client.api_key,
            secret_key=self.client.secret_key,
            paper=self.client.paper,
        )
        stream.subscribe_trade_updates(self._on_trade_update)
        try:
            stream.run()
        except Exception as exc:
            logger.error("TradingStream error: %s — live loop will handle reconnect.", exc)

    async def _on_trade_update(self, update) -> None:
        """Called by TradingStream on every order lifecycle event."""
        event = str(update.event)
        if event not in ("fill", "partial_fill"):
            return

        order = update.order
        sym = order.symbol
        fill_price = float(order.filled_avg_price or 0)
        fill_qty = float(order.filled_qty or 0)
        side = str(order.side).lower()

        with self._lock:
            if side == "buy":
                if sym in self._positions:
                    pos = self._positions[sym]
                    total_qty = pos.qty + fill_qty
                    # Volume-weighted average entry price
                    pos.entry_price = (
                        (pos.entry_price * pos.qty + fill_price * fill_qty) / total_qty
                    )
                    pos.qty = total_qty
                else:
                    self._positions[sym] = TrackedPosition(
                        symbol=sym,
                        qty=fill_qty,
                        entry_price=fill_price,
                        entry_time=datetime.utcnow(),
                        current_price=fill_price,
                        stop_level=0.0,
                        take_profit=None,
                        regime_at_entry="UNKNOWN",
                        regime_current="UNKNOWN",
                        trade_id=str(order.client_order_id or ""),
                    )
            else:  # sell / close
                pos = self._positions.get(sym)
                if pos:
                    pos.qty -= fill_qty
                    if pos.qty <= 0.001:
                        del self._positions[sym]

        logger.info(
            "Fill [%s]: %s x%.2f @ $%.2f", event.upper(), sym, fill_qty, fill_price
        )

        # Notify all registered callbacks (PortfolioState, CircuitBreaker updates)
        for cb in self._fill_callbacks:
            try:
                cb(sym, fill_price, fill_qty, side)
            except Exception as exc:
                logger.warning("Fill callback error: %s", exc)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_fill_callback(self, callback: Callable) -> None:
        """
        Register a function called on every fill:
          callback(symbol: str, fill_price: float, fill_qty: float, side: str)
        Use this to update PortfolioState and the CircuitBreaker with real P&L.
        """
        self._fill_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Position state updates
    # ------------------------------------------------------------------

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current_price for all tracked positions."""
        with self._lock:
            for sym, price in prices.items():
                if sym in self._positions:
                    self._positions[sym].current_price = price

    def update_regime(self, symbol: str, regime: str) -> None:
        """Update the current regime for a tracked position."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].regime_current = regime

    def set_entry_regime(self, symbol: str, regime: str) -> None:
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].regime_at_entry = regime

    def set_stop(self, symbol: str, stop: float) -> None:
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].stop_level = stop

    def set_take_profit(self, symbol: str, tp: float) -> None:
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].take_profit = tp

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[TrackedPosition]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, TrackedPosition]:
        with self._lock:
            return dict(self._positions)

    def open_position_count(self) -> int:
        return len(self._positions)

    def portfolio_market_value(self) -> float:
        with self._lock:
            return sum(p.qty * p.current_price for p in self._positions.values())

    def total_unrealized_pnl(self) -> float:
        with self._lock:
            return sum(p.unrealized_pnl for p in self._positions.values())

    def current_weights(self, total_equity: float) -> Dict[str, float]:
        """Return each position's market value as a fraction of total equity."""
        if total_equity <= 0:
            return {}
        with self._lock:
            return {
                sym: (p.qty * p.current_price) / total_equity
                for sym, p in self._positions.items()
            }
