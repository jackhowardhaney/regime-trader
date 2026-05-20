"""Order placement, modification, and cancellation — Phase 6."""

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from .alpaca_client import AlpacaClient

logger = logging.getLogger("regime_trader")

_LIMIT_OFFSET = 0.001   # 0.1% — limit price offset from entry for guaranteed-fill limit
_CANCEL_TIMEOUT = 30    # seconds before cancelling an unfilled limit order


@dataclass
class OrderRecord:
    """Links signal → risk_decision → order → fill with a single trade_id."""
    trade_id: str
    signal_symbol: str
    order_id: str
    client_order_id: str
    side: str
    qty: int
    order_type: str
    limit_price: Optional[float]
    stop_price: Optional[float]
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    status: str = "pending"


class OrderExecutor:
    """
    Handles the full order lifecycle.

    Every submission generates a unique trade_id so every fill can be
    traced back to the signal and RiskDecision that approved it.
    """

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client
        self._records: Dict[str, OrderRecord] = {}   # trade_id → OrderRecord

    # ------------------------------------------------------------------
    # Primary order submission
    # ------------------------------------------------------------------

    def submit_order(self, signal, retry_at_market: bool = False) -> OrderRecord:
        """
        Submit a LIMIT order at ±0.1% of entry_price.
        Cancels automatically after 30s if unfilled; retries at market if
        retry_at_market=True.

        signal must have: symbol, direction, entry_price, position_size_pct, leverage
        """
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        trade_id = str(uuid.uuid4())
        side = OrderSide.BUY if signal.direction == "LONG" else OrderSide.SELL

        # Buy slightly above, sell slightly below — increases fill probability
        if side == OrderSide.BUY:
            limit_price = round(signal.entry_price * (1 + _LIMIT_OFFSET), 2)
        else:
            limit_price = round(signal.entry_price * (1 - _LIMIT_OFFSET), 2)

        qty = self._calc_qty(signal)
        if qty < 1:
            raise ValueError(
                f"Calculated qty < 1 for {signal.symbol} — position too small."
            )

        client_order_id = f"rt-{trade_id[:8]}"
        req = LimitOrderRequest(
            symbol=signal.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )

        order = self.client.trading.submit_order(req)
        record = OrderRecord(
            trade_id=trade_id,
            signal_symbol=signal.symbol,
            order_id=str(order.id),
            client_order_id=client_order_id,
            side=str(side),
            qty=qty,
            order_type="limit",
            limit_price=limit_price,
            stop_price=None,
            submitted_at=datetime.utcnow(),
        )
        self._records[trade_id] = record

        logger.info(
            "LIMIT submitted: %s %s x%d @ $%.2f [trade_id=%s]",
            side, signal.symbol, qty, limit_price, trade_id,
        )

        # Schedule cancellation if unfilled after _CANCEL_TIMEOUT seconds
        t = threading.Timer(
            _CANCEL_TIMEOUT,
            self._cancel_if_unfilled,
            args=(record, retry_at_market, signal),
        )
        t.daemon = True
        t.start()

        return record

    def submit_bracket_order(self, signal) -> OrderRecord:
        """
        Submit entry + stop_loss + take_profit via Alpaca OCO bracket order.

        signal must have: symbol, direction, entry_price, stop_loss,
                          take_profit (Optional), position_size_pct, leverage
        """
        from alpaca.trading.requests import (
            LimitOrderRequest, TakeProfitRequest, StopLossRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        trade_id = str(uuid.uuid4())
        side = OrderSide.BUY if signal.direction == "LONG" else OrderSide.SELL

        if side == OrderSide.BUY:
            limit_price = round(signal.entry_price * (1 + _LIMIT_OFFSET), 2)
        else:
            limit_price = round(signal.entry_price * (1 - _LIMIT_OFFSET), 2)

        qty = self._calc_qty(signal)
        if qty < 1:
            raise ValueError(
                f"Calculated qty < 1 for {signal.symbol} — position too small."
            )

        client_order_id = f"rt-brk-{trade_id[:8]}"
        take_profit_req = (
            TakeProfitRequest(limit_price=round(float(signal.take_profit), 2))
            if signal.take_profit
            else None
        )

        req = LimitOrderRequest(
            symbol=signal.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            client_order_id=client_order_id,
            order_class=OrderClass.BRACKET,
            take_profit=take_profit_req,
            stop_loss=StopLossRequest(
                stop_price=round(float(signal.stop_loss), 2)
            ),
        )

        order = self.client.trading.submit_order(req)
        record = OrderRecord(
            trade_id=trade_id,
            signal_symbol=signal.symbol,
            order_id=str(order.id),
            client_order_id=client_order_id,
            side=str(side),
            qty=qty,
            order_type="bracket",
            limit_price=limit_price,
            stop_price=float(signal.stop_loss),
            submitted_at=datetime.utcnow(),
        )
        self._records[trade_id] = record

        logger.info(
            "BRACKET submitted: %s %s x%d entry=%.2f stop=%.2f tp=%s [trade_id=%s]",
            side, signal.symbol, qty, limit_price,
            signal.stop_loss,
            f"${signal.take_profit:.2f}" if signal.take_profit else "None",
            trade_id,
        )
        return record

    # ------------------------------------------------------------------
    # Stop modification — only tightening allowed
    # ------------------------------------------------------------------

    def modify_stop(self, symbol: str, order_id: str, new_stop: float) -> bool:
        """
        Move a stop leg to new_stop. Silently rejects if the move would
        widen the stop (new_stop farther from entry than current stop).

        Returns True if the modification was submitted, False if rejected.
        """
        try:
            order = self.client.trading.get_order_by_id(order_id)
        except Exception as exc:
            logger.warning("modify_stop: cannot fetch order %s: %s", order_id, exc)
            return False

        stop_leg = None
        for leg in (order.legs or []):
            if str(leg.order_type).lower() in ("stop", "stop_limit"):
                stop_leg = leg
                break

        if stop_leg is None:
            logger.warning("modify_stop: no stop leg on order %s", order_id)
            return False

        current_stop = float(stop_leg.stop_price or 0)
        is_long = str(order.side).lower() == "buy"

        # Reject if this would widen the stop
        if is_long and new_stop <= current_stop:
            logger.debug(
                "modify_stop SKIPPED (widen): current=%.2f new=%.2f", current_stop, new_stop
            )
            return False
        if not is_long and new_stop >= current_stop:
            logger.debug(
                "modify_stop SKIPPED (widen): current=%.2f new=%.2f", current_stop, new_stop
            )
            return False

        try:
            self.client.trading.replace_order_by_id(
                str(stop_leg.id),
                stop_price=round(new_stop, 2),
            )
            logger.info(
                "Stop tightened: %s $%.2f → $%.2f", symbol, current_stop, new_stop
            )
            return True
        except Exception as exc:
            logger.warning("modify_stop replace failed for %s: %s", symbol, exc)
            return False

    # ------------------------------------------------------------------
    # Cancellation and position closing
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> None:
        try:
            self.client.trading.cancel_order_by_id(order_id)
            logger.info("Order cancelled: %s", order_id)
        except Exception as exc:
            logger.warning("cancel_order %s failed: %s", order_id, exc)

    def close_position(self, symbol: str) -> None:
        """Submit a market order to flatten the full position in symbol."""
        try:
            self.client.trading.close_position(symbol)
            logger.info("Position closed: %s", symbol)
        except Exception as exc:
            logger.warning("close_position %s failed: %s", symbol, exc)

    def close_all_positions(self) -> None:
        """Close all open positions and cancel all open orders."""
        try:
            self.client.trading.close_all_positions(cancel_orders=True)
            logger.warning("ALL POSITIONS CLOSED.")
        except Exception as exc:
            logger.error("close_all_positions failed: %s", exc)

    # ------------------------------------------------------------------
    # Record lookup
    # ------------------------------------------------------------------

    def get_record(self, trade_id: str) -> Optional[OrderRecord]:
        return self._records.get(trade_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calc_qty(self, signal) -> int:
        """Derive share count from signal sizing and current account equity."""
        acct = self.client.get_account()
        equity = float(acct["equity"])
        notional = equity * signal.position_size_pct * signal.leverage
        return max(1, int(notional / signal.entry_price))

    def _cancel_if_unfilled(
        self, record: OrderRecord, retry_at_market: bool, signal
    ) -> None:
        """Timer callback — cancels the limit order if still open after 30s."""
        try:
            order = self.client.trading.get_order_by_id(record.order_id)
            if str(order.status) in ("filled", "partially_filled", "canceled"):
                return
            self.cancel_order(record.order_id)
            record.status = "canceled_timeout"
            logger.info(
                "Limit order %s cancelled after %ds (unfilled).",
                record.order_id, _CANCEL_TIMEOUT,
            )
            if retry_at_market:
                self._retry_market(signal, record.trade_id)
        except Exception as exc:
            logger.warning("_cancel_if_unfilled error for %s: %s", record.order_id, exc)

    def _retry_market(self, signal, original_trade_id: str) -> None:
        """Retry a timed-out limit order as a market order."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side = OrderSide.BUY if signal.direction == "LONG" else OrderSide.SELL
        qty = self._calc_qty(signal)

        req = MarketOrderRequest(
            symbol=signal.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"rt-mkt-{original_trade_id[:8]}",
        )
        try:
            order = self.client.trading.submit_order(req)
            logger.info(
                "Market retry submitted: %s %s x%d [orig=%s new=%s]",
                side, signal.symbol, qty, original_trade_id, order.id,
            )
        except Exception as exc:
            logger.error("Market retry failed for %s: %s", signal.symbol, exc)
