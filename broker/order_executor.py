"""Order placement, modification, and cancellation."""

from typing import Optional
from .alpaca_client import AlpacaClient


class OrderExecutor:
    """Handles order lifecycle: submit, modify, cancel, and status checks."""

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client

    def submit_market_order(self, symbol: str, qty: float, side: str) -> dict:
        """Submit a market order. side must be 'buy' or 'sell'."""
        ...

    def submit_limit_order(
        self, symbol: str, qty: float, side: str, limit_price: float
    ) -> dict:
        """Submit a limit order."""
        ...

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order by ID."""
        ...

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        ...

    def get_order_status(self, order_id: str) -> dict:
        """Return current status of an order."""
        ...

    def rebalance_to_weights(
        self, target_weights: dict[str, float], portfolio_value: float
    ) -> list[dict]:
        """Generate and submit orders to move current holdings to target weights."""
        ...
