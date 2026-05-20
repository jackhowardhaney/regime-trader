"""Track open positions and real-time P&L."""

from dataclasses import dataclass, field
from typing import Dict
from .alpaca_client import AlpacaClient


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


class PositionTracker:
    """Fetches and caches open positions; computes portfolio-level P&L."""

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client
        self.positions: Dict[str, Position] = {}

    def refresh(self) -> None:
        """Pull latest positions from Alpaca and update internal cache."""
        ...

    def get_position(self, symbol: str) -> Position | None:
        """Return the current Position for a symbol, or None if flat."""
        ...

    def portfolio_value(self) -> float:
        """Return total equity value of all open positions."""
        ...

    def total_unrealized_pnl(self) -> float:
        """Return aggregate unrealized P&L across all positions."""
        ...

    def current_weights(self, total_equity: float) -> Dict[str, float]:
        """Return each position's share of total equity."""
        ...
