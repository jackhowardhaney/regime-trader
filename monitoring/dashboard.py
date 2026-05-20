"""Terminal-based live dashboard using Rich."""

from typing import Dict
from rich.live import Live
from rich.table import Table
from rich.layout import Layout


class Dashboard:
    """Renders a live terminal dashboard showing regime, positions, and P&L."""

    def __init__(self, refresh_seconds: int = 5) -> None:
        self.refresh_seconds = refresh_seconds
        self._live: Live | None = None

    def start(self) -> None:
        """Begin the live render loop."""
        ...

    def stop(self) -> None:
        """Stop the live render loop."""
        ...

    def update(
        self,
        regime: str,
        portfolio_value: float,
        positions: Dict[str, float],
        daily_pnl: float,
        drawdown: float,
    ) -> None:
        """Push new data to the dashboard for the next render cycle."""
        ...

    def _build_layout(self) -> Layout:
        """Construct the Rich Layout with regime, positions, and metrics panels."""
        ...

    def _positions_table(self, positions: Dict[str, float]) -> Table:
        """Build a Rich Table of current positions and weights."""
        ...
