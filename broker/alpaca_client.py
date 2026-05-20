"""Alpaca API wrapper."""

from typing import Optional
import alpaca_trade_api as tradeapi


class AlpacaClient:
    """Thin wrapper around the Alpaca REST API."""

    def __init__(self, api_key: str, secret_key: str, base_url: str, paper: bool = True) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.paper = paper
        self._api: Optional[tradeapi.REST] = None

    def connect(self) -> None:
        """Initialize the Alpaca REST client."""
        ...

    def get_account(self) -> dict:
        """Return account info including equity, buying power, and status."""
        ...

    def get_clock(self) -> dict:
        """Return market clock (is_open, next_open, next_close)."""
        ...

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> list:
        """Fetch historical OHLCV bars for a symbol."""
        ...

    def get_latest_trade(self, symbol: str) -> dict:
        """Return the most recent trade price for a symbol."""
        ...

    def is_market_open(self) -> bool:
        """Return True if the market is currently open."""
        ...
