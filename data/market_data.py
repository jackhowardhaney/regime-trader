"""Real-time and historical market data fetching."""

from typing import Dict, List
import pandas as pd
from broker.alpaca_client import AlpacaClient


class MarketDataFetcher:
    """Retrieves OHLCV data and real-time quotes via Alpaca."""

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client

    def fetch_historical(
        self,
        symbols: List[str],
        timeframe: str,
        start: str,
        end: str,
    ) -> Dict[str, pd.DataFrame]:
        """Return a dict of {symbol: OHLCV DataFrame} for the given date range."""
        ...

    def fetch_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Return the most recent trade price for each symbol."""
        ...

    def fetch_daily_returns(
        self, symbols: List[str], lookback_days: int
    ) -> pd.DataFrame:
        """Return a DataFrame of daily log-returns (columns=symbols)."""
        ...

    def fetch_realized_volatility(
        self, symbols: List[str], window: int = 21
    ) -> Dict[str, float]:
        """Return annualized realized volatility for each symbol over `window` days."""
        ...
