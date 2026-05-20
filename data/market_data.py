"""Real-time and historical market data via Alpaca — Phase 6."""

import logging
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger("regime_trader")


class MarketDataFetcher:
    """
    Historical and real-time market data via alpaca-py.

    Historical:  StockHistoricalDataClient (REST)
    Real-time:   StockDataStream (WebSocket) for bars and quotes
    Gap handling: short gaps (weekends, single holidays) are forward-filled;
                  long gaps (trading halts, delistings) are left as NaN then dropped.
    """

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client
        self._latest_bars: Dict[str, dict] = {}
        self._latest_quotes: Dict[str, dict] = {}
        self._bar_callback: Optional[Callable] = None
        self._quote_callback: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Historical — single symbol
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for one symbol.
        Returns a DataFrame with DatetimeIndex (America/New_York) and columns:
        open, high, low, close, volume.
        """
        from alpaca.data.requests import StockBarsRequest

        tf = self._parse_timeframe(timeframe)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=pd.Timestamp(start, tz="America/New_York"),
            end=pd.Timestamp(end, tz="America/New_York"),
        )
        bars = self.client.data.get_stock_bars(req)
        df = bars.df

        if df.empty:
            logger.warning("get_historical_bars: empty response for %s", symbol)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Flatten MultiIndex (symbol is outer level when fetching a single symbol)
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        df = df.sort_index()
        df = self._handle_gaps(df, timeframe)
        return df

    # ------------------------------------------------------------------
    # Historical — multiple symbols
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbols: List[str],
        timeframe: str,
        start: str,
        end: str,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols. Returns {symbol: DataFrame}."""
        return {
            sym: self.get_historical_bars(sym, timeframe, start, end)
            for sym in symbols
        }

    def fetch_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Return the most recent close price for each symbol."""
        prices = {}
        for sym in symbols:
            bar = self.get_latest_bar(sym)
            if bar:
                prices[sym] = bar.get("close", 0.0)
        return prices

    def fetch_daily_returns(
        self, symbols: List[str], lookback_days: int
    ) -> pd.DataFrame:
        """Return a DataFrame of daily log-returns (columns = symbols)."""
        import numpy as np

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")
        frames = {}
        for sym in symbols:
            df = self.get_historical_bars(sym, "1Day", start, end)
            if not df.empty:
                frames[sym] = np.log(df["close"] / df["close"].shift(1)).dropna()
        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames).dropna()

    def fetch_realized_volatility(
        self, symbols: List[str], window: int = 21
    ) -> Dict[str, float]:
        """Return annualized realized volatility for each symbol over window days."""
        import numpy as np

        returns = self.fetch_daily_returns(symbols, window + 5)
        if returns.empty:
            return {sym: 0.0 for sym in symbols}
        return {
            sym: float(returns[sym].tail(window).std() * np.sqrt(252))
            for sym in symbols
            if sym in returns.columns
        }

    # ------------------------------------------------------------------
    # Real-time — latest snapshot values
    # ------------------------------------------------------------------

    def get_latest_bar(self, symbol: str) -> Optional[dict]:
        """
        Return the most recently received bar for symbol.
        Falls back to a REST request if the WebSocket hasn't delivered one yet.
        """
        if symbol in self._latest_bars:
            return self._latest_bars[symbol]
        return self._fetch_latest_bar_rest(symbol)

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """
        Return the most recent bid/ask quote for symbol.
        Falls back to REST if WebSocket hasn't delivered one yet.
        """
        if symbol in self._latest_quotes:
            return self._latest_quotes[symbol]
        return self._fetch_latest_quote_rest(symbol)

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        """
        Return a full snapshot (latest trade, bid/ask, daily OHLCV) for symbol.
        """
        from alpaca.data.requests import StockSnapshotRequest

        try:
            req = StockSnapshotRequest(symbol_or_symbols=symbol)
            snaps = self.client.data.get_stock_snapshot(req)
            snap = snaps.get(symbol)
            if snap is None:
                return None
            return {
                "symbol": symbol,
                "latest_trade_price": float(snap.latest_trade.price),
                "bid": float(snap.latest_quote.bid_price),
                "ask": float(snap.latest_quote.ask_price),
                "daily_open": float(snap.daily_bar.open),
                "daily_high": float(snap.daily_bar.high),
                "daily_low": float(snap.daily_bar.low),
                "daily_close": float(snap.daily_bar.close),
                "daily_volume": float(snap.daily_bar.volume),
            }
        except Exception as exc:
            logger.warning("get_snapshot %s failed: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    def subscribe_bars(
        self, symbols: List[str], timeframe: str, callback: Callable
    ) -> None:
        """
        Subscribe to real-time bar updates via StockDataStream WebSocket.
        callback(bar_dict) is called on each new completed bar.
        Runs in a background daemon thread.
        """
        self._bar_callback = callback
        threading.Thread(
            target=self._run_bar_stream,
            args=(symbols,),
            daemon=True,
            name="BarStream",
        ).start()
        logger.info("Bar WebSocket stream started for %s", symbols)

    def subscribe_quotes(self, symbols: List[str], callback: Callable) -> None:
        """
        Subscribe to real-time bid/ask quote updates via WebSocket.
        callback(quote_dict) is called on each update.
        Used for spread checks before order submission.
        """
        self._quote_callback = callback
        threading.Thread(
            target=self._run_quote_stream,
            args=(symbols,),
            daemon=True,
            name="QuoteStream",
        ).start()
        logger.info("Quote WebSocket stream started for %s", symbols)

    def _run_bar_stream(self, symbols: List[str]) -> None:
        from alpaca.data.live import StockDataStream

        stream = StockDataStream(
            api_key=self.client.api_key,
            secret_key=self.client.secret_key,
        )

        async def _bar_handler(bar):
            bar_dict = {
                "symbol": bar.symbol,
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            self._latest_bars[bar.symbol] = bar_dict
            if self._bar_callback:
                try:
                    self._bar_callback(bar_dict)
                except Exception as exc:
                    logger.warning("Bar callback error: %s", exc)

        stream.subscribe_bars(_bar_handler, *symbols)
        try:
            stream.run()
        except Exception as exc:
            logger.error("BarStream error: %s", exc)

    def _run_quote_stream(self, symbols: List[str]) -> None:
        from alpaca.data.live import StockDataStream

        stream = StockDataStream(
            api_key=self.client.api_key,
            secret_key=self.client.secret_key,
        )

        async def _quote_handler(quote):
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            mid = (bid + ask) / 2.0
            quote_dict = {
                "symbol": quote.symbol,
                "timestamp": quote.timestamp,
                "bid": bid,
                "ask": ask,
                "bid_size": float(quote.bid_size),
                "ask_size": float(quote.ask_size),
                "spread_pct": (ask - bid) / mid if mid > 0 else 0.0,
            }
            self._latest_quotes[quote.symbol] = quote_dict
            if self._quote_callback:
                try:
                    self._quote_callback(quote_dict)
                except Exception as exc:
                    logger.warning("Quote callback error: %s", exc)

        stream.subscribe_quotes(_quote_handler, *symbols)
        try:
            stream.run()
        except Exception as exc:
            logger.error("QuoteStream error: %s", exc)

    # ------------------------------------------------------------------
    # Gap handling
    # ------------------------------------------------------------------

    def _handle_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Handle missing bars from weekends, holidays, and trading halts.

        Daily bars: reindex to business-day calendar and forward-fill gaps
        of ≤ 3 bars (weekends = 2 missing bars, single holiday = 1).
        Gaps > 3 bars are left NaN (trading halt / delist) then dropped.

        Intraday bars: gaps are left as-is — synthesizing intraday bars
        during halts produces misleading feature values.
        """
        if df.empty or "Day" not in timeframe:
            return df

        full_idx = pd.bdate_range(
            df.index.min(), df.index.max(), tz=df.index.tz
        )
        original_len = len(df)
        df = df.reindex(full_idx)

        gap_count = int(df["close"].isna().sum())
        if gap_count > 0:
            logger.debug(
                "_handle_gaps: %d missing bars in %d-bar series (weekends/holidays/halts)",
                gap_count, original_len,
            )

        # Forward-fill only short gaps
        df = df.ffill(limit=3)
        df["volume"] = df["volume"].fillna(0.0)

        remaining_nans = int(df["close"].isna().sum())
        if remaining_nans > 0:
            logger.warning(
                "_handle_gaps: %d bars still NaN after forward-fill "
                "(likely trading halt or delist). Dropping.",
                remaining_nans,
            )
            df = df.dropna(subset=["close"])

        return df

    # ------------------------------------------------------------------
    # REST fallbacks for latest bar / quote
    # ------------------------------------------------------------------

    def _fetch_latest_bar_rest(self, symbol: str) -> Optional[dict]:
        from alpaca.data.requests import StockLatestBarRequest

        try:
            req = StockLatestBarRequest(symbol_or_symbols=symbol)
            bars = self.client.data.get_stock_latest_bar(req)
            bar = bars.get(symbol)
            if bar is None:
                return None
            result = {
                "symbol": symbol,
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            self._latest_bars[symbol] = result
            return result
        except Exception as exc:
            logger.warning("_fetch_latest_bar_rest %s failed: %s", symbol, exc)
            return None

    def _fetch_latest_quote_rest(self, symbol: str) -> Optional[dict]:
        from alpaca.data.requests import StockLatestQuoteRequest

        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.client.data.get_stock_latest_quote(req)
            quote = quotes.get(symbol)
            if quote is None:
                return None
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            mid = (bid + ask) / 2.0
            result = {
                "symbol": symbol,
                "timestamp": quote.timestamp,
                "bid": bid,
                "ask": ask,
                "spread_pct": (ask - bid) / mid if mid > 0 else 0.0,
            }
            self._latest_quotes[symbol] = result
            return result
        except Exception as exc:
            logger.warning("_fetch_latest_quote_rest %s failed: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Timeframe parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timeframe(timeframe: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        mapping = {
            "1Min":  TimeFrame.Minute,
            "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day":  TimeFrame.Day,
        }
        tf = mapping.get(timeframe)
        if tf is None:
            logger.warning(
                "Unknown timeframe '%s' — defaulting to 1Day.", timeframe
            )
            tf = TimeFrame.Day
        return tf
