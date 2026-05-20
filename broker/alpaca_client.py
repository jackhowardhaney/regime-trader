"""Alpaca broker client — alpaca-py SDK wrapper — Phase 6."""

import logging
import sys
import time
from typing import Dict, List, Optional

logger = logging.getLogger("regime_trader")

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

_MAX_BACKOFF = 60  # seconds


class AlpacaClient:
    """
    alpaca-py SDK wrapper.

    Credentials come exclusively from .env — NEVER hardcoded. Paper trading
    is the default. Switching to live requires typing the full confirmation
    phrase to prevent accidental live execution.

    Usage:
        client = AlpacaClient(api_key=..., secret_key=..., paper=True)
        client.connect()   # health check + optional live confirmation
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.base_url = base_url or (PAPER_URL if paper else LIVE_URL)
        self._trading_client = None
        self._data_client = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Initialize SDK clients, run health check, and confirm live mode
        interactively if paper=False.
        """
        if not self.paper:
            self._confirm_live_mode()
        self._init_clients()
        self._health_check()
        self._connected = True
        logger.info("AlpacaClient connected (%s)", "PAPER" if self.paper else "LIVE")

    def _confirm_live_mode(self) -> None:
        """Require explicit typed phrase before enabling live trading."""
        print(
            "\n⚠️  LIVE TRADING MODE. "
            "Type 'YES I UNDERSTAND THE RISKS' to confirm: ",
            end="",
            flush=True,
        )
        answer = input().strip()
        if answer != "YES I UNDERSTAND THE RISKS":
            logger.critical("Live trading confirmation rejected. Exiting.")
            sys.exit(1)
        logger.warning("LIVE TRADING MODE CONFIRMED.")

    def _init_clients(self) -> None:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        self._trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

    def _health_check(self) -> None:
        """Verify connectivity by fetching the account. Raises on failure."""
        try:
            acct = self._trading_client.get_account()
            logger.info(
                "Health check OK — status=%s equity=$%s buying_power=$%s",
                acct.status, acct.equity, acct.buying_power,
            )
        except Exception as exc:
            raise ConnectionError(f"Alpaca health check failed: {exc}") from exc

    def reconnect(self, max_attempts: int = 10) -> None:
        """
        Re-initialize clients with exponential backoff.
        Delays: 1, 2, 4, 8, 16, 32, 60, 60, … seconds.
        """
        self._connected = False
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info("Reconnect attempt %d / %d …", attempt, max_attempts)
                self._init_clients()
                self._health_check()
                self._connected = True
                logger.info("Reconnected on attempt %d.", attempt)
                return
            except Exception as exc:
                logger.warning("Reconnect %d failed: %s", attempt, exc)
                time.sleep(min(delay, _MAX_BACKOFF))
                delay *= 2

        raise RuntimeError(
            f"AlpacaClient failed to reconnect after {max_attempts} attempts."
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return account info as a plain dict."""
        acct = self._trading_client.get_account()
        return {
            "id": str(acct.id),
            "status": str(acct.status),
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "pattern_day_trader": bool(acct.pattern_day_trader),
            "trading_blocked": bool(acct.trading_blocked),
            "account_blocked": bool(acct.account_blocked),
        }

    def get_positions(self) -> List[dict]:
        """Return a list of open position dicts."""
        positions = self._trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": str(p.side),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]

    def get_order_history(self, limit: int = 100) -> List[dict]:
        """Return recent orders (all statuses) as plain dicts."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        orders = self._trading_client.get_orders(filter=req)
        return [
            {
                "id": str(o.id),
                "client_order_id": str(o.client_order_id),
                "symbol": o.symbol,
                "side": str(o.side),
                "type": str(o.order_type),
                "qty": float(o.qty or 0),
                "filled_qty": float(o.filled_qty or 0),
                "filled_avg_price": float(o.filled_avg_price or 0),
                "status": str(o.status),
                "created_at": str(o.created_at),
                "filled_at": str(o.filled_at) if o.filled_at else None,
            }
            for o in orders
        ]

    def get_available_margin(self) -> float:
        """Return available buying power (includes margin if enabled)."""
        acct = self._trading_client.get_account()
        return float(acct.buying_power)

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Return True if the US equities market is currently open."""
        return bool(self._trading_client.get_clock().is_open)

    def get_clock(self) -> dict:
        """Return market clock as a plain dict."""
        clock = self._trading_client.get_clock()
        return {
            "is_open": bool(clock.is_open),
            "timestamp": str(clock.timestamp),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }

    # ------------------------------------------------------------------
    # SDK handle properties (used by OrderExecutor and MarketDataFetcher)
    # ------------------------------------------------------------------

    @property
    def trading(self):
        """Raw TradingClient for OrderExecutor."""
        return self._trading_client

    @property
    def data(self):
        """Raw StockHistoricalDataClient for MarketDataFetcher."""
        return self._data_client
