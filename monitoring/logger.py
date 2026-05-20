"""Structured JSON logging with rotating files — Phase 8.

Four log files, each rotating at 10 MB, retaining 30 backups:
  logs/main.log    — all events
  logs/trades.log  — regime_trader.trades sub-logger
  logs/alerts.log  — regime_trader.alerts sub-logger
  logs/regime.log  — regime_trader.regime sub-logger

Every JSON entry includes the live trading context: timestamp, regime,
regime_probability, equity, positions, daily_pnl. Call update_log_context()
from the live loop to keep these fields current.
"""

import json
import logging
import logging.handlers
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_LOGS_DIR = Path("logs")
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 30               # 30 backup files ≈ 30 days at typical rotation

# Shared context injected into every JSON log entry
_LOG_CONTEXT: Dict[str, Any] = {
    "regime": "UNKNOWN",
    "regime_probability": 0.0,
    "equity": 0.0,
    "positions": {},
    "daily_pnl": 0.0,
}
_CONTEXT_LOCK = threading.Lock()


def update_log_context(**kwargs: Any) -> None:
    """
    Update the fields that appear in every log entry.
    Call this from the live trading loop after each bar:

        update_log_context(
            regime="BULL",
            regime_probability=0.87,
            equity=105230.0,
            positions={"SPY": 0.95},
            daily_pnl=340.0,
        )
    """
    with _CONTEXT_LOCK:
        _LOG_CONTEXT.update(kwargs)


class _JSONFormatter(logging.Formatter):
    """Formats every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        with _CONTEXT_LOCK:
            ctx = dict(_LOG_CONTEXT)

        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "regime": ctx["regime"],
            "regime_probability": round(float(ctx["regime_probability"]), 4),
            "equity": round(float(ctx["equity"]), 2),
            "positions": ctx["positions"],
            "daily_pnl": round(float(ctx["daily_pnl"]), 2),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            entry.update(record.extra)

        return json.dumps(entry, default=str)


def _make_rotating_handler(log_path: Path) -> logging.handlers.RotatingFileHandler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(_JSONFormatter())
    return handler


def setup_logger(
    name: str = "regime_trader",
    logs_dir: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure the full regime_trader logger hierarchy.

    Calling this more than once is safe — it detects existing handlers
    and returns the already-configured logger.

    Returns the root 'regime_trader' logger.
    """
    log_dir = Path(logs_dir) if logs_dir else _LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(name)
    if root.handlers:
        return root   # already configured

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler — human-readable (not JSON)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(console)

    # main.log — receives all events from the root logger
    root.addHandler(_make_rotating_handler(log_dir / "main.log"))

    # Sub-loggers with dedicated files; each also propagates to main.log
    _sub_configs = [
        ("trades", "trades.log"),
        ("alerts", "alerts.log"),
        ("regime", "regime.log"),
    ]
    for sub_name, log_file in _sub_configs:
        sub = logging.getLogger(f"{name}.{sub_name}")
        sub.addHandler(_make_rotating_handler(log_dir / log_file))
        sub.propagate = True   # events also flow up to main.log

    return root


def get_logger(name: str = "regime_trader") -> logging.Logger:
    """Retrieve an already-configured logger by name."""
    return logging.getLogger(name)


def log_trade(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    trade_id: str,
    regime: str,
    **extra: Any,
) -> None:
    """Convenience function — writes a structured trade record to trades.log."""
    logger = logging.getLogger("regime_trader.trades")
    record = logger.makeRecord(
        logger.name, logging.INFO,
        fn="log_trade", lno=0,
        msg=f"TRADE {side} {symbol} x{qty} @ ${price:.2f} [{trade_id}]",
        args=(), exc_info=None,
    )
    record.extra = {
        "trade_id": trade_id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "regime_at_trade": regime,
        **extra,
    }
    logger.handle(record)


def log_regime_change(old_regime: str, new_regime: str, probability: float) -> None:
    """Convenience function — writes a structured regime change to regime.log."""
    logger = logging.getLogger("regime_trader.regime")
    record = logger.makeRecord(
        logger.name, logging.WARNING,
        fn="log_regime_change", lno=0,
        msg=f"REGIME CHANGE: {old_regime} → {new_regime} (p={probability:.3f})",
        args=(), exc_info=None,
    )
    record.extra = {
        "old_regime": old_regime,
        "new_regime": new_regime,
        "probability": round(probability, 4),
    }
    logger.handle(record)
