"""Structured logging setup."""

import logging
from pathlib import Path


def setup_logger(
    name: str = "regime_trader",
    log_file: str | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """Return a logger with console + optional file handlers and structured formatting."""
    ...


def get_logger(name: str = "regime_trader") -> logging.Logger:
    """Retrieve an already-configured logger by name."""
    ...
