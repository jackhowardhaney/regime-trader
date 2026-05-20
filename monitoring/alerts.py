"""Alert system for critical trading events — Phase 8.

Triggers:
  regime_change, circuit_breaker, large_pnl, data_feed_down,
  api_lost, hmm_retrained, flicker_exceeded

Delivery:
  console (always) | log file (always) | email (optional) | webhook (optional)

Rate limit: 1 alert per event type per 15 minutes.
"""

import json
import logging
import smtplib
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Dict, Optional

logger = logging.getLogger("regime_trader.alerts")

_LARGE_PNL_THRESHOLD = 0.02    # 2% daily P&L swing triggers an alert


@dataclass
class Alert:
    event_type: str              # machine-readable key for rate-limiting
    level: str                   # "INFO" | "WARNING" | "CRITICAL"
    title: str
    message: str
    timestamp: datetime
    metadata: dict = None        # extra structured data

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AlertManager:
    """
    Dispatches alerts via configured channels with per-event-type rate limiting.

    Usage:
        am = AlertManager(
            smtp_config={...},          # optional
            webhook_url="https://...",  # optional — Slack or generic
            rate_limit_minutes=15,
        )
        am.alert_regime_change("BULL", "BEAR", probability=0.89)
        am.alert_circuit_breaker("halt_daily", drawdown_pct=3.1)
    """

    def __init__(
        self,
        smtp_config: Optional[dict] = None,
        webhook_url: Optional[str] = None,
        rate_limit_minutes: int = 15,
    ) -> None:
        self.smtp_config = smtp_config or {}
        self.webhook_url = webhook_url
        self.rate_limit_minutes = rate_limit_minutes
        self._last_sent: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Named trigger methods
    # ------------------------------------------------------------------

    def alert_regime_change(
        self, old_regime: str, new_regime: str, probability: float = 0.0
    ) -> None:
        self._send(Alert(
            event_type="regime_change",
            level="INFO",
            title=f"Regime Change: {old_regime} → {new_regime}",
            message=(
                f"The HMM confirmed a regime transition from {old_regime} to "
                f"{new_regime} (p={probability:.1%}). Strategy allocation will "
                f"adjust on the next bar."
            ),
            timestamp=datetime.utcnow(),
            metadata={
                "old_regime": old_regime,
                "new_regime": new_regime,
                "probability": round(probability, 4),
            },
        ))

    def alert_circuit_breaker(
        self, cb_status: str, drawdown_pct: float = 0.0
    ) -> None:
        level = "CRITICAL" if "halt" in cb_status else "WARNING"
        self._send(Alert(
            event_type="circuit_breaker",
            level=level,
            title=f"Circuit Breaker: {cb_status.upper()}",
            message=(
                f"Circuit breaker triggered: status={cb_status}, "
                f"drawdown={drawdown_pct:.2f}%. "
                + ("ALL TRADING HALTED." if "halt" in cb_status
                   else "Position sizes reduced 50%.")
            ),
            timestamp=datetime.utcnow(),
            metadata={"status": cb_status, "drawdown_pct": round(drawdown_pct, 3)},
        ))

    def alert_large_pnl(self, pnl_pct: float, equity: float = 0.0) -> None:
        """Fire when daily P&L exceeds ±2% of equity."""
        if abs(pnl_pct) < _LARGE_PNL_THRESHOLD * 100:
            return
        direction = "gain" if pnl_pct > 0 else "loss"
        level = "INFO" if pnl_pct > 0 else "WARNING"
        self._send(Alert(
            event_type="large_pnl",
            level=level,
            title=f"Large Daily {direction.title()}: {pnl_pct:+.2f}%",
            message=(
                f"Daily P&L {direction}: {pnl_pct:+.2f}% "
                f"(equity=${equity:,.0f}). "
                f"Threshold: ±{_LARGE_PNL_THRESHOLD * 100:.0f}%."
            ),
            timestamp=datetime.utcnow(),
            metadata={"pnl_pct": round(pnl_pct, 3), "equity": round(equity, 2)},
        ))

    def alert_data_feed_down(
        self, symbols: list, duration_sec: float = 0.0
    ) -> None:
        self._send(Alert(
            event_type="data_feed_down",
            level="WARNING",
            title="Data Feed Down",
            message=(
                f"No bar received for {duration_sec:.0f}s on {symbols}. "
                f"Signals paused; stops remain active."
            ),
            timestamp=datetime.utcnow(),
            metadata={"symbols": symbols, "duration_sec": round(duration_sec, 1)},
        ))

    def alert_api_lost(self, error_msg: str = "") -> None:
        self._send(Alert(
            event_type="api_lost",
            level="CRITICAL",
            title="Alpaca API Connection Lost",
            message=(
                f"Lost connection to Alpaca API. Reconnect in progress. "
                f"Error: {error_msg}"
            ),
            timestamp=datetime.utcnow(),
            metadata={"error": error_msg},
        ))

    def alert_hmm_retrained(
        self, n_regimes: int = 0, bic: float = 0.0, symbol: str = ""
    ) -> None:
        self._send(Alert(
            event_type="hmm_retrained",
            level="INFO",
            title=f"HMM Retrained: {symbol}",
            message=(
                f"HMM model retrained for {symbol}. "
                f"Selected n_regimes={n_regimes} (BIC={bic:.0f})."
            ),
            timestamp=datetime.utcnow(),
            metadata={"n_regimes": n_regimes, "bic": round(bic, 2), "symbol": symbol},
        ))

    def alert_flicker_exceeded(
        self, flicker_rate: int = 0, threshold: int = 4
    ) -> None:
        self._send(Alert(
            event_type="flicker_exceeded",
            level="WARNING",
            title=f"Regime Flicker Exceeded: {flicker_rate}/{threshold}",
            message=(
                f"Regime flicker rate {flicker_rate} exceeds threshold {threshold}. "
                f"Uncertainty mode active — leverage forced to 1.0x."
            ),
            timestamp=datetime.utcnow(),
            metadata={"flicker_rate": flicker_rate, "threshold": threshold},
        ))

    # Legacy names from Phase 1 stub kept as aliases
    def drawdown_alert(self, drawdown_pct: float, threshold_pct: float) -> None:
        self.alert_circuit_breaker("reduce_peak", drawdown_pct=drawdown_pct)

    def regime_change_alert(self, old_regime: str, new_regime: str) -> None:
        self.alert_regime_change(old_regime, new_regime)

    def order_error_alert(self, symbol: str, error: str) -> None:
        self._send(Alert(
            event_type="order_error",
            level="WARNING",
            title=f"Order Error: {symbol}",
            message=f"Order failed for {symbol}: {error}",
            timestamp=datetime.utcnow(),
            metadata={"symbol": symbol, "error": error},
        ))

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def _send(self, alert: Alert) -> None:
        """Route alert to all configured channels, respecting rate limit."""
        if self._is_rate_limited(alert.event_type):
            logger.debug(
                "Alert '%s' suppressed by rate limit (%dm)",
                alert.event_type, self.rate_limit_minutes,
            )
            return

        self._last_sent[alert.event_type] = datetime.utcnow()

        self._send_console(alert)
        self._send_log(alert)

        if self.smtp_config:
            try:
                self._send_email(alert)
            except Exception as exc:
                logger.warning("Email alert failed: %s", exc)

        if self.webhook_url:
            try:
                self._send_webhook(alert)
            except Exception as exc:
                logger.warning("Webhook alert failed: %s", exc)

    def _is_rate_limited(self, event_type: str) -> bool:
        last = self._last_sent.get(event_type)
        if last is None:
            return False
        return (datetime.utcnow() - last) < timedelta(minutes=self.rate_limit_minutes)

    # ------------------------------------------------------------------
    # Delivery channels
    # ------------------------------------------------------------------

    def _send_console(self, alert: Alert) -> None:
        level_colors = {
            "INFO": "\033[36m",      # cyan
            "WARNING": "\033[33m",   # yellow
            "CRITICAL": "\033[31m",  # red
        }
        reset = "\033[0m"
        color = level_colors.get(alert.level, "")
        print(
            f"{color}[{alert.level}] {alert.timestamp.strftime('%H:%M:%S')} "
            f"— {alert.title}{reset}"
        )
        print(f"  {alert.message}")

    def _send_log(self, alert: Alert) -> None:
        level_map = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "CRITICAL": logging.CRITICAL,
        }
        log_level = level_map.get(alert.level, logging.INFO)
        record = logger.makeRecord(
            logger.name, log_level,
            fn="_send_log", lno=0,
            msg=f"ALERT [{alert.event_type}] {alert.title}: {alert.message}",
            args=(), exc_info=None,
        )
        record.extra = {
            "event_type": alert.event_type,
            "alert_title": alert.title,
            **alert.metadata,
        }
        logger.handle(record)

    def _send_email(self, alert: Alert) -> None:
        """Send via SMTP. smtp_config keys: host, port, user, password, to."""
        cfg = self.smtp_config
        subject = f"[regime-trader] {alert.level}: {alert.title}"
        body = (
            f"Time:    {alert.timestamp.isoformat()}\n"
            f"Level:   {alert.level}\n"
            f"Event:   {alert.event_type}\n\n"
            f"{alert.message}\n\n"
            f"Metadata: {json.dumps(alert.metadata, indent=2)}"
        )
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = cfg.get("user", "regime-trader@localhost")
        msg["To"] = cfg.get("to", "")

        with smtplib.SMTP(cfg.get("host", "localhost"), cfg.get("port", 587)) as smtp:
            smtp.ehlo()
            if cfg.get("tls", True):
                smtp.starttls()
            if cfg.get("user") and cfg.get("password"):
                smtp.login(cfg["user"], cfg["password"])
            smtp.sendmail(msg["From"], [msg["To"]], msg.as_string())

        logger.debug("Email alert sent to %s: %s", cfg.get("to"), subject)

    def _send_webhook(self, alert: Alert) -> None:
        """
        POST alert to webhook URL.
        Auto-detects Slack format (URL contains 'hooks.slack.com').
        All others receive a generic JSON payload.
        """
        if "hooks.slack.com" in (self.webhook_url or ""):
            payload = {
                "text": f"*[{alert.level}] {alert.title}*\n{alert.message}",
                "attachments": [{
                    "color": (
                        "danger" if alert.level == "CRITICAL"
                        else "warning" if alert.level == "WARNING"
                        else "good"
                    ),
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in alert.metadata.items()
                    ],
                }],
            }
        else:
            payload = {
                "event_type": alert.event_type,
                "level": alert.level,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"Webhook returned HTTP {resp.status}")

        logger.debug("Webhook alert sent: %s", alert.title)
