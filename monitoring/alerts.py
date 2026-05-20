"""Email and webhook alerts for critical trading events."""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Alert:
    level: str  # "INFO" | "WARNING" | "CRITICAL"
    title: str
    message: str
    timestamp: datetime


class AlertManager:
    """Dispatches alerts via email and/or Slack webhook with rate-limiting."""

    def __init__(
        self,
        smtp_config: dict | None = None,
        slack_webhook: str | None = None,
        rate_limit_minutes: int = 15,
    ) -> None:
        self.smtp_config = smtp_config
        self.slack_webhook = slack_webhook
        self.rate_limit_minutes = rate_limit_minutes
        self._last_sent: dict[str, datetime] = {}

    def send(self, alert: Alert) -> None:
        """Route the alert to configured channels, respecting the rate limit."""
        ...

    def drawdown_alert(self, drawdown_pct: float, threshold_pct: float) -> None:
        """Fire a CRITICAL alert when drawdown exceeds a threshold."""
        ...

    def regime_change_alert(self, old_regime: str, new_regime: str) -> None:
        """Fire an INFO alert on regime transitions."""
        ...

    def order_error_alert(self, symbol: str, error: str) -> None:
        """Fire a WARNING alert when an order fails."""
        ...

    def _send_email(self, alert: Alert) -> None:
        """Send alert via SMTP."""
        ...

    def _send_slack(self, alert: Alert) -> None:
        """POST alert to Slack webhook."""
        ...
