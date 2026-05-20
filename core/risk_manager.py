"""Position sizing, leverage control, and drawdown limits."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskParams:
    max_portfolio_drawdown: float = 0.15
    max_position_size: float = 0.25
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.05
    volatility_target: float = 0.12


class RiskManager:
    def __init__(self, params: RiskParams | None = None, initial_capital: float = 100_000):
        self.params = params or RiskParams()
        self.initial_capital = initial_capital
        self.peak_value = initial_capital

    def update_peak(self, portfolio_value: float) -> None:
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

    def current_drawdown(self, portfolio_value: float) -> float:
        return (self.peak_value - portfolio_value) / self.peak_value

    def is_drawdown_breach(self, portfolio_value: float) -> bool:
        return self.current_drawdown(portfolio_value) >= self.params.max_portfolio_drawdown

    def clamp_position_size(self, weight: float) -> float:
        return min(weight, self.params.max_position_size)

    def volatility_scalar(self, realized_vol: float) -> float:
        """Scale exposure so portfolio vol targets params.volatility_target."""
        if realized_vol <= 0:
            return 1.0
        return min(self.params.volatility_target / realized_vol, self.params.max_leverage)

    def apply_risk_limits(
        self, weights: Dict[str, float], portfolio_value: float, portfolio_vol: float
    ) -> Dict[str, float]:
        if self.is_drawdown_breach(portfolio_value):
            return {"CASH": 1.0}

        scalar = self.volatility_scalar(portfolio_vol)
        scaled = {}
        for asset, w in weights.items():
            if asset == "CASH":
                scaled[asset] = w
            else:
                scaled[asset] = self.clamp_position_size(w * scalar)

        total_equity = sum(v for k, v in scaled.items() if k != "CASH")
        scaled["CASH"] = max(0.0, 1.0 - total_equity)
        return scaled

    def stop_loss_triggered(self, entry_price: float, current_price: float) -> bool:
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self.params.stop_loss_pct
