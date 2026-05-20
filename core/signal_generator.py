"""Combines HMM regime detection + strategy allocations into tradeable signals."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .hmm_engine import HMMEngine
from .regime_strategies import RegimeStrategyManager
from .risk_manager import RiskManager


@dataclass
class Signal:
    regime: str
    regime_label: int
    target_weights: Dict[str, float]
    risk_adjusted_weights: Dict[str, float]
    drawdown_breach: bool


class SignalGenerator:
    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy_manager: RegimeStrategyManager,
        risk_manager: RiskManager,
    ):
        self.hmm = hmm_engine
        self.strategy = strategy_manager
        self.risk = risk_manager

    def generate(
        self,
        features: np.ndarray,
        realized_vols: Dict[str, float],
        portfolio_value: float,
        portfolio_vol: float,
    ) -> Signal:
        regime_label = self.hmm.predict_current(features)
        regime_name = self.hmm.regime_name(regime_label)

        target_weights = self.strategy.compute_weights(regime_name, realized_vols)
        risk_weights = self.risk.apply_risk_limits(target_weights, portfolio_value, portfolio_vol)

        self.risk.update_peak(portfolio_value)

        return Signal(
            regime=regime_name,
            regime_label=regime_label,
            target_weights=target_weights,
            risk_adjusted_weights=risk_weights,
            drawdown_breach=self.risk.is_drawdown_breach(portfolio_value),
        )
