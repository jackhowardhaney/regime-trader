"""Combines HMM regime detection + strategy allocations into validated signals — Phase 5 ready."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .hmm_engine import HMMEngine, RegimeState
from .regime_strategies import Signal, StrategyOrchestrator
from .risk_manager import PortfolioState, RiskDecision, RiskManager


class SignalGenerator:
    """
    Orchestrates the full signal pipeline for one bar:
      1. HMM forward inference → regime label + probabilities
      2. Stability filter → RegimeState
      3. Strategy orchestrator → List[Signal]
      4. Risk validation → List[RiskDecision]

    Used by the live trading loop. WalkForwardBacktester runs the same
    pipeline inline for walk-forward efficiency.
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy_manager: StrategyOrchestrator,
        risk_manager: RiskManager,
    ) -> None:
        self.hmm = hmm_engine
        self.strategy = strategy_manager
        self.risk = risk_manager

    def generate(
        self,
        symbols: List[str],
        features_history: np.ndarray,
        bars_history: Dict[str, pd.DataFrame],
        portfolio_state: PortfolioState,
    ) -> Tuple[List[RiskDecision], RegimeState]:
        """
        Run the full pipeline for one bar.

        Args:
            symbols:           list of symbols to generate signals for
            features_history:  (T, n_features) array — all features up to and
                               including the current bar (no future data)
            bars_history:      {symbol: OHLCV DataFrame} up to current bar
            portfolio_state:   current account snapshot for risk checks

        Returns:
            (decisions, regime_state) — decisions is a list of RiskDecision
            objects, one per symbol. approved=True means safe to execute.
        """
        # 1. Forward inference — strictly no look-ahead
        label_sequence = self.hmm.predict_regime_filtered(features_history)
        proba_matrix = self.hmm.predict_regime_proba(features_history)
        label_idx = int(label_sequence[-1])
        proba = proba_matrix[-1]

        # 2. Stability filter
        regime_state = self.hmm.update_stability_filter(label_idx, proba)

        # 3. Generate strategy signals
        signals: List[Signal] = self.strategy.generate_signals(
            symbols, bars_history, regime_state, self.hmm.is_flickering()
        )

        # 4. Inject live flicker rate and CB status before validation
        portfolio_state.flicker_rate = self.hmm.get_regime_flicker_rate()
        portfolio_state.circuit_breaker_status = self.risk.circuit_breaker.status

        decisions: List[RiskDecision] = [
            self.risk.validate_signal(signal, portfolio_state)
            for signal in signals
        ]

        return decisions, regime_state
