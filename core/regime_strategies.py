"""Volatility-based allocation strategies — Phase 3.

DESIGN: The HMM detects VOLATILITY ENVIRONMENTS, not direction.
- Low vol  → fully invested (calm markets trend up ~70% of the time)
- Mid vol  → stay invested if trend intact, reduce if trend broken
- High vol → reduce but stay partially long (catch V-shaped rebounds)

ALWAYS LONG. NEVER SHORT. The correct response to high vol is REDUCING
allocation, not reversing direction.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState

logger = logging.getLogger("regime_trader")


# ------------------------------------------------------------------
# Signal dataclass
# ------------------------------------------------------------------

@dataclass
class Signal:
    symbol: str
    direction: str                   # "LONG" or "FLAT"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size_pct: float         # 0.60 to 0.95
    leverage: float                  # 1.0 or 1.25
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: datetime
    reasoning: str
    strategy_name: str
    metadata: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# BaseStrategy ABC
# ------------------------------------------------------------------

class BaseStrategy(ABC):
    """Abstract base class for all volatility-regime strategies."""

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """Generate a trading signal for `symbol`. Returns None if no action."""
        ...

    @staticmethod
    def _ema(prices: pd.Series, span: int) -> float:
        return float(prices.ewm(span=span, adjust=False).mean().iloc[-1])

    @staticmethod
    def _atr(bars: pd.DataFrame, period: int = 14) -> float:
        high, low, close = bars["high"], bars["low"], bars["close"]
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        return float(tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])


# ------------------------------------------------------------------
# Three strategy classes — selected by volatility rank, not label
# ------------------------------------------------------------------

class LowVolBullStrategy(BaseStrategy):
    """
    Lowest-third volatility regime: be fully invested.

    Calm markets trend upward most of the time. Use modest leverage.
    Allocation: 95%  |  Leverage: 1.25x
    Stop: max(price - 3*ATR, 50 EMA - 0.5*ATR)
    """

    NAME = "LowVolBullStrategy"
    ALLOCATION = 0.95
    LEVERAGE = 1.25

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        bars = bars.rename(columns=str.lower)
        price = float(bars["close"].iloc[-1])
        atr = self._atr(bars)
        ema50 = self._ema(bars["close"], 50)
        stop_loss = max(price - 3.0 * atr, ema50 - 0.5 * atr)

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=self.ALLOCATION,
            leverage=self.LEVERAGE,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=datetime.utcnow(),
            reasoning=(
                f"Low-vol regime ({regime_state.label}): calm market, fully invested "
                f"at {self.ALLOCATION*100:.0f}% with {self.LEVERAGE}x leverage."
            ),
            strategy_name=self.NAME,
        )


class MidVolCautiousStrategy(BaseStrategy):
    """
    Middle-third volatility regime: stay invested if trend intact, reduce if not.

    price > 50 EMA → 95% allocation, 1.0x (trend intact, stay invested)
    price < 50 EMA → 60% allocation, 1.0x (trend broken, reduce)
    Stop: 50 EMA - 0.5*ATR
    """

    NAME = "MidVolCautiousStrategy"
    LEVERAGE = 1.0
    ALLOCATION_TREND = 0.95
    ALLOCATION_NO_TREND = 0.60

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        bars = bars.rename(columns=str.lower)
        price = float(bars["close"].iloc[-1])
        atr = self._atr(bars)
        ema50 = self._ema(bars["close"], 50)

        trend_intact = price > ema50
        allocation = self.ALLOCATION_TREND if trend_intact else self.ALLOCATION_NO_TREND
        stop_loss = ema50 - 0.5 * atr
        trend_desc = "trend intact" if trend_intact else "trend broken"

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=allocation,
            leverage=self.LEVERAGE,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=datetime.utcnow(),
            reasoning=(
                f"Mid-vol regime ({regime_state.label}): {trend_desc}. "
                f"Allocation {allocation*100:.0f}% at {self.LEVERAGE}x leverage."
            ),
            strategy_name=self.NAME,
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """
    Highest-third volatility regime: reduce but stay partially invested.

    Never short — staying 60% long catches the sharp V-shaped rebounds
    that happen fast and which the HMM is 2-3 days late detecting.
    Allocation: 60%  |  Leverage: 1.0x
    Stop: 50 EMA - 1.0*ATR (wider stop for volatile conditions)
    """

    NAME = "HighVolDefensiveStrategy"
    ALLOCATION = 0.60
    LEVERAGE = 1.0

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        bars = bars.rename(columns=str.lower)
        price = float(bars["close"].iloc[-1])
        atr = self._atr(bars)
        ema50 = self._ema(bars["close"], 50)
        stop_loss = ema50 - 1.0 * atr

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=self.ALLOCATION,
            leverage=self.LEVERAGE,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=datetime.utcnow(),
            reasoning=(
                f"High-vol regime ({regime_state.label}): defensive. "
                f"Reduced to {self.ALLOCATION*100:.0f}% to limit drawdown; "
                f"staying LONG for rebound."
            ),
            strategy_name=self.NAME,
        )


# ------------------------------------------------------------------
# Backward-compatible aliases
# ------------------------------------------------------------------

CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

# Maps every possible regime label → strategy class.
# Independent of the volatility-rank sort used by StrategyOrchestrator —
# this is a convenience fallback when no RegimeInfo vol data is available.
LABEL_TO_STRATEGY: Dict[str, type] = {
    "CRASH":       HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "BEAR":        HighVolDefensiveStrategy,
    "WEAK_BEAR":   MidVolCautiousStrategy,
    "NEUTRAL":     MidVolCautiousStrategy,
    "WEAK_BULL":   MidVolCautiousStrategy,
    "BULL":        LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA":    LowVolBullStrategy,
}


# ------------------------------------------------------------------
# StrategyOrchestrator
# ------------------------------------------------------------------

class StrategyOrchestrator:
    """
    Maps HMM regime_ids → strategy classes via volatility rank.

    Sort order is by expected_volatility (ascending), INDEPENDENT of the
    label sort (which is by return). "BULL" does not mean low vol.

    Volatility rank mapping:
      position = vol_rank / (n_regimes - 1)   # 0.0 = lowest, 1.0 = highest
      position <= 0.33  → LowVolBullStrategy
      position >= 0.67  → HighVolDefensiveStrategy
      else              → MidVolCautiousStrategy
    """

    def __init__(self, config: dict, regime_infos: Dict[int, RegimeInfo]) -> None:
        self.config = config
        self.min_confidence: float = (
            config.get("hmm", {}).get("min_confidence", 0.55)
        )
        self.rebalance_threshold: float = (
            config.get("strategy", {}).get("rebalance_threshold", 0.10)
        )
        self._strategy_map: Dict[int, BaseStrategy] = {}
        self.update_regime_infos(regime_infos)

    def update_regime_infos(self, regime_infos: Dict[int, RegimeInfo]) -> None:
        """Rebuild the regime_id → strategy mapping after an HMM retrain."""
        if not regime_infos:
            return

        n = len(regime_infos)
        sorted_ids = sorted(
            regime_infos.keys(),
            key=lambda rid: regime_infos[rid].expected_volatility,
        )

        self._strategy_map = {}
        for vol_rank, regime_id in enumerate(sorted_ids):
            position = vol_rank / max(n - 1, 1)

            if position <= 0.33:
                strategy: BaseStrategy = LowVolBullStrategy()
            elif position >= 0.67:
                strategy = HighVolDefensiveStrategy()
            else:
                strategy = MidVolCautiousStrategy()

            self._strategy_map[regime_id] = strategy
            logger.info(
                "Regime %d (%s)  vol_rank=%d  position=%.2f  → %s",
                regime_id,
                regime_infos[regime_id].regime_name,
                vol_rank,
                position,
                strategy.NAME,
            )

    def generate_signals(
        self,
        symbols: List[str],
        bars: Dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool,
    ) -> List[Signal]:
        """
        Generate signals for all symbols using the strategy for the current regime.

        Uncertainty mode activates when prob < min_confidence or is_flickering:
        - halve all position sizes
        - force leverage to 1.0x
        - append "[UNCERTAINTY — size halved]" to reasoning
        """
        strategy = self._strategy_map.get(regime_state.state_id)
        if strategy is None:
            logger.warning(
                "No strategy mapped for regime_id=%d; defaulting to HighVolDefensive.",
                regime_state.state_id,
            )
            strategy = HighVolDefensiveStrategy()

        uncertainty = (
            regime_state.probability < self.min_confidence or is_flickering
        )

        signals: List[Signal] = []
        for symbol in symbols:
            symbol_bars = bars.get(symbol)
            if symbol_bars is None or len(symbol_bars) < 60:
                continue

            signal = strategy.generate_signal(symbol, symbol_bars, regime_state)
            if signal is None:
                continue

            if uncertainty:
                signal.position_size_pct = round(signal.position_size_pct * 0.5, 4)
                signal.leverage = 1.0
                signal.reasoning += " [UNCERTAINTY — size halved]"

            signals.append(signal)

        return signals

    def needs_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> bool:
        """
        True if any symbol's weight differs from target by > rebalance_threshold.
        Prevents churn from minor probability fluctuations.
        """
        all_symbols = set(current_weights) | set(target_weights)
        return any(
            abs(target_weights.get(sym, 0.0) - current_weights.get(sym, 0.0))
            > self.rebalance_threshold
            for sym in all_symbols
        )


# Ergonomic alias so existing imports of RegimeStrategyManager still work
RegimeStrategyManager = StrategyOrchestrator
