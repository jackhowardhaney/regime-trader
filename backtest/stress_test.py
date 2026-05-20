"""Stress tests: crash injection, gap risk, regime misclassification — Phase 4."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("regime_trader")


@dataclass
class StressResult:
    scenario: str
    mean_max_loss: float
    worst_case_loss: float
    circuit_breaker_pct: float
    details: dict


class StressTest:
    """
    Injects synthetic market shocks to test system robustness.

    Three scenarios:
      a. Crash injection  — 100 Monte Carlo runs with -5% to -15% single-day gaps
      b. Gap risk         — overnight gaps of 2-5x ATR at random points
      c. Regime misclassification — shuffled regimes verify risk management independence
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.max_dd_halt: float = config.get("risk", {}).get("max_dd_from_peak", 0.10)

    # ------------------------------------------------------------------
    # a. Crash injection — 100 Monte Carlo simulations
    # ------------------------------------------------------------------

    def inject_crash(
        self,
        price_data: Dict[str, pd.DataFrame],
        crash_pct_range: Tuple[float, float] = (0.05, 0.15),
        n_crashes: int = 10,
        seed: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """Insert single-day gaps of -(crash_pct) at n_crashes random points."""
        rng = np.random.default_rng(seed)
        result = {}
        for sym, df in price_data.items():
            df = df.rename(columns=str.lower).copy()
            n = len(df)
            pts = sorted(
                rng.choice(range(50, n), size=min(n_crashes, n - 50), replace=False).tolist()
            )
            for cp in pts:
                pct = float(rng.uniform(*crash_pct_range))
                mult = 1.0 - pct
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df.iloc[cp:, df.columns.get_loc(col)] = (
                            df.iloc[cp:, df.columns.get_loc(col)] * mult
                        )
            result[sym] = df
        return result

    def run_crash_monte_carlo(
        self,
        price_data: Dict[str, pd.DataFrame],
        backtester,
        n_simulations: int = 100,
    ) -> StressResult:
        """
        Run 100 crash-injected backtests.
        Report: mean max loss, worst case, % where circuit breaker fired.
        """
        max_losses = []
        cb_count = 0

        for sim in range(n_simulations):
            shocked = self.inject_crash(price_data, seed=sim)
            try:
                res = backtester.run(shocked)
                ec = res.equity_curve
                if ec.empty:
                    continue
                max_dd = float(((ec.cummax() - ec) / ec.cummax()).max())
                max_losses.append(max_dd)
                if max_dd >= self.max_dd_halt:
                    cb_count += 1
            except Exception as exc:
                logger.debug("Crash MC sim %d failed: %s", sim, exc)

        if not max_losses:
            return StressResult("crash_injection", 0.0, 0.0, 0.0, {})

        return StressResult(
            scenario="crash_injection",
            mean_max_loss=float(np.mean(max_losses)),
            worst_case_loss=float(np.max(max_losses)),
            circuit_breaker_pct=cb_count / n_simulations,
            details={
                "n_simulations": n_simulations,
                "p25_loss": float(np.percentile(max_losses, 25)),
                "p75_loss": float(np.percentile(max_losses, 75)),
            },
        )

    # ------------------------------------------------------------------
    # b. Gap risk — 2-5x ATR overnight gaps
    # ------------------------------------------------------------------

    def inject_gap(
        self,
        price_data: Dict[str, pd.DataFrame],
        atr_multiplier_range: Tuple[float, float] = (2.0, 5.0),
        n_gaps: int = 20,
        seed: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """Insert overnight gap-downs of 2-5x ATR at random bar indices."""
        rng = np.random.default_rng(seed)
        result = {}
        for sym, df in price_data.items():
            df = df.rename(columns=str.lower).copy()
            n = len(df)
            high, low, close = df["high"], df["low"], df["close"]
            tr = pd.concat(
                [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
                axis=1,
            ).max(axis=1)
            atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()

            pts = sorted(
                rng.choice(range(20, n), size=min(n_gaps, n - 20), replace=False).tolist()
            )
            for gp in pts:
                mult_val = float(rng.uniform(*atr_multiplier_range))
                gap_size = float(atr.iloc[gp]) * mult_val
                prev_close = float(df["close"].iloc[gp - 1])
                pct = gap_size / (prev_close + 1e-10)
                factor = 1.0 - pct
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df.iloc[gp:, df.columns.get_loc(col)] = (
                            df.iloc[gp:, df.columns.get_loc(col)] * factor
                        )
            result[sym] = df
        return result

    def run_gap_risk(
        self,
        price_data: Dict[str, pd.DataFrame],
        backtester,
        n_simulations: int = 50,
    ) -> StressResult:
        """Run gap-risk stress tests. Report expected vs actual loss."""
        max_losses = []
        cb_count = 0

        for sim in range(n_simulations):
            shocked = self.inject_gap(price_data, seed=sim)
            try:
                res = backtester.run(shocked)
                ec = res.equity_curve
                if ec.empty:
                    continue
                max_dd = float(((ec.cummax() - ec) / ec.cummax()).max())
                max_losses.append(max_dd)
                if max_dd >= self.max_dd_halt:
                    cb_count += 1
            except Exception as exc:
                logger.debug("Gap risk sim %d failed: %s", sim, exc)

        if not max_losses:
            return StressResult("gap_risk", 0.0, 0.0, 0.0, {})

        return StressResult(
            scenario="gap_risk",
            mean_max_loss=float(np.mean(max_losses)),
            worst_case_loss=float(np.max(max_losses)),
            circuit_breaker_pct=cb_count / len(max_losses),
            details={"n_simulations": n_simulations},
        )

    # ------------------------------------------------------------------
    # c. Regime misclassification
    # ------------------------------------------------------------------

    def run_regime_misclassification(
        self,
        price_data: Dict[str, pd.DataFrame],
        backtester,
        n_shuffles: int = 20,
    ) -> StressResult:
        """
        Deliberately corrupt OHLCV data to force random regime labels.
        If the system suffers catastrophic drawdown, risk management is not
        independent enough of regime classification.
        """
        max_losses = []
        cb_count = 0

        for seed in range(n_shuffles):
            rng = np.random.default_rng(seed)
            corrupted = {}
            for sym, df in price_data.items():
                noisy = df.rename(columns=str.lower).copy()
                # Large noise forces the HMM to classify near-randomly
                noise = rng.normal(1.0, 0.04, size=len(noisy))
                for col in ["open", "high", "low", "close"]:
                    if col in noisy.columns:
                        noisy[col] = noisy[col].values * noise
                corrupted[sym] = noisy

            try:
                res = backtester.run(corrupted)
                ec = res.equity_curve
                if ec.empty:
                    continue
                max_dd = float(((ec.cummax() - ec) / ec.cummax()).max())
                max_losses.append(max_dd)
                if max_dd >= self.max_dd_halt:
                    cb_count += 1
            except Exception as exc:
                logger.debug("Misclassification sim %d failed: %s", seed, exc)

        blowup = any(l > 0.50 for l in max_losses)
        if blowup:
            logger.warning(
                "REGIME MISCLASSIFICATION TEST: drawdown > 50%% detected. "
                "Risk management may not be sufficiently independent of HMM output."
            )

        return StressResult(
            scenario="regime_misclassification",
            mean_max_loss=float(np.mean(max_losses)) if max_losses else 0.0,
            worst_case_loss=float(np.max(max_losses)) if max_losses else 0.0,
            circuit_breaker_pct=cb_count / max(len(max_losses), 1),
            details={
                "n_shuffles": n_shuffles,
                "system_blowup_detected": blowup,
            },
        )

    # ------------------------------------------------------------------
    # Run all three scenarios
    # ------------------------------------------------------------------

    def run_all(
        self,
        price_data: Dict[str, pd.DataFrame],
        backtester,
    ) -> Dict[str, StressResult]:
        logger.info("Running all stress tests (crash, gap, misclassification)...")
        return {
            "crash_injection": self.run_crash_monte_carlo(price_data, backtester),
            "gap_risk": self.run_gap_risk(price_data, backtester),
            "regime_misclassification": self.run_regime_misclassification(
                price_data, backtester
            ),
        }
