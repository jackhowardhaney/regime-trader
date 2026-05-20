"""HMM regime detection engine — volatility classifier with no look-ahead bias."""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from hmmlearn import hmm

logger = logging.getLogger("regime_trader")

# Regime label sets sorted ascending by mean return (most bearish → most bullish)
REGIME_LABELS: Dict[int, List[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}


@dataclass
class RegimeInfo:
    """Static metadata about a regime learned from training data."""
    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Current runtime state of the regime detector."""
    label: str
    state_id: int
    probability: float
    state_probabilities: np.ndarray
    timestamp: datetime
    is_confirmed: bool
    consecutive_bars: int


class HMMEngine:
    """
    Gaussian HMM volatility classifier with automatic model selection (BIC).

    Uses forward-algorithm-only inference to guarantee no look-ahead bias.
    Regime labels are sorted ascending by mean return after training for human
    readability. The strategy layer re-sorts independently by volatility.
    """

    MIN_TRAIN_BARS = 504  # minimum 2 years of daily data

    def __init__(
        self,
        n_candidates: Optional[List[int]] = None,
        n_init: int = 10,
        covariance_type: str = "full",
        stability_bars: int = 3,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
        min_confidence: float = 0.55,
    ) -> None:
        self.n_candidates = n_candidates or [3, 4, 5, 6, 7]
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.min_confidence = min_confidence

        self.model: Optional[hmm.GaussianHMM] = None
        self.n_regimes: int = 0
        self.regime_labels: List[str] = []
        self.regime_info: Dict[int, RegimeInfo] = {}
        # maps HMM internal state index → sorted label rank (0 = most bearish)
        self._state_to_label_idx: np.ndarray = np.array([])
        self.bic: float = float("inf")
        self.training_date: Optional[datetime] = None
        self.is_fitted: bool = False

        # Stability filter runtime state
        self._confirmed_regime: Optional[str] = None
        self._candidate_regime: Optional[str] = None
        self._consecutive_candidate: int = 0
        self._consecutive_confirmed: int = 0
        self._recent_changes: List[int] = []  # bar indices where confirmed changes occurred
        self._bar_count: int = 0

        # Cached forward-pass alpha vector for incremental live updates
        self._cached_alpha: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray, returns: np.ndarray) -> "HMMEngine":
        """
        Fit the HMM using automatic model selection (lowest BIC wins).

        Args:
            features: (T, n_features) standardized feature array
            returns:  (T,) log-return array used only for regime label sorting
        """
        if len(features) < self.MIN_TRAIN_BARS:
            raise ValueError(
                f"Need at least {self.MIN_TRAIN_BARS} bars, got {len(features)}"
            )

        best_model: Optional[hmm.GaussianHMM] = None
        best_bic = float("inf")
        best_n = None
        bic_scores: Dict[int, float] = {}

        for n in self.n_candidates:
            bic, model = self._fit_candidate(features, n)
            bic_scores[n] = round(bic, 2)
            logger.info("HMM candidate n_regimes=%d  BIC=%.2f", n, bic)
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n

        logger.info(
            "Model selection complete: n_regimes=%d (BIC=%.2f). All scores: %s",
            best_n, best_bic, bic_scores,
        )

        self.model = best_model
        self.n_regimes = best_n
        self.bic = best_bic
        self.training_date = datetime.utcnow()
        self.is_fitted = True
        self._cached_alpha = None

        self._assign_regime_labels(features, returns)
        self._build_regime_info(features, returns)
        return self

    def _fit_candidate(
        self, features: np.ndarray, n: int
    ) -> Tuple[float, hmm.GaussianHMM]:
        """Train n_init models for a given n; return (BIC, best model)."""
        best_ll = -np.inf
        best_model: Optional[hmm.GaussianHMM] = None

        for _ in range(self.n_init):
            candidate = hmm.GaussianHMM(
                n_components=n,
                covariance_type=self.covariance_type,
                n_iter=200,
                tol=1e-4,
                random_state=None,
            )
            try:
                candidate.fit(features)
                ll = candidate.score(features)
                logger.debug(
                    "  n=%d init attempt: ll=%.4f converged=%s iters=%d",
                    n, ll, candidate.monitor_.converged, candidate.monitor_.iter,
                )
                if ll > best_ll:
                    best_ll = ll
                    best_model = candidate
            except Exception as exc:
                logger.debug("  n=%d init failed: %s", n, exc)
                continue

        if best_model is None:
            raise RuntimeError(f"All HMM fits failed for n_components={n}")

        n_params = self._count_params(n, features.shape[1])
        bic = -2.0 * best_ll + n_params * np.log(len(features))
        logger.info(
            "  n=%d best ll=%.4f  n_params=%d  BIC=%.2f  converged=%s",
            n, best_ll, n_params, bic, best_model.monitor_.converged,
        )
        return bic, best_model

    @staticmethod
    def _count_params(n: int, n_features: int) -> int:
        """Free parameters in a full-covariance Gaussian HMM."""
        transition = n * (n - 1)
        start = n - 1
        means = n * n_features
        covars = n * n_features * (n_features + 1) // 2
        return transition + start + means + covars

    def _assign_regime_labels(self, features: np.ndarray, returns: np.ndarray) -> None:
        """
        Sort HMM states by mean return (ascending) and map to human-readable labels.
        Viterbi is used here ONLY for training-time labeling, never for live inference.
        """
        state_sequence = self.model.predict(features)
        mean_returns = np.array(
            [returns[state_sequence == s].mean() if (state_sequence == s).any() else 0.0
             for s in range(self.n_regimes)]
        )
        sorted_states = np.argsort(mean_returns)  # ascending: index 0 = most bearish state

        self._state_to_label_idx = np.empty(self.n_regimes, dtype=int)
        for rank, state in enumerate(sorted_states):
            self._state_to_label_idx[state] = rank

        self.regime_labels = REGIME_LABELS[self.n_regimes]
        label_map = {
            self.regime_labels[self._state_to_label_idx[s]]: round(float(mean_returns[s]), 5)
            for s in range(self.n_regimes)
        }
        logger.info("Regime labels assigned (label: mean_return): %s", label_map)

    def _build_regime_info(self, features: np.ndarray, returns: np.ndarray) -> None:
        """Populate RegimeInfo for each HMM state."""
        state_sequence = self.model.predict(features)
        for state_id in range(self.n_regimes):
            mask = state_sequence == state_id
            label_idx = int(self._state_to_label_idx[state_id])
            label = self.regime_labels[label_idx]
            exp_ret = float(returns[mask].mean()) if mask.any() else 0.0
            exp_vol = float(returns[mask].std() * np.sqrt(252)) if mask.any() else 0.0

            if label in ("CRASH", "STRONG_BEAR", "BEAR"):
                strategy_type, max_lev, max_pos, min_conf = "defensive", 0.5, 0.10, 0.65
            elif label in ("NEUTRAL", "WEAK_BEAR", "WEAK_BULL"):
                strategy_type, max_lev, max_pos, min_conf = "moderate", 0.8, 0.15, 0.55
            else:
                strategy_type, max_lev, max_pos, min_conf = "aggressive", 1.25, 0.25, 0.50

            self.regime_info[state_id] = RegimeInfo(
                regime_id=state_id,
                regime_name=label,
                expected_return=exp_ret,
                expected_volatility=exp_vol,
                recommended_strategy_type=strategy_type,
                max_leverage_allowed=max_lev,
                max_position_size_pct=max_pos,
                min_confidence_to_act=min_conf,
            )

    # ------------------------------------------------------------------
    # Inference — forward algorithm ONLY (no look-ahead bias)
    # ------------------------------------------------------------------

    def predict_regime_filtered(self, features_up_to_now: np.ndarray) -> np.ndarray:
        """
        Compute P(state_t | observations_1:t) using the forward algorithm.
        Uses ONLY past and present data. No future data.

        DO NOT use model.predict() here — Viterbi processes the entire sequence
        and revises past states using future observations (look-ahead bias).

        Returns an integer array of length T with label indices.
        """
        if not self.is_fitted:
            raise RuntimeError("HMMEngine must be fitted before inference.")

        T = len(features_up_to_now)
        alphas = np.zeros((T, self.n_regimes))

        # 1. alpha_0 = startprob * emission_prob(obs_0)
        alphas[0] = self.model.startprob_ * self._emission_prob(features_up_to_now[0])
        alphas[0] /= alphas[0].sum() + 1e-300

        for t in range(1, T):
            # 2. alpha_t = (alpha_{t-1} @ transmat) * emission_prob(obs_t)
            prior = alphas[t - 1] @ self.model.transmat_
            alphas[t] = prior * self._emission_prob(features_up_to_now[t])
            # 3. Normalize at each step
            norm = alphas[t].sum()
            alphas[t] /= norm + 1e-300

        # 4. Cache final alpha for incremental live updates
        self._cached_alpha = alphas[-1].copy()

        raw_states = np.argmax(alphas, axis=1)
        return np.array([int(self._state_to_label_idx[s]) for s in raw_states])

    def predict_regime_filtered_incremental(
        self, obs: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        Update the forward pass with one new observation using cached alpha.
        O(n_regimes^2) per step — efficient for the live trading loop.

        Returns (label_idx, state_probabilities).
        """
        if not self.is_fitted:
            raise RuntimeError("HMMEngine must be fitted before inference.")

        if self._cached_alpha is None:
            alpha = self.model.startprob_ * self._emission_prob(obs)
        else:
            alpha = (self._cached_alpha @ self.model.transmat_) * self._emission_prob(obs)

        norm = alpha.sum()
        alpha /= norm + 1e-300
        self._cached_alpha = alpha.copy()

        raw_state = int(np.argmax(alpha))
        label_idx = int(self._state_to_label_idx[raw_state])
        return label_idx, alpha

    def predict_regime_proba(self, features_up_to_now: np.ndarray) -> np.ndarray:
        """
        Return (T, n_regimes) forward-filtered state probability array.
        Same forward algorithm as predict_regime_filtered; no look-ahead.
        """
        if not self.is_fitted:
            raise RuntimeError("HMMEngine must be fitted before inference.")

        T = len(features_up_to_now)
        alphas = np.zeros((T, self.n_regimes))
        alphas[0] = self.model.startprob_ * self._emission_prob(features_up_to_now[0])
        alphas[0] /= alphas[0].sum() + 1e-300

        for t in range(1, T):
            prior = alphas[t - 1] @ self.model.transmat_
            alphas[t] = prior * self._emission_prob(features_up_to_now[t])
            alphas[t] /= alphas[t].sum() + 1e-300

        return alphas

    def _emission_prob(self, obs: np.ndarray) -> np.ndarray:
        """Gaussian log-space emission probabilities for each HMM state."""
        n_features = len(obs)
        probs = np.zeros(self.n_regimes)
        for s in range(self.n_regimes):
            diff = obs - self.model.means_[s]
            cov = self.model.covars_[s]
            try:
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    probs[s] = 1e-300
                    continue
                inv_cov = np.linalg.inv(cov)
                exponent = -0.5 * float(diff @ inv_cov @ diff)
                log_norm = -0.5 * (n_features * np.log(2 * np.pi) + logdet)
                probs[s] = np.exp(log_norm + exponent)
            except np.linalg.LinAlgError:
                probs[s] = 1e-300
        return probs

    # ------------------------------------------------------------------
    # Stability filter
    # ------------------------------------------------------------------

    def update_stability_filter(
        self, label_idx: int, proba: np.ndarray
    ) -> RegimeState:
        """
        Apply the regime stability filter to a raw forward-pass label.

        - A regime change is only confirmed after persisting stability_bars consecutive bars.
        - During a pending transition, the previous (confirmed) regime is returned
          with is_confirmed=False so the strategy layer can reduce position sizes by 25%.
        - Tracks flicker rate; if flicker > threshold, forces uncertainty mode.
        """
        self._bar_count += 1
        label = self.regime_labels[label_idx]
        probability = float(np.max(proba))

        if self._confirmed_regime is None:
            self._confirmed_regime = label
            self._candidate_regime = label
            self._consecutive_candidate = 1
            self._consecutive_confirmed = 1

        if label == self._candidate_regime:
            self._consecutive_candidate += 1
        else:
            self._candidate_regime = label
            self._consecutive_candidate = 1

        is_confirmed = False
        if self._candidate_regime == self._confirmed_regime:
            self._consecutive_confirmed += 1
            is_confirmed = True
        else:
            # Regime change pending
            if self._consecutive_candidate >= self.stability_bars:
                old = self._confirmed_regime
                self._confirmed_regime = self._candidate_regime
                self._consecutive_confirmed = self._consecutive_candidate
                self._recent_changes.append(self._bar_count)
                is_confirmed = True
                logger.warning(
                    "Regime change CONFIRMED: %s → %s  (p=%.3f, consecutive=%d)",
                    old, self._confirmed_regime, probability, self._consecutive_confirmed,
                )
            else:
                logger.info(
                    "Regime transition PENDING: %s → %s  (%d/%d bars)",
                    self._confirmed_regime, self._candidate_regime,
                    self._consecutive_candidate, self.stability_bars,
                )

        consecutive = (
            self._consecutive_confirmed if is_confirmed else self._consecutive_candidate
        )

        return RegimeState(
            label=self._confirmed_regime,
            state_id=self.regime_labels.index(self._confirmed_regime),
            probability=probability,
            state_probabilities=proba.copy(),
            timestamp=datetime.utcnow(),
            is_confirmed=is_confirmed,
            consecutive_bars=consecutive,
        )

    def get_regime_stability(self) -> int:
        """Consecutive bars in the current confirmed regime."""
        return self._consecutive_confirmed

    def get_transition_matrix(self) -> np.ndarray:
        """Learned HMM transition probability matrix."""
        if not self.is_fitted:
            raise RuntimeError("HMMEngine must be fitted first.")
        return self.model.transmat_.copy()

    def detect_regime_change(self) -> bool:
        """True only on the bar a regime change is first confirmed."""
        return (
            self._confirmed_regime == self._candidate_regime
            and self._consecutive_confirmed == self.stability_bars
        )

    def get_regime_flicker_rate(self) -> int:
        """Number of confirmed regime changes within the last flicker_window bars."""
        cutoff = self._bar_count - self.flicker_window
        return sum(1 for b in self._recent_changes if b >= cutoff)

    def is_flickering(self) -> bool:
        """True if the flicker rate exceeds the configured threshold."""
        return self.get_regime_flicker_rate() > self.flicker_threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Pickle the full engine to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(
            "HMM engine saved: %s  (n_regimes=%d, BIC=%.2f, trained=%s)",
            path, self.n_regimes, self.bic, self.training_date,
        )

    @classmethod
    def load(cls, path: Path) -> "HMMEngine":
        """Load a pickled HMMEngine from disk."""
        with open(path, "rb") as f:
            engine: HMMEngine = pickle.load(f)
        logger.info(
            "HMM engine loaded: %s  (n_regimes=%d, trained=%s)",
            path, engine.n_regimes, engine.training_date,
        )
        return engine
