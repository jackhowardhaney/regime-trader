"""Technical indicators and feature computation for the HMM engine.

All top-level functions are pure: they take pandas Series/DataFrames and return
Series/DataFrames. FeatureEngineer.build_hmm_features() assembles the full
14-feature matrix and applies rolling z-score standardization.
"""

from typing import List

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Pure feature functions
# ------------------------------------------------------------------

def log_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    """Log returns over `period` bars."""
    return np.log(prices / prices.shift(period))


def realized_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility: rolling std of log returns * sqrt(252)."""
    return log_returns(prices).rolling(window).std() * np.sqrt(252)


def vol_ratio(prices: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """Short-window realized vol divided by long-window realized vol."""
    rets = log_returns(prices)
    short_vol = rets.rolling(short).std()
    long_vol = rets.rolling(long).std()
    return short_vol / (long_vol + 1e-10)


def normalized_volume(volume: pd.Series, window: int = 50) -> pd.Series:
    """Volume z-score vs its rolling mean (window-period)."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / (std + 1e-10)


def volume_trend(volume: pd.Series, sma_window: int = 10) -> pd.Series:
    """First difference of the volume SMA — proxy for slope of the 10-period SMA."""
    return volume.rolling(sma_window).mean().diff()


def _true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Wilder True Range: max(H-L, |H-C_prev|, |L-C_prev|)."""
    return pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average Directional Index (Wilder smoothing, period=14)."""
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_dm = high.diff()
    minus_dm = -(low.diff())
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def sma_slope(prices: pd.Series, window: int = 50) -> pd.Series:
    """First difference of the rolling SMA (slope proxy)."""
    return prices.rolling(window).mean().diff()


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_zscore(prices: pd.Series, period: int = 14, z_window: int = 252) -> pd.Series:
    """RSI(14) standardized as a rolling z-score over z_window bars."""
    rsi_vals = rsi(prices, period)
    return rolling_zscore(rsi_vals, z_window)


def distance_from_sma(prices: pd.Series, window: int = 200) -> pd.Series:
    """(price - SMA) / price — distance from the 200-period SMA as a fraction of price."""
    sma = prices.rolling(window).mean()
    return (prices - sma) / (prices + 1e-10)


def rate_of_change(prices: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change: (price_t / price_{t-period}) - 1."""
    return prices / prices.shift(period) - 1.0


def normalized_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """14-period ATR divided by close price."""
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr / (close + 1e-10)


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Standardize a series using a rolling z-score (window-period lookback)."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-10)


# ------------------------------------------------------------------
# FeatureEngineer — assembles the full HMM feature matrix
# ------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms raw OHLCV data into the 14-feature matrix used by HMMEngine.

    All features are standardized with a 252-period rolling z-score so the HMM
    sees stationary, unit-variance inputs regardless of market level.
    """

    FEATURE_NAMES: List[str] = [
        "ret_1",
        "ret_5",
        "ret_20",
        "realized_vol_20",
        "vol_ratio_5_20",
        "norm_volume_50",
        "volume_trend_10",
        "adx_14",
        "sma50_slope",
        "rsi14_zscore",
        "dist_sma200",
        "roc_10",
        "roc_20",
        "norm_atr_14",
    ]

    def build_hmm_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 14 features from an OHLCV DataFrame and standardize each
        with a 252-period rolling z-score. Rows with NaN are dropped.

        Expected columns: open, high, low, close, volume (case-insensitive).
        Returns a DataFrame aligned to ohlcv's index with NaN rows removed.
        """
        ohlcv = ohlcv.rename(columns=str.lower)
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]
        volume = ohlcv["volume"]

        # Compute log returns once and reuse across all return-derived features.
        # realized_volatility and vol_ratio each call log_returns(close) internally
        # too, but isolating those computations here avoids redundant rolling windows.
        log_ret_1 = np.log(close / close.shift(1))

        raw = pd.DataFrame(index=ohlcv.index)

        # Returns
        raw["ret_1"] = log_ret_1
        raw["ret_5"] = np.log(close / close.shift(5))
        raw["ret_20"] = np.log(close / close.shift(20))

        # Volatility (build from the pre-computed 1-period returns)
        raw["realized_vol_20"] = log_ret_1.rolling(20).std() * np.sqrt(252)
        short_vol = log_ret_1.rolling(5).std()
        long_vol = log_ret_1.rolling(20).std()
        raw["vol_ratio_5_20"] = short_vol / (long_vol + 1e-10)

        # Volume
        raw["norm_volume_50"] = normalized_volume(volume, 50)
        raw["volume_trend_10"] = volume_trend(volume, 10)

        # Trend (ADX and normalized ATR share a single True Range computation)
        tr = _true_range(high, low, close)
        period = 14
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

        plus_dm = high.diff()
        minus_dm = -(low.diff())
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-10)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        raw["adx_14"] = dx.ewm(alpha=1.0 / period, adjust=False).mean()
        raw["norm_atr_14"] = atr / (close + 1e-10)

        raw["sma50_slope"] = sma_slope(close, 50)

        # Mean reversion
        raw["rsi14_zscore"] = rsi_zscore(close, 14, 252)
        raw["dist_sma200"] = distance_from_sma(close, 200)

        # Momentum
        raw["roc_10"] = rate_of_change(close, 10)
        raw["roc_20"] = rate_of_change(close, 20)

        # Standardize all features with rolling z-scores (252-period lookback)
        mean = raw.rolling(252).mean()
        std = raw.rolling(252).std()
        standardized = (raw - mean) / (std + 1e-10)

        return standardized.dropna()

    # ------------------------------------------------------------------
    # Individual accessors (for testing / external inspection)
    # ------------------------------------------------------------------

    def compute_returns(self, prices: pd.Series, period: int = 1) -> pd.Series:
        """Log returns over `period` bars."""
        return log_returns(prices, period)

    def compute_realized_vol(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Annualized realized volatility."""
        return realized_volatility(prices, window)

    def compute_volume_ratio(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume z-score vs rolling mean."""
        return normalized_volume(volume, window)

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Wilder RSI."""
        return rsi(prices, period)

    def compute_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """MACD line, signal line, and histogram."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        })
