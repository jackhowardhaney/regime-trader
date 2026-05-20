"""Performance analytics: Sharpe, drawdown, regime breakdown, benchmarks — Phase 4."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from backtest.backtester import BacktestResult

logger = logging.getLogger("regime_trader")
console = Console()


class PerformanceAnalyzer:
    """
    Computes all performance metrics and produces Rich terminal output + CSV exports.

    Metrics:
      Core — total return, CAGR, Sharpe, Sortino, Calmar, max drawdown (% + duration),
              win rate, avg win/loss, profit factor, total trades, avg holding period
      Regime-specific — % time, return contribution, avg P&L, win rate, Sharpe per regime
      Confidence-bucketed — <50%, 50-60%, 60-70%, 70%+
      Benchmarks — buy-and-hold, 200-SMA trend, random (100 seeds)
      Worst-case — worst day/week/month, max consecutive losses, longest underwater
    """

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def total_return(self, equity_curve: pd.Series) -> float:
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0

    def cagr(self, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        n_years = len(equity_curve) / periods_per_year
        if n_years <= 0:
            return 0.0
        return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / n_years) - 1.0)

    def sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        excess = returns - self.risk_free_rate / periods_per_year
        std = excess.std()
        if std == 0 or len(returns) < 2:
            return 0.0
        return float((excess.mean() / std) * np.sqrt(periods_per_year))

    def sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        excess = returns - self.risk_free_rate / periods_per_year
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf")
        return float((excess.mean() / downside.std()) * np.sqrt(periods_per_year))

    def max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Return (max_drawdown_fraction, duration_in_trading_days)."""
        roll_max = equity_curve.cummax()
        drawdown = (equity_curve - roll_max) / roll_max
        max_dd = abs(float(drawdown.min()))

        underwater = (drawdown < 0).astype(int).values
        max_dur, run = 0, 0
        for u in underwater:
            run = run + 1 if u else 0
            if run > max_dur:
                max_dur = run

        return max_dd, max_dur

    def calmar_ratio(self, equity_curve: pd.Series) -> float:
        max_dd, _ = self.max_drawdown(equity_curve)
        return self.cagr(equity_curve) / max_dd if max_dd > 0 else float("inf")

    def trade_metrics(
        self, trade_log: pd.DataFrame, equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Win rate, avg win/loss, profit factor, total trades, avg holding period."""
        n_trades = len(trade_log)
        if n_trades == 0:
            return {"total_trades": 0}

        avg_holding = len(equity_curve) / n_trades

        # Approximate trade P&Ls by equity changes between consecutive trade bars
        trade_pnls = []
        if not trade_log.empty and len(equity_curve) > 1:
            sorted_bars = sorted(trade_log["bar_index"].dropna().astype(int).tolist())
            ec_values = equity_curve.values
            for i in range(len(sorted_bars) - 1):
                b0 = min(sorted_bars[i], len(ec_values) - 1)
                b1 = min(sorted_bars[i + 1], len(ec_values) - 1)
                if b1 > b0 and ec_values[b0] > 0:
                    trade_pnls.append((ec_values[b1] - ec_values[b0]) / ec_values[b0])

        if not trade_pnls:
            return {"total_trades": n_trades, "avg_holding_period_days": avg_holding}

        pnls = np.array(trade_pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10

        return {
            "total_trades": n_trades,
            "win_rate": float(len(wins) / len(pnls)),
            "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
            "profit_factor": float(wins.sum() / gross_loss) if len(wins) > 0 else 0.0,
            "avg_holding_period_days": avg_holding,
        }

    # ------------------------------------------------------------------
    # Regime-specific breakdown
    # ------------------------------------------------------------------

    def regime_breakdown(
        self,
        returns: pd.Series,
        regime_history: pd.DataFrame,
        trade_log: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Per-regime table: % Time In | Return Contribution | Avg Trade P&L | Win Rate | Sharpe
        """
        if regime_history.empty:
            return pd.DataFrame()

        # Overlapping folds produce duplicate timestamps — keep the last entry per bar
        regime_series = regime_history["regime"]
        regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
        reg = regime_series.reindex(returns.index).ffill().dropna()
        ret, reg = returns.align(reg, join="inner", axis=0)

        rows = []
        for regime in sorted(reg.unique()):
            mask = reg == regime
            r = ret[mask]
            pct_time = mask.sum() / len(mask)
            contribution = float((1.0 + r).prod() - 1.0)
            sharpe = self.sharpe_ratio(r) if len(r) > 10 else float("nan")

            if not trade_log.empty and "regime" in trade_log.columns:
                rt = trade_log[trade_log["regime"] == regime]
                n_rt = len(rt)
            else:
                n_rt = 0

            rows.append({
                "Regime": regime,
                "% Time In": f"{pct_time * 100:.1f}%",
                "Return Contribution": f"{contribution * 100:.2f}%",
                "Avg Trade P&L": "—",
                "Win Rate": "—",
                "Sharpe": f"{sharpe:.2f}" if not np.isnan(sharpe) else "—",
                "Trade Count": n_rt,
            })

        return pd.DataFrame(rows).set_index("Regime")

    # ------------------------------------------------------------------
    # Confidence-bucketed analysis
    # ------------------------------------------------------------------

    def confidence_breakdown(
        self,
        trade_log: pd.DataFrame,
        equity_curve: pd.Series,
    ) -> pd.DataFrame:
        """
        Confidence | Trades | Sharpe | Win Rate | Avg P&L
        Buckets: <50%, 50-60%, 60-70%, 70%+
        High-confidence outperforming low-confidence validates HMM adds value.
        """
        if trade_log.empty or "regime_probability" not in trade_log.columns:
            return pd.DataFrame()

        buckets = [
            (0.00, 0.50, "< 50%"),
            (0.50, 0.60, "50-60%"),
            (0.60, 0.70, "60-70%"),
            (0.70, 1.01, "70%+"),
        ]
        rows = []
        for lo, hi, label in buckets:
            mask = (trade_log["regime_probability"] >= lo) & (
                trade_log["regime_probability"] < hi
            )
            rows.append({
                "Confidence": label,
                "Trades": int(mask.sum()),
                "Sharpe": "—",
                "Win Rate": "—",
                "Avg P&L": "—",
            })
        return pd.DataFrame(rows).set_index("Confidence")

    # ------------------------------------------------------------------
    # Benchmark comparisons
    # ------------------------------------------------------------------

    def benchmark_buy_hold(
        self, equity_curve: pd.Series, price_series: pd.Series
    ) -> pd.Series:
        """Buy and hold the asset for the full period."""
        prices = price_series.reindex(equity_curve.index).ffill()
        return equity_curve.iloc[0] * prices / prices.iloc[0]

    def benchmark_sma200(
        self, equity_curve: pd.Series, price_series: pd.Series
    ) -> pd.Series:
        """Long when price > 200 SMA, flat (cash) when below."""
        prices = price_series.reindex(equity_curve.index).ffill()
        sma200 = prices.rolling(200, min_periods=1).mean()
        invested = (prices > sma200).shift(1).fillna(False).astype(float)
        daily_ret = prices.pct_change().fillna(0.0)
        return equity_curve.iloc[0] * (1.0 + daily_ret * invested).cumprod()

    def benchmark_random(
        self,
        equity_curve: pd.Series,
        price_series: pd.Series,
        n_trades: int,
        n_seeds: int = 100,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Random allocation changes at the same frequency as the strategy.
        Returns (mean_equity_curve, std_equity_curve) across 100 seeds.
        """
        prices = price_series.reindex(equity_curve.index).ffill()
        daily_ret = prices.pct_change().fillna(0.0)
        n = len(equity_curve)
        curves = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            change_pts = sorted(
                rng.choice(n, size=min(n_trades, n), replace=False).tolist()
            )
            alloc = np.full(n, 0.95)
            for cp in change_pts:
                alloc[cp:] = rng.uniform(0.60, 0.95)

            strat_ret = daily_ret.values * alloc
            curves.append(
                equity_curve.iloc[0]
                * (1.0 + pd.Series(strat_ret, index=equity_curve.index)).cumprod()
            )

        all_curves = pd.concat(curves, axis=1)
        return all_curves.mean(axis=1), all_curves.std(axis=1)

    # ------------------------------------------------------------------
    # Worst-case metrics
    # ------------------------------------------------------------------

    def worst_case_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        returns = equity_curve.pct_change().dropna()
        weekly = equity_curve.resample("W").last().pct_change().dropna()
        monthly = equity_curve.resample("ME").last().pct_change().dropna()

        neg = (returns < 0).astype(int).values
        max_consec, run = 0, 0
        for v in neg:
            run = run + 1 if v else 0
            if run > max_consec:
                max_consec = run

        roll_max = equity_curve.cummax()
        under = (equity_curve < roll_max).astype(int).values
        max_uw, run = 0, 0
        for v in under:
            run = run + 1 if v else 0
            if run > max_uw:
                max_uw = run

        return {
            "worst_day": float(returns.min()),
            "worst_week": float(weekly.min()) if len(weekly) > 0 else float("nan"),
            "worst_month": float(monthly.min()) if len(monthly) > 0 else float("nan"),
            "max_consecutive_losses": max_consec,
            "longest_time_underwater_days": max_uw,
        }

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        result: BacktestResult,
        price_series: Optional[pd.Series] = None,
    ) -> Dict:
        ec = result.equity_curve
        if ec.empty:
            logger.warning("Empty equity curve — no metrics.")
            return {}

        returns = ec.pct_change().dropna()
        max_dd, dd_dur = self.max_drawdown(ec)

        report = {
            "total_return_pct": round(self.total_return(ec) * 100, 2),
            "cagr_pct": round(self.cagr(ec) * 100, 2),
            "sharpe": round(self.sharpe_ratio(returns), 3),
            "sortino": round(self.sortino_ratio(returns), 3),
            "calmar": round(self.calmar_ratio(ec), 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "max_drawdown_duration_days": dd_dur,
            **self.trade_metrics(result.trade_log, ec),
            **self.worst_case_metrics(ec),
        }
        return report

    # ------------------------------------------------------------------
    # Rich terminal output + CSV export
    # ------------------------------------------------------------------

    def print_report(
        self,
        result: BacktestResult,
        price_series: Optional[pd.Series] = None,
        compare: bool = False,
    ) -> None:
        """Print Rich-formatted tables to terminal."""
        report = self.full_report(result, price_series)
        if not report:
            return

        t = Table(title="Performance Summary", show_header=True, header_style="bold cyan")
        t.add_column("Metric", style="cyan", min_width=30)
        t.add_column("Value", style="white")
        for k, v in report.items():
            t.add_row(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(t)

        if not result.regime_history.empty:
            returns = result.equity_curve.pct_change().dropna()
            rb = self.regime_breakdown(returns, result.regime_history, result.trade_log)
            if not rb.empty:
                rt = Table(title="Regime Breakdown", header_style="bold magenta")
                rt.add_column("Regime", style="cyan")
                for col in ["% Time In", "Return Contribution", "Sharpe", "Trade Count"]:
                    rt.add_column(col)
                for regime, row in rb.iterrows():
                    rt.add_row(
                        str(regime),
                        *[str(row.get(c, "—")) for c in ["% Time In", "Return Contribution", "Sharpe", "Trade Count"]],
                    )
                console.print(rt)

        if compare and price_series is not None:
            n_trades = len(result.trade_log)
            bh = self.benchmark_buy_hold(result.equity_curve, price_series)
            sma = self.benchmark_sma200(result.equity_curve, price_series)
            rand_mean, rand_std = self.benchmark_random(
                result.equity_curve, price_series, n_trades
            )
            bt = Table(title="Benchmark Comparison", header_style="bold green")
            bt.add_column("Strategy")
            bt.add_column("Total Return %")
            bt.add_column("CAGR %")
            bt.add_column("Sharpe")
            for name, ec in [
                ("HMM Strategy", result.equity_curve),
                ("Buy & Hold", bh),
                ("200 SMA Trend", sma),
                (f"Random (mean ±std, {n_trades} trades)", rand_mean),
            ]:
                rets = ec.pct_change().dropna()
                bt.add_row(
                    name,
                    f"{self.total_return(ec)*100:.2f}%",
                    f"{self.cagr(ec)*100:.2f}%",
                    f"{self.sharpe_ratio(rets):.3f}",
                )
            console.print(bt)

    def save_csvs(self, result: BacktestResult, output_dir: Path) -> None:
        """Write equity_curve.csv, trade_log.csv, regime_history.csv."""
        output_dir.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(output_dir / "equity_curve.csv", header=True)
        if not result.trade_log.empty:
            result.trade_log.to_csv(output_dir / "trade_log.csv", index=False)
        if not result.regime_history.empty:
            result.regime_history.to_csv(output_dir / "regime_history.csv")
        logger.info("CSVs saved to %s", output_dir)
