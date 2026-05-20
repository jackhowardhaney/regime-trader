"""Web dashboard for regime-trader — run with: streamlit run dashboard_web.py"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Regime Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
LIVE_STATE = ROOT / "live_state.json"
SETTINGS = ROOT / "config" / "settings.yaml"
LOCK_FILE = ROOT / "trading_halted.lock"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

REGIME_COLORS = {
    "BULL": "#22c55e",
    "STRONG_BULL": "#16a34a",
    "WEAK_BULL": "#86efac",
    "EUPHORIA": "#a78bfa",
    "NEUTRAL": "#94a3b8",
    "WEAK_BEAR": "#fca5a5",
    "BEAR": "#f87171",
    "STRONG_BEAR": "#ef4444",
    "CRASH": "#dc2626",
    "UNKNOWN": "#64748b",
}

REGIME_ORDER = [
    "CRASH", "STRONG_BEAR", "BEAR", "WEAK_BEAR",
    "NEUTRAL",
    "WEAK_BULL", "BULL", "STRONG_BULL", "EUPHORIA",
]


def load_equity_curve() -> Optional[pd.DataFrame]:
    f = RESULTS_DIR / "equity_curve.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
    df.columns = ["equity"]
    return df


def load_trade_log() -> Optional[pd.DataFrame]:
    f = RESULTS_DIR / "trade_log.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    return df


def load_regime_history() -> Optional[pd.DataFrame]:
    f = RESULTS_DIR / "regime_history.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
    return df


def load_live_state() -> Optional[dict]:
    if not LIVE_STATE.exists():
        return None
    try:
        return json.loads(LIVE_STATE.read_text())
    except Exception:
        return None


def load_settings() -> dict:
    if SETTINGS.exists():
        return yaml.safe_load(SETTINGS.read_text())
    return {}


def fmt_pct(v: float, decimals: int = 2) -> str:
    color = "green" if v >= 0 else "red"
    sign = "+" if v >= 0 else ""
    return f":{color}[{sign}{v:.{decimals}f}%]"


def fmt_dollar(v: float) -> str:
    color = "green" if v >= 0 else "red"
    sign = "+" if v >= 0 else ""
    return f":{color}[{sign}${v:,.2f}]"


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("Regime Trader")
    st.caption("HMM Volatility-Regime Bot")

    page = st.radio(
        "View",
        ["Backtest Results", "Live Trading", "Settings"],
        index=0,
    )

    st.divider()

    settings = load_settings()
    broker = settings.get("broker", {})
    st.caption(f"**Symbols:** {', '.join(broker.get('symbols', []))}")
    st.caption(f"**Timeframe:** {broker.get('timeframe', '1Day')}")
    st.caption(f"**Mode:** {'PAPER' if broker.get('paper_trading') else 'LIVE'}")

    if LOCK_FILE.exists():
        st.error("TRADING HALTED — delete `trading_halted.lock` to resume")

    st.divider()
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    if st.button("Refresh now"):
        st.rerun()


# ==================================================================
# Page 1: Backtest Results
# ==================================================================

if page == "Backtest Results":
    st.header("Backtest Results")

    ec = load_equity_curve()
    tl = load_trade_log()
    rh = load_regime_history()

    if ec is None:
        st.info(
            "No backtest results found. Run:\n"
            "```\npython3 main.py backtest --symbols SPY QQQ "
            "--start 2019-01-01 --end 2024-12-31 --compare\n```"
        )
        st.stop()

    # ── Summary metrics row ──────────────────────────────────────
    start_eq = ec["equity"].iloc[0]
    end_eq = ec["equity"].iloc[-1]
    total_ret = (end_eq / start_eq - 1) * 100
    days = (ec.index[-1] - ec.index[0]).days
    years = days / 365.25
    cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100 if years > 0 else 0
    rets = ec["equity"].pct_change().dropna()
    rf = 0.045 / 252
    excess = rets - rf
    sharpe = (excess.mean() / excess.std()) * (252 ** 0.5) if excess.std() > 0 else 0
    roll_max = ec["equity"].cummax()
    max_dd = ((ec["equity"] - roll_max) / roll_max).min() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return", f"{total_ret:+.2f}%", delta_color="normal")
    c2.metric("CAGR", f"{cagr:+.2f}%")
    c3.metric("Sharpe", f"{sharpe:.3f}")
    c4.metric("Max Drawdown", f"{max_dd:.2f}%")
    c5.metric("Final Equity", f"${end_eq:,.0f}")

    # ── Equity curve chart ───────────────────────────────────────
    st.subheader("Equity Curve")

    fig = go.Figure()

    # Strategy line
    fig.add_trace(go.Scatter(
        x=ec.index, y=ec["equity"],
        name="HMM Strategy",
        line=dict(color="#6366f1", width=2),
    ))

    # Drawdown shading
    fig.add_trace(go.Scatter(
        x=ec.index, y=roll_max,
        name="Peak Equity",
        line=dict(color="#94a3b8", width=1, dash="dot"),
        opacity=0.6,
    ))

    # Shade by regime if history available
    if rh is not None and "regime" in rh.columns:
        rh_aligned = rh["regime"].reindex(ec.index, method="ffill")
        for regime in rh_aligned.unique():
            if pd.isna(regime):
                continue
            mask = rh_aligned == regime
            color = REGIME_COLORS.get(str(regime), "#94a3b8")
            # Find contiguous blocks
            blocks = mask.astype(int).diff().fillna(0)
            starts = ec.index[blocks == 1].tolist()
            ends = ec.index[blocks == -1].tolist()
            if mask.iloc[0]:
                starts = [ec.index[0]] + starts
            if mask.iloc[-1]:
                ends = ends + [ec.index[-1]]
            for s, e in zip(starts, ends):
                fig.add_vrect(
                    x0=s, x1=e,
                    fillcolor=color,
                    opacity=0.08,
                    layer="below",
                    line_width=0,
                )

    # Trade markers
    if tl is not None:
        buys = tl[tl["shares_delta"] > 0]
        sells = tl[tl["shares_delta"] < 0]
        if not buys.empty:
            buy_eq = [
                float(ec["equity"].asof(pd.Timestamp(ts)))
                for ts in buys["timestamp"]
            ]
            fig.add_trace(go.Scatter(
                x=buys["timestamp"], y=buy_eq,
                mode="markers",
                name="Buy / Increase",
                marker=dict(symbol="triangle-up", size=8, color="#22c55e"),
            ))
        if not sells.empty:
            sell_eq = [
                float(ec["equity"].asof(pd.Timestamp(ts)))
                for ts in sells["timestamp"]
            ]
            fig.add_trace(go.Scatter(
                x=sells["timestamp"], y=sell_eq,
                mode="markers",
                name="Sell / Reduce",
                marker=dict(symbol="triangle-down", size=8, color="#ef4444"),
            ))

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b", showgrid=True),
        yaxis=dict(gridcolor="#1e293b", showgrid=True, tickprefix="$", tickformat=","),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Drawdown chart ───────────────────────────────────────────
    with st.expander("Drawdown Chart"):
        dd_series = ((ec["equity"] - roll_max) / roll_max) * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series,
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.3)",
            line=dict(color="#ef4444", width=1),
            name="Drawdown %",
        ))
        fig_dd.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
            showlegend=False,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Regime timeline ──────────────────────────────────────────
    if rh is not None and "regime" in rh.columns:
        st.subheader("Regime Timeline")
        rh_aligned = rh["regime"].reindex(ec.index, method="ffill").dropna()

        counts = rh_aligned.value_counts()
        regime_names = [r for r in REGIME_ORDER if r in counts.index]
        colors = [REGIME_COLORS.get(r, "#94a3b8") for r in regime_names]

        fig_r = go.Figure(go.Bar(
            x=[counts[r] for r in regime_names],
            y=regime_names,
            orientation="h",
            marker_color=colors,
            text=[f"{counts[r]/len(rh_aligned)*100:.1f}%" for r in regime_names],
            textposition="outside",
        ))
        fig_r.update_layout(
            height=250,
            margin=dict(l=0, r=80, t=10, b=0),
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e293b", title="Trading Days"),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Trade log ────────────────────────────────────────────────
    if tl is not None and not tl.empty:
        st.subheader("Trade Log")
        display = tl.copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d")
        display["old_allocation"] = (display["old_allocation"] * 100).round(1).astype(str) + "%"
        display["new_allocation"] = (display["new_allocation"] * 100).round(1).astype(str) + "%"
        display["fill_price"] = display["fill_price"].map("${:,.2f}".format)
        display["slippage_cost"] = display["slippage_cost"].map("${:,.2f}".format)
        display["regime_probability"] = (display["regime_probability"] * 100).round(1).astype(str) + "%"
        cols = ["timestamp", "symbol", "regime", "regime_probability",
                "old_allocation", "new_allocation", "shares_delta",
                "fill_price", "slippage_cost", "strategy"]
        st.dataframe(
            display[[c for c in cols if c in display.columns]],
            use_container_width=True,
            hide_index=True,
        )

    # ── Run new backtest ─────────────────────────────────────────
    st.divider()
    st.subheader("Run Backtest")
    col_sym, col_start, col_end = st.columns(3)
    syms = col_sym.text_input("Symbols (space-separated)", value="SPY QQQ")
    start = col_start.text_input("Start date", value="2019-01-01")
    end = col_end.text_input("End date", value="2024-12-31")
    compare = st.checkbox("Include benchmark comparison", value=True)

    if st.button("Run Backtest", type="primary"):
        cmd = [
            sys.executable, "-W", "ignore", str(ROOT / "main.py"),
            "backtest", "--symbols", *syms.split(),
            "--start", start, "--end", end,
            "--output-dir", str(RESULTS_DIR),
        ]
        if compare:
            cmd.append("--compare")
        with st.spinner("Running backtest…"):
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        if result.returncode == 0:
            st.success("Backtest complete — refresh to see updated results.")
        else:
            st.error("Backtest failed:")
            st.code(result.stderr[-3000:])


# ==================================================================
# Page 2: Live Trading
# ==================================================================

elif page == "Live Trading":
    st.header("Live Trading")

    state = load_live_state()

    if state is None:
        st.info(
            "No live session running. Start paper trading with:\n"
            "```\npython3 main.py live --paper\n```\n"
            "The dashboard refreshes from `live_state.json` — "
            "enable **Auto-refresh** in the sidebar to track live updates."
        )
    else:
        ts = state.get("timestamp", "unknown")
        st.caption(f"Last update: {ts}")

        # Regime panel
        regime = state.get("regime", "UNKNOWN")
        prob = state.get("regime_probability", 0) * 100
        stability = state.get("stability_bars", 0)
        flickering = state.get("is_flickering", False)
        color = REGIME_COLORS.get(regime, "#94a3b8")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Current Regime", regime)
        r2.metric("Confidence", f"{prob:.1f}%")
        r3.metric("Stability", f"{stability} bars")
        r4.metric("Flickering", "YES" if flickering else "no")

        st.divider()

        # Portfolio metrics
        equity = state.get("equity", 0)
        daily_pnl = state.get("daily_pnl", 0)
        weekly_pnl = state.get("weekly_pnl", 0)
        peak_equity = state.get("peak_equity", equity)
        peak_dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Equity", f"${equity:,.2f}")
        p2.metric("Daily P&L", f"${daily_pnl:+,.2f}", delta=f"{daily_pnl/equity*100:+.2f}%" if equity > 0 else "")
        p3.metric("Weekly P&L", f"${weekly_pnl:+,.2f}")
        p4.metric("Drawdown from Peak", f"{peak_dd:.2f}%")

        st.divider()

        # CB status
        cb = state.get("circuit_breaker_status", "normal")
        if cb == "normal":
            st.success(f"Circuit Breaker: {cb}")
        elif cb.startswith("reduce"):
            st.warning(f"Circuit Breaker: {cb} — position sizes halved")
        else:
            st.error(f"Circuit Breaker: {cb} — trading halted")

        # Positions
        positions = state.get("positions", {})
        if positions:
            st.subheader("Positions")
            pos_df = pd.DataFrame([
                {"Symbol": sym, "Market Value": f"${v:,.2f}", "Weight": f"{v/equity*100:.1f}%"}
                for sym, v in positions.items()
            ])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)

        # Recent signals
        signals = state.get("recent_signals", [])
        if signals:
            st.subheader("Recent Signals")
            sig_df = pd.DataFrame(signals)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

    st.divider()

    # Start/stop controls
    st.subheader("Controls")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Start Paper Trading", type="primary"):
            subprocess.Popen(
                [sys.executable, "-W", "ignore", str(ROOT / "main.py"), "live", "--paper"],
                cwd=str(ROOT),
            )
            st.success("Paper trading started. Enable Auto-refresh to monitor.")

    with col_b:
        if st.button("Dry Run (no orders)"):
            subprocess.Popen(
                [sys.executable, "-W", "ignore", str(ROOT / "main.py"), "live", "--paper", "--dry-run"],
                cwd=str(ROOT),
            )
            st.success("Dry run started.")

    with col_c:
        if LOCK_FILE.exists():
            if st.button("Resume Trading (delete lock)", type="secondary"):
                LOCK_FILE.unlink()
                st.success("Lock file removed — trading will resume on next bar.")
        else:
            st.button("Resume Trading (no lock active)", disabled=True)


# ==================================================================
# Page 3: Settings
# ==================================================================

elif page == "Settings":
    st.header("Settings")
    st.caption(f"Editing: `{SETTINGS}`")

    settings = load_settings()

    with st.form("settings_form"):
        st.subheader("Broker")
        col1, col2 = st.columns(2)
        syms_str = col1.text_input(
            "Symbols (comma-separated)",
            value=", ".join(settings.get("broker", {}).get("symbols", ["SPY"])),
        )
        paper = col2.checkbox(
            "Paper Trading",
            value=settings.get("broker", {}).get("paper_trading", True),
        )

        st.subheader("HMM")
        c1, c2, c3 = st.columns(3)
        min_conf = c1.slider(
            "Min Confidence",
            0.50, 0.90,
            float(settings.get("hmm", {}).get("min_confidence", 0.55)),
            0.05,
        )
        stab_bars = c2.number_input(
            "Stability Bars",
            1, 10,
            int(settings.get("hmm", {}).get("stability_bars", 3)),
        )
        n_init = c3.number_input(
            "HMM n_init",
            1, 50,
            int(settings.get("hmm", {}).get("n_init", 10)),
        )

        st.subheader("Strategy Allocations")
        s1, s2, s3, s4 = st.columns(4)
        strat = settings.get("strategy", {})
        low_alloc = s1.slider("Low-Vol Alloc", 0.50, 1.00, float(strat.get("low_vol_allocation", 0.95)), 0.05)
        mid_trend = s2.slider("Mid-Vol (trend)", 0.50, 1.00, float(strat.get("mid_vol_allocation_trend", 0.95)), 0.05)
        mid_no = s3.slider("Mid-Vol (no trend)", 0.20, 0.80, float(strat.get("mid_vol_allocation_no_trend", 0.60)), 0.05)
        high_alloc = s4.slider("High-Vol Alloc", 0.10, 0.70, float(strat.get("high_vol_allocation", 0.60)), 0.05)

        st.subheader("Risk Limits")
        r1, r2, r3 = st.columns(3)
        risk = settings.get("risk", {})
        max_dd = r1.slider(
            "Max DD from Peak (halt)",
            0.05, 0.30,
            float(risk.get("max_dd_from_peak", 0.10)),
            0.01,
        )
        daily_halt = r2.slider(
            "Daily DD Halt",
            0.01, 0.10,
            float(risk.get("daily_dd_halt", 0.03)),
            0.005,
        )
        weekly_halt = r3.slider(
            "Weekly DD Halt",
            0.03, 0.20,
            float(risk.get("weekly_dd_halt", 0.07)),
            0.01,
        )

        submitted = st.form_submit_button("Save Settings", type="primary")

    if submitted:
        settings["broker"]["symbols"] = [s.strip() for s in syms_str.split(",")]
        settings["broker"]["paper_trading"] = paper
        settings["hmm"]["min_confidence"] = round(min_conf, 2)
        settings["hmm"]["stability_bars"] = int(stab_bars)
        settings["hmm"]["n_init"] = int(n_init)
        settings["strategy"]["low_vol_allocation"] = round(low_alloc, 2)
        settings["strategy"]["mid_vol_allocation_trend"] = round(mid_trend, 2)
        settings["strategy"]["mid_vol_allocation_no_trend"] = round(mid_no, 2)
        settings["strategy"]["high_vol_allocation"] = round(high_alloc, 2)
        settings["risk"]["max_dd_from_peak"] = round(max_dd, 3)
        settings["risk"]["daily_dd_halt"] = round(daily_halt, 3)
        settings["risk"]["weekly_dd_halt"] = round(weekly_halt, 2)

        with open(str(SETTINGS), "w") as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
        st.success("Settings saved. Re-run the backtest or restart live trading to apply.")

    with st.expander("Raw YAML"):
        st.code(SETTINGS.read_text() if SETTINGS.exists() else "(not found)", language="yaml")
