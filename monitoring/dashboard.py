"""Rich terminal dashboard — Phase 8.

Layout (refreshes every 5 seconds):
  ┌─ REGIME ─────────────────────────────────────────────┐
  │ BULL (72%) │ Stability: 14 bars │ Flicker: 1/20      │
  ├─ PORTFOLIO ───────────────────────────────────────── │
  │ Equity: $105,230 │ Daily: +$340 (+0.32%)             │
  │ Allocation: 95% │ Leverage: 1.25x                    │
  ├─ POSITIONS ───────────────────────────────────────── │
  │ SPY │ LONG │ $520.30 │ +1.2% │ Stop: $508 │ 3h      │
  ├─ RECENT SIGNALS ──────────────────────────────────── │
  │ 14:30 │ SPY │ Rebalance 60%→95% │ Low vol            │
  ├─ RISK STATUS ─────────────────────────────────────── │
  │ Daily DD: 0.3%/3% ✅ │ From Peak: 1.2%/10% ✅       │
  ├─ SYSTEM ──────────────────────────────────────────── │
  │ Data: ✅ │ API: ✅ 23ms │ HMM: 2d ago │ PAPER        │
  └──────────────────────────────────────────────────────┘
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

_console = Console()


# ======================================================================
# DashboardState — all data the dashboard needs in one object
# ======================================================================

@dataclass
class DashboardState:
    # --- Regime ---
    regime: str = "UNKNOWN"
    regime_probability: float = 0.0
    stability_bars: int = 0
    flicker_rate: int = 0
    flicker_window: int = 20
    is_flickering: bool = False

    # --- Portfolio ---
    equity: float = 0.0
    daily_pnl: float = 0.0
    total_allocation_pct: float = 0.0
    leverage: float = 1.0

    # --- Positions ---
    # Each dict: symbol, direction, price, pnl_pct, stop, holding_minutes
    positions: List[dict] = field(default_factory=list)

    # --- Recent signals ---
    # Each dict: timestamp, symbol, description, regime
    recent_signals: List[dict] = field(default_factory=list)

    # --- Risk ---
    daily_dd_pct: float = 0.0
    daily_dd_limit_pct: float = 3.0
    peak_dd_pct: float = 0.0
    peak_dd_limit_pct: float = 10.0
    circuit_breaker: str = "normal"

    # --- System ---
    data_feed_ok: bool = True
    api_latency_ms: float = 0.0
    hmm_age_days: int = 0
    paper_mode: bool = True
    last_update: datetime = field(default_factory=datetime.utcnow)
    dry_run: bool = False


# ======================================================================
# Dashboard class
# ======================================================================

class Dashboard:
    """
    Rich terminal dashboard for the live trading loop.

    Standalone usage:
        state = DashboardState(...)
        dash = Dashboard(refresh_seconds=5)
        dash.start()
        dash.update(state)   # call after each bar
        dash.stop()

    Integration (no internal Live context):
        dash = Dashboard()
        renderable = dash.render(state)   # pass to your own Rich.Live
    """

    def __init__(self, refresh_seconds: int = 5) -> None:
        self.refresh_seconds = refresh_seconds
        self._state: DashboardState = DashboardState()
        self._live: Optional[Live] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin the live render loop (standalone mode)."""
        self._live = Live(
            self.render(self._state),
            console=_console,
            refresh_per_second=1.0 / self.refresh_seconds,
            screen=False,
        )
        self._live.__enter__()

    def stop(self) -> None:
        """Stop the live render loop."""
        if self._live:
            self._live.__exit__(None, None, None)
            self._live = None

    # ------------------------------------------------------------------
    # Data update
    # ------------------------------------------------------------------

    def update(
        self,
        regime: str = "UNKNOWN",
        portfolio_value: float = 0.0,
        positions: Optional[Dict[str, float]] = None,
        daily_pnl: float = 0.0,
        drawdown: float = 0.0,
    ) -> None:
        """
        Simple update — compatible with the Phase 1 stub signature.
        Converts to DashboardState and delegates to update_state().
        """
        pos_list = [
            {"symbol": sym, "direction": "LONG", "price": 0.0,
             "pnl_pct": 0.0, "stop": 0.0, "holding_minutes": 0}
            for sym in (positions or {})
        ]
        self.update_state(DashboardState(
            regime=regime,
            equity=portfolio_value,
            daily_pnl=daily_pnl,
            peak_dd_pct=drawdown,
            positions=pos_list,
            last_update=datetime.utcnow(),
        ))

    def update_state(self, state: DashboardState) -> None:
        """Full update — pass a DashboardState from the live trading loop."""
        with self._lock:
            self._state = state
        if self._live:
            self._live.update(self.render(state))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, state: Optional[DashboardState] = None) -> Panel:
        """Return a Rich Panel renderable. Safe to call at any time."""
        s = state or self._state
        sections = []

        # ── REGIME ──────────────────────────────────────────────────
        regime_color = self._regime_color(s.regime)
        flicker_style = "bold red" if s.is_flickering else "default"
        regime_line = Text()
        regime_line.append(f" {s.regime}", style=f"bold {regime_color}")
        regime_line.append(f" ({s.regime_probability * 100:.0f}%)")
        regime_line.append(f" │ Stability: {s.stability_bars} bars")
        regime_line.append(
            f" │ Flicker: {s.flicker_rate}/{s.flicker_window}",
            style=flicker_style,
        )
        if s.dry_run:
            regime_line.append(" │ DRY-RUN", style="bold yellow")

        sections += [Rule("REGIME", style="cyan"), regime_line]

        # ── PORTFOLIO ────────────────────────────────────────────────
        pnl_style = "green" if s.daily_pnl >= 0 else "red"
        pnl_pct = (s.daily_pnl / s.equity * 100) if s.equity > 0 else 0.0

        portfolio_line1 = Text()
        portfolio_line1.append(f" Equity: ${s.equity:,.0f}")
        portfolio_line1.append(" │ Daily: ")
        portfolio_line1.append(
            f"{s.daily_pnl:+,.0f} ({pnl_pct:+.2f}%)", style=pnl_style
        )

        portfolio_line2 = Text()
        portfolio_line2.append(f" Allocation: {s.total_allocation_pct:.0f}%")
        portfolio_line2.append(f" │ Leverage: {s.leverage:.2f}x")

        sections += [
            Rule("PORTFOLIO", style="cyan"),
            portfolio_line1,
            portfolio_line2,
        ]

        # ── POSITIONS ────────────────────────────────────────────────
        sections.append(Rule("POSITIONS", style="cyan"))
        if s.positions:
            for pos in s.positions:
                pnl = pos.get("pnl_pct", 0.0)
                mins = pos.get("holding_minutes", 0)
                hold_str = f"{mins // 60}h{mins % 60:02d}m" if mins >= 60 else f"{mins}m"
                pline = Text()
                pline.append(f" {pos.get('symbol','?')}")
                pline.append(f" │ {pos.get('direction','?')}")
                pline.append(f" │ ${pos.get('price', 0):.2f}")
                pline.append(" │ ")
                pline.append(f"{pnl:+.1f}%", style="green" if pnl >= 0 else "red")
                pline.append(f" │ Stop: ${pos.get('stop', 0):.2f}")
                pline.append(f" │ {hold_str}")
                sections.append(pline)
        else:
            sections.append(Text(" No open positions.", style="dim"))

        # ── RECENT SIGNALS ───────────────────────────────────────────
        sections.append(Rule("RECENT SIGNALS", style="cyan"))
        if s.recent_signals:
            for sig in s.recent_signals[-5:]:
                ts = sig.get("timestamp", "")
                if hasattr(ts, "strftime"):
                    ts = ts.strftime("%H:%M")
                elif isinstance(ts, str) and "T" in ts:
                    ts = ts[11:16]
                sline = Text()
                sline.append(f" {ts}")
                sline.append(f" │ {sig.get('symbol', '?')}")
                sline.append(f" │ {sig.get('description', '')}")
                sline.append(f" │ {sig.get('regime', '')}", style="dim")
                sections.append(sline)
        else:
            sections.append(Text(" No recent signals.", style="dim"))

        # ── RISK STATUS ──────────────────────────────────────────────
        sections.append(Rule("RISK STATUS", style="cyan"))

        daily_pct = min(s.daily_dd_pct / s.daily_dd_limit_pct, 1.0) if s.daily_dd_limit_pct > 0 else 0
        peak_pct = min(s.peak_dd_pct / s.peak_dd_limit_pct, 1.0) if s.peak_dd_limit_pct > 0 else 0

        risk_line = Text()
        risk_line.append(" Daily DD: ")
        risk_line.append(
            f"{s.daily_dd_pct:.1f}%/{s.daily_dd_limit_pct:.0f}%",
            style=self._risk_color(daily_pct),
        )
        risk_line.append(" " + self._status_icon(daily_pct))
        risk_line.append(" │ From Peak: ")
        risk_line.append(
            f"{s.peak_dd_pct:.1f}%/{s.peak_dd_limit_pct:.0f}%",
            style=self._risk_color(peak_pct),
        )
        risk_line.append(" " + self._status_icon(peak_pct))

        cb_style = "green" if s.circuit_breaker == "normal" else "bold red"
        cb_line = Text()
        cb_line.append(f" Circuit Breaker: ")
        cb_line.append(s.circuit_breaker.upper(), style=cb_style)

        sections += [risk_line, cb_line]

        # ── SYSTEM ───────────────────────────────────────────────────
        sections.append(Rule("SYSTEM", style="cyan"))

        data_icon = "✅" if s.data_feed_ok else "🔴"
        api_icon = "✅" if s.api_latency_ms < 500 else "🔴"
        hmm_str = f"{s.hmm_age_days}d ago" if s.hmm_age_days >= 0 else "unknown"
        mode_str = "PAPER" if s.paper_mode else "LIVE"
        mode_style = "cyan" if s.paper_mode else "bold red"
        last_upd = s.last_update.strftime("%H:%M:%S")

        sys_line = Text()
        sys_line.append(f" Data: {data_icon}")
        sys_line.append(f" │ API: {api_icon} {s.api_latency_ms:.0f}ms")
        sys_line.append(f" │ HMM: {hmm_str}")
        sys_line.append(" │ ")
        sys_line.append(mode_str, style=mode_style)
        sys_line.append(f" │ {last_upd}")

        sections.append(sys_line)

        return Panel(Group(*sections), title="[bold cyan]regime-trader[/]")

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_color(regime: str) -> str:
        if any(x in regime for x in ("CRASH", "STRONG_BEAR", "BEAR")):
            return "red"
        if any(x in regime for x in ("BULL", "EUPHORIA")):
            return "green"
        return "yellow"

    @staticmethod
    def _risk_color(fraction: float) -> str:
        """fraction = current / limit (0–1)."""
        if fraction >= 0.80:
            return "bold red"
        if fraction >= 0.50:
            return "yellow"
        return "green"

    @staticmethod
    def _status_icon(fraction: float) -> str:
        return "🔴" if fraction >= 0.80 else ("🟡" if fraction >= 0.50 else "✅")

    # ------------------------------------------------------------------
    # Alias kept for backtester / legacy callers
    # ------------------------------------------------------------------

    def _build_layout(self):
        return self.render()

    def _positions_table(self, positions: Dict[str, float]) -> Table:
        t = Table(show_header=True, header_style="bold")
        t.add_column("Symbol")
        t.add_column("Weight", justify="right")
        for sym, weight in positions.items():
            t.add_row(sym, f"{weight * 100:.1f}%")
        return t
