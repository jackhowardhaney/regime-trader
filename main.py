"""Entry point for the regime-trader bot.

CLI usage:
  python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
  python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
  python main.py backtest --stress-test
  python main.py live --paper
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="regime-trader")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- backtest subcommand ---
    bt = sub.add_parser("backtest", help="Run walk-forward backtest or stress test")
    bt.add_argument("--symbols", nargs="+", default=["SPY"], help="Symbols to backtest")
    bt.add_argument("--start", default="2019-01-01", help="Start date YYYY-MM-DD")
    bt.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    bt.add_argument("--compare", action="store_true", help="Run benchmark comparisons")
    bt.add_argument("--stress-test", action="store_true", help="Run stress tests instead")
    bt.add_argument(
        "--output-dir", default="results", help="Directory for CSV output files"
    )
    bt.add_argument(
        "--config", default="config/settings.yaml", help="Path to settings.yaml"
    )

    # --- live subcommand ---
    live = sub.add_parser("live", help="Run live / paper trading loop")
    live.add_argument("--paper", action="store_true", help="Use paper trading account")
    live.add_argument(
        "--config", default="config/settings.yaml", help="Path to settings.yaml"
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load and return settings.yaml as a dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_credentials() -> dict:
    """Load API keys from environment (set via .env or shell)."""
    import os
    return {
        "alpaca_api_key": os.getenv("ALPACA_API_KEY", ""),
        "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
        "paper": os.getenv("ALPACA_PAPER", "true").lower() == "true",
    }


def _fetch_price_data(
    symbols: list,
    start: str,
    end: str,
    credentials: dict,
    config: dict,
) -> Dict:
    """Fetch historical OHLCV data from Alpaca for each symbol."""
    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketDataFetcher

    client = AlpacaClient(
        api_key=credentials["alpaca_api_key"],
        secret_key=credentials["alpaca_secret_key"],
        base_url=config["broker"].get(
            "base_url", "https://paper-api.alpaca.markets"
        ),
        paper=credentials["paper"],
    )
    client.connect()
    fetcher = MarketDataFetcher(client)
    timeframe = config["broker"].get("timeframe", "1Day")
    return fetcher.fetch_historical(symbols, timeframe, start, end)


def run_backtest(args: argparse.Namespace, config: dict, credentials: dict) -> None:
    """Walk-forward backtest with optional benchmark comparison."""
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer

    price_data = _fetch_price_data(
        args.symbols, args.start, args.end, credentials, config
    )

    backtester = WalkForwardBacktester(config)
    result = backtester.run(price_data, symbols=args.symbols)

    analyzer = PerformanceAnalyzer(
        risk_free_rate=config.get("backtest", {}).get("risk_free_rate", 0.045)
    )

    # Use primary symbol's close as the price series for benchmarks
    primary_bars = price_data[args.symbols[0]]
    primary_bars = primary_bars.rename(columns=str.lower)
    price_series = primary_bars["close"]

    analyzer.print_report(result, price_series=price_series, compare=args.compare)
    analyzer.save_csvs(result, Path(args.output_dir))


def run_stress_test(args: argparse.Namespace, config: dict, credentials: dict) -> None:
    """Run all three stress-test scenarios and print results."""
    from backtest.backtester import WalkForwardBacktester
    from backtest.stress_test import StressTest
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Fetch a few years of data for the primary symbol
    price_data = _fetch_price_data(
        args.symbols, "2015-01-01", args.end, credentials, config
    )

    backtester = WalkForwardBacktester(config)
    stress = StressTest(config)
    results = stress.run_all(price_data, backtester)

    t = Table(title="Stress Test Results", header_style="bold red")
    t.add_column("Scenario", style="cyan")
    t.add_column("Mean Max Loss")
    t.add_column("Worst Case")
    t.add_column("Circuit Breaker %")
    t.add_column("Notes")

    for scenario, r in results.items():
        notes = "; ".join(f"{k}={v}" for k, v in r.details.items())
        t.add_row(
            scenario,
            f"{r.mean_max_loss * 100:.2f}%",
            f"{r.worst_case_loss * 100:.2f}%",
            f"{r.circuit_breaker_pct * 100:.1f}%",
            notes,
        )
    console.print(t)


def run_live(args: argparse.Namespace, config: dict, credentials: dict) -> None:
    """Start the live / paper trading loop (Phase 5)."""
    logging.getLogger("regime_trader").info(
        "Live trading not yet implemented (Phase 5). "
        "Run with --paper flag: paper=%s", args.paper
    )


def main() -> None:
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    config_path = Path(getattr(args, "config", "config/settings.yaml"))

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    credentials = load_credentials()

    if args.command == "backtest":
        if args.stress_test:
            run_stress_test(args, config, credentials)
        else:
            run_backtest(args, config, credentials)
    elif args.command == "live":
        run_live(args, config, credentials)


if __name__ == "__main__":
    main()
