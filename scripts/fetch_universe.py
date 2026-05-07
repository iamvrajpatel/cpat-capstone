"""
CPAT — Universe Data Fetcher
==============================
CLI script to download historical OHLCV data for the full universe
and store it in the Parquet data store.

Usage::

    python scripts/fetch_universe.py
    python scripts/fetch_universe.py --config config/settings.yaml
    python scripts/fetch_universe.py --start 2020-01-01 --end 2024-12-31
    python scripts/fetch_universe.py --force  # re-download even if cached
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parents[1]))

import click

from cpat.config.loader import load_config
from cpat.data.pipeline import DataPipeline
from cpat.infrastructure.logging import setup_logging


@click.command()
@click.option("--config", default="config/settings.yaml", help="Path to settings.yaml")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--force", is_flag=True, default=False, help="Re-download even if cached")
@click.option("--log-level", default="INFO", help="Log level")
def main(config: str, start: str | None, end: str | None, force: bool, log_level: str) -> None:
    """Download and store historical OHLCV data for the CPAT universe."""
    cfg = load_config(config)
    setup_logging(
        log_level=log_level,
        log_format="console",
        log_dir=cfg.logging.log_dir,
    )

    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    pipeline = DataPipeline.from_config(cfg)
    result = pipeline.run(start=start_date, end=end_date, force_refresh=force)

    print("---> ", result)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Universe: {cfg.universe.name}")
    click.echo(f"  Symbols:  {result.total_symbols}")
    click.echo(f"  Success:  {result.successful} ({result.success_rate:.1%})")
    click.echo(f"  Failed:   {result.failed or 'none'}")
    click.echo(f"  Time:     {result.elapsed_seconds:.1f}s")
    click.echo(f"{'='*60}\n")

    if result.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
