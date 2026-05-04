"""
CPAT — Trade Log
=================
Structured, auditable trade log persisted as CSV and Parquet.

The TradeLog records every Fill as a row with full metadata.
It is the canonical audit trail for compliance and performance analysis.

Columns:
    fill_id, order_id, timestamp, symbol, side, quantity,
    fill_price, commission, slippage, gross_value, net_value,
    cost_bps, strategy_id (if available)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cpat.core.models import Fill
from cpat.core.enums import OrderSide

logger = logging.getLogger(__name__)


class TradeLog:
    """Append-only trade log backed by an in-memory list.

    Supports CSV and Parquet export for compliance and analytics.

    Args:
        output_dir: Directory where trade logs are written on ``save()``.
    """

    def __init__(self, output_dir: str | Path = "data/results") -> None:
        self._output_dir = Path(output_dir)
        self._fills: list[Fill] = []

    def record(self, fill: Fill) -> None:
        """Append a fill to the log.

        Args:
            fill: Confirmed Fill to record.
        """
        self._fills.append(fill)

    def record_all(self, fills: list[Fill]) -> None:
        """Append multiple fills.

        Args:
            fills: List of fills to record.
        """
        self._fills.extend(fills)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade log to DataFrame.

        Returns:
            DataFrame with one row per fill. Returns empty DataFrame if no fills.
        """
        if not self._fills:
            return pd.DataFrame(columns=[
                "fill_id", "order_id", "timestamp", "symbol", "side",
                "quantity", "fill_price", "commission", "slippage",
                "gross_value", "net_value", "cost_bps",
            ])

        rows = []
        for fill in self._fills:
            rows.append({
                "fill_id": str(fill.fill_id),
                "order_id": str(fill.order_id),
                "timestamp": fill.timestamp,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "fill_price": fill.fill_price,
                "commission": fill.commission,
                "slippage": fill.slippage,
                "gross_value": fill.gross_value,
                "net_value": fill.net_value,
                "cost_bps": fill.cost_bps,
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp").sort_index()

    def save(self, filename_stem: str = "trade_log") -> dict[str, Path]:
        """Save trade log to CSV and Parquet.

        Args:
            filename_stem: Base filename (without extension).

        Returns:
            Dict mapping format → output path.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()

        csv_path = self._output_dir / f"{filename_stem}.csv"
        parquet_path = self._output_dir / f"{filename_stem}.parquet"

        df.to_csv(csv_path)
        df.to_parquet(parquet_path, compression="snappy")

        logger.info(
            "Trade log saved: %d fills → %s", len(self._fills), self._output_dir
        )

        return {"csv": csv_path, "parquet": parquet_path}

    def summary(self) -> dict[str, object]:
        """Return a summary dict of trade log statistics.

        Returns:
            Dict with count, symbols, total commission, total slippage.
        """
        if not self._fills:
            return {"n_fills": 0}

        df = self.to_dataframe()
        return {
            "n_fills": len(self._fills),
            "n_symbols": df["symbol"].nunique(),
            "symbols": sorted(df["symbol"].unique().tolist()),
            "total_commission": round(df["commission"].sum(), 2),
            "total_slippage": round(df["slippage"].sum(), 2),
            "total_cost_bps": round(df["cost_bps"].mean(), 2),
            "n_buys": int((df["side"] == OrderSide.BUY.value).sum()),
            "n_sells": int((df["side"] == OrderSide.SELL.value).sum()),
        }

    def __len__(self) -> int:
        return len(self._fills)

    def __repr__(self) -> str:
        return f"TradeLog(n_fills={len(self._fills)})"
