"""
CPAT — Portfolio Manager
=========================
Full multi-asset portfolio state management, extracted from BacktestEngine.

Responsibilities:
    - Track cash balance
    - Maintain per-symbol Position objects (FIFO cost basis)
    - Mark-to-market equity computation
    - Exposure analysis (gross, net, per-symbol, per-sector)
    - Periodic equity snapshots for the equity curve

Design:
    PortfolioManager is the single owner of all financial state.
    It is mutated only by ``apply_fill()`` — no other component
    changes positions or cash directly.

    The BacktestEngine calls ``apply_fill()`` synchronously after
    every confirmed fill.  No fill is ever applied twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from cpat.core.enums import OrderSide
from cpat.core.models import Bar, Fill, Position

logger = logging.getLogger(__name__)


# ── Snapshot dataclass ─────────────────────────────────────────────────────────


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state for the equity curve and analytics.

    Attributes:
        timestamp: UTC snapshot time.
        cash: Available cash.
        positions_value: Mark-to-market value of all open positions.
        total_equity: cash + positions_value.
        gross_exposure: Sum of absolute position values.
        net_exposure: Long value − short value.
        n_longs: Number of symbols with positive quantity.
        n_shorts: Number of symbols with negative quantity.
        realised_pnl: Cumulative realised P&L across all positions.
    """

    timestamp: pd.Timestamp
    cash: float
    positions_value: float
    total_equity: float
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    n_longs: int
    n_shorts: int
    n_open_positions: int
    realised_pnl: float


# ── Portfolio Manager ──────────────────────────────────────────────────────────


class PortfolioManager:
    """Multi-asset portfolio state manager.

    Args:
        initial_capital: Starting cash balance (USD).
        universe_sectors: Optional mapping of symbol → sector for
                          sector-level exposure tracking.
    """

    def __init__(
        self,
        initial_capital: float,
        universe_sectors: Optional[dict[str, str]] = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")

        self._initial_capital = initial_capital
        self._cash: float = initial_capital
        self._positions: dict[str, Position] = {}
        self._universe_sectors = universe_sectors or {}

        # Audit trail: every fill ever applied, in order
        self._fills: list[Fill] = []
        self._snapshots: list[PortfolioSnapshot] = []

        logger.info("PortfolioManager initialised: capital=%.2f", initial_capital)

    # ── Mutation ───────────────────────────────────────────────────────────────

    def apply_fill(self, fill: Fill) -> None:
        """Apply a confirmed fill to cash and position state.

        Cash accounting:
            BUY:  cash -= (fill_price × quantity) + commission
            SELL: cash += (fill_price × quantity) - commission

        Args:
            fill: The confirmed Fill to apply.
        """
        # Update cash
        if fill.side == OrderSide.BUY:
            cash_change = -(fill.fill_price * fill.quantity) - fill.commission
        else:  # SELL
            cash_change = (fill.fill_price * fill.quantity) - fill.commission

        self._cash += cash_change

        # Update position
        if fill.symbol not in self._positions:
            self._positions[fill.symbol] = Position(symbol=fill.symbol)
        self._positions[fill.symbol].apply_fill(fill)

        self._fills.append(fill)

        logger.debug(
            "Fill applied: %s %s %.0f @ %.4f | cash: %.2f",
            fill.side.value, fill.symbol, fill.quantity, fill.fill_price, self._cash,
        )

    # ── Valuation ──────────────────────────────────────────────────────────────

    def total_equity(self, current_prices: dict[str, float]) -> float:
        """Compute mark-to-market total portfolio value.

        Args:
            current_prices: Latest price for each symbol (symbol → price).

        Returns:
            Total portfolio value: cash + sum(position market values).
        """
        equity = self._cash
        for symbol, position in self._positions.items():
            if position.is_flat:
                continue
            price = current_prices.get(symbol)
            if price is None or price <= 0:
                continue
            equity += position.market_value(price)
        return equity

    def positions_value(self, current_prices: dict[str, float]) -> float:
        """Mark-to-market value of all open positions.

        Args:
            current_prices: Latest prices.

        Returns:
            Signed sum of position market values.
        """
        total = 0.0
        for symbol, pos in self._positions.items():
            if pos.is_flat:
                continue
            price = current_prices.get(symbol)
            if price and price > 0:
                total += pos.market_value(price)
        return total

    def gross_exposure(self, current_prices: dict[str, float]) -> float:
        """Sum of absolute position values (long + short).

        Args:
            current_prices: Latest prices.

        Returns:
            Gross exposure in currency units.
        """
        total = 0.0
        for symbol, pos in self._positions.items():
            if pos.is_flat:
                continue
            price = current_prices.get(symbol)
            if price and price > 0:
                total += abs(pos.market_value(price))
        return total

    def net_exposure(self, current_prices: dict[str, float]) -> float:
        """Signed net exposure (long - short).

        Args:
            current_prices: Latest prices.

        Returns:
            Net exposure: positive = net long, negative = net short.
        """
        long_val = sum(
            pos.market_value(current_prices[sym])
            for sym, pos in self._positions.items()
            if pos.is_long and sym in current_prices and current_prices[sym] > 0
        )
        short_val = sum(
            pos.market_value(current_prices[sym])
            for sym, pos in self._positions.items()
            if pos.is_short and sym in current_prices and current_prices[sym] > 0
        )
        return long_val + short_val  # short market_value is already negative

    def sector_exposure(self, current_prices: dict[str, float]) -> dict[str, float]:
        """Compute exposure by GICS sector.

        Args:
            current_prices: Latest prices.

        Returns:
            Mapping of sector → absolute exposure value.
        """
        sector_exp: dict[str, float] = {}
        for symbol, pos in self._positions.items():
            if pos.is_flat:
                continue
            price = current_prices.get(symbol)
            if not price or price <= 0:
                continue
            sector = self._universe_sectors.get(symbol, "Unknown")
            sector_exp[sector] = sector_exp.get(sector, 0.0) + abs(pos.market_value(price))
        return sector_exp

    def symbol_weight(self, symbol: str, current_prices: dict[str, float]) -> float:
        """Position weight as fraction of total equity.

        Args:
            symbol: Symbol to query.
            current_prices: Latest prices.

        Returns:
            Weight in [−1, 1+] (can exceed 1 if leveraged).
        """
        equity = self.total_equity(current_prices)
        if equity <= 0:
            return 0.0
        pos = self._positions.get(symbol)
        if pos is None or pos.is_flat:
            return 0.0
        price = current_prices.get(symbol, 0.0)
        return pos.market_value(price) / equity

    # ── Snapshot ───────────────────────────────────────────────────────────────

    def snapshot(
        self,
        timestamp: pd.Timestamp,
        current_prices: dict[str, float],
    ) -> PortfolioSnapshot:
        """Create and store a point-in-time portfolio snapshot.

        Args:
            timestamp: Current bar timestamp.
            current_prices: Latest prices for all symbols.

        Returns:
            PortfolioSnapshot for this timestamp.
        """
        pos_value = self.positions_value(current_prices)
        equity = self._cash + pos_value
        gross = self.gross_exposure(current_prices)
        net = self.net_exposure(current_prices)
        long_exposure = sum(
            pos.market_value(current_prices[sym])
            for sym, pos in self._positions.items()
            if pos.is_long and sym in current_prices and current_prices[sym] > 0
        )
        short_exposure = sum(
            abs(pos.market_value(current_prices[sym]))
            for sym, pos in self._positions.items()
            if pos.is_short and sym in current_prices and current_prices[sym] > 0
        )

        n_longs = sum(1 for p in self._positions.values() if p.is_long)
        n_shorts = sum(1 for p in self._positions.values() if p.is_short)
        n_open_positions = sum(1 for p in self._positions.values() if not p.is_flat)
        realised = sum(p.realised_pnl for p in self._positions.values())

        snap = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self._cash,
            positions_value=pos_value,
            total_equity=equity,
            gross_exposure=gross,
            net_exposure=net,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            n_longs=n_longs,
            n_shorts=n_shorts,
            n_open_positions=n_open_positions,
            realised_pnl=realised,
        )
        self._snapshots.append(snap)
        return snap

    def equity_curve(self) -> pd.Series:
        """Return the equity curve from all recorded snapshots.

        Returns:
            Series indexed by UTC timestamp, values = total equity.
        """
        if not self._snapshots:
            return pd.Series(dtype=float, name="equity")
        idx = [s.timestamp for s in self._snapshots]
        vals = [s.total_equity for s in self._snapshots]
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")

    def daily_returns(self) -> pd.Series:
        """Compute daily returns from the equity curve.

        Returns:
            Series of percentage returns (decimal, e.g. 0.01 = 1%).
        """
        curve = self.equity_curve()
        if curve.empty:
            return pd.Series(dtype=float, name="returns")
        return curve.pct_change().dropna().rename("returns")

    def snapshots_df(self) -> pd.DataFrame:
        """Return all snapshots as a DataFrame.

        Returns:
            DataFrame with one row per snapshot.
        """
        if not self._snapshots:
            return pd.DataFrame()
        rows = [
            {
                "timestamp": s.timestamp,
                "cash": s.cash,
                "positions_value": s.positions_value,
                "total_equity": s.total_equity,
                "gross_exposure": s.gross_exposure,
                "net_exposure": s.net_exposure,
                "long_exposure": s.long_exposure,
                "short_exposure": s.short_exposure,
                "n_longs": s.n_longs,
                "n_shorts": s.n_shorts,
                "n_open_positions": s.n_open_positions,
                "realised_pnl": s.realised_pnl,
            }
            for s in self._snapshots
        ]
        return pd.DataFrame(rows).set_index("timestamp")

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        """Current available cash."""
        return self._cash

    @property
    def initial_capital(self) -> float:
        """Starting capital."""
        return self._initial_capital

    @property
    def positions(self) -> dict[str, Position]:
        """Read-only view of all Position objects (including flat)."""
        return dict(self._positions)

    @property
    def open_positions(self) -> dict[str, Position]:
        """Only non-flat positions."""
        return {sym: pos for sym, pos in self._positions.items() if not pos.is_flat}

    @property
    def fills(self) -> list[Fill]:
        """All fills applied to this portfolio, in order."""
        return list(self._fills)

    def get_position(self, symbol: str) -> Position:
        """Return the Position for a symbol (creates flat position if missing).

        Args:
            symbol: Ticker symbol.

        Returns:
            Current Position (may be flat).
        """
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    def unrealised_pnl(self, current_prices: dict[str, float]) -> float:
        """Total unrealised P&L across all open positions.

        Args:
            current_prices: Latest prices.

        Returns:
            Total unrealised P&L in currency units.
        """
        total = 0.0
        for symbol, pos in self._positions.items():
            if pos.is_flat:
                continue
            price = current_prices.get(symbol)
            if price and price > 0:
                total += pos.unrealised_pnl(price)
        return total

    def realised_pnl(self) -> float:
        """Total realised P&L across all positions."""
        return sum(p.realised_pnl for p in self._positions.values())
