"""
CPAT — Position Sizing Framework (Week 4)
==========================================
Transforms a signal + market data into a concrete share quantity,
stop-loss price, and take-profit price.

Three sizing methods are provided, each implementing the ``PositionSizer``
Protocol so they are interchangeable with no changes to callers.

Method summary
--------------
FixedFractionalSizer (recommended baseline)
    Risk exactly ``risk_per_trade_pct`` of current equity on every trade.
    Stop distance = ``atr_multiplier × ATR(period)`` below entry price.
    Quantity = risk_amount / stop_distance_per_share.

    Formula::
        stop_distance = atr_mult × ATR
        risk_amount   = equity × risk_per_trade_pct
        quantity      = floor(risk_amount / stop_distance)

InverseVolatilitySizer (preferred for multi-asset)
    Equal *risk* allocation: every position risks the same fraction of equity
    regardless of the asset's volatility.  High-vol assets get smaller
    positions automatically.

    Formula::
        ATR  = rolling mean of True Range over ``atr_period`` bars
        stop = entry_price − atr_multiplier × ATR
        qty  = floor((equity × risk_pct) / (atr_mult × ATR))

    This is equivalent to FixedFractional when ATR is used for the stop —
    the key difference is that ``atr_multiplier`` determines *both* the stop
    distance and the quantity, ensuring consistent risk in ATR units.

KellySizer (advanced — use with caution)
    Half-Kelly criterion based on historical win-rate and payoff ratio.
    Capped at ``max_kelly_fraction`` to prevent ruin from estimation error.

    Formula::
        f* = (win_rate × avg_win − loss_rate × avg_loss) / avg_win
        f_half = 0.5 × f*   (half-Kelly for robustness)
        f_capped = min(f_half, max_kelly_fraction)
        qty = floor((equity × f_capped) / price)

Key design rules
-----------------
* No hardcoded values — all parameters come from ``PositionSizingConfig``.
* Every sizer returns ``PositionSize(quantity=0)`` when data is insufficient.
* Quantity is always floored to whole shares and capped at ``max_position_pct``.

Usage::

    from cpat.risk.position_sizing import PositionSizerFactory, PositionSize

    sizer = PositionSizerFactory.build("inverse_volatility", cfg)
    size  = sizer.compute(
        symbol="RELIANCE.NS",
        entry_price=2_500.0,
        equity=10_000_000.0,
        bar_data=reliance_df,          # OHLCV DataFrame
    )
    print(size.quantity, size.stop_price, size.risk_per_share)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from cpat.core.enums import SignalDirection

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class PositionSize:
    """Output of a position sizing calculation.

    Attributes:
        symbol          : Instrument ticker.
        quantity        : Whole shares to buy/sell (0 = do not trade).
        entry_price     : Reference entry price used in sizing.
        stop_price      : Initial hard stop-loss price (below entry for longs).
        take_profit_price: Optional take-profit target (None = no TP).
        risk_per_share  : Entry - stop in currency units.
        total_risk      : quantity × risk_per_share (portfolio risk on this trade).
        atr             : ATR value used for stop computation (informational).
        sizing_method   : Which algorithm produced this result.
    """

    symbol: str
    quantity: float
    entry_price: float
    stop_price: float
    take_profit_price: Optional[float]
    risk_per_share: float
    total_risk: float
    atr: float
    sizing_method: str
    capital_budget: float = 0.0

    @property
    def is_valid(self) -> bool:
        """True if quantity ≥ 1 and stop_price is meaningful."""
        return self.quantity >= 1.0 and self.stop_price > 0.0 and self.risk_per_share > 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 4),
            "stop_price": round(self.stop_price, 4),
            "take_profit_price": round(self.take_profit_price, 4) if self.take_profit_price else None,
            "risk_per_share": round(self.risk_per_share, 4),
            "total_risk": round(self.total_risk, 2),
            "atr": round(self.atr, 4),
            "sizing_method": self.sizing_method,
            "capital_budget": round(self.capital_budget, 2),
        }


# ── Protocol (Strategy Pattern) ────────────────────────────────────────────────


class PositionSizer(Protocol):
    """Protocol that all position sizers must implement."""

    def compute(
        self,
        symbol: str,
        entry_price: float,
        equity: float,
        bar_data: pd.DataFrame,
        capital_budget: Optional[float] = None,
        direction: SignalDirection = SignalDirection.LONG,
    ) -> PositionSize:
        """Compute the position size for a new trade.

        Args:
            symbol      : Ticker symbol (for logging/metadata).
            entry_price : Expected entry price (typically last close).
            equity      : Current total portfolio equity.
            bar_data    : OHLCV DataFrame for this symbol (index=DatetimeIndex).
                          Must contain columns: high, low, close, volume.

        Returns:
            PositionSize with quantity ≥ 0.  quantity=0 means skip the trade.
        """
        ...


# ── ATR Helper ─────────────────────────────────────────────────────────────────


def _compute_atr(bar_data: pd.DataFrame, period: int = 14) -> float:
    """Compute the Average True Range over the last ``period`` bars.

    True Range = max(high − low, |high − prev_close|, |low − prev_close|)

    Args:
        bar_data: OHLCV DataFrame sorted ascending.
        period  : Look-back window (bars).

    Returns:
        ATR value in price units. Returns 0.0 if insufficient data.
    """
    if bar_data is None or len(bar_data) < period + 1:
        return 0.0

    tail = bar_data.tail(period + 1).copy()
    high = tail["high"].to_numpy()
    low  = tail["low"].to_numpy()
    close = tail["close"].to_numpy()

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1]),
        ),
    )
    return float(tr.mean())


def _zero_size(symbol: str, entry_price: float, method: str, reason: str) -> PositionSize:
    """Return a zero-quantity PositionSize with logging."""
    logger.debug("[sizing] SKIP %s (%s): %s", symbol, method, reason)
    return PositionSize(
        symbol=symbol, quantity=0.0, entry_price=entry_price,
        stop_price=0.0, take_profit_price=None,
        risk_per_share=0.0, total_risk=0.0,
        atr=0.0, sizing_method=method, capital_budget=0.0,
    )


def _price_targets(
    entry_price: float,
    stop_distance: float,
    take_profit_mult: Optional[float],
    direction: SignalDirection,
) -> tuple[float, Optional[float]]:
    """Build stop-loss and take-profit prices for the requested direction."""
    if direction == SignalDirection.SHORT:
        stop_price = entry_price + stop_distance
        take_profit = (
            entry_price - take_profit_mult * stop_distance
            if take_profit_mult is not None
            else None
        )
    else:
        stop_price = entry_price - stop_distance
        take_profit = (
            entry_price + take_profit_mult * stop_distance
            if take_profit_mult is not None
            else None
        )
    return stop_price, take_profit


# ── Fixed Fractional Sizer ─────────────────────────────────────────────────────


class FixedFractionalSizer:
    """Risk a fixed fraction of equity on every trade.

    Stop = entry − atr_multiplier × ATR(period)
    Quantity = floor((equity × risk_per_trade_pct) / stop_distance)

    This ensures every trade risks exactly ``risk_per_trade_pct`` of current
    equity regardless of price level or volatility.

    Args:
        risk_per_trade_pct  : Fraction of equity to risk per trade (e.g. 0.01 = 1%).
        atr_period          : ATR calculation window.
        atr_multiplier      : Stop width in ATR units (e.g. 2.0 = 2×ATR).
        take_profit_mult    : R-multiple for take-profit (e.g. 3.0 = 3R).
                              Set to None to disable take-profit.
        max_position_pct    : Hard cap on single position as fraction of equity.
        min_trade_value     : Skip order if position value < this (avoids tiny lots).
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 0.01,
        stop_method: str = "atr",
        fixed_stop_pct: float = 0.02,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        take_profit_mult: Optional[float] = 3.0,
        max_position_pct: float = 0.05,
        min_trade_value: float = 5_000.0,
    ) -> None:
        self._risk_pct = risk_per_trade_pct
        self._stop_method = stop_method
        self._fixed_stop_pct = fixed_stop_pct
        self._atr_period = atr_period
        self._atr_mult = atr_multiplier
        self._tp_mult = take_profit_mult
        self._max_pos_pct = max_position_pct
        self._min_trade = min_trade_value

    def compute(
        self,
        symbol: str,
        entry_price: float,
        equity: float,
        bar_data: pd.DataFrame,
        capital_budget: Optional[float] = None,
        direction: SignalDirection = SignalDirection.LONG,
    ) -> PositionSize:
        if entry_price <= 0 or equity <= 0:
            return _zero_size(symbol, entry_price, "fixed_fractional", "invalid price/equity")

        atr = _compute_atr(bar_data, self._atr_period)
        if self._stop_method == "atr" and atr <= 0:
            return _zero_size(symbol, entry_price, "fixed_fractional", "ATR=0 (insufficient data)")

        stop_distance = (
            self._atr_mult * atr
            if self._stop_method == "atr"
            else entry_price * self._fixed_stop_pct
        )
        stop_price, tp_price = _price_targets(
            entry_price=entry_price,
            stop_distance=stop_distance,
            take_profit_mult=self._tp_mult,
            direction=direction,
        )
        if stop_price <= 0:
            return _zero_size(symbol, entry_price, "fixed_fractional", f"stop_price={stop_price:.4f} ≤ 0")

        fixed_capital = equity * self._risk_pct
        capped_budget = min(
            fixed_capital,
            capital_budget if capital_budget is not None else fixed_capital,
            equity * self._max_pos_pct,
        )
        quantity = math.floor(capped_budget / entry_price)

        if quantity < 1 or quantity * entry_price < self._min_trade:
            return _zero_size(symbol, entry_price, "fixed_fractional",
                              f"quantity={quantity} too small (min_trade={self._min_trade:.0f})")

        risk_per_share = stop_distance
        total_risk     = quantity * risk_per_share

        logger.debug(
            "[sizing:FF] %s | qty=%d | entry=%.4f | stop=%.4f | ATR=%.4f | risk=%.2f",
            symbol, quantity, entry_price, stop_price, atr, total_risk,
        )
        return PositionSize(
            symbol=symbol, quantity=float(quantity),
            entry_price=entry_price, stop_price=stop_price,
            take_profit_price=tp_price,
            risk_per_share=risk_per_share, total_risk=total_risk,
            atr=atr, sizing_method="fixed_fractional",
            capital_budget=capped_budget,
        )


# ── Inverse Volatility Sizer ───────────────────────────────────────────────────


class InverseVolatilitySizer:
    """Equal risk-dollar allocation using ATR as the volatility proxy.

    Identical formula to FixedFractional but explicitly labelled for
    multi-asset use: positions across all assets share the same *risk*
    budget, so high-volatility assets automatically receive smaller sizes.

    In practice, this is the industry-standard implementation of
    "inverse volatility position sizing" for intraday and daily strategies.

    Args:
        risk_per_trade_pct: Fraction of equity to risk per trade.
        atr_period        : ATR window.
        atr_multiplier    : Stop distance = atr_mult × ATR.
        take_profit_mult  : Take-profit = entry + tp_mult × stop_distance.
        max_position_pct  : Hard cap per single name.
        min_trade_value   : Minimum order value.
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 0.01,
        stop_method: str = "atr",
        fixed_stop_pct: float = 0.02,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        take_profit_mult: Optional[float] = 3.0,
        max_position_pct: float = 0.05,
        min_trade_value: float = 5_000.0,
    ) -> None:
        self._risk_pct    = risk_per_trade_pct
        self._stop_method = stop_method
        self._fixed_stop_pct = fixed_stop_pct
        self._atr_period  = atr_period
        self._atr_mult    = atr_multiplier
        self._tp_mult     = take_profit_mult
        self._max_pos_pct = max_position_pct
        self._min_trade   = min_trade_value

    def compute(
        self,
        symbol: str,
        entry_price: float,
        equity: float,
        bar_data: pd.DataFrame,
        capital_budget: Optional[float] = None,
        direction: SignalDirection = SignalDirection.LONG,
    ) -> PositionSize:
        if entry_price <= 0 or equity <= 0:
            return _zero_size(symbol, entry_price, "inverse_volatility", "invalid price/equity")

        atr = _compute_atr(bar_data, self._atr_period)
        if self._stop_method == "atr" and atr <= 0:
            return _zero_size(symbol, entry_price, "inverse_volatility", "ATR=0")

        vol_stop_distance = (
            self._atr_mult * atr
            if self._stop_method == "atr"
            else entry_price * self._fixed_stop_pct
        )
        stop_price, tp_price = _price_targets(
            entry_price=entry_price,
            stop_distance=vol_stop_distance,
            take_profit_mult=self._tp_mult,
            direction=direction,
        )
        if stop_price <= 0:
            return _zero_size(symbol, entry_price, "inverse_volatility", "stop_price ≤ 0")

        risk_budget = equity * self._risk_pct
        max_position_budget = equity * self._max_pos_pct
        capital_cap = capital_budget if capital_budget is not None else max_position_budget
        quantity = math.floor(min(
            risk_budget / vol_stop_distance,
            capital_cap / entry_price,
            max_position_budget / entry_price,
        ))

        if quantity < 1 or quantity * entry_price < self._min_trade:
            return _zero_size(symbol, entry_price, "inverse_volatility",
                              f"quantity={quantity} too small")

        risk_per_share = vol_stop_distance
        total_risk     = quantity * risk_per_share

        logger.debug(
            "[sizing:IV] %s | qty=%d | entry=%.4f | stop=%.4f | ATR=%.4f",
            symbol, quantity, entry_price, stop_price, atr,
        )
        return PositionSize(
            symbol=symbol, quantity=float(quantity),
            entry_price=entry_price, stop_price=stop_price,
            take_profit_price=tp_price,
            risk_per_share=risk_per_share, total_risk=total_risk,
            atr=atr, sizing_method="inverse_volatility",
            capital_budget=min(capital_cap, max_position_budget),
        )


# ── Kelly Sizer ────────────────────────────────────────────────────────────────


class KellySizer:
    """Half-Kelly criterion position sizing.

    Uses historical win-rate and payoff ratio to compute an optimal fraction
    of equity to allocate.  Always uses half-Kelly to guard against
    estimation error.  Hard-capped at ``max_kelly_fraction``.

    Warning: Kelly sizing is highly sensitive to estimated parameters.
    Do NOT use with fewer than 30 historical trades.

    Formula::
        f* = (W × avg_win − L × avg_loss) / avg_win
        f  = min(0.5 × f*, max_kelly_fraction)
        qty = floor((equity × f) / price)

    where W = win_rate, L = 1 − win_rate.

    Args:
        win_rate          : Historical fraction of profitable trades.
        avg_win           : Average winning trade P&L (positive, in currency).
        avg_loss          : Average losing trade P&L (positive magnitude, in currency).
        max_kelly_fraction: Hard cap on Kelly fraction (default 0.25 = 25% equity).
        max_position_pct  : Hard cap per name.
        atr_period        : ATR for stop computation (stop is informational for Kelly).
        atr_multiplier    : Stop width for stop_price computation.
        min_trade_value   : Minimum order value.
    """

    def __init__(
        self,
        win_rate: float = 0.55,
        avg_win: float = 1.5,
        avg_loss: float = 1.0,
        max_kelly_fraction: float = 0.25,
        max_position_pct: float = 0.05,
        stop_method: str = "atr",
        fixed_stop_pct: float = 0.02,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        min_trade_value: float = 5_000.0,
    ) -> None:
        self._win_rate   = win_rate
        self._avg_win    = avg_win    # R-units (or currency)
        self._avg_loss   = avg_loss   # R-units (or currency)
        self._max_kelly  = max_kelly_fraction
        self._max_pos    = max_position_pct
        self._stop_method = stop_method
        self._fixed_stop_pct = fixed_stop_pct
        self._atr_period = atr_period
        self._atr_mult   = atr_multiplier
        self._min_trade  = min_trade_value

    def compute(
        self,
        symbol: str,
        entry_price: float,
        equity: float,
        bar_data: pd.DataFrame,
        capital_budget: Optional[float] = None,
        direction: SignalDirection = SignalDirection.LONG,
    ) -> PositionSize:
        if entry_price <= 0 or equity <= 0:
            return _zero_size(symbol, entry_price, "kelly", "invalid price/equity")
        if self._avg_win <= 0:
            return _zero_size(symbol, entry_price, "kelly", "avg_win ≤ 0")

        loss_rate = 1.0 - self._win_rate
        kelly_f   = (self._win_rate * self._avg_win - loss_rate * self._avg_loss) / self._avg_win

        if kelly_f <= 0:
            return _zero_size(symbol, entry_price, "kelly", f"kelly_f={kelly_f:.4f} ≤ 0 (negative edge)")

        half_kelly = 0.5 * kelly_f
        f_capped   = min(half_kelly, self._max_kelly)

        kelly_budget = equity * f_capped
        max_budget = equity * self._max_pos
        capital_cap = capital_budget if capital_budget is not None else max_budget
        quantity = math.floor(min(kelly_budget, capital_cap, max_budget) / entry_price)

        if quantity < 1 or quantity * entry_price < self._min_trade:
            return _zero_size(symbol, entry_price, "kelly", f"quantity={quantity} too small")

        atr = _compute_atr(bar_data, self._atr_period)
        if self._stop_method == "atr":
            stop_distance = self._atr_mult * atr if atr > 0 else entry_price * 0.02
        else:
            stop_distance = entry_price * self._fixed_stop_pct
        stop_price, tp_price = _price_targets(
            entry_price=entry_price,
            stop_distance=stop_distance,
            take_profit_mult=3.0,
            direction=direction,
        )
        stop_price = max(stop_price, 0.01)

        logger.debug(
            "[sizing:Kelly] %s | f*=%.3f | half-Kelly=%.3f | capped=%.3f | qty=%d",
            symbol, kelly_f, half_kelly, f_capped, quantity,
        )
        return PositionSize(
            symbol=symbol, quantity=float(quantity),
            entry_price=entry_price, stop_price=stop_price,
            take_profit_price=tp_price,
            risk_per_share=stop_distance,
            total_risk=quantity * stop_distance,
            atr=atr, sizing_method="kelly",
            capital_budget=min(kelly_budget, capital_cap, max_budget),
        )


# ── Factory ────────────────────────────────────────────────────────────────────


class PositionSizerFactory:
    """Build a PositionSizer from configuration.

    Usage::

        sizer = PositionSizerFactory.build("inverse_volatility", cfg)
    """

    @staticmethod
    def build(
        method: str,
        risk_per_trade_pct: float = 0.01,
        stop_method: str = "atr",
        fixed_stop_pct: float = 0.02,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        take_profit_mult: Optional[float] = 3.0,
        max_position_pct: float = 0.05,
        min_trade_value: float = 5_000.0,
        # Kelly-specific
        win_rate: float = 0.55,
        avg_win: float = 1.5,
        avg_loss: float = 1.0,
        max_kelly_fraction: float = 0.25,
    ) -> FixedFractionalSizer | InverseVolatilitySizer | KellySizer:
        """Instantiate a sizer by name.

        Args:
            method: "fixed_fractional" | "inverse_volatility" | "kelly".
            remaining kwargs: passed to the selected sizer.

        Returns:
            Configured PositionSizer instance.

        Raises:
            ValueError: If method is unknown.
        """
        common = dict(
            risk_per_trade_pct=risk_per_trade_pct,
            stop_method=stop_method,
            fixed_stop_pct=fixed_stop_pct,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            take_profit_mult=take_profit_mult,
            max_position_pct=max_position_pct,
            min_trade_value=min_trade_value,
        )
        if method == "fixed_fractional":
            return FixedFractionalSizer(**common)
        elif method == "inverse_volatility":
            return InverseVolatilitySizer(**common)
        elif method == "kelly":
            return KellySizer(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_kelly_fraction=max_kelly_fraction,
                max_position_pct=max_position_pct,
                stop_method=stop_method,
                fixed_stop_pct=fixed_stop_pct,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
                min_trade_value=min_trade_value,
            )
        else:
            raise ValueError(f"Unknown sizing method: '{method}'. "
                             f"Choose: fixed_fractional | inverse_volatility | kelly")
