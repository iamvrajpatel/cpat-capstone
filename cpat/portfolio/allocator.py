"""
CPAT — Capital Allocator (Week 4)
===================================
Distributes available portfolio capital across active signals before
the position sizing step.

Two allocation methods are provided:

EqualWeightAllocator
    Divides deployable capital equally across all active (LONG) signals.
    Simple, robust, easy to reason about.

    Rule::
        capital_per_signal = deployable_capital / max(n_long_signals, 1)

    Cap: each slice is also capped by ``max_position_pct × equity``.

VolatilityAdjustedAllocator
    Weights each signal inversely proportional to its recent volatility
    (measured as 20-day rolling standard deviation of close-to-close returns).
    High-volatility assets receive smaller capital slices automatically.

    Rule::
        weight_i = (1/σ_i) / Σ(1/σ_j)  for all active signals j
        capital_i = deployable_capital × weight_i

    Cap: each slice is capped by ``max_position_pct × equity``.
    If σ_i cannot be computed (insufficient data), the symbol receives
    equal weight as a fallback.

Deployable capital
    Both allocators only deploy ``(1 − min_cash_pct) × cash``, preserving
    the cash buffer enforced by the RiskEngine.

Usage::

    from cpat.portfolio.allocator import VolatilityAdjustedAllocator, CapitalSlice

    allocator = VolatilityAdjustedAllocator(
        max_position_pct=0.04,
        min_cash_pct=0.05,
        vol_window=20,
    )
    slices = allocator.allocate(
        signals=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
        equity=10_000_000.0,
        cash=3_000_000.0,
        bar_data={"RELIANCE.NS": reliance_df, ...},
    )
    for sl in slices:
        print(sl.symbol, sl.target_value, sl.weight)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class CapitalSlice:
    """Capital allocation for one symbol.

    Attributes:
        symbol       : Ticker symbol.
        target_value : Target position value in currency units.
        weight       : Fraction of total deployable capital allocated.
        sigma        : Annualised volatility used for weighting (0.0 = equal weight).
    """

    symbol: str
    target_value: float
    weight: float
    sigma: float = 0.0

    @property
    def is_valid(self) -> bool:
        """True if the slice has a meaningful allocation."""
        return self.target_value > 0.0 and self.weight > 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "target_value": round(self.target_value, 2),
            "weight": round(self.weight, 6),
            "sigma": round(self.sigma, 6),
        }


# ── Protocol ───────────────────────────────────────────────────────────────────


class CapitalAllocator(Protocol):
    """Protocol that all capital allocators must implement."""

    def allocate(
        self,
        signals: list[str],
        equity: float,
        cash: float,
        bar_data: dict[str, pd.DataFrame],
    ) -> list[CapitalSlice]:
        """Distribute capital across active signals.

        Args:
            signals : List of symbols with active LONG signals.
            equity  : Total portfolio equity (cash + positions).
            cash    : Available cash balance.
            bar_data: Historical bar data per symbol for volatility estimation.

        Returns:
            List of CapitalSlice, one per signal.
            Symbols that cannot be traded (e.g. insufficient data) receive
            a slice with target_value=0.
        """
        ...


# ── Helpers ────────────────────────────────────────────────────────────────────


def _compute_sigma(bar_data: pd.DataFrame, window: int = 20) -> float:
    """Annualised close-to-close return volatility.

    Args:
        bar_data: OHLCV DataFrame.
        window  : Rolling window for standard deviation.

    Returns:
        Annualised sigma (√252 scaling). 0.0 if insufficient data.
    """
    if bar_data is None or len(bar_data) < window + 2:
        return 0.0
    closes = bar_data["close"].tail(window + 1)
    if (closes <= 0).any():
        return 0.0
    rets = closes.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    return float(rets.std(ddof=1)) * (252 ** 0.5)


def _apply_caps(
    raw_weights: dict[str, float],
    deployable_capital: float,
    max_per_position: float,
) -> dict[str, float]:
    """Apply per-position caps and redistribute leftover capital iteratively."""
    if deployable_capital <= 0 or not raw_weights:
        return {symbol: 0.0 for symbol in raw_weights}

    allocations = {symbol: 0.0 for symbol in raw_weights}
    remaining = set(raw_weights.keys())
    capital_left = deployable_capital

    while remaining and capital_left > 1e-9:
        weight_total = sum(raw_weights[symbol] for symbol in remaining)
        if weight_total <= 0:
            break

        capped_this_round: set[str] = set()
        for symbol in list(remaining):
            normalized_weight = raw_weights[symbol] / weight_total
            desired_value = capital_left * normalized_weight
            room_left = max(0.0, max_per_position - allocations[symbol])
            allocated = min(desired_value, room_left)
            allocations[symbol] += allocated

            if room_left - allocated <= 1e-9:
                capped_this_round.add(symbol)

        capital_left = max(0.0, deployable_capital - sum(allocations.values()))
        if not capped_this_round:
            break
        remaining.difference_update(capped_this_round)

    return allocations


# ── Equal Weight Allocator ─────────────────────────────────────────────────────


class EqualWeightAllocator:
    """Divide deployable capital equally across all active signals.

    Args:
        max_position_pct: Hard cap per position as fraction of equity.
        min_cash_pct    : Minimum cash buffer — only ``(1−min_cash_pct) × cash``
                          is available for deployment.
    """

    def __init__(
        self,
        max_position_pct: float = 0.05,
        min_cash_pct: float = 0.05,
    ) -> None:
        self._max_pos_pct = max_position_pct
        self._min_cash_pct = min_cash_pct

    def allocate(
        self,
        signals: list[str],
        equity: float,
        cash: float,
        bar_data: dict[str, pd.DataFrame],
    ) -> list[CapitalSlice]:
        """Allocate capital equally.

        Args:
            signals : Active signal symbols.
            equity  : Total portfolio equity.
            cash    : Available cash.
            bar_data: Not used by equal-weight; present for API consistency.

        Returns:
            List of CapitalSlice with equal allocations.
        """
        if not signals or equity <= 0:
            return []

        deployable = max(0.0, cash * (1.0 - self._min_cash_pct))
        max_per_position = equity * self._max_pos_pct

        raw_weights = {sym: 1.0 / len(signals) for sym in signals}
        capped_allocations = _apply_caps(raw_weights, deployable, max_per_position)

        slices: list[CapitalSlice] = []
        for sym in signals:
            target = capped_allocations[sym]
            weight = target / deployable if deployable > 0 else 0.0
            slices.append(CapitalSlice(
                symbol=sym,
                target_value=target,
                weight=weight,
                sigma=0.0,
            ))
            logger.debug("[alloc:EQ] %s | target=%.0f | weight=%.3f", sym, target, weight)

        return slices


# ── Volatility-Adjusted Allocator ─────────────────────────────────────────────


class VolatilityAdjustedAllocator:
    """Allocate capital inversely proportional to 20-day volatility.

    Symbols with higher volatility receive smaller allocations, creating
    a portfolio where each position contributes roughly equal risk.

    Fallback: symbols without sufficient history for volatility estimation
    receive equal weight based on the average of computable sigmas.

    Args:
        max_position_pct: Hard cap per position as fraction of equity.
        min_cash_pct    : Minimum cash buffer fraction.
        vol_window      : Return window for sigma estimation (bars).
    """

    def __init__(
        self,
        max_position_pct: float = 0.05,
        min_cash_pct: float = 0.05,
        vol_window: int = 20,
    ) -> None:
        self._max_pos_pct  = max_position_pct
        self._min_cash_pct = min_cash_pct
        self._vol_window   = vol_window

    def allocate(
        self,
        signals: list[str],
        equity: float,
        cash: float,
        bar_data: dict[str, pd.DataFrame],
    ) -> list[CapitalSlice]:
        """Allocate capital with volatility-inverse weighting.

        Args:
            signals : Active signal symbols.
            equity  : Total portfolio equity.
            cash    : Available cash.
            bar_data: Historical bar data for sigma estimation.

        Returns:
            List of CapitalSlice with volatility-adjusted allocations.
        """
        if not signals or equity <= 0:
            return []

        deployable      = max(0.0, cash * (1.0 - self._min_cash_pct))
        max_per_pos     = equity * self._max_pos_pct

        # Compute per-symbol sigma
        sigmas: dict[str, float] = {}
        for sym in signals:
            df = bar_data.get(sym)
            sigmas[sym] = _compute_sigma(df, self._vol_window) if df is not None else 0.0

        # Fallback: symbols with sigma=0 get the mean of computable sigmas
        valid_sigmas = [s for s in sigmas.values() if s > 0]
        fallback_sigma = float(np.mean(valid_sigmas)) if valid_sigmas else 1.0

        inv_sigmas: dict[str, float] = {
            sym: 1.0 / max(sig, fallback_sigma)
            for sym, sig in sigmas.items()
        }
        total_inv = sum(inv_sigmas.values())
        if total_inv <= 0:
            # Degenerate: fall back to equal weight
            logger.warning("[alloc:VA] All sigmas zero — falling back to equal weight.")
            return EqualWeightAllocator(self._max_pos_pct, self._min_cash_pct).allocate(
                signals, equity, cash, bar_data
            )

        raw_weights = {sym: inv_sigmas[sym] / total_inv for sym in signals}
        capped_allocations = _apply_caps(raw_weights, deployable, max_per_pos)
        slices: list[CapitalSlice] = []
        for sym in signals:
            target = capped_allocations[sym]
            final_weight = target / deployable if deployable > 0 else 0.0

            slices.append(CapitalSlice(
                symbol=sym,
                target_value=target,
                weight=final_weight,
                sigma=sigmas[sym],
            ))
            logger.debug(
                "[alloc:VA] %s | σ=%.4f | target=%.0f | weight=%.3f",
                sym, sigmas[sym], target, final_weight,
            )

        return slices


# ── Factory ────────────────────────────────────────────────────────────────────


class AllocatorFactory:
    """Build a CapitalAllocator from a method name.

    Usage::

        alloc = AllocatorFactory.build("volatility_adjusted",
                                       max_position_pct=0.04,
                                       min_cash_pct=0.05)
    """

    @staticmethod
    def build(
        method: str,
        max_position_pct: float = 0.05,
        min_cash_pct: float = 0.05,
        vol_window: int = 20,
    ) -> EqualWeightAllocator | VolatilityAdjustedAllocator:
        """Instantiate an allocator by name.

        Args:
            method: "equal_weight" | "volatility_adjusted".
            max_position_pct: Max single-name weight.
            min_cash_pct    : Cash reserve fraction.
            vol_window      : Volatility estimation window.

        Returns:
            Configured allocator instance.

        Raises:
            ValueError: If method is unknown.
        """
        if method == "equal_weight":
            return EqualWeightAllocator(max_position_pct, min_cash_pct)
        elif method == "volatility_adjusted":
            return VolatilityAdjustedAllocator(max_position_pct, min_cash_pct, vol_window)
        else:
            raise ValueError(
                f"Unknown allocator: '{method}'. "
                f"Choose: equal_weight | volatility_adjusted"
            )
