"""
CPAT — Performance Analytics
==============================
Computes all performance metrics from an equity curve and trade log.

Metrics computed:
    ┌─────────────────────────┬─────────────────────────────────────────┐
    │ Metric                  │ Formula                                 │
    ├─────────────────────────┼─────────────────────────────────────────┤
    │ Total Return            │ (V_final - V_initial) / V_initial       │
    │ Annualised Return       │ (1 + TR)^(252/N) - 1                   │
    │ Annualised Volatility   │ std(daily_returns) × √252               │
    │ Sharpe Ratio            │ E[r - rf] / std(r - rf) × √252         │
    │ Sortino Ratio           │ E[r - rf] / downside_std × √252        │
    │ Max Drawdown            │ max(V_peak - V_t) / V_peak              │
    │ Max Drawdown Duration   │ longest consecutive drawdown (days)     │
    │ Calmar Ratio            │ Ann. Return / abs(Max Drawdown)         │
    │ Win Rate                │ profitable fills / total fills          │
    │ Profit Factor           │ gross_profit / abs(gross_loss)         │
    └─────────────────────────┴─────────────────────────────────────────┘

All results are returned as a typed ``PerformanceReport`` dataclass, making
them easy to serialise to JSON for dashboards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from cpat.core.models import Fill
from cpat.core.enums import OrderSide

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ── Report dataclass ───────────────────────────────────────────────────────────


@dataclass
class PerformanceReport:
    """Full performance analytics report.

    All rates/ratios are annualised unless otherwise noted.

    Attributes:
        total_return: Decimal total return (e.g. 0.15 = 15%).
        ann_return: Annualised return.
        ann_volatility: Annualised return standard deviation.
        sharpe_ratio: Sharpe ratio (excess return / volatility).
        sortino_ratio: Sortino ratio (excess return / downside deviation).
        max_drawdown: Maximum peak-to-trough drawdown (negative, e.g. -0.15).
        max_drawdown_duration_days: Longest drawdown duration in calendar days.
        calmar_ratio: Ann. return / abs(max drawdown).
        win_rate: Fraction of fills that were profitable.
        profit_factor: Gross profit / abs(gross loss).
        n_trades: Total number of fills.
        n_profitable: Number of profitable fills.
        avg_win: Average winning fill P&L.
        avg_loss: Average losing fill P&L (negative).
        initial_capital: Starting capital.
        final_equity: Ending equity.
        risk_free_rate: Annual risk-free rate used for Sharpe/Sortino.
    """

    total_return: float
    ann_return: float
    ann_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    n_trades: int
    n_profitable: int
    avg_win: float
    avg_loss: float
    initial_capital: float
    final_equity: float
    risk_free_rate: float = 0.04
    # ── Week 3 extensions ──────────────────────────────────────────────────────
    expectancy: float = 0.0           # (win_rate × avg_win) + (loss_rate × avg_loss)
    skewness: float = 0.0             # return distribution skewness
    kurtosis: float = 0.0             # excess kurtosis (0 = Normal)
    tail_ratio: float = 1.0           # abs(95th pct) / abs(5th pct)
    ulcer_index: float = 0.0          # RMS of drawdown depths (pain index)
    recovery_factor: float = 0.0      # total_return / abs(max_drawdown)

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to plain dict for JSON/CSV export."""
        return {
            "total_return_pct": round(self.total_return * 100, 4),
            "ann_return_pct": round(self.ann_return * 100, 4),
            "ann_volatility_pct": round(self.ann_volatility * 100, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown * 100, 4),
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "calmar_ratio": round(self.calmar_ratio, 4),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 4),
            "n_trades": self.n_trades,
            "n_profitable": self.n_profitable,
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.final_equity, 2),
            # Week 3
            "expectancy": round(self.expectancy, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "tail_ratio": round(self.tail_ratio, 4),
            "ulcer_index": round(self.ulcer_index, 4),
            "recovery_factor": round(self.recovery_factor, 4),
        }

    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  CPAT Performance Report\n"
            f"{'='*60}\n"
            f"  Total Return:          {self.total_return:>+.2%}\n"
            f"  Ann. Return:           {self.ann_return:>+.2%}\n"
            f"  Ann. Volatility:       {self.ann_volatility:>.2%}\n"
            f"  Sharpe Ratio:          {self.sharpe_ratio:>6.3f}\n"
            f"  Sortino Ratio:         {self.sortino_ratio:>6.3f}\n"
            f"  Calmar Ratio:          {self.calmar_ratio:>6.3f}\n"
            f"  Ulcer Index:           {self.ulcer_index:>6.3f}\n"
            f"  Recovery Factor:       {self.recovery_factor:>6.3f}\n"
            f"  Max Drawdown:          {self.max_drawdown:>+.2%}\n"
            f"  Max DD Duration:       {self.max_drawdown_duration_days} days\n"
            f"  Win Rate:              {self.win_rate:.1%}\n"
            f"  Profit Factor:         {self.profit_factor:.2f}\n"
            f"  Expectancy:            {self.expectancy:>+.2f}\n"
            f"  Skewness:              {self.skewness:>+.3f}\n"
            f"  Kurtosis (excess):     {self.kurtosis:>+.3f}\n"
            f"  Tail Ratio:            {self.tail_ratio:>6.3f}\n"
            f"  Total Trades:          {self.n_trades}\n"
            f"  Profitable Trades:     {self.n_profitable}\n"
            f"  Avg Win:               {self.avg_win:>+10.2f}\n"
            f"  Avg Loss:              {self.avg_loss:>+10.2f}\n"
            f"  Initial Capital:       {self.initial_capital:>14,.0f}\n"
            f"  Final Equity:          {self.final_equity:>14,.0f}\n"
            f"{'='*60}\n"
        )


# ── Performance tracker ────────────────────────────────────────────────────────


class PerformanceTracker:
    """Computes and accumulates performance metrics from a backtest run.

    Usage::

        tracker = PerformanceTracker(initial_capital=1_000_000)
        # During backtest loop:
        tracker.record(timestamp, portfolio.total_equity(prices))
        # After backtest:
        report = tracker.compute(fills=result.fills)
        print(report)

    Args:
        initial_capital: Starting portfolio value.
        risk_free_rate: Annualised risk-free rate (default: 4%).
    """

    def __init__(
        self,
        initial_capital: float,
        risk_free_rate: float = 0.04,
    ) -> None:
        self._initial_capital = initial_capital
        self._risk_free_rate = risk_free_rate
        self._equity_records: list[tuple[pd.Timestamp, float]] = []

    def record(self, timestamp: pd.Timestamp, equity: float) -> None:
        """Record an equity snapshot.

        Args:
            timestamp: UTC bar timestamp.
            equity: Total portfolio value at this timestamp.
        """
        self._equity_records.append((timestamp, equity))

    def equity_curve(self) -> pd.Series:
        """Return equity curve as a Series with DatetimeIndex.

        Returns:
            Series of equity values (UTC indexed).
        """
        if not self._equity_records:
            return pd.Series(dtype=float, name="equity")
        idx = [r[0] for r in self._equity_records]
        vals = [r[1] for r in self._equity_records]
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")

    def daily_returns(self) -> pd.Series:
        """Compute period-over-period returns from the equity curve."""
        curve = self.equity_curve()
        if len(curve) < 2:
            return pd.Series(dtype=float, name="returns")
        return curve.pct_change().dropna().rename("returns")

    def drawdown_series(self) -> pd.Series:
        """Compute rolling drawdown from peak (negative values).

        Returns:
            Series of drawdown values in decimal (e.g. -0.05 = -5%).
        """
        curve = self.equity_curve()
        if curve.empty:
            return pd.Series(dtype=float, name="drawdown")
        peak = curve.cummax()
        dd = (curve - peak) / peak.replace(0, float("nan"))
        return dd.fillna(0.0).rename("drawdown")

    def compute(self, fills: list[Fill] | None = None) -> PerformanceReport:
        """Compute the full performance report.

        Args:
            fills: List of all fills from the backtest (for trade-level metrics).

        Returns:
            PerformanceReport with all metrics populated.
        """
        fills = fills or []
        curve = self.equity_curve()
        returns = self.daily_returns()

        # ── Basic return metrics ───────────────────────────────────────────────
        final_equity = float(curve.iloc[-1]) if not curve.empty else self._initial_capital
        total_return = (final_equity - self._initial_capital) / self._initial_capital

        n = len(returns)
        ann_return = (1 + total_return) ** (TRADING_DAYS / max(n, 1)) - 1

        # ── Volatility ────────────────────────────────────────────────────────
        ann_volatility = float(returns.std()) * (TRADING_DAYS ** 0.5) if n > 1 else 0.0

        # ── Sharpe ratio ──────────────────────────────────────────────────────
        daily_rf = self._risk_free_rate / TRADING_DAYS
        excess_returns = returns - daily_rf
        sharpe = (
            float(excess_returns.mean()) / float(excess_returns.std()) * (TRADING_DAYS ** 0.5)
            if n > 1 and excess_returns.std() > 0
            else 0.0
        )

        # ── Sortino ratio ─────────────────────────────────────────────────────
        downside = excess_returns[excess_returns < 0]
        downside_std = float(downside.std()) * (TRADING_DAYS ** 0.5) if len(downside) > 1 else 1e-10
        sortino = (ann_return - self._risk_free_rate) / downside_std if downside_std > 0 else 0.0

        # ── Drawdown metrics ──────────────────────────────────────────────────
        dd_series = self.drawdown_series()
        max_drawdown = float(dd_series.min()) if not dd_series.empty else 0.0
        max_dd_duration = self._compute_max_drawdown_duration(curve)

        # ── Calmar ratio ──────────────────────────────────────────────────────
        calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        # ── Trade-level metrics ────────────────────────────────────────────────
        trade_pnl = self._compute_trade_pnl(fills)
        n_trades = len(trade_pnl)
        wins = [p for p in trade_pnl if p > 0]
        losses = [p for p in trade_pnl if p < 0]
        n_profitable = len(wins)

        win_rate = n_profitable / n_trades if n_trades > 0 else 0.0
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # ── Week 3: extended metrics ──────────────────────────────────────────
        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # Return distribution statistics
        arr = returns.to_numpy()
        if len(arr) >= 4:
            skewness = float(scipy_stats.skew(arr, bias=False))
            kurtosis = float(scipy_stats.kurtosis(arr, fisher=True, bias=False))
            p5 = float(np.percentile(arr, 5))
            p95 = float(np.percentile(arr, 95))
            tail_ratio = abs(p95) / max(abs(p5), 1e-10)
        else:
            skewness = kurtosis = 0.0
            tail_ratio = 1.0

        # Ulcer Index
        from cpat.analytics.drawdown import ulcer_index as _ulcer_index
        ui = _ulcer_index(curve)

        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        return PerformanceReport(
            total_return=total_return,
            ann_return=ann_return,
            ann_volatility=ann_volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=max_dd_duration,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            n_trades=n_trades,
            n_profitable=n_profitable,
            avg_win=avg_win,
            avg_loss=avg_loss,
            initial_capital=self._initial_capital,
            final_equity=final_equity,
            risk_free_rate=self._risk_free_rate,
            expectancy=expectancy,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            ulcer_index=ui,
            recovery_factor=recovery_factor,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_max_drawdown_duration(equity_curve: pd.Series) -> int:
        """Compute maximum drawdown duration in calendar days.

        Returns:
            Duration of the longest drawdown period in days.
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return 0

        peak = equity_curve.cummax()
        in_drawdown = equity_curve < peak

        max_duration = 0
        current_duration = 0
        start_ts: Optional[pd.Timestamp] = None

        for ts, is_down in in_drawdown.items():
            if is_down:
                if start_ts is None:
                    start_ts = ts
                current_duration = (ts - start_ts).days
                max_duration = max(max_duration, current_duration)
            else:
                start_ts = None
                current_duration = 0

        return max_duration

    @staticmethod
    def _compute_trade_pnl(fills: list[Fill]) -> list[float]:
        """Compute round-trip P&L for each completed trade pair.

        Simple FIFO matching: matches buys and sells chronologically.
        Only computes P&L for closed round-trips.

        Args:
            fills: List of all fills.

        Returns:
            List of P&L values for each closed trade.
        """
        # Group by symbol
        symbol_fills: dict[str, list[Fill]] = {}
        for fill in fills:
            symbol_fills.setdefault(fill.symbol, []).append(fill)

        all_pnl: list[float] = []

        for symbol, sym_fills in symbol_fills.items():
            # FIFO matching
            buy_queue: list[tuple[float, float]] = []  # (price, qty)
            for fill in sym_fills:
                if fill.side == OrderSide.BUY:
                    buy_queue.append((fill.fill_price, fill.quantity))
                else:
                    sell_qty = fill.quantity
                    while sell_qty > 0 and buy_queue:
                        buy_price, buy_qty = buy_queue[0]
                        matched_qty = min(sell_qty, buy_qty)
                        pnl = (fill.fill_price - buy_price) * matched_qty - fill.commission
                        all_pnl.append(pnl)
                        sell_qty -= matched_qty
                        remaining_buy = buy_qty - matched_qty
                        if remaining_buy < 1e-9:
                            buy_queue.pop(0)
                        else:
                            buy_queue[0] = (buy_price, remaining_buy)

        return all_pnl
