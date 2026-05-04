# CPAT Week 1 — Delivery Walkthrough

## What Was Built

A complete **foundational infrastructure layer** for a production-grade multi-asset algorithmic trading system. Every module is independently testable, type-annotated, and follows SOLID principles.

---

## File Inventory (34 files)

| File | Purpose |
|------|---------|
| `pyproject.toml` | Build config, all tool settings (mypy/ruff/pytest/coverage) |
| `config/settings.yaml` | Master YAML config — no hardcoded values anywhere |
| `cpat/core/enums.py` | All system enumerations (OrderSide, EventType, Interval…) |
| `cpat/core/models.py` | Frozen domain models: Bar, Signal, Order, Fill, Position, CostConfig |
| `cpat/core/events.py` | Typed event dataclasses: MarketEvent, SignalEvent, OrderEvent, FillEvent, RiskEvent |
| `cpat/config/loader.py` | Pydantic v2 config loader with env-var overrides |
| `cpat/data/base.py` | AbstractDataAdapter — canonical column schema + validation |
| `cpat/data/adapters/yahoo.py` | Yahoo Finance adapter (batch multi-ticker, retry, UTC normalisation) |
| `cpat/data/adapters/csv_adapter.py` | CSV adapter for offline/replay use |
| `cpat/data/store.py` | ParquetDataStore — merge-on-save, snappy compression, manifest |
| `cpat/data/universe.py` | 57-symbol universe with metadata and query methods |
| `cpat/data/pipeline.py` | Orchestrator — fetch, validate, store, load_bars |
| `cpat/backtest/event_queue.py` | Thread-safe priority queue (Market→Signal→Order→Fill ordering) |
| `cpat/backtest/handlers.py` | Protocol interfaces for all handler types |
| `cpat/backtest/engine.py` | Full event-driven BacktestEngine + SimulatedExecutionEngine |
| `cpat/strategies/base.py` | AbstractStrategy with bar history buffer and emit_signal |
| `cpat/strategies/momentum.py` | Cross-sectional momentum (Jegadeesh & Titman 1993) |
| `cpat/strategies/mean_reversion.py` | Bollinger Band z-score + Wilder RSI confirmation |
| `cpat/infrastructure/logging.py` | Console (colour) + JSON (production) rotating log handler |
| `scripts/fetch_universe.py` | CLI: download and store all universe data |
| `scripts/run_backtest.py` | CLI: run backtest, print Sharpe/DD/return summary |
| `tests/conftest.py` | Shared fixtures (no network calls) |
| `tests/test_models.py` | 22 domain model tests |
| `tests/test_data_pipeline.py` | 22 data pipeline tests |
| `tests/test_event_engine.py` | 15 backtest engine tests |
| `tests/test_strategies.py` | 16 strategy + RSI tests |

---

## Test Results

```
75 passed in 0.78s
```

| Module | Tests | Coverage Areas |
|--------|-------|---------------|
| `test_models.py` | 22 | Bar invariants, Signal bounds, Order validation, Fill P&L, Position FIFO, CostConfig |
| `test_data_pipeline.py` | 22 | Validator, normaliser, Parquet CRUD/merge, universe queries, CSV adapter |
| `test_event_engine.py` | 15 | Queue priority, next-open fill simulation, slippage/commission, engine dispatch |
| `test_strategies.py` | 16 | RSI correctness, momentum signal ranking, mean reversion entry/exit/flat |

---

## Key Engineering Decisions

### 1. Next-Bar-Open Execution Model
Orders submitted at bar T are filled at bar T+1's **open price**. This prevents the ubiquitous "buy today's close, count today's return" bias that inflates backtests by 10-30 bps per trade.

### 2. Priority Event Queue
`MarketEvent(1) < SignalEvent(2) < OrderEvent(3) < FillEvent(4)` — enforced by the priority queue. Within any bar, causality is guaranteed: you see the market before signalling, signal before ordering, order before filling.

### 3. Frozen Domain Models
`Bar`, `Signal`, `Order`, `Fill` are all `frozen=True` dataclasses. This prevents accidental mutation across handler boundaries and makes them hashable (safe as dict keys and in sets).

### 4. Dependency Injection Throughout
`BacktestEngine`, `DataPipeline`, `AbstractStrategy` all receive their dependencies via constructor injection. Zero module-level singletons. Easy to mock in tests.

### 5. Parquet Merge-on-Save
Re-downloading data merges with existing store (new data wins on timestamp conflicts). The pipeline is **idempotent** — safe to re-run without duplication.

### 6. Wilder RSI (not simple SMA RSI)
The mean reversion strategy uses the canonical Wilder's smoothed RSI (alpha = 1/period), not the simplified SMA version. This matches the standard used in all professional trading platforms.

---

## Bias Prevention Summary

| Bias | Technique |
|------|-----------|
| Look-ahead (execution) | Next-bar open fill model |
| Look-ahead (indicator) | Rolling windows: only `prices.iloc[:-skip]` used in formation |
| Survivorship | Universe frozen at `start_date`; no dynamic addition |
| Data snooping | Parameters from academic literature (J&T 1993, Bollinger 1992) |
| Overfitting | No in-sample parameter search in Week 1 |
| Transaction cost bias | 5 bps slippage + 5 bps commission applied at every fill |

---

## How to Run

```bash
# 1. Activate venv and install
source .venv/bin/activate && pip install -e .

# 2. Download data (~2 min)
python scripts/fetch_universe.py

# 3. Run backtest
python scripts/run_backtest.py --strategy both

# Expected output:
# ============================================================
#   CPAT Backtest Results
# ============================================================
#   Period:         2019-01-01 → 2024-12-31
#   Initial Capital: $  1,000,000.00
#   Final Equity:    $  X,XXX,XXX.XX
#   Total Return:   +XX.XX%
#   Ann. Return:    +XX.XX%
#   Sharpe Ratio:    X.XX
#   Max Drawdown:   -XX.XX%
#   Total Trades:      XXX
```

---

## Week 2 Roadmap

| Component | Description |
|-----------|------------|
| **Risk Engine** | `max_position_pct`, `max_drawdown_pct`, correlation constraints, position halt |
| **Portfolio Manager** | Inverse-volatility position sizing, equal-weight baseline, rebalance scheduling |
| **Performance Analytics** | Sharpe, Sortino, Calmar, Alpha, Beta vs benchmark, monthly returns heatmap |
| **Alpaca Adapter** | Paper trading integration: submit orders, poll fills, websocket feed |
| **Walk-Forward Validation** | Rolling train/test windows to assess out-of-sample robustness |
| **Signal Combination** | Ensemble: momentum score × (1 - mean_reversion_z) weighting |

---

## Limitations (Week 1)

- Strategies are complete but portfolio manager is a stub — signals don't translate to sized positions yet (requires Week 2 risk engine)
- No live data feed support
- Execution engine is simulated (Alpaca wired in Week 2)
- Performance metrics computed inline in the run script (full analytics module in Week 3)
- No walk-forward or out-of-sample testing yet
