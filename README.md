# CPAT — Multi-Asset Algorithmic Trading System

[![Tests](https://img.shields.io/badge/tests-357%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-85%25-green)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Architecture](https://img.shields.io/badge/architecture-event--driven-purple)]()
[![Week](https://img.shields.io/badge/week-5%20live-orange)]()

A **production-grade, event-driven multi-asset algorithmic trading system** for quantitative research and live trading. Supports the **India universe** (69 NSE equities + 11 Indian indices + 9 global commodity futures), two strategy families (momentum + mean reversion), a fully integrated risk engine, ATR-based position sizing, stop-loss management, institutional-grade performance analytics, parameter optimization, and walk-forward validation.

---

## Architecture

```
cpat/
├── core/                Domain models (Bar, Signal, Order, Fill, Position) — frozen, typed
├── config/              YAML-driven config with Pydantic v2 validation
├── data/
│   ├── handler.py       Forward-only DataHandler (look-ahead bias prevention)
│   ├── adapters/        Yahoo Finance + CSV adapters
│   └── store.py         Parquet-backed local storage
├── backtest/            Event-driven engine v2 (Market → Signal → Order → Fill)
├── strategies/          Momentum (cross-sectional) + Mean Reversion (Bollinger + RSI)
├── portfolio/
│   ├── manager.py       PortfolioManager — cash, positions, equity, exposure
│   ├── translator.py    SignalOrderTranslator — signal → sized order
│   └── allocator.py     CapitalAllocator — equal/vol-adjusted distribution  [Week 4]
├── execution/
│   └── engine.py        ExecutionEngine v2 — 3 slippage models, partial fills
├── risk/
│   ├── engine.py          RiskEngine — 5 configurable pre-trade constraints
│   ├── position_sizing.py FixedFractional + InverseVol + Kelly sizers  [Week 4]
│   └── risk_manager.py    StopLossTracker + PortfolioRiskManager  [Week 4]
├── analytics/
│   ├── performance.py   PerformanceTracker — 20 metrics (Sharpe, Sortino, Calmar…)
│   ├── drawdown.py      DrawdownPeriod table, Ulcer Index  [Week 3]
│   ├── distributions.py ReturnDistribution, VaR, CVaR, Jarque-Bera  [Week 3]
│   └── trade_log.py     TradeLog — CSV + Parquet audit trail
├── optimization/
│   └── optimizer.py     GridSearch + RandomSearch (train split only)  [Week 3]
├── validation/
│   ├── splitter.py      Forward-only train/test split + CV folds  [Week 3]
│   ├── walk_forward.py  Rolling OOS walk-forward harness  [Week 3]
│   └── overfitting.py   Parameter sensitivity + degradation report  [Week 3]
├── brokers/             PaperBroker + DhanBroker adapters  [Week 5]
└── infrastructure/      OMS, scheduler, live engine, structured logging  [Week 5]
```

**Event flow (per bar):**

```
DataHandler.get_next()
    │
    ├─ 0. StopLossTracker.check_stops()          ← stops fire at bar open  [Week 4]
    │       └─ Emits FLAT SignalEvents for breached stops/TPs
    │
    ├─ 1. ExecutionEngine.process_pending()      ← fills at THIS bar's open (anti-bias)
    │       └─ PortfolioManager.apply_fill()
    │       └─ TradeLog.record()
    │
    ├─ 2. queue.put(MarketEvent)                 ← strategy sees bar AFTER fill
    │
    ├─ 3. drain_queue()
    │       MarketEvent  → Strategy.on_market()  → SignalEvent
    │       SignalEvent  → PortfolioAllocator.allocate()   [Week 4]
    │                   → PositionSizer.compute()         [Week 4]
    │                   → PortfolioRiskManager.can_open() [Week 4]
    │                   → SignalOrderTranslator → RiskEngine.check() → OrderEvent
    │       OrderEvent   → ExecutionEngine.submit()  (deferred to next bar)
    │
    └─ 4. PerformanceTracker.record(equity)
```

**Week 3 Analytics Flow:**

```
BacktestResult.equity_curve
        │
        ├── drawdown.py          → DrawdownPeriod table, ulcer_index
        ├── distributions.py     → skewness, kurtosis, VaR95, CVaR95, Jarque-Bera
        └── performance.py       → PerformanceReport (20 metrics)
                │
    ┌───────────┴──────────────────────────┐
    │                                      │
optimizer.py                       walk_forward.py
(train split only)                 (rolling IS + OOS folds)
    │                                      │
OptimizationResult[]              WalkForwardResult
results_to_dataframe()             ↓ combined_oos_equity
    │                              ↓ degradation_ratio
    └──────────────────────────────┘
                    │
          overfitting.py
          degradation_report()   → IS vs OOS structured comparison
          compute_sensitivity()  → CV per parameter
          stability_check()      → all-parameter fragility table
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install all dependencies
pip install -r requirements-dev.txt && pip install -e .

# 3. Download historical data (Yahoo Finance — Indian universe)
python scripts/fetch_universe.py

# 4. Run a standard backtest (full period, complete report)
python scripts/run_backtest.py --mode backtest --strategy momentum

# 5. Optimize parameters on training split, evaluate OOS
python scripts/run_backtest.py --mode optimize --strategy momentum

# 6. Walk-forward validation (fold table + stitched OOS equity)
python scripts/run_backtest.py --mode walk-forward --strategy momentum

# 7. Run tests with coverage
pytest tests/ -v --cov=cpat
```

---

## Universe — India Diversified (89 Symbols)

| Category | Count | Examples |
|----------|-------|---------|
| **NSE Equities** | 69 | TCS, INFY, HDFCBANK, RELIANCE, SUNPHARMA, ITC, MARUTI |
| **Indian Indices** | 11 | `^NSEI` (Nifty 50), `^BSESN`, `^NSEBANK`, `^CNXIT`, `^CNXPHARMA` |
| **Global Commodities** | 9 | `GC=F` (Gold), `CL=F` (WTI), `GS=F` (Silver), `NG=F` (NatGas) |

Equities span 10 sectors: IT, Banking & Finance, Energy, Power, Healthcare, FMCG, Auto, Metals, Infrastructure, Telecom.

---

## Configuration

All parameters live in `config/settings.yaml`. No hardcoded values anywhere.

```yaml
system:
  timezone: Asia/Kolkata        # IST — Indian Standard Time

backtest:
  start_date: "2019-01-01"
  end_date:   "2024-12-31"
  initial_capital: 10_000_000  # ₹1 Crore
  benchmark_symbol: "^NSEI"    # Nifty 50 benchmark
  warmup_bars: 252

costs:
  commission:
    model: PERCENTAGE
    value: 0.0003              # 3 bps — Zerodha/Upstox model
    min_commission: 20.0       # ₹20 minimum per order
  slippage:
    model: SPREAD_BASED        # FIXED_BPS | VOLUME_WEIGHTED | SPREAD_BASED
    bps: 8.0

risk:
  max_position_pct: 0.04       # 4% max per position
  max_sector_pct: 0.25         # 25% max per sector
  max_gross_exposure_pct: 1.0  # Long-only (cash segment)
  max_drawdown_pct: 0.12       # 12% drawdown halt
  min_cash_pct: 0.05           # 5% cash buffer (STT + settlement)
  risk_free_rate: 0.065        # 6.5% — RBI repo rate
  allocation_method: volatility_adjusted
  max_open_positions: 20
  max_daily_loss_pct: 0.02

position_sizing:
  method: inverse_volatility
  stop_method: atr             # atr | fixed_pct
  fixed_stop_pct: 0.02
  atr_period: 14
  atr_multiplier: 2.0
  take_profit_mult: 3.0
  trailing_stop: false
  vol_window: 20
  min_trade_value: 5000.0

optimization:
  method: random               # grid | random
  n_trials: 50
  metric: sharpe_ratio
  train_ratio: 0.70            # 70% train / 30% test

validation:
  walk_forward_train_bars: 756  # ~3 years
  walk_forward_test_bars: 252   # ~1 year
  walk_forward_step_bars: 63    # quarterly step
  optimize_on_fold: false
```

## Live Trading (Week 5)

### System Architecture

```
Scheduler
  -> LiveExecutionEngine.run_tick()
      -> data provider / broker quote sync
      -> StopLossTracker
      -> Strategy.on_market() via MarketEvent
      -> signal batch -> allocator -> position sizer -> risk checks
      -> OrderManager (OMS)
      -> BrokerInterface adapter (PaperBroker / DhanBroker)
      -> fill polling -> PortfolioManager update
      -> structured logs + heartbeat + reconciliation
```

### Broker Integration

- `PaperBroker`: mandatory paper-trading acceptance path, instant in-memory fills, no API keys required.
- `DhanBroker`: live Indian broker adapter using `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN`.
- `BrokerInterface` is the canonical adapter contract; broker-specific logic is isolated behind the adapter.
- Limitation: the current live path is candle-driven and uses a rolling OHLCV provider, not tick-by-tick execution.

### OMS Lifecycle

- Internal `OrderStatus` remains the source of truth for code paths.
- Audit/log lifecycle labels are exported as:
  - `CREATED`
  - `SENT`
  - `PARTIALLY_FILLED`
  - `FILLED`
  - `REJECTED`
  - `CANCELLED`
  - `EXPIRED`
- OMS supports idempotent broker status application, stale-order expiry, and JSONL audit-log replay on restart.

### Failure Handling

- Broker connectivity errors retry with exponential backoff.
- Stale or missing market data causes a tick or symbol to be skipped, never a process crash.
- Reconciliation mismatches between broker state and local portfolio block new opens and emit critical logs.
- Scheduler overlap protection skips a tick instead of running concurrent execution loops.

### Deployment

```bash
# Paper/demo validation
python scripts/run_live.py --mode paper --demo

# Paper with dry-run order generation only
python scripts/run_live.py --mode paper --dry-run --symbols RELIANCE.NS TCS.NS

# Live Dhan mode (requires env vars + local seed data)
export DHAN_CLIENT_ID=...
export DHAN_ACCESS_TOKEN=...
python scripts/run_live.py --mode dhan --symbols RELIANCE.NS TCS.NS
```

- Logs are written to `logs/live/`.
- OMS audit trail is written to `logs/live/oms_audit.jsonl`.
- For unattended deployment, run `scripts/run_live.py` under `systemd`, Docker, or another process supervisor.

## Streamlit Control Console

The project now includes a modular Streamlit orchestration layer under `ui/`.

```bash
streamlit run ui/app.py
```

### UI Architecture

- `ui/app.py`: application entrypoint, navigation shell, theme injection
- `ui/pages/`: dashboard, backtest, optimization, live trading, portfolio/risk, analytics, logs, settings
- `ui/services/`: config parsing, backend execution wrappers, analytics adapters, strategy metadata
- `ui/components/`: reusable forms, charts, metric cards, tables, live controls
- `ui/state/session_manager.py`: typed `st.session_state` abstraction

### What the UI Controls

- Dynamic config editing from `config/settings.yaml`
- Backtest, compare, optimize, and walk-forward execution
- Parameter-sweep overlays without mutating scalar YAML defaults
- Managed `scripts/run_live.py` subprocess control for paper/live sessions
- Saved research artifacts in `data/results/`
- Runtime, live JSONL, and OMS audit logs in `logs/`

### Mock Layout Description

- `Dashboard`: dense KPI header, equity and drawdown panels, exposure strip, runtime state card
- `Backtesting`: left-side run controls with inline config overrides, right-side performance charts and trade tables
- `Optimization`: sweep definition panel, combination counter, results grid, 2D heatmap surface
- `Live Trading`: broker mode selector, arming checkbox, start/stop/kill controls, OMS table, process log tail
- `Settings`: full sectioned config editor generated from the typed schema

---

## Strategies

### Momentum (Cross-Sectional)
Based on Jegadeesh & Titman (1993). Ranks all universe instruments by 12-1 month log return, goes long the top 20%.

**Mathematical formulation:**
```
r_i(t) = log(P_i(t - skip) / P_i(t - lookback_long))   # Formation return
z_i(t) = (r_i - μ_r) / σ_r                              # Cross-sectional z-score
LONG if rank(z_i) ∈ top 20%, else FLAT
```

### Mean Reversion (Bollinger + RSI)
Time-series mean reversion with Wilder RSI confirmation filter.

**Mathematical formulation:**
```
z(t) = (close(t) - rolling_mean(t, N)) / rolling_std(t, N)
RSI(t) = 100 - 100/(1 + avg_gain/avg_loss)

Entry: z(t) < -2.0 AND RSI(t) < 30  → LONG
Exit:  z(t) > -0.5                   → FLAT
```

---

## Risk Engine

Five configurable pre-trade constraints applied to every order. Closing trades (risk-reducing) always bypass limits.

| Constraint | India Default | Behavior on Breach |
|-----------|-------------|-------------------|
| `max_position_pct` | 4% | Reduce order quantity |
| `max_sector_pct` | 25% | Reject order |
| `max_gross_exposure_pct` | 100% | Reject order |
| `max_drawdown_pct` | 12% | Trading halt (new opens only) |
| `min_cash_pct` | 5% | Reject order |

---

## Execution Engine

Three pluggable slippage models — switch via `config/settings.yaml` with no code changes.

| Model | Formula | Use Case |
|-------|---------|---------| 
| `FIXED_BPS` | `open × bps/10000` | Baseline, liquid large-caps |
| `VOLUME_WEIGHTED` | `base_bps + impact × (qty/volume)` | Realistic for large orders |
| `SPREAD_BASED` | `0.5 × spread_fraction × (high - low)` | NSE mid/large-caps (default) |

All orders execute at the **next bar's open price** — trade-on-next-open eliminates look-ahead bias in fill prices.

---

## Performance Analytics (Week 3 — 20 Metrics)

`PerformanceTracker` computes 20 metrics from the equity curve and fill log:

### Return & Risk Metrics (Week 2)
| Metric | Formula |
|--------|---------|
| Total Return | `(V_final - V_initial) / V_initial` |
| Annualised Return | `(1 + TR)^(252/N) - 1` |
| Annualised Volatility | `std(daily_returns) × √252` |
| **Sharpe Ratio** | `E[r - rf] / std(r - rf) × √252` |
| **Sortino Ratio** | `E[r - rf] / downside_std × √252` |
| **Max Drawdown** | `max(V_peak - V_t) / V_peak` |
| Max DD Duration | Longest consecutive drawdown (days) |
| **Calmar Ratio** | `Ann. Return / abs(Max Drawdown)` |
| Win Rate | `profitable_fills / total_fills` |
| Profit Factor | `gross_profit / abs(gross_loss)` |
| Avg Win / Avg Loss | Per-trade P&L averages |

### Extended Metrics (Week 3)
| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Expectancy** | `win_rate × avg_win + loss_rate × avg_loss` | Expected P&L per trade — must be > 0 |
| **Skewness** | Third standardised moment | Negative = left-tail dominance (crash risk) |
| **Kurtosis** | Excess kurtosis | > 0 = fat tails; Sharpe understates real risk |
| **Tail Ratio** | `\|P95\| / \|P5\|` | > 1 = gains outpace losses in the tails |
| **Ulcer Index** | `√mean(DD²)` | RMS drawdown pain — penalises depth + duration |
| **Recovery Factor** | `total_return / abs(max_drawdown)` | Times you earned back the max loss |

Results are returned as a typed `PerformanceReport` — serialisable to JSON.

---

## Optimization Framework (Week 3)

Parameter search runs exclusively on the **training split (70%)**. The test split is never touched until final OOS evaluation.

```python
from cpat.optimization.optimizer import RandomSearchOptimizer, results_to_dataframe
from cpat.validation.splitter import train_test_split

train_data, test_data = train_test_split(bar_data, train_ratio=0.70)

optimizer = RandomSearchOptimizer(strategy="momentum", metric="sharpe_ratio", n_trials=50)
results = optimizer.run(train_data, config)

df = results_to_dataframe(results)   # param table sorted best-first
print(df.head(10))
```

### Default parameter spaces

| Strategy | Parameters | Grid Size |
|----------|-----------|----------|
| Momentum | `lookback_long` × `lookback_short` × `skip` × `top_n_pct` × `rebalance_freq` | 243 combos |
| Mean Rev | `lookback_window` × `entry_zscore` × `exit_zscore` × `rsi_oversold` | 108 combos |

---

## Walk-Forward Validation (Week 3)

The gold standard for strategy robustness evaluation — no IS data ever contaminates the OOS measurement.

```python
from cpat.validation.walk_forward import WalkForwardValidator, WalkForwardConfig

wf = WalkForwardValidator(
    strategy="momentum",
    wf_config=WalkForwardConfig(train_bars=756, test_bars=252, step_bars=63),
)
result = wf.run(bar_data, config)

print(result)                        # summary with degradation ratio
print(result.summary_dataframe())    # per-fold IS vs OOS table
result.combined_oos_equity.to_csv("oos_equity.csv")
```

### Robustness thresholds

| Metric | Healthy | Warning | Failed |
|--------|---------|---------|--------|
| `degradation_ratio` (OOS Sharpe / IS Sharpe) | ≥ 0.50 | 0.25–0.50 | < 0.25 |
| `consistency_score` (% folds profitable) | ≥ 60% | 40–60% | < 40% |
| `parameter_sensitivity_cv` | < 0.30 | 0.30–0.50 | > 0.50 |

---

## Overfitting Detection (Week 3)

```python
from cpat.validation.overfitting import degradation_report, stability_check

# Structured IS vs OOS comparison
report = degradation_report(is_report, oos_report)
# → {"assessment": "ROBUST / ACCEPTABLE / MARGINAL / WEAK / FAILED", ...}

# Parameter fragility table (sorted most unstable first)
stability = stability_check(results_df, metric="sharpe_ratio")
# → DataFrame[param_name, mean, std, cv, is_stable]
```

---

## Bias Prevention

| Risk | Mitigation |
|------|-----------| 
| Look-ahead bias | `DataHandler` enforces forward-only iteration; fills at **next bar's open** |
| Indicator look-ahead | All rolling windows computed on strictly past bars only |
| Evaluation bias | Optimizer **never** accesses test split; OOS measured post-selection |
| Survivorship bias | Universe fixed at backtest start |
| Data snooping | Canonical parameters from academic literature as starting points |
| Transaction costs | 3 bps commission + 8 bps slippage (Indian broker model) |
| Causality | EventQueue enforces strict order: Market → Signal → Order → Fill |
| Overfitting | Walk-forward OOS + parameter sensitivity CV + degradation ratio |

---

## CLI Reference

```bash
# Standard backtest — full period, all 20 metrics, drawdown table
python scripts/run_backtest.py --mode backtest --strategy momentum

# Parameter optimization — train split only, prints OOS evaluation
python scripts/run_backtest.py --mode optimize --strategy momentum
python scripts/run_backtest.py --mode optimize --strategy mean_reversion

# Walk-forward validation — fold table + combined OOS curve
python scripts/run_backtest.py --mode walk-forward --strategy momentum

# Baseline vs managed risk comparison
python scripts/run_backtest.py --mode compare --strategy momentum

# Run both strategies together
python scripts/run_backtest.py --mode backtest --strategy both
```

**Output files** (saved to `data/results/`):

| File | Contents |
|------|---------|
| `equity_curve_{strategy}.csv` | Full-period equity curve |
| `comparison_{strategy}.csv` | Baseline vs managed risk summary table |
| `risk_history_{strategy}.csv` | Portfolio risk telemetry over time |
| `trade_risk_{strategy}.csv` | Planned risk, stops, fills, and stop reasons |
| `optimization_{strategy}.csv` | All param combinations × metrics table |
| `walk_forward_oos_{strategy}.csv` | Stitched OOS equity curve |

---

## Position Sizing (Week 4)

All new positions are sized by the `PositionSizer` *before* reaching the translator.  Three methods are available:

| Method | Formula | Best For |
|--------|---------|----------|
| **Fixed Fractional** | `qty = floor(min(capital_budget, equity × risk_pct) / entry_price)` | Fixed capital deployment per trade |
| **Inverse Volatility** *(default)* | `qty = floor(min(risk_budget / stop_distance, capital_budget / price))` | Multi-asset portfolios |
| **Kelly** | `f* = (W·avg_win − L·avg_loss) / avg_win`; half-Kelly capped at 25% | High win-rate strategies |

**Stop-loss is mandatory** — every position gets:
- **ATR stop**: `entry − atr_mult × ATR(14)` (default: 2× ATR below entry)
- **Fixed % stop**: `entry × (1 − fixed_stop_pct)` when `stop_method: fixed_pct`
- **Take-profit**: `entry + tp_mult × stop_distance`
- **Trailing stop**: optional, ratchets up as price moves favourably

---

## Capital Allocation (Week 4)

| Method | Rule | Use Case |
|--------|------|---------|
| `equal_weight` | Deploy `deployable_cash / n_signals` per symbol | Simple, robust baseline |
| `volatility_adjusted` | Weight ∝ `1/σ` (20-day std dev) | Reduces concentration in volatile assets |

Deployable cash = `cash × (1 − min_cash_pct)`.  Each slice is capped at `max_position_pct × equity`.

---

## Portfolio-Level Risk Controls (Week 4)

| Control | Default | Effect |
|---------|---------|-------|
| `max_open_positions` | 20 | Blocks new opens when 20 positions are open |
| `max_daily_loss_pct` | 2% | Halts all new opens if today's equity drop > 2% |
| `max_drawdown_pct` *(existing)* | 12% | Halts all new opens during drawdown |
| `max_position_pct` *(existing)* | 4% | Reduces order to respect single-name cap |
| `max_sector_pct` *(existing)* | 25% | Rejects order breaching sector limit |

---

## Test Suite

```bash
pytest tests/ -v             # 357 tests, ~51s
pytest tests/ --cov=cpat     # With coverage report (85% overall)
```

**Test breakdown:**

| Module | Tests | What's Covered |
|--------|-------|---------------|
| `test_models.py` | 16 | Bar/Signal/Order/Fill/Position/CostConfig invariants |
| `test_data_pipeline.py` | 8 | Adapter validation, Parquet store, Universe |
| `test_event_engine.py` | 18 | EventQueue priority, execution, BacktestEngine |
| `test_strategies.py` | 11 | RSI, Momentum signals, MeanReversion entry/exit |
| `test_portfolio.py` | 40 | PortfolioManager, SignalOrderTranslator, DataHandler |
| `test_execution.py` | 30 | 3 slippage models, commission, partial fills, edge cases |
| `test_risk_analytics.py` | 35 | RiskEngine constraints, PerformanceTracker, TradeLog |
| `test_analytics_ext.py` | 39 | Drawdown table, distributions, extended Report fields |
| `test_optimizer.py` | 28 | GridSearch, RandomSearch, param validation, result table |
| `test_validation.py` | 35 | Splitter, walk-forward, overfitting detection |
| `test_position_sizing.py` | 36 | ATR, FixedFractional, InverseVol, Kelly, Factory |
| `test_risk_manager.py` | 33 | StopLevel, StopLossTracker, PortfolioRiskManager |
| `test_allocator.py` | 28 | EqualWeight, VolatilityAdjusted, sigma, AllocatorFactory |

---

## Week 5 Roadmap

- [ ] **Zerodha Kite adapter** — Indian live trading via Kite Connect API
- [ ] **Parallel optimizer** — `concurrent.futures` for 80% speed improvement
- [ ] **Benchmark analytics** — Alpha / Beta vs `^NSEI` in `PerformanceReport`
- [ ] **Plotly dashboard** — equity curve + drawdown + monthly return heatmap
- [ ] **Multi-strategy ensemble** — weighted signal aggregation layer
