# CPAT Execution Flow

## 1. Purpose of This Document

This document explains where the code starts, how control moves through the project, and where each execution path ends.

CPAT has four practical starting points:

- `scripts/fetch_universe.py`
- `scripts/run_backtest.py`
- `scripts/run_live.py`
- `ui/app.py`

Each one enters the system for a different purpose, but all of them reuse the same core modules.

---

## 2. High-Level Start Points

## 2.1 Historical Data Start

`scripts/fetch_universe.py`

Use when the goal is to build or refresh the historical dataset.

## 2.2 Research Start

`scripts/run_backtest.py`

Use when the goal is backtesting, comparison, optimization, or walk-forward validation.

## 2.3 Live Trading Start

`scripts/run_live.py`

Use when the goal is paper trading, demo trading, or broker-connected trading.

## 2.4 UI Start

`ui/app.py`

Use when the goal is to control the project visually through Streamlit.

---

## 3. Data Ingestion Flow

### Entry

`scripts/fetch_universe.py -> main()`

### Flow

1. CLI arguments are parsed.
2. `load_config()` reads `config/settings.yaml`.
3. `setup_logging()` configures console/file logging.
4. `DataPipeline.from_config()` constructs:
   - the data adapter
   - the parquet store
   - the universe definition
5. `pipeline.run()` begins the ingestion job.
6. The adapter fetches bars for the requested symbols and date range.
7. The store persists cleaned data under `data/processed/`.
8. A `PipelineResult` summary is printed.

### End

The flow ends with:

- processed parquet data written locally
- a success/failure summary in the CLI

### Compact Sequence

```text
scripts/fetch_universe.py
  -> load_config()
  -> DataPipeline.from_config()
  -> adapter.fetch_universe()
  -> ParquetDataStore.save()
  -> processed data in data/processed/
```

---

## 4. Backtest Flow

## 4.1 Entry

`scripts/run_backtest.py -> main()`

### Step 1: Bootstrap

1. Parse CLI options:
   - `--mode`
   - `--strategy`
   - `--config`
2. Load typed config with `load_config()`.
3. Set up logging.
4. Construct `DataPipeline.from_config(cfg)`.
5. Call `pipeline.load_bars()` to load stored historical bars.

If no stored data exists, the script stops and tells the user to run `fetch_universe.py` first.

### Step 2: Mode Dispatch

After data is loaded, `main()` dispatches to one of four mode handlers:

- `_run_backtest()`
- `_run_compare()`
- `_run_optimize()`
- `_run_walk_forward()`

---

## 4.2 Standard Backtest Flow

### Entry

`_run_backtest(bar_data, cfg, strategy)`

### Engine Construction

1. `BacktestEngine.from_config(cfg, managed_risk=True)` is created.
2. `_register_strategies()` attaches:
   - `MomentumStrategy`
   - `MeanReversionStrategy`
   - or both
3. `engine.run(bar_data)` starts the simulation.

### What `BacktestEngine.run()` Does

#### Phase A: One-time setup

1. reset internal state
2. build `DataHandler(bar_data, warmup_bars=...)`
3. push `SystemEvent.start`

#### Phase B: Per-bar loop

For each timestamp:

1. `ExecutionEngine.process_pending(current_bars, ts)`
   - executes previously queued orders at this bar's open
   - produces fills
2. each fill is applied:
   - `PortfolioManager.apply_fill()`
   - `TradeLog.record()`
   - strategy `on_fill()` callbacks
3. current portfolio equity is computed
4. `PortfolioRiskManager.update_daily_equity()` updates day-level state
5. `StopLossTracker.check_stops()` runs before strategy processing
6. a `MarketEvent` is pushed into the queue
7. `_drain_queue()` processes queued events

Inside `_drain_queue()`:

- `MARKET` events go to strategy `on_market()`
- strategies emit `SIGNAL` events
- managed mode stores signals into a batch
- baseline mode directly translates signals into orders
- `ORDER` events route through risk and execution submission

#### Phase C: Managed signal batch logic

When managed-risk mode is enabled, `_process_signal_batch()` applies:

1. deduplicate latest signal per symbol
2. prioritize stop-generated flat signals
3. build close orders first
4. check portfolio-level open-position and daily-loss guardrails
5. allocate capital across valid opening signals
6. size each trade with stop and take-profit levels
7. translate sized intents into market orders
8. route each order through `RiskEngine.check()`
9. submit approved orders to `ExecutionEngine.submit()`

#### Phase D: End-of-bar accounting

1. compute current equity
2. `PerformanceTracker.record(ts, equity)`
3. `PortfolioManager.snapshot(ts, prices)`
4. `RiskEngine.update_equity(equity)`
5. record risk history row

#### Phase E: End of run

1. compute final equity from the last prices
2. push `SystemEvent.stop`
3. build:
   - equity curve
   - returns
   - drawdown series
   - performance report
   - trade log dataframe
   - portfolio snapshots dataframe
   - risk history dataframe
   - trade risk dataframe
4. return `BacktestResult`

### Final Outputs

Back in `_run_backtest()`:

1. performance is printed
2. drawdown and distribution summaries are printed
3. CSV artifacts are saved to `data/results/`

### End

The backtest flow ends with:

- terminal performance report
- `BacktestResult` in memory
- research artifacts in `data/results/`

### Compact Sequence

```text
scripts/run_backtest.py
  -> load_config()
  -> DataPipeline.load_bars()
  -> BacktestEngine.from_config()
  -> register strategies
  -> engine.run()
      -> DataHandler iteration
      -> process pending orders
      -> apply fills
      -> stop checks
      -> MarketEvent
      -> strategy signals
      -> allocation + sizing + risk
      -> submit orders for next bar
      -> performance + snapshots
  -> save CSV artifacts
```

---

## 4.3 Compare Flow

### Entry

`_run_compare(bar_data, cfg, strategy)`

### Flow

1. run `BacktestEngine.from_config(..., managed_risk=False)`
2. run `BacktestEngine.from_config(..., managed_risk=True)`
3. summarize both outcomes into a comparison dataframe
4. save comparison and both equity curves

### End

The flow ends with a clear before/after view of unmanaged versus managed portfolio logic.

---

## 4.4 Optimization Flow

### Entry

`_run_optimize(bar_data, cfg, strategy)`

### Flow

1. split data with `train_test_split()`
2. build either:
   - `GridSearchOptimizer`
   - `RandomSearchOptimizer`
3. optimizer runs repeated fresh backtests on the train split only
4. results are sorted by the chosen metric
5. best parameters are selected
6. best parameters are re-applied to config
7. a final out-of-sample test backtest runs on the held-out test split
8. `degradation_report()` compares train and test performance
9. `stability_check()` evaluates parameter fragility
10. results are saved to CSV

### End

The optimization flow ends with:

- best parameter candidates
- test-split validation result
- parameter stability analysis

---

## 4.5 Walk-Forward Flow

### Entry

`_run_walk_forward(bar_data, cfg, strategy)`

### Flow

1. `WalkForwardConfig` is built from validation settings
2. `WalkForwardValidator.run()` receives the full dataset
3. the timeline is split into repeated train/test windows
4. for each fold:
   - optional optimization on train data
   - in-sample backtest
   - out-of-sample backtest
   - fold metrics are stored
5. all OOS equity segments are stitched together
6. a combined OOS performance report is built
7. fold summary and aggregate summary are printed
8. combined OOS equity is saved

### End

The flow ends with the most robust evaluation artifact in the project:

- stitched out-of-sample equity curve

---

## 5. Strategy-Level Internal Flow

Both strategies follow the same contract:

1. receive `MarketEvent`
2. update internal history
3. check whether enough bars exist
4. compute indicators from past data only
5. emit `SignalEvent`

### Momentum internal logic

```text
MarketEvent
  -> update bar history
  -> wait until rebalance frequency is reached
  -> compute formation returns
  -> normalize via z-score
  -> rank cross-section
  -> emit LONG or FLAT signals
```

### Mean reversion internal logic

```text
MarketEvent
  -> update bar history
  -> compute rolling mean and std
  -> compute z-score and RSI
  -> if oversold: emit LONG
  -> if reverted: emit FLAT
```

---

## 6. Backtest Order Lifecycle

This is the exact lifecycle of a trade inside the backtest:

1. strategy emits `SignalEvent`
2. signal is either batched or translated immediately
3. translator builds `Order`
4. `RiskEngine.check()` approves, reduces, or rejects
5. approved order is stored in `_submitted_orders`
6. approved order is queued in `ExecutionEngine.submit()`
7. on the next bar, `ExecutionEngine.process_pending()` executes it
8. `Fill` is created
9. portfolio cash and positions are updated
10. trade log and risk records are updated
11. if opening a long, stop-loss tracking is registered

That lifecycle is one of the key causal guarantees in the project.

---

## 7. Live Trading Flow

## 7.1 Entry

`scripts/run_live.py -> main()`

### Step 1: Bootstrap

1. parse CLI options
2. load config
3. set up logging
4. choose mode:
   - paper
   - dhan
   - demo
5. choose symbol list
6. create broker adapter
7. create data provider
8. call `build_live_engine(...)`

### Step 2: Live Engine Assembly

`build_live_engine()` creates:

- `EventQueue`
- `LiveTradeLogger`
- `OrderManager`
- `PortfolioManager`
- `RiskEngine`
- `PositionSizer`
- `CapitalAllocator`
- `StopLossTracker`
- `PortfolioRiskManager`
- `MomentumStrategy`
- `LiveExecutionEngine`

### Step 3A: Demo Flow

If `--demo` is used:

1. loop over a fixed number of ticks
2. call `engine.run_tick()` each time
3. print final session summary
4. disconnect broker

### Step 3B: Scheduled Live Flow

If not in demo:

1. create `TradingScheduler(callback=engine.run_tick, ...)`
2. register shutdown signal handlers
3. start the scheduler
4. keep the process alive until stopped

---

## 7.2 What Happens Inside `LiveExecutionEngine.run_tick()`

Every scheduler tick runs the same sequence:

1. ensure broker connection
2. fetch the latest rolling bar data from the provider
3. remove stale or missing symbol data
4. convert latest rows into `Bar` objects
5. update paper prices if in paper mode
6. poll existing order updates from the OMS/broker
7. expire stale live orders if needed
8. compute current equity
9. update risk and day-level loss state
10. run stop-loss checks
11. collect strategy signals from a `MarketEvent`
12. reconcile broker state against local portfolio state
13. process signal batch:
    - close orders first
    - portfolio risk gate
    - allocator
    - sizer
    - pre-trade risk engine
    - OMS submission
14. poll order updates again
15. apply any fills received
16. emit heartbeat log

### End

The live tick ends with updated:

- portfolio state
- OMS state
- live logs
- broker reconciliation status

---

## 7.3 Live Order Lifecycle

In live mode the order path is longer than in backtest mode:

1. strategy emits signal
2. translator builds order
3. `RiskEngine.check()` validates
4. `OrderManager.create()` registers the order
5. broker submission happens
6. OMS marks order as submitted
7. order status polling or paper fills update the OMS
8. fill is applied to portfolio
9. stop tracker is synchronized
10. heartbeat and audit logs reflect the new state

This makes the OMS the central truth for live execution.

---

## 8. Streamlit UI Flow

## 8.1 Entry

`ui/app.py -> main()`

### Flow

1. Streamlit page config is set
2. theme CSS is injected
3. `build_backend_services()` constructs long-lived service instances
4. `UIServiceBundle` is assembled
5. `SessionManager` initializes UI state
6. sidebar chooses the active page
7. the corresponding page renderer is called

### Key Architectural Point

The UI does not talk directly to low-level engine internals. It talks through services:

- `ConfigService`
- `ExecutionService`
- `AnalyticsService`
- `StrategyService`

That keeps UI code separate from business logic.

---

## 8.2 UI Research Flow

When the user runs a backtest from the UI:

1. a page collects settings and overrides
2. `ExecutionService.run_backtest()` is called
3. it loads bars, builds the engine, registers strategies, and runs the backtest
4. result artifacts are saved
5. charts and tables are rendered from the returned data

The same pattern applies to compare, optimize, and walk-forward modes.

---

## 8.3 UI Live Process Flow

When the user starts live trading from the UI:

1. `ExecutionService.start_live_process()` builds a subprocess command
2. the subprocess runs `scripts/run_live.py`
3. runtime metadata is stored in `logs/live/ui_runtime_status.json`
4. stdout/stderr are appended to `logs/live/ui_process.log`
5. the UI reads status and logs for operator visibility

So the UI is a controller for the live runner, not a replacement for it.

---

## 9. Where Each Path Ends

### Data path ends at

- `data/processed/`

### Backtest path ends at

- `BacktestResult`
- `data/results/*.csv`

### Optimization path ends at

- ranked parameter results
- held-out test evaluation
- optimization CSV artifacts

### Walk-forward path ends at

- fold summary table
- combined OOS equity file

### Live path ends at

- broker/OMS state
- `logs/live/*.jsonl`
- `logs/live/oms_audit.jsonl`

### UI path ends at

- rendered dashboards
- saved config/state
- managed subprocess state

---

## 10. Single-Slide Summary Flow

If you need one compact slide, use this:

```text
Historical Data
  -> Data Pipeline
  -> Stored Bars
  -> Backtest / Optimize / Walk-Forward
  -> Strategy Signals
  -> Allocation + Position Sizing
  -> Risk Checks
  -> Execution
  -> Portfolio Update
  -> Performance Analytics
  -> CSV / Logs / UI / Live Monitoring
```

---

## 11. Short Answer: Where the Code Starts and Ends

### Research path

Starts at:

- `scripts/run_backtest.py`

Ends at:

- `BacktestResult`
- `data/results/` artifacts

### Live path

Starts at:

- `scripts/run_live.py`

Ends at:

- broker/OMS state updates
- live trade logs
- session summary

### UI path

Starts at:

- `ui/app.py`

Ends at:

- page rendering plus backend artifact generation or live subprocess control
