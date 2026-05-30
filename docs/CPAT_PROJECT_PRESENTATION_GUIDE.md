# CPAT Project Presentation Guide

## 1. Project Overview

**CPAT** is a modular, event-driven algorithmic trading platform built for the full quantitative trading lifecycle:

- historical data ingestion
- strategy research
- realistic backtesting
- portfolio construction
- risk-controlled execution
- parameter optimization
- walk-forward validation
- paper/live trading orchestration
- operator-facing monitoring through a Streamlit console

At a high level, the project answers one core question:

> How can we build a trading system that is not just profitable in research, but also structured, testable, risk-aware, and operationally ready for live deployment?

The codebase is designed as a production-style system rather than a notebook-only prototype. Most decisions in the repository aim to reduce common quant-research mistakes such as look-ahead bias, state leakage, poor separation of concerns, and weak risk controls.

---

## 2. What Problem CPAT Solves

Most student or early-stage quant systems stop at signal generation. CPAT goes further and models the entire trading pipeline:

1. market data must be collected and stored reliably
2. strategies must generate signals using only past information
3. those signals must be converted into tradeable position sizes
4. portfolio and trade-level risk must be checked before execution
5. fills must be simulated realistically or routed to a broker
6. performance must be evaluated with robust out-of-sample validation
7. the operator must have a usable interface for control and monitoring

Because of that, CPAT is better presented as a **quant trading platform** rather than just a pair of strategies.

---

## 3. Core Design Philosophy

The codebase follows the same engineering philosophy throughout:

- **Event-driven architecture**: all major actions flow through market, signal, order, and fill events.
- **Strict separation of concerns**: data, strategies, portfolio, risk, execution, analytics, and UI are isolated into dedicated modules.
- **Config-driven behavior**: key parameters live in `config/settings.yaml`, not hardcoded in the implementation.
- **Bias prevention first**: the engine is intentionally designed to prevent unrealistic backtest behavior.
- **Extensibility**: new strategies, allocators, sizers, brokers, and validation methods can be added without rewriting the whole engine.

This matches the repository structure in `cpat/`, where each folder owns one responsibility.

---

## 4. Architecture by Layer

## 4.1 Domain and Contracts

The `cpat/core/` package defines the canonical trading language of the system:

- `models.py`: immutable trading objects like `Bar`, `Signal`, `Order`, `Fill`
- `events.py`: typed event wrappers like `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`
- `enums.py`: shared enums such as order side, order type, signal direction, event type

This layer is intentionally simple and framework-independent. It acts as the foundation for every other module.

## 4.2 Configuration

`cpat/config/loader.py` loads `config/settings.yaml` into a typed `CPATConfig` model using Pydantic.

This gives the project:

- validation before runtime
- one central configuration contract
- easy dependency injection into all engines and services

## 4.3 Data Layer

The `cpat/data/` package handles the historical data workflow:

- adapters fetch or read data
- `DataPipeline` orchestrates downloading, validation, and persistence
- `ParquetDataStore` persists processed bars
- `DataHandler` exposes a forward-only iterator during backtests

This layer is critical because the engine depends on the data being time-safe and consistent.

## 4.4 Strategy Layer

The `cpat/strategies/` package contains the alpha logic:

- `MomentumStrategy`: cross-sectional ranking across the universe
- `MeanReversionStrategy`: time-series mean reversion using Bollinger-style z-score and RSI confirmation
- `AbstractStrategy`: common event and history behavior

Strategies do not place orders directly. They only emit signal events.

## 4.5 Portfolio and Risk Layer

This layer translates raw signals into controlled trades:

- `PortfolioManager`: owns cash, positions, equity, exposures, snapshots
- `allocator.py`: allocates capital across multiple valid signals
- `position_sizing.py`: converts capital budgets into quantities and stop levels
- `risk/engine.py`: pre-trade checks like max position size and cash buffer
- `risk/risk_manager.py`: stop-loss tracking, daily loss halt, max open positions

This is where the project shifts from “signal generation” to “portfolio construction”.

## 4.6 Execution Layer

Two different execution paths exist:

- `cpat/execution/engine.py`: simulated backtest execution at next-bar open
- `cpat/infrastructure/execution_engine_live.py`: live or paper order routing through the OMS and broker adapters

Backtest execution models cost realism.
Live execution models operational realism.

## 4.7 Analytics and Validation Layer

This layer evaluates whether performance is meaningful:

- `analytics/performance.py`: return, risk, and trade-level metrics
- `analytics/drawdown.py`: drawdown tables and ulcer index
- `analytics/distributions.py`: skew, kurtosis, tail metrics, VaR/CVaR
- `optimization/optimizer.py`: grid and random parameter search
- `validation/splitter.py`: forward-only train/test splits
- `validation/walk_forward.py`: rolling out-of-sample evaluation
- `validation/overfitting.py`: degradation and parameter stability analysis

This is one of the strongest parts of the project for presentation because it shows the system is trying to validate robustness, not just maximize in-sample results.

## 4.8 Infrastructure and UI

The infrastructure and UI layers make the research engine operable:

- `infrastructure/order_manager.py`: live OMS
- `infrastructure/scheduler.py`: market-hours tick scheduling
- `brokers/paper.py` and `brokers/dhan.py`: broker adapters
- `ui/`: Streamlit multipage control console
- `ui/services/`: backend-facing service layer for the UI

This makes the project suitable for a “research-to-operations” demonstration.

---

## 5. Supported Project Modes

CPAT supports several distinct operating modes. This is useful in a presentation because you can show that one codebase supports multiple stages of the trading lifecycle.

## 5.1 Data Fetching Mode

Entry point: `scripts/fetch_universe.py`

Purpose:

- downloads historical data for the configured universe
- validates and stores it in processed parquet format

Output:

- stored bar data in `data/processed/`

## 5.2 Backtest Mode

Entry point: `scripts/run_backtest.py --mode backtest`

Purpose:

- runs a standard historical simulation
- evaluates strategy performance over the full period

Output:

- performance report in terminal
- CSV artifacts in `data/results/`

## 5.3 Compare Mode

Entry point: `scripts/run_backtest.py --mode compare`

Purpose:

- compares baseline flow against managed-risk flow
- demonstrates the value of allocator, sizer, stop-loss, and portfolio guardrails

Output:

- comparison table
- baseline and managed result files

## 5.4 Optimization Mode

Entry point: `scripts/run_backtest.py --mode optimize`

Purpose:

- searches strategy parameter combinations only on the training split
- evaluates the selected best configuration on held-out test data

Output:

- optimization table
- train-vs-test degradation analysis

## 5.5 Walk-Forward Validation Mode

Entry point: `scripts/run_backtest.py --mode walk-forward`

Purpose:

- rolls train and test windows through time
- measures whether the strategy generalizes across market regimes

Output:

- per-fold IS/OOS table
- stitched out-of-sample equity curve

## 5.6 Paper Trading Mode

Entry point: `scripts/run_live.py --mode paper`

Purpose:

- executes the live trading path without real capital
- tests OMS, scheduler, risk controls, and live logging

Output:

- live logs in `logs/live/`
- OMS audit trail

## 5.7 Demo Mode

Entry point: `scripts/run_live.py --mode paper --demo`

Purpose:

- runs a small finite live session using synthetic data
- useful for demos because it avoids market-hours dependency

## 5.8 Broker Live Mode

Entry point: `scripts/run_live.py --mode dhan`

Purpose:

- routes trades through the Dhan broker adapter
- reuses the same managed-risk logic as paper mode

## 5.9 UI Console Mode

Entry point: `streamlit run ui/app.py`

Purpose:

- gives a control center for research, artifact inspection, and live process management

Pages include:

- Dashboard
- Backtesting
- Optimization
- Portfolio / Risk
- Analytics
- Live Trading
- Logs
- Settings

---

## 6. Strategies Implemented

## 6.1 Momentum Strategy

File: `cpat/strategies/momentum.py`

This is a **cross-sectional momentum** strategy.

Logic:

1. collect enough history for all tracked symbols
2. compute formation return using a long lookback
3. skip the most recent period to avoid short-term reversal effects
4. normalize scores cross-sectionally with z-scores
5. rank symbols
6. go long the top fraction
7. flatten the rest

Why it matters:

- this strategy tests relative strength across many assets
- it fits well with the multi-asset architecture of the project

## 6.2 Mean Reversion Strategy

File: `cpat/strategies/mean_reversion.py`

This is a **single-symbol mean reversion** strategy.

Logic:

1. compute rolling mean and rolling standard deviation
2. compute z-score of the latest price
3. compute RSI as a confirmation filter
4. enter long when the asset is deeply oversold
5. exit when price reverts toward the mean
6. optionally support short-side logic if enabled

Why it matters:

- it contrasts nicely with momentum
- it shows the platform can support different alpha families

---

## 7. Risk and Trade Management Logic

One of the strongest parts of CPAT is that it does not assume every signal should become a full trade.

The managed-risk path applies four sequential control layers:

1. **Portfolio guardrails**
   `PortfolioRiskManager` blocks new positions if the system already has too many open trades or the day-level loss limit has been breached.
2. **Capital allocation**
   The allocator decides how much capital each signal is allowed to consume.
3. **Position sizing**
   The sizer converts that capital budget into quantity, stop-loss, and take-profit levels.
4. **Pre-trade risk engine**
   `RiskEngine` checks concentration, sector, gross exposure, drawdown halt, and minimum cash buffer.

Only after passing all of those layers does an order get routed for execution.

This gives you a very strong presentation point:

> CPAT is not signal-driven only. It is portfolio- and risk-driven.

---

## 8. Backtesting Logic

The backtest engine is event-driven and designed to avoid look-ahead bias.

Important backtest principles:

- the `DataHandler` is forward-only
- strategies only see current and past bars
- orders created on bar `T` are executed on bar `T+1` open
- stop checks happen before strategy logic on the new bar
- performance is recorded after fills and portfolio valuation update

This means the backtest is intentionally conservative and causally valid.

---

## 9. Live Trading Logic

The live path reuses most of the managed-risk design rather than creating a separate ad hoc flow.

Key live components:

- `LiveExecutionEngine`
- `OrderManager`
- `TradingScheduler`
- `BrokerInterface` adapters
- `LiveTradeLogger`

Live mode therefore preserves strategy logic, risk logic, portfolio logic, and stop logic while swapping the execution backend from simulated fills to broker-managed order lifecycle.

That is important academically and practically because it reduces the research-to-production gap.

---

## 10. Why the Project Is Strong Technically

These are the strongest technical talking points to emphasize:

- it is not notebook-only; it is a structured Python application
- it uses typed domain models and event contracts
- it separates strategy, portfolio, risk, execution, analytics, and UI cleanly
- it prevents look-ahead bias in both data access and execution timing
- it includes realistic transaction costs and slippage
- it supports train/test optimization and walk-forward validation
- it includes paper/live operational infrastructure
- it has test coverage across core modules

If you are presenting to faculty or interviewers, these points usually matter more than claiming one strategy has a high Sharpe ratio.

---

## 11. How to Present the Project

A strong presentation sequence is:

1. **Problem statement**
   Most retail backtests are unrealistic and fail in deployment.
2. **Solution**
   CPAT is an end-to-end event-driven trading platform with research, validation, risk, and execution layers.
3. **Architecture**
   Explain the `core -> data -> strategy -> portfolio/risk -> execution -> analytics -> UI/live` flow.
4. **Strategies**
   Show that the platform supports both momentum and mean reversion.
5. **Risk controls**
   Explain why raw signals are filtered through allocation, sizing, and risk checks.
6. **Validation**
   Highlight optimization, train/test separation, and walk-forward evaluation.
7. **Operations**
   Show paper/live mode, OMS, scheduler, and logs.
8. **Demo**
   Use the Streamlit UI or the demo live runner.

This framing presents CPAT as a serious systems project, not just a strategy experiment.

---

## 12. Suggested Demo Script

If you need a live demo narrative, use this:

1. run `scripts/fetch_universe.py` and explain that data is fetched once and stored locally
2. run a backtest and show the event-driven simulation output
3. show the saved result artifacts in `data/results/`
4. run compare mode to show the difference between baseline and managed risk
5. run optimization or walk-forward to show robustness analysis
6. open `streamlit run ui/app.py` and walk through the dashboard, analytics, and settings
7. if needed, run `scripts/run_live.py --mode paper --demo` to show the live execution stack without broker dependency

---

## 13. End State of the System

Depending on the mode, the system ends in one of four places:

- processed historical market data in parquet storage
- research artifacts in `data/results/`
- live logs and OMS audit artifacts in `logs/live/`
- UI-visible runtime state and analytics visualizations

So the project does not “end” at signal generation. It ends in persisted outputs that can be inspected, validated, audited, and presented.

---

## 14. One-Line Summary for Presentation

> CPAT is a production-style, event-driven quantitative trading platform that takes a strategy from data ingestion to backtesting, risk-managed portfolio construction, validation, and paper/live execution.
