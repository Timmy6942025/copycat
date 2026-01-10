# SANDBOX KNOWLEDGE BASE

**Generated:** Fri Jan 09 2026
**Commit:** Current working tree
**Branch:** main

## OVERVIEW

Paper trading simulation with virtual money, real-time market data, realistic execution model, and comprehensive performance analytics.

## STRUCTURE

```
sandbox/
├── data_providers/   # Unified data: CoinGecko, Yahoo, Polymarket (6 files)
├── data/             # MarketDataCache (725 lines, COMPLEX), HistoricalLoader
├── models/           # VirtualOrder, VirtualPosition, SimulationState
├── managers/         # VirtualPortfolioManager
├── executors/        # VirtualOrderExecutor
├── engines/          # BacktestEngine
├── trackers/         # PerformanceTracker
├── reporters/        # SuccessCriteriaEvaluator
├── tests/            # 196 tests (~95% coverage)
├── runner.py         # SandboxRunner (435 lines)
├── executor.py       # VirtualOrderExecutor (409 lines)
├── portfolio.py      # VirtualPortfolioManager (378 lines)
├── analytics.py      # PerformanceTracker (571 lines)
├── backtest.py       # BacktestEngine (305 lines)
├── success_criteria.py # Live trading approval (637 lines)
├── dashboard.py      # TUI dashboard (665 lines)
└── config.py         # SandboxConfig (module-level)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Main runner | `runner.py` | SandboxRunner (435 lines) |
| Order execution | `executor.py` | VirtualOrderExecutor with slippage/fees |
| Portfolio tracking | `portfolio.py` | VirtualPortfolioManager (378 lines) |
| Performance metrics | `analytics.py` | PerformanceTracker (571 lines) |
| Market data cache | `data/market_data_cache.py` | ⚠️ 725 lines, LRU/TTL/FIFO |
| Live approval | `success_criteria.py` | SuccessCriteriaEvaluator (637 lines) |
| Data providers | `data_providers/unified.py` | UnifiedDataProvider (CoinGecko/Yahoo/Polymarket) |
| Backtesting | `backtest.py` | BacktestEngine (305 lines) |
| CLI | `cli.py` | Command-line interface |
| TUI | `dashboard.py` | Rich terminal dashboard (665 lines) |

## CONVENTIONS

**Virtual trading simulation**: Slippage, fees, fill probability modeled
**Performance metrics**: Sharpe ratio, max drawdown, win rate, profit factor, consistency
**Cache TTLs**: 30s for prediction markets, 60s for crypto/stocks
**State management**: SimulationState, VirtualOrder, VirtualPosition dataclasses
**Real-time data**: RealtimeDataFeed (polling-based) fetches live market data per order
**Data flow**: UnifiedDataProvider → CoinGecko/Yahoo/Polymarket → MarketDataCache → SandboxRunner
**Configuration**: Module-level SandboxConfig via `get_sandbox_mode()` env var (default: SANDBOX)
**Live approval**: SuccessCriteriaEvaluator enforces 7 metrics before allowing live trading

## ANTI-PATTERNS (SANDBOX)

❌ **Complexity hotspot**: `data/market_data_cache.py` (725 lines) - LRU/TTL/FIFO policies, thread safety, persistence
❌ **Large module**: `analytics.py` (571 lines) - PerformanceTracker growing complex
❌ **Data model duplication**: OrderSide, OrderType, OrderStatus defined here AND in `api_clients/base.py`
