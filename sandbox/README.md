# CopyCat Sandbox Module

Risk-free environment for testing and validating copy trading strategies using virtual money with real-time market data.

## Overview

The Sandbox module provides a complete simulation environment that mimics real prediction market trading without risking actual capital. It includes:

- **Virtual Order Execution**: Simulates order fills with realistic slippage, fees, and fill probabilities
- **Portfolio Management**: Tracks positions, balances, and exposure in real-time
- **Performance Analytics**: Calculates comprehensive metrics (Sharpe ratio, max drawdown, profit factor, etc.)
- **Backtesting Engine**: Tests strategies against historical data
- **Success Criteria Evaluation**: Validates simulation results against live trading requirements

## Quick Start

```python
from sandbox import SandboxConfig, SandboxRunner, VirtualOrder
from datetime import datetime

# Configure sandbox
config = SandboxConfig(
    mode="sandbox",  # or "live" for real trading
    initial_balance=10000.0,
    max_orders_per_day=10,
)

# Create runner
runner = SandboxRunner(config)

# Execute a virtual order
order = VirtualOrder(
    order_id="order_001",
    market_id="bitcoin",
    side="buy",
    quantity=100,
    order_type="market",
    timestamp=datetime.utcnow(),
)

result = runner.execute_order(order)
print(f"Order status: {result.status}")
```

## Configuration

### SandboxConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `sandbox` | Trading mode: `sandbox` or `live` |
| `initial_balance` | `10000.0` | Starting USD balance |
| `max_orders_per_day` | `50` | Maximum orders per day |
| `max_position_size_pct` | `0.10` | Max 10% per single trade |
| `max_total_exposure_pct` | `0.50` | Max 50% total exposure |
| `simulate_slippage` | `True` | Enable slippage simulation |
| `simulate_fees` | `True` | Enable fee simulation |

## Success Criteria for Live Trading

Before deploying to live trading, your sandbox simulation must meet these minimum requirements:

### Performance Thresholds

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Monthly Return | > 3% | > 5% | > 8% |
| Win Rate | > 52% | > 58% | > 65% |
| Sharpe Ratio | > 0.8 | > 1.2 | > 1.5 |
| Max Drawdown | < 25% | < 15% | < 10% |
| Profit Factor | > 1.2 | > 1.5 | > 2.0 |
| Simulation Duration | 90 days | 180 days | 365 days |
| Consistency | 4/6 positive | 5/6 positive | 6/6 positive |

### Evaluating Your Results

```python
from sandbox import (
    PerformanceSnapshot,
    evaluate_success_criteria,
    format_evaluation_result,
)

# Create snapshot from your simulation results
snapshot = PerformanceSnapshot(
    monthly_return_pct=0.05,
    win_rate=0.55,
    sharpe_ratio=1.0,
    max_drawdown=0.20,
    profit_factor=1.5,
    simulation_days=100,
    # ... other metrics
    monthly_returns=[
        (datetime(2024, 1, 1), 0.02),
        (datetime(2024, 2, 1), 0.03),
        # ... 6 months required for consistency check
    ]
)

# Evaluate against success criteria
result = evaluate_success_criteria(snapshot)

# Print formatted results
print(format_evaluation_result(result))

# Check approval status
if result.is_approved_for_live:
    print("✓ Approved for live trading!")
else:
    print("✗ Not approved. Check failed metrics:")
    for metric, reason in result.failure_reasons.items():
        print(f"  - {metric}: {reason}")
```

### Critical Success Factors

Beyond metrics, these factors are essential for live trading approval:

1. **Consistency > Speed**: Steady returns matter more than occasional big wins
2. **Risk Control**: Never lose more than 20% of portfolio in worst case
3. **Reproducibility**: Results must be consistent across multiple simulation runs
4. **Transparency**: Every trade must be logged with full context
5. **Learning**: Bot should improve over time as it "learns" from more trader data

## Components

### VirtualOrderExecutor

Simulates order execution with realistic market behavior:

```python
from sandbox import VirtualOrderExecutor, SandboxConfig

config = SandboxConfig(initial_balance=10000)
executor = VirtualOrderExecutor(config)

# Execute order and get result
result = executor.execute(order)
print(f"Filled: {result.filled_quantity} @ ${result.average_price}")
```

### VirtualPortfolioManager

Manages portfolio state and position tracking:

```python
from sandbox import VirtualPortfolioManager, SandboxConfig

config = SandboxConfig(initial_balance=10000)
portfolio = VirtualPortfolioManager(config)

# Get current portfolio summary
summary = portfolio.get_summary()
print(f"Balance: ${summary.balance}")
print(f"Positions: {summary.position_count}")
```

### PerformanceTracker

Calculates and tracks performance metrics:

```python
from sandbox import PerformanceTracker

tracker = PerformanceTracker(initial_balance=10000)

# Record completed trades
tracker.record_trade(trade)

# Calculate comprehensive metrics
metrics = tracker.calculate_metrics()
print(f"Sharpe Ratio: {metrics.sharpe_ratio}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
print(f"Win Rate: {metrics.win_rate:.2%}")
```

## CLI Usage

Run sandbox simulations from the command line:

```bash
# Run sandbox with default settings
python -m sandbox.cli

# Run with custom balance
python -m sandbox.cli --initial-balance 50000

# Run backtest on historical data
python -m sandbox.cli --backtest --data-file history.csv
```

## Testing

Run the sandbox test suite:

```bash
# Run all sandbox tests
python -m pytest sandbox/tests/ -v

# Run specific test file
python -m pytest sandbox/tests/test_success_criteria.py -v

# Run integration tests
python -m pytest sandbox/tests/test_success_criteria_integration.py -v
```

### Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| `test_executor.py` | 32 | Order execution tests |
| `test_portfolio.py` | 28 | Portfolio management tests |
| `test_analytics.py` | 37 | Performance tracking tests |
| `test_runner.py` | 19 | Sandbox runner tests |
| `test_sandbox.py` | 26 | Core sandbox tests |
| `test_success_criteria.py` | 41 | Success criteria tests |
| `test_success_criteria_integration.py` | 9 | Integration tests |
| `test_data_providers.py` | 4 | Data provider tests |

**Total: 196 tests passing**

## File Structure

```
sandbox/
├── __init__.py           # Main exports
├── config.py             # Configuration dataclasses
├── success_criteria.py   # Success criteria evaluation
├── executor.py           # VirtualOrderExecutor
├── portfolio.py          # VirtualPortfolioManager
├── analytics.py          # PerformanceTracker
├── backtest.py           # BacktestEngine
├── cli.py                # Command-line interface
├── runner.py             # SandboxRunner
├── dashboard.py          # TUI Dashboard
├── data/                 # Data loading and caching
│   ├── __init__.py
│   ├── historical_loader.py
│   └── market_data_cache.py
└── tests/                # Test suite
    ├── __init__.py
    ├── test_*.py         # Unit tests
    └── test_*_integration.py  # Integration tests
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COPYCAT_SANDBOX` | `true` | Set to `false` for live mode |
| `COPYCAT_DATA_PATH` | `./data` | Path to market data |

## Next Steps

1. **Configure your strategy** - Set parameters in `SandboxConfig`
2. **Run simulation** - Execute via `SandboxRunner` or CLI
3. **Evaluate results** - Use `evaluate_success_criteria()`
4. **Iterate** - Adjust strategy based on results
5. **Deploy** - Once approved, switch to live mode

## Support

- See [plan.md](../plan.md) for detailed architecture
- See [LIVE_DATA_SETUP.md](../LIVE_DATA_SETUP.md) for live trading setup
