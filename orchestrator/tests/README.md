# Sandbox-Orchestrator Integration

This document describes the integration between the CopyCat Orchestrator and the Sandbox/Paper Trading Simulation module.

## Overview

The Sandbox module provides a risk-free environment for testing and validating the trading bot's core logic before any real funds are deployed. It uses real-time market data while simulating all trading operations with virtual money.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CopyCat Orchestrator                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐│
│  │   Trader           │    │   Bot              │    │   Copy             ││
│  │   Identification   │───►│   Filtering        │───►│   Trading          ││
│  └────────────────────┘    └────────────────────┘    └─────────┬──────────┘│
│                                                                │            │
└────────────────────────────────────────────────────────────────┼────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Sandbox Simulation Layer                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐│
│  │   SandboxRunner    │───►│   VirtualPortfolio │───►│   Performance      ││
│  │   (Orchestrator)   │    │   Manager          │    │   Tracker          ││
│  └────────────────────┘    └────────────────────┘    └────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### SandboxRunner (`sandbox/runner.py`)

Main orchestrator for sandbox simulation with real-time data support.

```python
from sandbox import SandboxRunner, SandboxConfig

config = SandboxConfig(initial_balance=10000.0)
runner = SandboxRunner(config)

# Set up market data callback
runner.set_market_data_callback(lambda market_id: {
    "market_id": market_id,
    "current_price": 0.5,
    "previous_price": 0.5,
    "volatility": 0.02,
})

# Execute virtual orders
order = VirtualOrder(
    order_id="order_001",
    market_id="test_market",
    side="buy",
    quantity=100.0,
    order_type="market",
    outcome="YES",
)
result = await runner.execute_order(order)
```

### VirtualOrder (`sandbox/config.py`)

Represents a virtual order for simulation.

```python
@dataclass
class VirtualOrder:
    order_id: str
    market_id: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str  # "market" or "limit"
    outcome: str = "YES"
    source_trader: Optional[str] = None
```

### CopyCatOrchestrator (`orchestrator/engine.py`)

Main orchestrator that coordinates all trading modules.

```python
from orchestrator.engine import CopyCatOrchestrator
from orchestrator.config import (
    OrchestratorConfig,
    TradingMode,
    MarketPlatform,
)

config = OrchestratorConfig(
    mode=TradingMode.SANDBOX,
    platform=MarketPlatform.POLYMARKET,
)

orchestrator = CopyCatOrchestrator(config)
```

## Integration Points

### 1. Market Data Callback

The orchestrator provides a callback to fetch real market data:

```python
def get_real_market_data(market_id: str) -> Dict[str, Any]:
    """Sync wrapper for async market data fetching."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(fetch_market_data(market_id))
    finally:
        loop.close()

runner.set_market_data_callback(get_real_market_data)
```

### 2. Copy Trade Execution

When a copied trader makes a trade, the orchestrator executes it in the sandbox:

```python
async def _execute_copy_trade(self, trader_address: str, trade: Trade):
    """Execute a copy trade in sandbox mode."""
    order = VirtualOrder(
        order_id=f"copy_{trade.trade_id}_{uuid.uuid4().hex[:8]}",
        market_id=trade.market_id,
        side=trade.side.value,
        quantity=copy_config.base_position_size,
        order_type="market",
        outcome=trade.outcome,
        source_trader=trader_address,
    )
    
    result = await self.trading_runner.execute_order(order)
    
    if result.status == "FILLED":
        self.state.trades_executed += 1
```

### 3. Performance Tracking

Trades are automatically recorded in the performance tracker:

```python
# In SandboxRunner.execute_order
if result.status == OrderStatus.FILLED.value:
    trade_result = await self.portfolio_manager.execute_copy_trade(...)
    
    # Record trade in tracker
    if self.portfolio_manager.completed_trades:
        latest_trade = self.portfolio_manager.completed_trades[-1]
        self.tracker.record_trade(latest_trade)

# Get performance metrics
metrics = runner.get_performance_metrics()
print(f"Total P&L: ${metrics.total_pnl:,.2f}")
print(f"Win rate: {metrics.win_rate:.1%}")
```

## Configuration

### Sandbox Config

```python
from sandbox.config import SandboxConfig

config = SandboxConfig(
    initial_balance=10000.0,      # Starting balance
    simulate_slippage=True,       # Simulate slippage
    simulate_fees=True,           # Simulate platform fees
    max_position_size_pct=0.10,   # Max 10% per trade
    max_total_exposure_pct=0.50,  # Max 50% total exposure
    min_order_size=1.0,           # Min $1 per trade
)
```

### Orchestrator Config

```python
from orchestrator.config import (
    OrchestratorConfig,
    TradingMode,
    SandboxConfigOrchestrator,
    CopyTradingConfig,
)

config = OrchestratorConfig(
    mode=TradingMode.SANDBOX,
    platform=MarketPlatform.POLYMARKET,
    sandbox=SandboxConfigOrchestrator(
        initial_balance=10000.0,
    ),
    copy_trading=CopyTradingConfig(
        base_position_size=100.0,
        position_size_pct=0.05,
    ),
)
```

## Running Integration Tests

```bash
# Run integration tests
PYTHONPATH=/home/timmy/copycat python orchestrator/tests/test_sandbox_orchestrator_integration.py

# Run all tests
PYTHONPATH=/home/timmy/copycat python -m pytest sandbox/tests/test_sandbox.py -v
PYTHONPATH=/home/timmy/copycat python -m pytest orchestrator/tests/test_orchestrator.py -v
```

## Test Coverage

### Integration Tests

| Test | Description |
|------|-------------|
| `test_orchestrator_sandbox_initialization` | Verifies orchestrator initializes with sandbox mode |
| `test_sandbox_runner_standalone` | Tests sandbox runner independently |
| `test_orchestrator_copy_trade_execution` | Tests copy trade execution through orchestrator |
| `test_performance_metrics` | Verifies performance metrics tracking |
| `test_orchestrator_status` | Tests status reporting |
| `test_sandbox_real_time_data` | Tests real-time data provider integration |

### Expected Results

```
============================================================
SANDBOX-ORCHESTRATOR INTEGRATION TESTS
============================================================

Total: 6 | Passed: 6 | Failed: 0
```

## Features

- **Real-Time Data**: Uses real Polymarket data for accurate simulation
- **Realistic Execution**: Simulates slippage, fees, and fill probability
- **Risk Management**: Enforces position limits and exposure caps
- **Performance Tracking**: Calculates P&L, win rate, Sharpe ratio, max drawdown
- **Portfolio Management**: Tracks positions, balance, and unrealized P&L
- **Copy Trading**: Automatically copies trades from identified traders

## Next Steps

1. **Trader Identification Module** - Identify profitable traders to copy
2. **Bot Filtering System** - Filter out HFT and arbitrage bots
3. **Live Trading Mode** - Extend sandbox to real trading
4. **Web Dashboard** - Visualize performance and manage settings

## Files Modified

| File | Changes |
|------|---------|
| `orchestrator/engine.py` | Fixed copy trade execution, async callback wrapper |
| `sandbox/config.py` | Added outcome field to VirtualOrder |
| `sandbox/portfolio.py` | Added _update_position_prices method |
| `sandbox/runner.py` | Fixed trade recording to performance tracker |
| `orchestrator/tests/test_sandbox_orchestrator_integration.py` | New integration tests |
