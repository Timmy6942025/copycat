# CopyCat - Polymarket/Kalshi Copy Trading Bot

A sophisticated copy trading bot that identifies and copies successful traders on prediction markets (Polymarket, Kalshi) while avoiding HFT/bots.

## Features

- **Trader Identification**: Identify profitable traders with high win rates and insider trading patterns
- **Bot Filtering**: Filter out HFT bots, arbitrage bots, and wash traders
- **Copy Trading**: Automatically copy trades from selected traders with configurable position sizing
- **Sandbox Simulation**: Test strategies with real market data using virtual money
- **Risk Management**: Position limits, exposure caps, and automatic risk controls
- **Performance Analytics**: Track P&L, win rate, Sharpe ratio, max drawdown, and more

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CopyCat Trading Bot                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐│
│  │   Trader           │    │   Bot              │    │   Copy             ││
│  │   Identification   │───►│   Filtering        │───►│   Trading          ││
│  │   Engine           │    │   System           │    │   Engine           ││
│  └────────────────────┘    └────────────────────┘    └─────────┬──────────┘│
│                                                                │            │
│  ┌────────────────────┐    ┌────────────────────┐              │            │
│  │   API Clients      │    │   Orchestrator     │◄─────────────┼────────────┘
│  │   (Polymarket,     │───►│   (Main Controller)│              │
│  │   Kalshi)          │    │                    │              │
│  └────────────────────┘    └────────────────────┘              │
│                                                                │
└────────────────────────────────────────────────────────────────┼────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Sandbox Simulation Layer                              │
│                    (Risk-free testing with real data)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐│
│  │   SandboxRunner    │───►│   VirtualPortfolio │───►│   Performance      ││
│  │   (Trade Sim)      │    │   Manager          │    │   Tracker          ││
│  └────────────────────┘    └────────────────────┘    └────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
copycat/
├── api_clients/           # API clients for market data
│   ├── base.py           # Base classes and interfaces
│   └── polymarket.py     # Polymarket API client
├── bot_filtering/         # HFT/bot detection system
├── orchestrator/          # Main orchestration layer
│   ├── engine.py         # CopyCatOrchestrator class
│   ├── config.py         # Configuration dataclasses
│   └── tests/            # Integration tests + README
├── sandbox/              # Paper trading simulation
│   ├── runner.py         # SandboxRunner orchestrator
│   ├── executor.py       # Virtual order execution
│   ├── portfolio.py      # Virtual portfolio management
│   ├── analytics.py      # Performance tracking
│   ├── backtest.py       # Historical backtesting
│   ├── config.py         # Sandbox configuration
│   ├── data_providers/   # Real-time data feeds
│   └── tests/            # Sandbox tests
├── trader_identification/ # Trader scoring system
├── example_usage.py      # Basic usage example
├── example_sandbox_standalone.py  # Sandbox demo
├── plan.md               # Detailed implementation plan
├── LIVE_DATA_SETUP.md    # Data setup guide
└── requirements.txt      # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/copycat.git
cd copycat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from orchestrator.engine import CopyCatOrchestrator
from orchestrator.config import (
    OrchestratorConfig,
    TradingMode,
    MarketPlatform,
)

# Configure the orchestrator
config = OrchestratorConfig(
    mode=TradingMode.SANDBOX,  # Use SANDBOX for testing, LIVE for real trading
    platform=MarketPlatform.POLYMARKET,
)

# Initialize and start
orchestrator = CopyCatOrchestrator(config)
await orchestrator.start()
```

### Sandbox Simulation

```python
from sandbox import SandboxRunner, SandboxConfig, VirtualOrder

# Create sandbox runner
config = SandboxConfig(initial_balance=10000.0)
runner = SandboxRunner(config)

# Set up market data callback
runner.set_market_data_callback(lambda mid: {
    "market_id": mid,
    "current_price": 0.5,
    "previous_price": 0.5,
    "volatility": 0.02,
})

# Execute virtual trades
order = VirtualOrder(
    order_id="trade_001",
    market_id="polymarket_market_123",
    side="buy",
    quantity=100.0,
    order_type="market",
    outcome="YES",
)

result = await runner.execute_order(order)
print(f"Order filled: {result.status} @ ${result.average_price:.4f}")

# Get performance metrics
metrics = runner.get_performance_metrics()
print(f"Total P&L: ${metrics.total_pnl:,.2f}")
print(f"Win rate: {metrics.win_rate:.1%}")
```

## Configuration

### Orchestrator Config

```python
from orchestrator.config import (
    OrchestratorConfig,
    TradingMode,
    SandboxConfigOrchestrator,
    CopyTradingConfig,
)

config = OrchestratorConfig(
    mode=TradingMode.SANDBOX,           # SANDBOX or LIVE
    platform=MarketPlatform.POLYMARKET, # POLYMARKET or KALSHI
    
    sandbox=SandboxConfigOrchestrator(
        initial_balance=10000.0,
        simulate_slippage=True,
        simulate_fees=True,
    ),
    
    copy_trading=CopyTradingConfig(
        base_position_size=100.0,       # $ per trade
        position_size_pct=0.05,          # 5% of portfolio
        max_traders_to_copy=10,
        min_trader_winrate=0.55,        # Min 55% win rate
        min_trader_trades=20,           # Min 20 trades
    ),
    
    trader_selection=TraderSelectionConfig(
        min_winrate=0.55,
        min_total_trades=20,
        min_profit_factor=1.2,
        max_avg_hold_time_hours=72,     # Avoid day traders
        min_volume_30d=1000.0,
    ),
)
```

### Sandbox Config

```python
from sandbox.config import SandboxConfig

config = SandboxConfig(
    initial_balance=10000.0,        # Starting balance
    simulate_slippage=True,         # Simulate price impact
    simulate_fees=True,             # Apply platform fees
    max_orders_per_day=50,          # Rate limiting
    max_position_size_pct=0.10,     # Max 10% per trade
    max_total_exposure_pct=0.50,    # Max 50% total exposure
    min_order_size=1.0,             # Min $1 per trade
)
```

## Testing

### Run All Tests

```bash
# Sandbox tests
PYTHONPATH=/home/timmy/copycat python -m pytest sandbox/tests/test_sandbox.py -v

# Orchestrator tests
PYTHONPATH=/home/timmy/copycat python -m pytest orchestrator/tests/test_orchestrator.py -v

# Integration tests
PYTHONPATH=/home/timmy/copycat python orchestrator/tests/test_sandbox_orchestrator_integration.py
```

### Demo Scripts

```bash
# Basic example
python example_usage.py

# Sandbox standalone demo
python example_sandbox_standalone.py

# Sandbox demo with real data
PYTHONPATH=/home/timmy/copycat python sandbox/demo_sandbox.py
```

## API Reference

### CopyCatOrchestrator

```python
class CopyCatOrchestrator:
    async def start()           # Start the trading bot
    async def stop()            # Stop and cleanup
    async def pause()           # Pause trading
    async def resume()          # Resume trading
    async def add_trader(address: str)  # Add trader to copy
    async def remove_trader(address: str)  # Remove trader
    def get_status() -> dict    # Get current status
    def get_performance_metrics() -> PerformanceMetrics  # Get metrics
```

### SandboxRunner

```python
class SandboxRunner:
    async def execute_order(order: VirtualOrder) -> VirtualOrderResult
    def get_portfolio_summary() -> PortfolioSummary
    def get_performance_metrics() -> PerformanceMetrics
    def save_report(filename: str = None) -> str
    def reset()  # Reset to initial state
```

### VirtualOrder

```python
@dataclass
class VirtualOrder:
    order_id: str
    market_id: str
    side: str           # "buy" or "sell"
    quantity: float
    order_type: str     # "market" or "limit"
    outcome: str = "YES"    # "YES" or "NO"
    source_trader: Optional[str] = None
```

## Data Sources

### Polymarket

- **Market Data**: Real-time prediction market prices
- **Trades**: Public trade history
- **Traders**: Trader profiles and statistics

### Sandbox Data Providers

```python
from sandbox.data_providers import (
    CoinGeckoProvider,      # Cryptocurrency prices
    YahooFinanceProvider,   # Stock prices
    PolymarketDataProvider, # Prediction markets
)
```

## Next Steps

1. **Trader Identification Module** - Implement scoring algorithm
2. **Bot Filtering System** - HFT and arbitrage detection
3. **Live Trading Mode** - Connect to real exchanges
4. **Web Dashboard** - Visual monitoring interface
5. **API Server** - REST API for external integration

## Documentation

- [Plan.md](plan.md) - Detailed implementation guide
- [Live Data Setup](LIVE_DATA_SETUP.md) - Data configuration
- [Sandbox-Orchestrator Integration](orchestrator/tests/README.md) - Integration docs

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## Disclaimer

This software is for educational and research purposes only. Always use sandbox mode first. Trading prediction markets involves financial risk. Use at your own discretion.
