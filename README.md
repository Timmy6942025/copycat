# CopyCat - AI Copy Trading Bot for Prediction Markets

<div align="center">

![CopyCat](https://img.shields.io/badge/CopyCat-AI%20Copy%20Trading-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![Tests](https://img.shields.io/badge/Tests-380%2B-passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Intelligent copy trading system that identifies profitable traders and automatically copies their trades on prediction markets like Polymarket and Kalshi.**

[Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Testing](#testing) • [Contributing](#contributing)

</div>

---

## What is CopyCat?

CopyCat is an intelligent copy trading bot designed for prediction markets. It:

1. **Discovers** profitable traders by analyzing historical performance
2. **Filters** out automated bots (HFT, arbitrage) to avoid copying machines
3. **Scores** traders using multi-factor analysis (win rate, Sharpe ratio, drawdown, etc.)
4. **Copies** trades automatically in sandbox mode with virtual money
5. **Tracks** performance with comprehensive analytics

### Why CopyCat?

- **Risk-Free Testing**: Sandbox mode uses real market data with virtual money
- **Bot Detection**: Advanced heuristics to identify and exclude automated trading
- **Multi-Factor Analysis**: Considers win rate, risk-adjusted returns, consistency
- **Position Sizing**: Multiple methods (fixed, percentage, scaled, Kelly criterion)
- **Production Ready**: Clean architecture with comprehensive test coverage

---

## Features

### 🔍 Trader Identification
- Historical trade analysis
- Performance metrics calculation (Sharpe ratio, max drawdown, profit factor)
- Win rate and consistency scoring
- Insider trading and suspicious pattern detection

### 🤖 Bot Filtering
- High-Frequency Trading (HFT) detection
- Arbitrage pattern identification
- Position sizing anomaly detection
- Trading hours pattern analysis

### 🎯 Copy Trading
- Automatic trade copying from identified traders
- Multiple position sizing methods
- Risk management (max position, max exposure)
- Performance monitoring and rebalancing

### 📊 Sandbox Simulation
- Real-time market data from Polymarket/Kalshi
- Realistic order execution with slippage and fees
- Comprehensive performance analytics
- Backtesting capabilities

### ⚙️ Orchestration
- Automated trading cycles
- Health checks and circuit breakers
- Error recovery and logging
- CLI and API interfaces

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/copycat.git
cd copycat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from orchestrator.engine import CopyCatOrchestrator
from orchestrator.config import OrchestratorConfig, TradingMode, MarketPlatform

# Configure the orchestrator
config = OrchestratorConfig(
    mode=TradingMode.SANDBOX,  # Use SANDBOX for paper trading
    platform=MarketPlatform.POLYMARKET,
    copy_trading={
        "base_position_size": 100.0,
        "position_size_pct": 0.05,  # 5% of portfolio per trade
        "position_sizing_method": "scaled",
    }
)

# Create and start the orchestrator
orchestrator = CopyCatOrchestrator(config)
import asyncio
asyncio.run(orchestrator.start())

# Check status
status = orchestrator.get_status()
print(f"Running: {status['is_running']}")
print(f"Mode: {status['mode']}")
print(f"Copied Traders: {status['copied_traders']}")

# Stop when done
asyncio.run(orchestrator.stop())
```

### CLI Usage

```bash
# Start orchestrator in sandbox mode
python -m orchestrator.cli start --mode sandbox

# Check status
python -m orchestrator.cli status

# Stop orchestrator
python -m orchestrator.cli stop
```

### Web Dashboard

CopyCat includes a Flask-based web dashboard for monitoring and controlling the bot:

```bash
# Install Flask (if not already in requirements.txt)
pip install flask

# Start the dashboard
python -m dashboard.app

# Dashboard will be available at http://localhost:5000
```

**Dashboard Features:**
- Real-time portfolio summary (P&L, win rate, Sharpe ratio)
- Start/Stop/Pause/Resume controls
- Add and remove traders to copy
- System health monitoring (API status, circuit breaker)
- Toast notifications for actions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CopyCat System                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   Polymarket    │    │    Kalshi       │    │    Other Markets        │  │
│  │   API Client    │    │   API Client    │    │    (Extensible)         │  │
│  └────────┬────────┘    └────────┬────────┘    └────────────┬────────────┘  │
│           │                      │                           │                │
│           └──────────────────────┼───────────────────────────┘                │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          Orchestrator Engine                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Trader    │──│    Bot       │──│    Copy     │──│  Sandbox    │  │   │
│  │  │ Identification│  │  Filtering │  │   Trading  │  │  Runner     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Overview

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `api_clients` | Market API integrations | `PolymarketAPIClient`, `KalshiAPIClient` |
| `trader_identification` | Analyze trader performance | `TraderIdentificationEngine` |
| `bot_filtering` | Detect automated traders | `BotFilter` |
| `sandbox` | Paper trading simulation | `SandboxRunner`, `VirtualPortfolioManager` |
| `orchestrator` | Main coordination | `CopyCatOrchestrator` |

### Trader Identification Flow

```
1. Fetch trader's historical trades
2. Calculate performance metrics:
   - Win rate, profit factor
   - Sharpe ratio, max drawdown
   - Average hold time, consistency
3. Evaluate suitability against criteria
4. Assign reputation and confidence scores
5. Output: TraderAnalysisResult
```

### Bot Filtering Flow

```
1. Analyze trade patterns:
   - Trade frequency (HFT detection)
   - Position sizing patterns
   - Arbitrage indicators
   - Trading hours patterns
2. Calculate bot scores (0-1)
3. Filter out high-confidence bots
4. Output: BotFilterResult
```

---

## Configuration

### Complete Configuration Example

```python
from orchestrator.config import (
    OrchestratorConfig,
    TradingMode,
    MarketPlatform,
    TraderSelectionConfig,
    BotFilterOrchestratorConfig,
    CopyTradingConfig,
    SandboxConfigOrchestrator,
)

config = OrchestratorConfig(
    # Mode settings
    mode=TradingMode.SANDBOX,
    platform=MarketPlatform.POLYMARKET,
    
    # Trader selection criteria
    trader_selection=TraderSelectionConfig(
        min_win_rate=0.55,
        min_trades=10,
        max_drawdown=0.25,
        min_sharpe_ratio=0.5,
        min_profit_factor=1.0,
        min_total_pnl=0.0,
        min_reputation_score=0.5,
    ),
    
    # Bot filtering
    bot_filter=BotFilterOrchestratorConfig(
        hft_max_hold_time_seconds=1.0,
        hft_min_trades_per_minute=5,
        arbitrage_max_profit_pct=0.5,
        min_hft_score_to_exclude=0.7,
        min_arbitrage_score_to_exclude=0.7,
    ),
    
    # Copy trading settings
    copy_trading=CopyTradingConfig(
        position_sizing_method="scaled",  # fixed, percentage, scaled, kelly
        base_position_size=100.0,
        position_size_pct=0.05,
        kelly_fraction=0.25,
        max_position_size_pct=0.10,
    ),
    
    # Sandbox settings
    sandbox=SandboxConfigOrchestrator(
        initial_balance=10000.0,
        simulate_slippage=True,
        simulate_fees=True,
    ),
    
    # Constraints
    max_traders_to_copy=10,
    max_traders_to_analyze_per_cycle=100,
    trader_data_refresh_interval_seconds=300,
)
```

### Configuration Options

#### TradingMode
| Value | Description |
|-------|-------------|
| `sandbox` | Paper trading with virtual money |
| `live` | Real trading with actual funds |

#### MarketPlatform
| Value | Description |
|-------|-------------|
| `polymarket` | Polymarket prediction market |
| `kalshi` | Kalshi prediction market |

#### Position Sizing Methods
| Method | Description |
|--------|-------------|
| `fixed_amount` | Fixed $ amount per trade |
| `percentage` | % of portfolio per trade |
| `scaled` | Scaled by trader confidence score |
| `kelly` | Kelly criterion optimization |

---

## Testing

CopyCat has a comprehensive test suite with **380+ tests** covering all modules.

### Running Tests

```bash
# Run all tests
python -m pytest -v

# Run specific module tests
python -m pytest trader_identification/tests/ -v
python -m pytest bot_filtering/tests/ -v
python -m pytest sandbox/tests/ -v
python -m pytest orchestrator/tests/ -v

# Run with coverage
python -m pytest --cov=copycat --cov-report=html

# Run integration tests
python -m pytest orchestrator/tests/test_sandbox_orchestrator_integration.py -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `sandbox` | 196 | ~95% |
| `trader_identification` | 61 | ~92% |
| `bot_filtering` | 47 | ~90% |
| `orchestrator` | 56 | ~88% |
| **Total** | **380+** | **~90%** |

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Module interaction testing
- **Edge Case Tests**: Boundary and error conditions
- **Performance Tests**: Scalability and stress testing

---

## Project Structure

```
copycat/
├── __init__.py              # Main package exports
├── requirements.txt         # Dependencies
├── plan.md                  # Detailed architecture plan
│
├── api_clients/             # Market API integrations
│   ├── __init__.py
│   ├── base.py              # Base API client
│   ├── polymarket.py        # Polymarket API
│   ├── kalshi.py            # Kalshi API
│   ├── mock.py              # Mock client for testing
│   └── tests/
│
├── trader_identification/   # Trader analysis engine
│   ├── __init__.py
│   ├── engine.py            # Main analysis engine
│   ├── config.py            # Configuration
│   └── tests/
│       └── test_trader_identification.py  # 61 tests
│
├── bot_filtering/           # Bot detection
│   ├── __init__.py
│   ├── engine.py            # Detection engine
│   ├── config.py            # Configuration
│   └── tests/
│       └── test_bot_filtering.py  # 47 tests
│
├── sandbox/                 # Paper trading simulation
│   ├── __init__.py
│   ├── config.py            # Sandbox configuration
│   ├── executor.py          # Order execution
│   ├── portfolio.py         # Portfolio management
│   ├── analytics.py         # Performance tracking
│   ├── runner.py            # Main runner
│   ├── success_criteria.py  # Live trading approval
│   ├── cli.py               # Command-line interface
│   ├── dashboard.py         # TUI dashboard
│   ├── data/                # Data providers
│   │   ├── __init__.py
│   │   ├── historical_loader.py
│   │   └── market_data_cache.py
│   └── tests/               # 196 tests
│
├── orchestrator/            # Main orchestration
│   ├── __init__.py
│   ├── engine.py            # CopyCatOrchestrator
│   ├── config.py            # All configuration
│   ├── cli.py               # CLI interface
│   ├── tests/
│   │   ├── test_orchestrator.py  # 50 tests
│   │   └── test_sandbox_orchestrator_integration.py  # 6 tests
│   └── tests/README.md      # Integration documentation
│
└── tests/                   # Root-level tests
    ├── test_sandbox_real_data.py
    └── test_standalone_sandbox.py
```

---

## API Reference

### TraderIdentificationEngine

```python
from trader_identification import TraderIdentificationEngine, TraderSelectionConfig

# Configure
config = TraderSelectionConfig(
    min_win_rate=0.55,
    min_trades=10,
    max_drawdown=0.25,
    min_sharpe_ratio=0.5,
)

engine = TraderIdentificationEngine(config)

# Analyze a trader
result = await engine.analyze_trader(
    trader_address="0x...",
    trades=trades_list
)

# Result attributes:
# - is_suitable: bool
# - reputation_score: float
# - confidence_score: float
# - performance: PerformanceMetrics
# - selection_reasons: List[str]
# - rejection_reasons: List[str]
```

### BotFilter

```python
from bot_filtering import BotFilter, BotFilterConfig

# Configure
config = BotFilterConfig(
    hft_max_hold_time_seconds=1.0,
    hft_min_trades_per_minute=5,
    min_hft_score_to_exclude=0.7,
)

filter = BotFilter(config)

# Analyze trades for bot patterns
result = filter.analyze_trades(trades_list)

# Result attributes:
# - is_bot: bool
# - hft_score: float
# - arbitrage_score: float
# - pattern_score: float
# - confidence: float
# - reasons: List[str]
```

### SandboxRunner

```python
from sandbox import SandboxRunner, SandboxConfig, VirtualOrder

# Configure
config = SandboxConfig(
    initial_balance=10000.0,
    simulate_slippage=True,
    simulate_fees=True,
)

runner = SandboxRunner(config)

# Execute virtual order
order = VirtualOrder(
    order_id="order_001",
    market_id="bitcoin",
    side="buy",
    quantity=100.0,
    order_type="market",
    outcome="YES",
)

result = await runner.execute_order(order)

# Get performance metrics
metrics = runner.get_performance_metrics()
print(f"Total P&L: ${metrics.total_pnl:,.2f}")
print(f"Win Rate: {metrics.win_rate:.1%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

---

## Success Criteria for Live Trading

Before deploying to live trading, your sandbox simulation must meet these minimum requirements:

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Monthly Return | > 3% | > 5% | > 8% |
| Win Rate | > 52% | > 58% | > 65% |
| Sharpe Ratio | > 0.8 | > 1.2 | > 1.5 |
| Max Drawdown | < 25% | < 15% | < 10% |
| Profit Factor | > 1.2 | > 1.5 | > 2.0 |
| Simulation Duration | 90 days | 180 days | 365 days |
| Consistency | 4/6 positive | 5/6 positive | 6/6 positive |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

---

## Roadmap

- [ ] API client tests (30+ tests)
- [ ] Live trading integration
- [ ] Web dashboard for monitoring
- [ ] Additional market integrations
- [ ] Advanced backtesting features
- [ ] Machine learning for trader selection

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- [Polymarket](https://polymarket.com) for their prediction market API
- [Kalshi](https://kalshi.com) for market data access
- All contributors and testers
- Made with [Mninimax-m2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) with [opencode](https://github.com/anomalyco/opencode) paired with [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode)

---

<div align="center">

**Built with ❤️ for the prediction market community**

</div>
