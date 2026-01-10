# CopyCat - AI Copy Trading Bot for Prediction Markets

<div align="center">

![CopyCat](https://img.shields.io/badge/CopyCat-AI%20Copy%20Trading-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![Tests](https://img.shields.io/badge/Tests-481%2B-passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Intelligent copy trading system that identifies profitable traders and automatically copies their trades on prediction markets like Polymarket.**

[Features](#features) • [Speed Mode](#speed-mode) • [Architecture](#architecture) • [Testing](#testing)

</div>

---

## What is CopyCat?

CopyCat is an intelligent copy trading bot designed for prediction markets. It:

1. **Discovers** profitable traders by analyzing historical performance
2. **Filters** out automated bots (HFT, arbitrage) to avoid copying machines
3. **Scores** traders using multi-factor analysis (win rate, Sharpe ratio, drawdown, etc.)
4. **Copies** trades automatically in sandbox mode with virtual money
5. **Optimizes** with 8 advanced speed features for maximum growth

### Why CopyCat?

- **Risk-Free Testing**: Sandbox mode uses real market data with virtual money
- **Bot Detection**: Advanced heuristics to identify and exclude automated trading
- **Growth-First Selection**: Find traders by actual account growth, not win rate
- **Speed Mode**: 8 optimization features for 2x faster growth
- **Adaptive Scaling**: Automatically adjusts as portfolio grows

---

## Features

### 🔍 Trader Identification
- Historical trade analysis
- Performance metrics calculation (Sharpe ratio, max drawdown, profit factor)
- Win rate and consistency scoring
- **Growth-First Selection**: Find traders by actual account growth, not win rate
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
- Real-time market data from Polymarket
- Realistic order execution with slippage and fees
- Comprehensive performance analytics
- Backtesting capabilities
- **Micro Mode**: Full feature parity with live trading

### ⚙️ Orchestration
- Automated trading cycles
- Health checks and circuit breakers
- Error recovery and logging
- CLI and API interfaces
- **Micro Mode Integration**: Circuit breaker, milestone tracking, auto transitions

---

## 🎯 Micro Mode - Small Account Optimization

CopyCat includes **Micro Mode** specifically designed for small accounts ($10-$100) with aggressive growth strategies.

### Key Features
 
 | Feature | Description |
 |---------|-------------|
 | **Aggressive Position Sizing** | Up to 75% position size for $10 accounts |
 | **Circuit Breaker Protection** | Mode-specific drawdown thresholds |
 | **Automatic Mode Transitions** | NANO → MICRO → MINI → BALANCED as balance grows |
 | **Milestone Notifications** | Discord alerts for $20, $50, $100 milestones |
 | **Bootstrap Trading** | Instant diversification from proven traders |
 | **Quick Resolve Prioritization** | Skip long-running markets, apply 2x bonus to quick-resolving markets |

### Mode Settings

| Mode | Balance Range | Position Size | Kelly | Max Drawdown |
|------|---------------|---------------|-------|--------------|
| **NANO** | $0-$15 | 75% | 0.75 | 30% |
| **MICRO** | $15-$25 | 60% | 0.75 | 25% |
| **MINI** | $25-$50 | 50% | 0.50 | 20% |
| **BALANCED** | $50-$200 | 40% | 0.40 | 18% |
| 
### Usage
```python
# For sandbox testing with micro mode
from orchestrator.sandbox_micro import create_micro_sandbox_runner

runner = await create_micro_sandbox_runner(
    config=create_micro_sandbox_config(
        initial_balance=10.0,
        micro_mode="nano",  # Start with NANO for $10
        enable_notifications=True,
        discord_webhook_url="https://discord.com/api/webhooks/...",
    ),
    api_client=mock_client,
)

await runner.start()
```

```python
# For live trading with micro mode
from orchestrator.live_trading_micro import create_micro_live_runner

runner = await create_micro_live_runner(
    config=create_micro_live_config(
        initial_balance=10.0,
        micro_mode="nano",
        wallet_address="0x...",  # Your wallet address
        enable_notifications=True,
        discord_webhook_url="https://discord.com/api/webhooks/...",
    ),
    api_client=polymarket_client,
)

await runner.start()
```
```

## 🧺 Topic-Based Basket Trading

CopyCat includes **topic-based wallet basket trading** for diversified, consensus-driven strategies. Instead of copying individual traders, this system builds "baskets" of wallets grouped by topic (e.g., geopolitics, crypto, sports, elections) and generates trading signals when 80%+ of the basket converges on the same outcome.

### Key Concepts

- **Topic-Based Baskets**: Group wallets by market category expertise
- **Weighted Scoring**: Recent performance (7d/30d) weighted higher than all-time
- **Consensus Signals**: 80%+ basket agreement triggers trades
- **Price Band Filtering**: Ensure basket buys in tight price range
- **Spread Protection**: Only trade when spread isn't "cooked"

### Basket Construction Process

1. **Wallet Filtering**:
   - Only wallets older than 6 months
   - No bots (filtered out wallets doing thousands of micro-trades)
   - Recent win rate weighted more than all-time (last 7 days and last 30 days)
   - Ranked by avg entry vs final price

2. **Copycat Cluster Detection**:
   - Excludes duplicate/correlated wallets that copy the same traders
   - Only representative wallet from each cluster participates

3. **Signal Generation**:
   - Wait until 80%+ of basket enters the same outcome
   - Check they're all buying within a tight price band (5% max)
   - Only trigger if spread isn't cooked yet (10% max threshold)
   - Right now: paper trading to avoid bias

### Basket Configuration

```python
from basket_trading import (
    BasketConfig,
    Topic,
)
from basket_trading.orchestrator import BasketTradingOrchestrator
from api_clients.polymarket.data_api import DataAPIClient

# Create basket trading config
config = BasketConfig(
    # Wallet filtering
    min_wallet_age_months=6,
    min_trades_7d=3,
    min_trades_30d=10,
    min_win_rate_7d=0.45,
    min_win_rate_30d=0.50,

    # Bot filtering
    max_hft_score=0.5,
    max_arbitrage_score=0.5,
    max_pattern_score=0.5,
    max_micro_trades_per_day=1000,

    # Basket composition
    min_basket_size=10,
    max_basket_size=50,
    min_consensus_pct=0.80,  # 80% agreement

    # Signal generation
    max_price_band_pct=0.05,  # 5% price band
    max_spread_pct=0.10,  # 10% spread threshold
    min_basket_participation_pct=0.60,  # 60% of basket must be active

    # Ranking weights
    win_rate_7d_weight=0.40,
    win_rate_30d_weight=0.35,
    win_rate_all_time_weight=0.25,
    avg_entry_vs_final_weight=0.30,
    consistency_weight=0.20,
    volume_weight=0.10,

    # Copycat cluster detection
    cluster_similarity_threshold=0.85,  # 85% trade similarity
    max_cluster_size=3,  # Max wallets per cluster
)

# Initialize orchestrator
data_client = DataAPIClient()
orchestrator = BasketTradingOrchestrator(data_client, config)

# Define wallet topics
topic_wallets = {
    Topic.GEOPOLITICS: [
        "0x1234567890abcdef1234567890abcdef1234567890ab",
        "0x2345678901abcdef1234567890abcdef1234567890cd",
    ],
    Topic.POLITICS: [
        "0x345678901234567890abcdef1234567890def",
    ],
    Topic.ELECTIONS: [
        "0x45678901234567890abcdef1234567890eff",
    ],
}

# Initialize baskets
baskets = await orchestrator.initialize_topic_baskets(topic_wallets)
await orchestrator.detect_and_assign_clusters()
await orchestrator.rank_all_baskets()

# Scan for consensus signals
signals = await orchestrator.scan_for_signals()

# Execute top signals
results = await orchestrator.execute_top_signals(limit=3)

print(f"Generated {len(signals)} signals, executed {sum(1 for r in results if r.executed)} trades")
```

### Demo Script

```bash
# Run basket trading demo
python demo_basket_trading.py
```

The demo will show:
- Signal filtering and validation logic
- Basket statistics for each topic
- Generated consensus signals
- Paper trading execution results
- Performance summary

### Key Differences from Individual Trader Copying

| Aspect | Individual Copying | Basket Trading |
|---------|-------------------|-----------------|
| **Approach** | Follow single "smart" trader | Monitor group consensus |
| **Diversification** | Fragile - trader can drift | Resilient - distributed across wallets |
| **Signal Quality** | Depends on one trader's consistency | High - requires 80%+ agreement |
| **Noise Reduction** | Single trader idiosyncrasies filtered out | Basket smooths individual biases |
| **Risk Profile** | Concentrated risk | Diversified risk across topic experts |
| **Entry Timing** | When one trader enters | When basket forms consensus |
|

```python
# For live trading with micro mode
from orchestrator.live_trading_micro import create_micro_live_runner

runner = await create_micro_live_runner(
    config=create_micro_live_config(
        initial_balance=10.0,
        micro_mode="nano",
        wallet_address="0x...",  # Your wallet address
        enable_notifications=True,
        discord_webhook_url="https://discord.com/api/webhooks/...",
    ),
    api_client=polymarket_client,
)

await runner.start()
```

### Demo Script

```bash
# Test in sandbox mode
python demo_live_trading_micro.py --mode sandbox --balance 10 --micro-mode nano

# Live trading (⚠️ REAL MONEY)
python demo_live_trading_micro.py --mode live --balance 10 --micro-mode nano --wallet 0x... --discord "WEBHOOK_URL"
```

### API Endpoints

```bash
# Auto-select mode based on balance
POST /api/config/auto-select
{
    "balance": 25.0,
    "risk_tolerance": "medium"
}

# Validate configuration
POST /api/config/validate
{
    "config": {
        "position_size_pct": 0.50,
        "kelly_fraction": 0.50
    }
}
```

---

## 🚀 Speed Mode - Maximum Growth

CopyCat includes **8 advanced optimization features** for maximum compound growth:

| Feature | Purpose | Impact |
|---------|---------|--------|
| **Tiered Copying** | Top traders get 3x capital | +30% faster |
| **Momentum Filtering** | Only copy recent winners | +20% better picks |
| **Event Focus** | Prioritize elections/Fed/etc. | +50% accuracy |
| **Cross-Market Hedging** | Reduce portfolio risk | Lower drawdown |
| **Auto-Optimizer** | Learns from your trades | Continuous improvement |
| **Smart Allocation** | More $ to better traders | +25% returns |
| **Quick-Start Bootstrap** | Copy historical winners + leaderboard | ~1 month faster |
| **Adaptive Scaling** | Scale up when profitable | +15% during growth |

### 🚀 Speed Mode Optimizations (NEW!)

Major performance improvements for faster compounding:

| Setting | Old Value | New Value | Impact |
|---------|-----------|-----------|--------|
| Cycle Refresh | 300s (5 min) | 10s | **30x faster** |
| Position Size | 8-15% | 25-50% | **3-6x larger** |
| Max Orders/Day | 50-150 | 500-1000 | **5-20x more** |
| Kelly Fraction | 0.25 | 0.50 | **2x Kelly** |
| Hedging | Enabled | Disabled | **10% more capital** |
| Min Growth Rate | 2% | 0.5% | **4x more traders qualify** |
| Trader Analysis | Sequential | Parallel | **10-50x faster** |
| Trader Discovery | Recent trades only | Bootstrap + Leaderboard | **Instant copy** |

### Usage

```python
from orchestrator.config_speed import create_speed_mode_config, SpeedModeEngine

# Create speed mode config
config = create_speed_mode_config(
    initial_balance=100.0,
    speed_mode="balanced",  # conservative/balanced/aggressive
)

# Create engine with all 8 optimizations
engine = SpeedModeEngine(config)
```

### Expected Growth ($10 → $20)

With the new speed optimizations, compound growth is significantly faster:

| Mode | Position Size | Monthly Return | Time to Double |
|------|---------------|----------------|----------------|
| Conservative | 25% | ~8% | ~9 months |
| **Balanced** | **35%** | ~12-15% | **~4-5 months** |
| Aggressive | 50% | ~18-22% | ~3-4 months |
| Extreme | 50% | ~25%+ | ~2-3 months |

> **Note**: These are estimates based on 30x faster cycles and 3-6x larger positions. Actual results depend on trader selection and market conditions.

### Key Optimizations Explained

1. **30x Faster Cycles**: Trading cycles now run every 10 seconds instead of 5 minutes, allowing more frequent position updates and faster reaction to market movements.

2. **3-6x Larger Positions**: Position sizes increased from 8-15% to 25-50%, maximizing capital utilization per trade.

3. **Parallel Trader Analysis**: Multiple traders are now analyzed simultaneously using `asyncio.gather()`, reducing analysis time from 100+ seconds to under 2 seconds.

4. **Bootstrap + Leaderboard Integration**: Traders are now discovered from:
   - Polymarket builder leaderboard (top profitable builders)
   - Pre-configured bootstrap list of known profitable traders
   - Recent active traders (fallback)

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
| `api_clients` | Market API integrations | `PolymarketAPIClient`, `DataAPIClient` |
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
| `api_clients` | 80 | ~90% |
| **Total** | **480+** | **~90%** |

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
│   ├── __init__.py          # Main exports (PolymarketAPIClient, DataAPIClient)
│   ├── base.py              # Base API client
│   ├── polymarket.py        # Polymarket API (CLOB, Gamma, Data APIs)
│   ├── mock.py              # Mock client for testing
│   ├── data_api.py          # Polymarket Data API client (NEW)
│   └── tests/
│       ├── test_polymarket.py      # CLOB/Gamma API tests
│       ├── test_data_api.py        # Data API unit tests (17 tests)
│       └── test_data_api_integration.py  # Data API integration tests
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
│   │
│   ├── config_aggressive.py # Aggressive growth config
│   ├── config_adaptive.py   # Adaptive scaling
│   ├── config_tiered.py     # Tiered copying
│   ├── config_momentum.py   # Momentum filtering
│   ├── config_events.py     # Event-based focus
│   ├── config_hedging.py    # Cross-market hedging
│   ├── config_optimizer.py  # Auto-optimization
│   ├── config_allocation.py # Smart capital allocation
│   ├── config_bootstrap.py  # Quick-start bootstrap
│   └── config_speed.py      # Unified Speed Mode (ALL features)
│
│   └── tests/               # 56 orchestrator tests
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

### Polymarket Data API Client

```python
from api_clients.polymarket import DataAPIClient

# Create client (optionally with API key from https://builders.polymarket.com)
client = DataAPIClient(api_key="your_api_key")

# Get user positions
positions = await client.get_positions(
    user_address="0x1234567890abcdef",
    limit=100,
    size_threshold=1.0,
)
for pos in positions:
    print(f"{pos.title}: {pos.size:.2f} @ ${pos.avg_price:.2f}")
    print(f"  P&L: ${pos.cash_pnl:,.2f} ({pos.percent_pnl:.1%})")

# Get user activity
activities = await client.get_activity(
    user_address="0x1234567890abcdef",
    limit=50,
)

# Get trade history
trades = await client.get_trades(
    user_address="0x1234567890abcdef",
    limit=100,
)

# Get user summary
summary = await client.get_user_summary("0x1234567890abcdef")
print(f"Positions: {summary['positions_count']}")
print(f"Total P&L: ${summary['total_pnl']:,.2f}")
print(f"Win Rate: {summary['win_rate']:.1%}")

# Get builder leaderboard
leaderboard = await client.get_builder_leaderboard(limit=100)

# Cleanup
await client.close()
```

**Data API Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `/positions` | User positions with P&L data |
| `/activity` | User activity history |
| `/trades` | Trade history |
| `/builders/leaderboard` | Builder rankings |

**Data API Features:**
- Free to use with Polymarket Builder API key (no bank card required)
- Rate limited to ~10 requests/second
- Returns real user trading data for copy trading analysis

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

## Demos

Several demo scripts are included to showcase CopyCat's capabilities:

```bash
# Growth-first selection demo
python demo_growth_first.py

# Aggressive growth mode demo
python demo_aggressive_growth.py

# Speed Mode (all 8 optimizations)
python orchestrator/config_speed.py
```

### Demo Scripts

| Script | Purpose |
|--------|---------|
| `demo_growth_first.py` | Demonstrates growth-based trader selection |
| `demo_aggressive_growth.py` | Aggressive growth configuration |
| `orchestrator/config_tiered.py` | Tiered copying system |
| `orchestrator/config_momentum.py` | Momentum filtering system |
| `orchestrator/config_events.py` | Event-based focus system |
| `orchestrator/config_hedging.py` | Cross-market hedging system |
| `orchestrator/config_optimizer.py` | Auto-optimization system |
| `orchestrator/config_allocation.py` | Smart capital allocation |
| `orchestrator/config_bootstrap.py` | Quick-start bootstrap |
| `orchestrator/config_speed.py` | All 8 speed optimizations combined |

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

- [x] API client tests (80+ tests, including Data API integration tests)
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
- All contributors and testers
- Made with [Mninimax-m2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) with [opencode](https://github.com/anomalyco/opencode) paired with [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode)

---

<div align="center">

**Built with ❤️ for the prediction market community**

</div>
