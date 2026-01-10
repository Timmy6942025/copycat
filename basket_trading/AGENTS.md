# BASKET TRADING KNOWLEDGE BASE

**Generated:** Sat Jan 10 2026
**Commit:** Current working tree
**Branch:** main

## OVERVIEW

Topic-based wallet basket construction and consensus-driven signal generation for prediction markets. Instead of copying individual traders, this system builds "baskets" of wallets grouped by topic (e.g., geopolitics, crypto, sports, elections) and generates trading signals when 80%+ of the basket converges on the same outcome.

**Stack:** Python 3.11+, asyncio, dataclasses, typing

## STRUCTURE

```
basket_trading/
├── __init__.py              # Core data models (Topic, BasketConfig, BasketWallet, BasketSignal)
├── builder.py               # WalletBasketBuilder - builds topic-based baskets
├── cluster_detector.py        # ClusterDetector - detects copycat clusters
├── engine.py                # BasketEngine - generates consensus signals
├── ranker.py                # BasketRanker - ranks wallets within baskets
├── orchestrator.py           # BasketTradingOrchestrator - integrates all components
├── sandbox_runner.py         # BasketSandboxRunner - paper trading simulation
└── tests/                   # Unit and integration tests
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Data models | `__init__.py` | Topic, BasketConfig, WalletBasket, BasketWallet, BasketSignal, BasketTradeResult |
| Basket building | `builder.py` | WalletBasketBuilder with filtering and performance calculation |
| Cluster detection | `cluster_detector.py` | ClusterDetector for copycat detection |
| Signal generation | `engine.py` | BasketEngine with 80% consensus logic |
| Wallet ranking | `ranker.py` | BasketRanker with multi-factor scoring |
| Orchestration | `orchestrator.py` | BasketTradingOrchestrator integrating all components |
| Sandbox execution | `sandbox_runner.py` | BasketSandboxRunner for paper trading |
| Demo | `demo_basket_trading.py` | Shows basket trading usage |

## CONVENTIONS

**Topic-based grouping**: Wallets are categorized by market expertise (geopolitics, politics, elections, crypto, economics, finance, sports, technology, climate, health, other)
**Weighted performance scoring**: Recent win rate (7d/30d) weighted higher than all-time (40%/35%/25%)
**Copycat cluster detection**: Multi-factor similarity scoring (market overlap 60%, timing 20%, position size 20%)
**Consensus signal generation**: 80%+ basket agreement required, with price band (5% max) and spread (10% max) validation
**Paper trading first**: All signals simulated in sandbox before live deployment
**Dataclass configs**: All configuration uses Python dataclasses with field(default_factory=...)
**Async-first architecture**: All I/O uses async/await (parallel wallet analysis, position fetching)

## CORE CONCEPTS

### Basket Construction

1. **Wallet Age Filter**: Only wallets older than 6 months (configurable via `min_wallet_age_months`)
2. **Bot Filtering**: Excludes wallets with:
   - HFT score > 0.5 (high-frequency trading detection)
   - Arbitrage score > 0.5 (arbitrage pattern detection)
   - Pattern score > 0.5 (automated behavior detection)
3. **Recent Activity**: Requires minimum trade counts:
   - 3+ trades in last 7 days
   - 10+ trades in last 30 days
4. **Win Rate Requirements**:
   - 7-day win rate >= 45%
   - 30-day win rate >= 50%
5. **Copycat Cluster Detection**:
   - Wallets with >85% trade similarity are in same cluster
   - Only representative wallet from each cluster participates in signals
   - Max 3 wallets per cluster in basket (configurable via `max_cluster_size`)

### Signal Generation

1. **Consensus Threshold**: 80%+ of basket must enter same outcome
2. **Price Band Validation**: All participating wallets must buy within 5% price band (configurable via `max_price_band_pct`)
3. **Spread Protection**: Signal only triggers if spread < 10% (configurable via `max_spread_pct`)
4. **Basket Participation**: Minimum 60% of basket must be active (last 30 days)
5. **Signal Strength**: Calculated as consensus_pct * (1 - price_range_pct)

### Basket Ranking

1. **Multi-Factor Score**:
   - Weighted win rate: 40% * 7d + 35% * 30d + 25% * all-time
   - Entry vs final price: 30% * average entry vs final price
   - Consistency: 20% * (Sharpe/3 - max_drawdown)
   - Volume: 10% * tiered volume scoring
2. **Cluster Representatives**: Only representative wallets from clusters participate in ranking
3. **Eligibility Filtering**: Remove wallets that don't meet performance criteria

## KEY ALGORITHMS

### Wallet Performance Calculation

```python
# In builder.py
def _calculate_wallet_performance(trades, now):
    # Calculate performance across 3 time windows
    trades_7d = [t for t in trades if (now - t.timestamp).days <= 7]
    trades_30d = [t for t in trades if (now - t.timestamp).days <= 30]

    win_rate_7d = len(winning_trades_7d) / len(trades_7d)
    win_rate_30d = len(winning_trades_30d) / len(trades_30d)
    win_rate_all_time = len(winning_trades_all) / len(trades_all)

    avg_entry_vs_final = _calculate_avg_entry_vs_final(trades)
```

### Copycat Cluster Detection

```python
# In cluster_detector.py
def _calculate_wallet_similarity(addr1, addr2, trades1, trades2):
    # Market overlap (60% weight)
    markets1 = set(t.market_id for t in trades1)
    markets2 = set(t.market_id for t in trades2)
    market_similarity = len(markets1 & markets2) / len(markets1 | markets2)

    # Timing similarity (20% weight)
    timing_similarity = _calculate_timing_similarity(trades1, trades2)

    # Position size similarity (20% weight)
    size_similarity = _calculate_size_similarity(trades1, trades2)

    # Combined score
    overall_similarity = (
        0.6 * market_similarity +
        0.2 * timing_similarity +
        0.2 * size_similarity
    )

    return overall_similarity
```

### Consensus Signal Generation

```python
# In engine.py
def _create_signal_for_market_outcome(positions, basket_wallets, config):
    # Calculate consensus percentage
    participating_addresses = set(p.wallet_address for p in positions)
    participating_wallets = [w for w in basket_wallets if w.wallet_address in participating_addresses]
    consensus_pct = len(participating_wallets) / len(basket_wallets)

    # Check 80% threshold
    if consensus_pct < config.min_consensus_pct:
        return None

    # Price band validation
    avg_entry_price = sum(p.entry_price for p in positions) / len(positions)
    price_band_low = min(p.entry_price for p in positions)
    price_band_high = max(p.entry_price for p in positions)
    price_range = (price_band_high - price_band_low) / avg_entry_price

    if price_range > config.max_price_band_pct:
        return None

    # Signal strength calculation
    signal_strength = min(1.0, consensus_pct * (1 - price_range))

    return BasketSignal(...)
```

## DATA FLOW

```
1. DataAPIClient.fetch_trades()
   └─> WalletBasketBuilder.analyze_wallet()
       └─> WalletPerformance (7d/30d/all-time metrics)

2. ClusterDetector.detect_clusters()
   └─> Similarity matrix (pairwise wallet comparison)
       └─> Clusters (groups of similar wallets)

3. BasketEngine.scan_for_signals()
   └─> Fetch wallet positions
       └─> Group by (market, outcome)
       └─> Calculate consensus
       └─> Validate price band & spread
       └─> BasketSignal

4. BasketSandboxRunner.run_basket_trading()
   └─> Initialize sandbox
       └─> Scan for signals (continuous loop)
       └─> Execute top signals
       └─> Update portfolio
```

## CONFIGURATION REFERENCE

### BasketConfig

```python
@dataclass
class BasketConfig:
    # Wallet filtering
    min_wallet_age_months: int = 6
    min_trades_7d: int = 3
    min_trades_30d: int = 10
    min_win_rate_7d: float = 0.45
    min_win_rate_30d: float = 0.50

    # Bot filtering
    max_hft_score: float = 0.5
    max_arbitrage_score: float = 0.5
    max_pattern_score: float = 0.5
    max_micro_trades_per_day: int = 1000

    # Basket composition
    min_basket_size: int = 10
    max_basket_size: int = 50
    min_consensus_pct: float = 0.80
    max_price_band_pct: float = 0.05
    max_spread_pct: float = 0.10
    min_basket_participation_pct: float = 0.60

    # Ranking weights
    win_rate_7d_weight: float = 0.40
    win_rate_30d_weight: float = 0.35
    win_rate_all_time_weight: float = 0.25
    avg_entry_vs_final_weight: float = 0.30
    consistency_weight: float = 0.20
    volume_weight: float = 0.10

    # Copycat cluster detection
    cluster_similarity_threshold: float = 0.85
    max_cluster_size: int = 3
```

### Topic Enum

```python
class Topic(Enum):
    GEOPOLITICS = "geopolitics"
    POLITICS = "politics"
    ELECTIONS = "elections"
    CRYPTO = "crypto"
    ECONOMICS = "economics"
    FINANCE = "finance"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    CLIMATE = "climate"
    HEALTH = "health"
    OTHER = "other"
```

## ANTI-PATTERNS (THIS MODULE)

❌ **No existing test infrastructure** - Tests file created but needs pytest validation

## INTEGRATION POINTS

### Existing CopyCat Modules

**Integration with trader_identification**:
- Uses TraderIdentificationEngine for performance metrics calculation
- Shares TraderSelectionConfig criteria (win rate, trades count)

**Integration with bot_filtering**:
- Uses BotFilter for automated trader detection
- Shares BotFilterConfig thresholds (HFT, arbitrage, pattern scores)

**Integration with sandbox**:
- Uses SandboxRunner for paper trading simulation
- Uses VirtualOrder for order execution
- Uses VirtualPortfolioManager for position tracking

**Integration with api_clients**:
- Uses DataAPIClient for fetching historical wallet data
- Uses Trade, Position data models from base.py

### New Data Flow

```
User Input (Topic + Wallet Addresses)
         ↓
BasketTradingOrchestrator.initialize_topic_baskets()
         ↓
WalletBasketBuilder.build_basket() for each topic
         ↓
   - Fetch trades from DataAPIClient
   - Calculate performance metrics (7d/30d/all-time)
   - Apply bot filtering (BotFilter.analyze_trades())
         ↓
WalletBasket with eligible wallets
         ↓
ClusterDetector.detect_clusters()
         ↓
   - Calculate pairwise similarity matrix
   - Group wallets by similarity threshold (85%)
   - Select representative wallet per cluster
         ↓
BasketWallet with cluster assignments
         ↓
BasketTradingOrchestrator.rank_all_baskets()
         ↓
BasketRanker.calculate_score() for each wallet
         ↓
   - Multi-factor scoring (win rate + entry price + consistency + volume)
         ↓
Ranked wallet list per topic
         ↓
BasketTradingOrchestrator.scan_for_signals()
         ↓
BasketEngine._scan_basket_for_signals() for each topic
         ↓
   - Fetch current positions (DataAPIClient.get_positions())
   - Group by (market, outcome)
   - Calculate consensus percentage
   - Validate price band (5% threshold)
   - Validate spread (10% threshold)
   - Calculate signal strength
         ↓
BasketSignal list
         ↓
BasketTradingOrchestrator.execute_top_signals()
         ↓
   - Sort by signal strength
   - Select top N signals
   - Execute in sandbox (BasketSandboxRunner)
         ↓
BasketTradeResult list
```

## COMMANDS

```bash
# Run basket trading demo
python demo_basket_trading.py

# Test basket trading module
python -m pytest basket_trading/tests/test_basket_trading.py -v
```

## TESTING

```bash
# Run all basket trading tests
python -m pytest basket_trading/tests/ -v

# Run with coverage
python -m pytest basket_trading/tests/ --cov=basket_trading --cov-report=html
```

## NOTES

**Why Topic-Based Baskets?**
- Individual trader copying is fragile: even best traders drift
- Baskets provide diversification across multiple wallets
- Consensus signals require 80%+ agreement, reducing false positives
- Expertise by topic: geopolitics specialists, crypto experts, etc.

**Why Paper Trading First?**
- Avoid confirmation bias by validating in sandbox before live
- Test strategies without risking real capital
- Measure performance metrics before deployment
- Gain confidence in signal generation logic

**Key Differences from Individual Copying:**
| Aspect | Individual Copying | Basket Trading |
|---------|-------------------|-----------------|
| Focus | Single trader expertise | Group consensus across topic experts |
| Signal Trigger | Trader enters position | 80%+ basket agreement |
| Risk Profile | Concentrated (one trader) | Diversified (10-50 wallets) |
| Noise Reduction | Filter trader bots | Filter bots AND copycat clusters |
| Signal Quality | Trader dependent | High consensus requirement |

**Current Limitations:**
- No automatic topic detection (manual topic assignment required)
- No market metadata integration (topics must be pre-defined)
- No learning/adaptation from basket performance
- Sandbox execution only (no live trading yet)

**Future Enhancements:**
- Automatic topic detection based on wallet trading patterns
- Machine learning for basket performance optimization
- Real-time cluster re-evaluation
- Integration with live trading module
- Multi-market basket arbitrage detection
