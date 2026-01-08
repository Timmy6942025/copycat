# Sandbox Data Providers - Implementation Summary

## Overview

Implemented real-time market data integration for the sandbox simulation module using **free and self-hostable APIs**. The implementation allows the trading bot to use accurate real-world data for paper trading simulations.

## Free APIs Implemented

### 1. CoinGecko API (Cryptocurrency)
- **URL**: `https://api.coingecko.com/api/v3`
- **Cost**: Free tier (10-50 calls/minute)
- **Authentication**: None required
- **Data**: Real-time crypto prices, market cap, volume, historical data
- **Coverage**: 10,000+ cryptocurrencies

### 2. Yahoo Finance (Stocks)
- **Library**: `yfinance`
- **Cost**: Free
- **Authentication**: None required
- **Data**: Stock prices, fundamentals, historical data
- **Coverage**: Global stock markets

### 3. Polymarket API (Prediction Markets)
- **Source**: Existing `api_clients.polymarket` module
- **Cost**: Free access to market data
- **Authentication**: API key optional
- **Data**: Prediction market prices, order books, trades
- **Coverage**: Political, sports, and event prediction markets

## Architecture

```
sandbox/data_providers/
├── __init__.py              # Exports all providers
├── coingecko.py             # CoinGecko API provider
├── yahoofinance.py          # Yahoo Finance provider
├── polymarket.py            # Polymarket data provider
├── unified.py               # Unified provider (multi-source)
└── feed.py                  # Real-time data feed handler
```

### Key Components

1. **CoinGeckoProvider**: Async client for CoinGecko REST API
2. **YahooFinanceProvider**: Async wrapper for yfinance (with simulated fallback)
3. **PolymarketDataProvider**: Uses existing Polymarket API client
4. **UnifiedDataProvider**: Single interface for all data sources
5. **RealtimeDataFeed**: Continuous polling-based data updates

## Usage

### Basic Usage with Real-Time Data

```python
from sandbox.runner import SandboxRunner
from sandbox.config import SandboxConfig

# Initialize with real-time data
runner = SandboxRunner()
runner.enable_realtime_data(
    enable_crypto=True,
    enable_stocks=True,
    enable_prediction=True,
)

# Start data feed
await runner.start_data_feed(["bitcoin", "AAPL", "ethereum"])

# Execute trades with real data
order = VirtualOrder(...)
result = await runner.execute_order(order)
```

### Fallback Mode (Simulated Data)

```python
# Works without any API dependencies
runner = SandboxRunner()

def get_market_data(market_id):
    return {
        "current_price": 0.5,
        "volatility": 0.02,
    }

runner.set_market_data_callback(get_market_data)
```

### Direct Provider Usage

```python
from sandbox.data_providers import CoinGeckoProvider, UnifiedDataProvider

# Direct CoinGecko usage
provider = CoinGeckoProvider(symbols=["bitcoin", "ethereum"])
markets = await provider.get_markets()

# Unified provider
provider = UnifiedDataProvider()
btc_data = await provider.get_market_data("bitcoin")
```

## Configuration

```python
from sandbox.data_providers import DataProviderConfig, FeedConfig

# Data provider config
data_config = DataProviderConfig(
    enable_crypto=True,
    enable_stocks=True,
    enable_prediction=True,
    cache_ttl_crypto=60,      # Cache TTL in seconds
    cache_ttl_stocks=60,
    cache_ttl_prediction=30,  # Markets move faster
)

# Real-time feed config
feed_config = FeedConfig(
    update_interval=1.0,      # Seconds between updates
    max_history=1000,         # Price history points
    auto_reconnect=True,
)
```

## Dependencies

```txt
# requirements.txt
aiohttp >= 3.8.0            # Async HTTP client
yfinance >= 0.2.0           # Yahoo Finance data
```

Optional:
- `pip install yfinance aiohttp` for live data
- Without dependencies: Uses simulated data for offline operation

## Running Tests

```bash
# Run integration tests
python3 sandbox/tests/test_data_providers.py

# Run with fallback mode (no dependencies)
python3 sandbox/runner.py --mode fallback

# Run with real-time data
python3 sandbox/runner.py --mode realtime
```

## Self-Hostable Alternatives

For production use, consider these self-hostable solutions:

1. **TARDIS Machine** (https://github.com/tardis-dev/tardis-machine)
   - Self-hosted crypto data server
   - Historical and real-time data
   - WebSocket API

2. **Hummingbot** (https://hummingbot.org/)
   - Open source market making framework
   - 100+ exchange connectors
   - Real-time data collection

3. **Cryptofeed** (https://github.com/bmoscon/cryptofeed)
   - Lightweight Python library
   - WebSocket data feeds
   - Backtesting support

## Features

- ✓ Real-time cryptocurrency prices (CoinGecko)
- ✓ Real-time stock prices (Yahoo Finance)
- ✓ Prediction market data (Polymarket)
- ✓ Automatic failover to simulated data
- ✓ Configurable caching
- ✓ Real-time data feed with callbacks
- ✓ Price history tracking
- ✓ Unified interface for all data sources

## API Reference

### UnifiedDataProvider

```python
data = await provider.get_market_data(market_id)  # Single market
all_data = await provider.get_all_market_data()   # All markets
prices = provider.get_sandbox_market_data(id)     # Sync fallback
```

### RealtimeDataFeed

```python
feed = RealtimeDataFeed()
await feed.start(market_ids, provider_callback)
prices = feed.get_all_latest_prices()
await feed.stop()
```

## Notes

- All providers include fallback implementations for offline operation
- Rate limiting is handled automatically (CoinGecko: ~1.5s between requests)
- Caching reduces API calls and improves performance
- The sandbox runner seamlessly switches between real and simulated data
