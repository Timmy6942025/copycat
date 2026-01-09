# Live Market Data Setup for CopyCat Sandbox

## Overview

The CopyCat sandbox can now use **real-time market data** from Polymarket APIs. This document explains how to set up and use live data.

## API Requirements

### Polymarket (Recommended)

| Feature | API Key Required? | Rate Limit |
|---------|------------------|------------|
| Get Markets | ❌ No | 4000 req/10s |
| Get Market Data | ❌ No | 9000 req/10s |
| Get Orderbook | ❌ No | 9000 req/10s |
| Get Trades | ❌ No | 9000 req/10s |
| Create Orders | ✅ Yes (for live trading) | 100 req/s |

**Base URLs:**
- Gamma API: `https://gamma.polymarket.com`
- CLOB API: `https://clob.polymarket.com`

## Quick Start

### 1. Copy Environment File

```bash
cp .env.example .env
```

### 2. Run Sandbox with Live Data (No API Key Needed)

```bash
# Start orchestrator in sandbox mode with live market data
python -m orchestrator.cli start --mode sandbox

# Or run the demo
python -m orchestrator.cli demo
```

### 3. Test API Connectivity

```bash
# Test Polymarket API
python -c "
import asyncio
from api_clients import PolymarketAPIClient

async def test():
    client = PolymarketAPIClient()
    markets = await client.get_markets(limit=5)
    print(f'Found {len(markets)} markets')
    for m in markets[:3]:
        print(f'  - {m.get(\"question\", \"Unknown\")}')

asyncio.run(test())
"
```

## How Live Data Works

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Orchestrator   │────►│  API Client      │────►│  Polymarket    │
│                 │     │  (CLOB/Gamma)    │     │  Live API      │
└────────┬────────┘     └──────────────────┘     └────────────────┘
         │
         │ Sets callback
         ▼
┌─────────────────┐     ┌──────────────────┐
│  SandboxRunner  │────►│  VirtualOrder    │
│                 │     │  Executor        │
└─────────────────┘     └──────────────────┘
```

1. **Orchestrator** initializes and sets up a callback function
2. **Callback** fetches real market data from Polymarket CLOB API
3. **SandboxRunner** uses real prices for order simulation
4. **VirtualOrderExecutor** calculates realistic slippage/fees using real orderbook data

## Configuration Options

### Sandbox Configuration

```python
from orchestrator.config import OrchestratorConfig, SandboxConfigOrchestrator

config = OrchestratorConfig(
    mode=TradingMode.SANDBOX,
    platform=MarketPlatform.POLYMARKET,
    sandbox=SandboxConfigOrchestrator(
        initial_balance=10000.0,
        simulate_slippage=True,      # Use real orderbook for slippage
        simulate_fees=True,           # Apply Polymarket fees (~2%)
        simulate_fill_probability=True,
    )
)
```

### Market Data Refresh

```python
# Configure how often to refresh market data
config = OrchestratorConfig(
    market_data_refresh_interval_seconds=60.0,  # Refresh every 60s
    trader_data_refresh_interval_seconds=300.0, # Refresh trader data every 5m
)
```

## Troubleshooting

### "No market data available"

This happens when the API is unreachable. The sandbox falls back to synthetic prices (0.5).

**Solutions:**
1. Check your internet connection
2. Verify Polymarket API is operational: `curl https://clob.polymarket.com/price/example`

### Rate Limiting Errors

If you see 429 errors, you're hitting rate limits. The default config has built-in rate limiting.

**Reduce request frequency:**
```python
config = OrchestratorConfig(
    market_data_refresh_interval_seconds=120.0,  # Slower refresh
    trader_data_refresh_interval_seconds=600.0,  # Much slower
)
```

### API Key for Live Trading

For **live trading** (not sandbox), you'll need an API key:

1. Go to [Polymarket API Keys](https://polymarket.com/api-key)
2. Create an API key with trading permissions
3. Add to `.env`:
   ```bash
   POLYMARKET_API_KEY=your_api_key_here
   ```

## API Endpoints Used

### Sandbox (Live Data Mode)

| Endpoint | Purpose |
|----------|---------|
| `GET /price/{market_id}` | Current price, bid/ask |
| `GET /book/{market_id}` | Orderbook for slippage |
| `GET /trades` | Recent trades for discovery |
| `GET /markets` | List all markets |

### Future: Live Trading

| Endpoint | Purpose |
|----------|---------|
| `POST /order` | Create order |
| `DELETE /order/{id}` | Cancel order |
| `GET /balance/{address}` | Get wallet balance |

## Performance Considerations

- **Memory**: Orderbooks are cached per market
- **Network**: ~1 API call per order execution
- **Latency**: Real-time prices (subject to API response time)

## Self-Hosting Option

For high-frequency usage, consider running your own API proxy:

```bash
# Run a simple cache proxy
pip install aiohttp-caching

# Or use cloudflare worker for caching
```

This reduces load on Polymarket API and improves response times.

## Summary

| Mode | API Key | Data Source | Use Case |
|------|---------|-------------|----------|
| Sandbox (Default) | ❌ No | Real-time from Polymarket | Paper trading with live prices |
| Sandbox (Simulated) | ❌ No | Synthetic prices (0.5) | Testing without network |
| Live Trading | ✅ Optional | Real-time + order execution | Production |

**Start with sandbox mode - no API key needed, real market data included!**
