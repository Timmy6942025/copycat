# PROJECT KNOWLEDGE BASE

**Generated:** Fri Jan 09 2026
**Commit:** Current working tree
**Branch:** main

## OVERVIEW

Polymarket API client layer with facade pattern combining CLOB (trading), Gamma (market data), and Data API (user data).

## STRUCTURE

```
polymarket/
├── __init__.py        # PolymarketAPIClient facade (261 lines)
├── clob.py           # CLOB API - order book, trading (459 lines)
├── gamma.py          # Gamma API - market metadata, prices (209 lines)
├── data_api.py       # Data API - user data, leaderboard (419 lines)
├── tests/            # Unit tests
└── base.py (parent)  # Abstract MarketAPIClient + data models
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Unified client | `__init__.py` | Facade pattern, delegates to 3 sub-APIs |
| Order book/trades | `clob.py` | Trading endpoints, real-time data |
| Market metadata | `gamma.py` | Markets, profiles, comments |
| User positions/P&L | `data_api.py` | Positions, activity, leaderboard |
| Fallback logic | `__init__.py:84-118` | get_market_data() tries CLOB→Gamma |
| Static parsing | `clob.py:110-156` | parse_market_data_from_gamma_response() |

## CONVENTIONS

**Facade pattern**: PolymarketAPIClient (`__init__.py`) combines 3 APIs (gamma, clob, data) with shared session management
**Session reuse**: Single `aiohttp.ClientSession` across all sub-APIs, created lazily via `_get_session()`
**Fallback strategy**: `get_market_data()` attempts CLOB first, falls back to Gamma API with parsing
**Rate limiting**: Each sub-API has its own `_rate_limit_delay` (CLOB: 0.011s, Gamma: 0.25s, Data: 0.1s)
**Optional auth**: API key from env var `POLYMARKET_API_KEY` or constructor param, free tier works without
**Static methods**: `parse_market_data_from_gamma_response()` for Gamma response parsing in CLOB client
**Error handling**: Methods return empty lists/None/False on errors instead of raising (except in base `_request()`)

## ANTI-PATTERNS (THIS MODULE)

❌ **No unified rate limiting**: Each sub-API implements its own delay separately (no centralized limiter)
❌ **Duplicate data models**: OrderSide, OrderType, OrderStatus defined in base.py AND sandbox/models/__init__.py
❌ **Inconsistent error handling**: Some methods swallow exceptions (return None), others propagate via `_request()`
❌ **Tight coupling**: Facade directly instantiates sub-APIs, not dependency-injectable
❌ **No request retry logic**: Single attempt per request, no exponential backoff on failures
❌ **Parsing duplication**: Gamma response parsing scattered across CLOB static methods and Data API
