"""
CoinGecko API Provider for real-time cryptocurrency data.

Provides free access to cryptocurrency prices, market data, and trading volume.
No API key required for basic endpoints. Rate limited to ~10-50 calls/minute.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CryptoMarketData:
    """Cryptocurrency market data from CoinGecko."""
    symbol: str
    name: str
    current_price: float
    market_cap: float
    volume_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    high_24h: float
    low_24h: float
    circulating_supply: float
    total_supply: float
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_sandbox_format(self) -> Dict[str, Any]:
        """Convert to sandbox-compatible format."""
        return {
            "market_id": f"crypto_{self.symbol}",
            "current_price": self.current_price,
            "previous_price": self.current_price / (1 + self.price_change_percentage_24h / 100) if self.price_change_percentage_24h != 0 else self.current_price,
            "mid_price": (self.high_24h + self.low_24h) / 2,
            "best_bid": self.current_price * 0.9999,  # Slightly below current
            "best_ask": self.current_price * 1.0001,  # Slightly above current
            "spread": (self.high_24h - self.low_24h) / self.current_price,
            "volume_24h": self.volume_24h,
            "liquidity": self.volume_24h * 0.1,  # Estimated liquidity
            "volatility": abs(self.price_change_percentage_24h) / 100,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class CryptoOrderBook:
    """Order book data for a cryptocurrency."""
    symbol: str
    bids: List[Dict[str, float]]  # [{"price": float, "size": float}]
    asks: List[Dict[str, float]]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CoinGeckoProvider:
    """
    CoinGecko API provider for cryptocurrency market data.
    
    Free API with no API key required for basic endpoints.
    - Rate limit: ~10-50 calls/minute (varies by endpoint)
    - No API key needed for basic market data
    - Pro API available for higher limits
    
    Endpoints used:
    - /coins/markets: List of coins with market data
    - /coins/{id}: Detailed coin data
    - /coins/{id}/market_chart: Historical price data
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Popular cryptocurrencies to track
    DEFAULT_SYMBOLS = [
        "bitcoin", "ethereum", "solana", "cardano", "polkadot",
        "avalanche-2", "polygon", "chainlink", "uniswap", "maker",
    ]
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        vs_currency: str = "usd",
        rate_limit_delay: float = 1.5,  # seconds between requests
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """Initialize CoinGecko provider."""
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.vs_currency = vs_currency
        self.rate_limit_delay = rate_limit_delay
        self._session = session
        self._last_request_time = 0.0
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"}
            )
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _rate_limited_request(self, url: str) -> Optional[Dict[str, Any]]:
        """Make a rate-limited HTTP request."""
        # Rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self._last_request_time = asyncio.get_event_loop().time()
        
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 429:
                    # Rate limited - wait and retry once
                    logger.warning("CoinGecko rate limit hit, waiting 60s...")
                    await asyncio.sleep(60)
                    self._last_request_time = 0  # Reset rate limit
                    return await self._rate_limited_request(url)
                
                if response.status != 200:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return None
                
                return await response.json()
                
        except asyncio.TimeoutError:
            logger.error("CoinGecko request timed out")
            return None
        except Exception as e:
            logger.error(f"CoinGecko request failed: {e}")
            return None
    
    async def get_markets(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 100,
        sparkline: bool = False,
    ) -> List[CryptoMarketData]:
        """
        Get market data for multiple cryptocurrencies.
        
        Args:
            symbols: List of coin IDs (e.g., ['bitcoin', 'ethereum'])
            limit: Maximum number of results
            sparkline: Include 7-day sparkline data
            
        Returns:
            List of CryptoMarketData objects
        """
        symbols = symbols or self.symbols
        # Filter to existing symbols
        ids_param = ",".join(symbols[:limit])
        url = f"{self.BASE_URL}/coins/markets?vs_currency={self.vs_currency}&ids={ids_param}&order=market_cap_desc&per_page={limit}&page=1&sparkline={str(sparkline).lower()}&price_change_percentage=24h"
        
        data = await self._rate_limited_request(url)
        if not data:
            return []
        
        markets = []
        for coin in data:
            try:
                market = CryptoMarketData(
                    symbol=coin.get("id", ""),
                    name=coin.get("name", ""),
                    current_price=coin.get("current_price", 0.0),
                    market_cap=coin.get("market_cap", 0.0),
                    volume_24h=coin.get("total_volume", 0.0),
                    price_change_24h=coin.get("price_change_24h", 0.0),
                    price_change_percentage_24h=coin.get("price_change_percentage_24h", 0.0),
                    high_24h=coin.get("high_24h", 0.0),
                    low_24h=coin.get("low_24h", 0.0),
                    circulating_supply=coin.get("circulating_supply", 0.0),
                    total_supply=coin.get("total_supply", 0.0),
                    last_updated=datetime.utcnow(),
                )
                markets.append(market)
            except Exception as e:
                logger.error(f"Error parsing CoinGecko market data: {e}")
                continue
        
        return markets
    
    async def get_market_data(self, symbol: str) -> Optional[CryptoMarketData]:
        """Get market data for a single cryptocurrency."""
        markets = await self.get_markets(symbols=[symbol], limit=1)
        return markets[0] if markets else None
    
    async def get_price_history(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get price history for a cryptocurrency.
        
        Args:
            symbol: Coin ID (e.g., 'bitcoin')
            days: Number of days of history
            
        Returns:
            List of price data points [{timestamp, price, market_cap}]
        """
        url = f"{self.BASE_URL}/coins/{symbol}/market_chart?vs_currency={self.vs_currency}&days={days}"
        
        data = await self._rate_limited_request(url)
        if not data or "prices" not in data:
            return []
        
        history = []
        for price_data in data["prices"]:
            try:
                history.append({
                    "timestamp": datetime.fromtimestamp(price_data[0] / 1000),
                    "price": price_data[1],
                    "market_cap": data["market_caps"][len(history)][1] if len(data["market_caps"]) > len(history) else 0,
                    "volume": data["total_volumes"][len(history)][1] if len(data["total_volumes"]) > len(history) else 0,
                })
            except Exception:
                continue
        
        return history
    
    async def get_trending_coins(self) -> List[Dict[str, Any]]:
        """Get trending cryptocurrencies."""
        url = f"{self.BASE_URL}/search/trending"
        
        data = await self._rate_limited_request(url)
        if not data or "coins" not in data:
            return []
        
        return [
            {
                "id": coin.get("item", {}).get("id"),
                "name": coin.get("item", {}).get("name"),
                "symbol": coin.get("item", {}).get("symbol"),
                "market_cap_rank": coin.get("item", {}).get("market_cap_rank"),
                "price": coin.get("item", {}).get("data", {}).get("price"),
            }
            for coin in data["coins"][:10]  # Top 10 trending
        ]
    
    async def get_global_data(self) -> Dict[str, Any]:
        """Get global cryptocurrency market data."""
        url = f"{self.BASE_URL}/global"
        
        data = await self._rate_limited_request(url)
        if not data or "data" not in data:
            return {}
        
        return {
            "total_market_cap": data["data"].get("total_market_cap", {}).get(self.vs_currency, 0),
            "total_volume": data["data"].get("total_volume", {}).get(self.vs_currency, 0),
            "btc_dominance": data["data"].get("market_cap_percentage", {}).get("btc", 0),
            "eth_dominance": data["data"].get("market_cap_percentage", {}).get("eth", 0),
            "market_cap_change_24h": data["data"].get("market_cap_change_percentage_24h_usd", 0),
            "active_cryptocurrencies": data["data"].get("active_cryptocurrencies", 0),
        }
    
    def get_sandbox_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data in sandbox-compatible format (synchronous, cached data)."""
        # This is a placeholder - real implementation would use cached async data
        return {
            "market_id": f"crypto_{symbol}",
            "current_price": 0.5,  # Placeholder
            "previous_price": 0.5,
            "mid_price": 0.5,
            "best_bid": 0.499,
            "best_ask": 0.501,
            "spread": 0.002,
            "volume_24h": 1000000,
            "liquidity": 100000,
            "volatility": 0.02,
            "last_updated": datetime.utcnow().isoformat(),
        }


async def test_coingecko_provider():
    """Test the CoinGecko provider."""
    provider = CoinGeckoProvider(symbols=["bitcoin", "ethereum"])
    
    try:
        # Get markets
        markets = await provider.get_markets(limit=10)
        print(f"Fetched {len(markets)} markets:")
        for market in markets[:3]:
            print(f"  {market.name}: ${market.current_price:,.2f} ({market.price_change_percentage_24h:+.2f}%)")
        
        # Get global data
        global_data = await provider.get_global_data()
        print(f"\nGlobal Market Cap: ${global_data.get('total_market_cap', 0):,.0f}")
        print(f"BTC Dominance: {global_data.get('btc_dominance', 0):.1f}%")
        
    finally:
        await provider.close()


if __name__ == "__main__":
    import sys
    asyncio.run(test_coingecko_provider())
