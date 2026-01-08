"""
Polymarket Data Provider for prediction market data.

Provides access to Polymarket prediction market prices and data.
Uses existing API clients from api_clients directory.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api_clients.polymarket import PolymarketAPIClient
from api_clients.base import MarketData

logger = logging.getLogger(__name__)


@dataclass
class PredictionMarketData:
    """Prediction market data from Polymarket."""
    market_id: str
    question: str
    outcome: str  # "YES" or "NO"
    current_price: float
    previous_price: float
    volume_24h: float
    liquidity: float
    volatility: float
    best_bid: float
    best_ask: float
    spread: float
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_sandbox_format(self) -> Dict[str, Any]:
        """Convert to sandbox-compatible format."""
        return {
            "market_id": self.market_id,
            "current_price": self.current_price,
            "previous_price": self.previous_price,
            "mid_price": (self.best_bid + self.best_ask) / 2,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "volatility": self.volatility,
            "last_updated": self.last_updated.isoformat(),
        }


class PolymarketDataProvider:
    """
    Polymarket data provider for prediction market data.
    
    Uses the existing Polymarket API client from api_clients.
    - Free access to market data
    - Rate limits apply
    - No authentication required for basic market data
    
    Note: For full API access, API key may be required.
    """
    
    def __init__(
        self,
        cache_ttl: int = 30,  # Cache TTL in seconds (markets can move fast)
    ):
        """Initialize Polymarket data provider."""
        self.client = PolymarketAPIClient()
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    async def close(self):
        """Close the API client."""
        await self.client.close()
    
    def _is_cache_valid(self, market_id: str) -> bool:
        """Check if cache entry is still valid."""
        if market_id not in self._cache_timestamps:
            return False
        elapsed = (datetime.utcnow() - self._cache_timestamps[market_id]).total_seconds()
        return elapsed < self.cache_ttl
    
    async def get_market_data(
        self,
        market_id: str,
        force_refresh: bool = False,
    ) -> Optional[PredictionMarketData]:
        """
        Get market data for a specific Polymarket.
        
        Args:
            market_id: Market ID (token_id)
            force_refresh: Force refresh from API
            
        Returns:
            PredictionMarketData object or None if not found
        """
        # Check cache
        if not force_refresh and self._is_cache_valid(market_id):
            cached = self._cache.get(market_id)
            if cached:
                return PredictionMarketData(**cached)
        
        try:
            # Get market data from API
            market_data = await self.client.get_market_data(market_id)
            
            if not market_data:
                logger.warning(f"No market data found for {market_id}")
                return None
            
            # Convert to our format
            prediction_data = PredictionMarketData(
                market_id=market_id,
                question=market_data.market_id if hasattr(market_data, 'market_id') else market_id,
                outcome="YES",  # Default outcome
                current_price=market_data.current_price,
                previous_price=market_data.previous_price,
                volume_24h=market_data.volume_24h,
                liquidity=market_data.liquidity,
                volatility=market_data.volatility,
                best_bid=market_data.best_bid or market_data.current_price * 0.99,
                best_ask=market_data.best_ask or market_data.current_price * 1.01,
                spread=market_data.spread or 0.02,
                last_updated=datetime.utcnow(),
            )
            
            # Cache the data
            self._cache[market_id] = {
                "market_id": prediction_data.market_id,
                "question": prediction_data.question,
                "outcome": prediction_data.outcome,
                "current_price": prediction_data.current_price,
                "previous_price": prediction_data.previous_price,
                "volume_24h": prediction_data.volume_24h,
                "liquidity": prediction_data.liquidity,
                "volatility": prediction_data.volatility,
                "best_bid": prediction_data.best_bid,
                "best_ask": prediction_data.best_ask,
                "spread": prediction_data.spread,
                "last_updated": prediction_data.last_updated.isoformat(),
            }
            self._cache_timestamps[market_id] = datetime.utcnow()
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error fetching Polymarket data for {market_id}: {e}")
            return None
    
    async def get_markets(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        force_refresh: bool = False,
    ) -> List[PredictionMarketData]:
        """
        Get list of Polymarkets.
        
        Args:
            category: Market category filter
            limit: Maximum number of markets
            force_refresh: Force refresh from API
            
        Returns:
            List of PredictionMarketData objects
        """
        try:
            markets = await self.client.get_markets(
                category=category,
                limit=limit,
            )
            
            prediction_markets = []
            for market in markets:
                try:
                    # Extract token_id for market_id
                    token_ids = market.get("clobTokenIds", "[]")
                    if token_ids:
                        import json
                        token_list = json.loads(token_ids)
                        market_id = token_list[0] if token_list else market.get("id", "")
                    else:
                        market_id = market.get("id", "")
                    
                    if not market_id:
                        continue
                    
                    # Get current price from outcome prices
                    outcome_prices = market.get("outcomePrices", {})
                    yes_price = outcome_prices.get("yes", outcome_prices.get("YES", 0.5))
                    no_price = outcome_prices.get("no", outcome_prices.get("NO", 0.5))
                    
                    prediction_data = PredictionMarketData(
                        market_id=market_id,
                        question=market.get("question", market.get("slug", "")),
                        outcome="YES",
                        current_price=yes_price,
                        previous_price=yes_price,  # Use current as previous if not available
                        volume_24h=market.get("volume24h", market.get("volume", 0)),
                        liquidity=market.get("liquidity", 0),
                        volatility=0.02,  # Default volatility
                        best_bid=yes_price * 0.99,
                        best_ask=yes_price * 1.01,
                        spread=abs(yes_price - no_price),
                        last_updated=datetime.utcnow(),
                    )
                    prediction_markets.append(prediction_data)
                    
                except Exception as e:
                    logger.error(f"Error parsing market: {e}")
                    continue
            
            return prediction_markets
            
        except Exception as e:
            logger.error(f"Error fetching Polymarkets: {e}")
            return []
    
    async def get_orderbook(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get order book for a Polymarket."""
        try:
            orderbook = await self.client.get_orderbook(market_id)
            if orderbook:
                return {
                    "bids": [{"price": b.price, "size": b.size} for b in orderbook.bids],
                    "asks": [{"price": a.price, "size": a.size} for a in orderbook.asks],
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching orderbook for {market_id}: {e}")
            return None
    
    async def get_recent_trades(
        self,
        market_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent trades."""
        try:
            trades = await self.client.get_trades(market_id=market_id, limit=limit)
            return [
                {
                    "trade_id": t.trade_id,
                    "market_id": t.market_id,
                    "price": t.price,
                    "quantity": t.quantity,
                    "side": t.side.value if hasattr(t.side, 'value') else t.side,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                }
                for t in trades
            ]
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []
    
    def clear_cache(self, market_id: Optional[str] = None):
        """Clear the cache."""
        if market_id:
            self._cache.pop(market_id, None)
            self._cache_timestamps.pop(market_id, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    def get_sandbox_market_data(self, market_id: str) -> Dict[str, Any]:
        """Get market data in sandbox-compatible format."""
        return {
            "market_id": market_id,
            "current_price": 0.5,
            "previous_price": 0.5,
            "mid_price": 0.5,
            "best_bid": 0.49,
            "best_ask": 0.51,
            "spread": 0.02,
            "volume_24h": 10000,
            "liquidity": 5000,
            "volatility": 0.02,
            "last_updated": datetime.utcnow().isoformat(),
        }


async def test_polymarket_provider():
    """Test the Polymarket data provider."""
    provider = PolymarketDataProvider()
    
    try:
        # Get markets
        markets = await provider.get_markets(limit=10)
        print(f"Fetched {len(markets)} markets:")
        for market in markets[:5]:
            print(f"  {market.question[:50]}: ${market.current_price:.2f}")
        
        # Get specific market data
        if markets:
            market_data = await provider.get_market_data(markets[0].market_id)
            if market_data:
                print(f"\nMarket: {market_data.question}")
                print(f"Price: ${market_data.current_price:.2f}")
                print(f"Volume: ${market_data.volume_24h:,.0f}")
        
    finally:
        await provider.close()


if __name__ == "__main__":
    import sys
    asyncio.run(test_polymarket_provider())
