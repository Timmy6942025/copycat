"""
Yahoo Finance Data Provider for stock market data.

Provides access to stock prices and market data through yfinance library.
Self-hostable with local data caching.

If yfinance is not available, uses simulated data for offline operation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Try to import yfinance, fall back to simulated data if not available
try:
    import yfinance
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yfinance = None

logger = logging.getLogger(__name__)


@dataclass
class StockMarketData:
    """Stock market data from Yahoo Finance."""
    symbol: str
    name: str
    current_price: float
    previous_close: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    price_change: float
    price_change_percent: float
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_sandbox_format(self) -> Dict[str, Any]:
        """Convert to sandbox-compatible format."""
        return {
            "market_id": f"stock_{self.symbol}",
            "current_price": self.current_price,
            "previous_price": self.previous_close,
            "mid_price": (self.high_price + self.low_price) / 2,
            "best_bid": self.current_price * 0.9999,
            "best_ask": self.current_price * 1.0001,
            "spread": (self.high_price - self.low_price) / self.current_price,
            "volume_24h": self.volume,
            "liquidity": self.volume * self.current_price * 0.01,  # 1% of daily volume
            "volatility": abs(self.price_change_percent) / 100,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class StockQuote:
    """Real-time stock quote."""
    symbol: str
    price: float
    change: float
    change_percent: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class YahooFinanceProvider:
    """
    Yahoo Finance data provider using yfinance library.
    
    Free, no API key required. Uses yfinance which scrapes Yahoo Finance data.
    - Rate limit: Respects Yahoo Finance rate limits
    - No API key needed
    - Local caching supported
    
    Note: This is a wrapper around yfinance library.
    """
    
    DEFAULT_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "JNJ",
    ]
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        cache_ttl: int = 60,  # Cache TTL in seconds
    ):
        """Initialize Yahoo Finance provider."""
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cache entry is still valid."""
        if symbol not in self._cache_timestamps:
            return False
        elapsed = (datetime.utcnow() - self._cache_timestamps[symbol]).total_seconds()
        return elapsed < self.cache_ttl
    
    async def get_market_data(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> Optional[StockMarketData]:
        """
        Get market data for a single stock.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            force_refresh: Force refresh from API even if cached
            
        Returns:
            StockMarketData object or None if not found
        """
        # Check cache
        if not force_refresh and self._is_cache_valid(symbol):
            cached = self._cache.get(symbol)
            if cached:
                return StockMarketData(**cached)
        
        # If yfinance is not available, use simulated data
        if not YFINANCE_AVAILABLE or yfinance is None:
            return self._get_simulated_data(symbol)
        
        try:
            # yfinance is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yfinance.Ticker(symbol))
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            if not info:
                return None
            
            # Extract data
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or current_price
            open_price = info.get("open") or info.get("regularMarketOpen") or current_price
            high_price = info.get("dayHigh") or info.get("regularMarketDayHigh") or current_price
            low_price = info.get("dayLow") or info.get("regularMarketDayLow") or current_price
            volume = info.get("volume") or info.get("regularMarketVolume") or 0
            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE")
            dividend_yield = info.get("dividendYield")
            
            price_change = info.get("regularMarketChange") or 0
            price_change_percent = info.get("regularMarketChangePercent") or 0
            
            data = StockMarketData(
                symbol=symbol,
                name=info.get("shortName") or info.get("longName") or symbol,
                current_price=current_price,
                previous_close=previous_close,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                volume=volume,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                dividend_yield=dividend_yield,
                price_change=price_change,
                price_change_percent=price_change_percent,
                last_updated=datetime.utcnow(),
            )
            
            # Cache the data
            self._cache[symbol] = {
                "symbol": data.symbol,
                "name": data.name,
                "current_price": data.current_price,
                "previous_close": data.previous_close,
                "open_price": data.open_price,
                "high_price": data.high_price,
                "low_price": data.low_price,
                "volume": data.volume,
                "market_cap": data.market_cap,
                "pe_ratio": data.pe_ratio,
                "dividend_yield": data.dividend_yield,
                "price_change": data.price_change,
                "price_change_percent": data.price_change_percent,
                "last_updated": data.last_updated.isoformat(),
            }
            self._cache_timestamps[symbol] = datetime.utcnow()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return None
    
    async def get_multiple_market_data(
        self,
        symbols: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> List[StockMarketData]:
        """
        Get market data for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            force_refresh: Force refresh from API
            
        Returns:
            List of StockMarketData objects
        """
        symbols = symbols or self.symbols
        
        if not YFINANCE_AVAILABLE or yfinance is None:
            # Return simulated data for all symbols
            return [self._get_simulated_data(symbol) for symbol in symbols]
        
        tasks = [self.get_market_data(symbol, force_refresh) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        markets = []
        for result in results:
            if isinstance(result, StockMarketData):
                markets.append(result)
        
        return markets
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            List of historical data points
        """
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yfinance.Ticker(symbol))
            history = await loop.run_in_executor(
                None,
                lambda: ticker.history(period=period, interval=interval)
            )
            
            if history is None or history.empty:
                return []
            
            data = []
            for index, row in history.iterrows():
                data.append({
                    "timestamp": index.isoformat(),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"],
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    async def search_symbol(self, query: str) -> List[Dict[str, str]]:
        """
        Search for stock symbols.
        
        Args:
            query: Search query
            
        Returns:
            List of matching symbols with name and type
        """
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yfinance.Ticker(query))
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            if info:
                return [{
                    "symbol": query,
                    "name": info.get("shortName", query),
                    "type": info.get("quoteType", "UNKNOWN"),
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching for symbol {query}: {e}")
            return []
    
    def _get_simulated_data(self, symbol: str) -> StockMarketData:
        """Get simulated stock data when yfinance is not available."""
        # Generate consistent simulated data based on symbol hash
        hash_value = abs(hash(symbol)) % 10000
        base_price = 100 + hash_value / 100
        price_change = (hash_value % 20 - 10) / 100  # -10% to +10%
        
        return StockMarketData(
            symbol=symbol,
            name=f"{symbol} (Simulated)",
            current_price=base_price * (1 + price_change),
            previous_close=base_price,
            open_price=base_price * (1 + price_change * 0.5),
            high_price=base_price * (1 + price_change * 1.2),
            low_price=base_price * (1 + price_change * 0.8),
            volume=1000000 + hash_value * 10000,
            market_cap=base_price * 1000000000,
            pe_ratio=15 + hash_value % 20,
            dividend_yield=0.01 + (hash_value % 50) / 10000,
            price_change=base_price * price_change,
            price_change_percent=price_change * 100,
            last_updated=datetime.utcnow(),
        )
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear the cache."""
        if symbol:
            self._cache.pop(symbol, None)
            self._cache_timestamps.pop(symbol, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    def get_sandbox_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data in sandbox-compatible format."""
        return {
            "market_id": f"stock_{symbol}",
            "current_price": 100.0,  # Placeholder
            "previous_price": 99.5,
            "mid_price": 100.0,
            "best_bid": 99.99,
            "best_ask": 100.01,
            "spread": 0.002,
            "volume_24h": 50000000,
            "liquidity": 5000000,
            "volatility": 0.02,
            "last_updated": datetime.utcnow().isoformat(),
        }


async def test_yahoo_finance_provider():
    """Test the Yahoo Finance provider."""
    provider = YahooFinanceProvider(symbols=["AAPL", "MSFT", "GOOGL"])
    
    try:
        # Get single market
        market = await provider.get_market_data("AAPL")
        if market:
            print(f"AAPL: ${market.current_price:.2f} ({market.price_change_percent:+.2f}%)")
        
        # Get multiple markets
        markets = await provider.get_multiple_market_data()
        print(f"\nFetched {len(markets)} markets:")
        for m in markets:
            print(f"  {m.symbol}: ${m.current_price:.2f}")
        
        # Get historical data
        history = await provider.get_historical_data("AAPL", period="5d")
        print(f"\nHistorical data points: {len(history)}")
        
    finally:
        pass


if __name__ == "__main__":
    import sys
    asyncio.run(test_yahoo_finance_provider())
