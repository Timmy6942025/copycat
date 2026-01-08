"""
Unified Data Provider - Integrates multiple data sources.

Provides a unified interface to access market data from various providers
including cryptocurrency, stocks, and prediction markets.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from .coingecko import CoinGeckoProvider, CryptoMarketData
from .yahoofinance import YahooFinanceProvider, StockMarketData
from .polymarket import PolymarketDataProvider, PredictionMarketData

logger = logging.getLogger(__name__)


class MarketType(Enum):
    """Types of markets available."""
    CRYPTO = "crypto"
    STOCK = "stock"
    PREDICTION = "prediction"
    FOREX = "forex"
    COMMODITY = "commodity"


@dataclass
class DataProviderConfig:
    """Configuration for unified data provider."""
    # Provider enable/disable
    enable_crypto: bool = True
    enable_stocks: bool = True
    enable_prediction: bool = True
    
    # Cache settings
    cache_ttl_crypto: int = 60
    cache_ttl_stocks: int = 60
    cache_ttl_prediction: int = 30
    
    # Symbols to track
    crypto_symbols: List[str] = field(default_factory=lambda: [
        "bitcoin", "ethereum", "solana", "cardano", "polkadot",
    ])
    stock_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    ])
    
    # Fallback settings
    use_fallback_data: bool = True


@dataclass
class UnifiedMarketData:
    """Unified market data format for all market types."""
    market_id: str
    market_type: MarketType
    current_price: float
    previous_price: float
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    volume_24h: float
    liquidity: float
    volatility: float
    name: str
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_sandbox_format(self) -> Dict[str, Any]:
        """Convert to sandbox-compatible format."""
        return {
            "market_id": self.market_id,
            "current_price": self.current_price,
            "previous_price": self.previous_price,
            "mid_price": self.mid_price,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "volatility": self.volatility,
            "last_updated": self.last_updated.isoformat(),
        }


class UnifiedDataProvider:
    """
    Unified data provider that integrates multiple market data sources.
    
    Provides a single interface to access:
    - Cryptocurrency prices (via CoinGecko)
    - Stock prices (via Yahoo Finance)
    - Prediction market prices (via Polymarket)
    
    Features:
    - Automatic failover between providers
    - Unified market data format
    - Caching with configurable TTL
    - Fallback to simulated data if APIs unavailable
    """
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        """Initialize unified data provider."""
        self.config = config or DataProviderConfig()
        self._providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all enabled data providers."""
        if self.config.enable_crypto:
            self._providers[MarketType.CRYPTO] = CoinGeckoProvider(
                symbols=self.config.crypto_symbols,
                rate_limit_delay=1.5,
            )
        
        if self.config.enable_stocks:
            self._providers[MarketType.STOCK] = YahooFinanceProvider(
                symbols=self.config.stock_symbols,
                cache_ttl=self.config.cache_ttl_stocks,
            )
        
        if self.config.enable_prediction:
            self._providers[MarketType.PREDICTION] = PolymarketDataProvider(
                cache_ttl=self.config.cache_ttl_prediction,
            )
    
    async def close(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
    
    async def get_market_data(
        self,
        market_id: str,
        market_type: Optional[MarketType] = None,
        force_refresh: bool = False,
    ) -> Optional[UnifiedMarketData]:
        """
        Get market data for a specific market.
        
        Args:
            market_id: Market identifier (symbol, token_id, etc.)
            market_type: Type of market (auto-detected if not provided)
            force_refresh: Force refresh from API
            
        Returns:
            UnifiedMarketData object or None if not found
        """
        # Detect market type if not provided
        if market_type is None:
            market_type = self._detect_market_type(market_id)
        
        provider = self._providers.get(market_type)
        if not provider:
            if self.config.use_fallback_data:
                return self._get_fallback_data(market_id, market_type)
            return None
        
        try:
            # Get data from provider
            if market_type == MarketType.CRYPTO:
                data = await provider.get_market_data(market_id)
                if data:
                    return self._convert_crypto_data(data, market_id)
            
            elif market_type == MarketType.STOCK:
                data = await provider.get_market_data(market_id)
                if data:
                    return self._convert_stock_data(data, market_id)
            
            elif market_type == MarketType.PREDICTION:
                data = await provider.get_market_data(market_id)
                if data:
                    return self._convert_prediction_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching {market_type.value} data for {market_id}: {e}")
        
        # Fallback to simulated data
        if self.config.use_fallback_data:
            return self._get_fallback_data(market_id, market_type)
        
        return None
    
    async def get_all_market_data(
        self,
        market_type: Optional[MarketType] = None,
        force_refresh: bool = False,
    ) -> List[UnifiedMarketData]:
        """
        Get market data for all available markets.
        
        Args:
            market_type: Filter by market type (all types if not provided)
            force_refresh: Force refresh from API
            
        Returns:
            List of UnifiedMarketData objects
        """
        all_data = []
        
        # Get data from each provider
        for mtype, provider in self._providers.items():
            if market_type is not None and mtype != market_type:
                continue
            
            try:
                if mtype == MarketType.CRYPTO:
                    markets = await provider.get_markets(
                        symbols=self.config.crypto_symbols
                    )
                    for market in markets:
                        all_data.append(self._convert_crypto_data(market, market.symbol))
                
                elif mtype == MarketType.STOCK:
                    markets = await provider.get_multiple_market_data(
                        symbols=self.config.stock_symbols
                    )
                    for market in markets:
                        all_data.append(self._convert_stock_data(market, market.symbol))
                
                elif mtype == MarketType.PREDICTION:
                    markets = await provider.get_markets()
                    for market in markets:
                        all_data.append(self._convert_prediction_data(market))
                
            except Exception as e:
                logger.error(f"Error fetching {mtype.value} markets: {e}")
                continue
        
        return all_data
    
    async def get_market_data_by_ids(
        self,
        market_ids: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, UnifiedMarketData]:
        """
        Get market data for multiple specific markets.
        
        Args:
            market_ids: List of market IDs
            force_refresh: Force refresh from API
            
        Returns:
            Dictionary mapping market_id to UnifiedMarketData
        """
        tasks = [
            self.get_market_data(market_id, force_refresh=force_refresh)
            for market_id in market_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for market_id, result in zip(market_ids, results):
            if isinstance(result, UnifiedMarketData):
                data_dict[market_id] = result
        
        return data_dict
    
    def _detect_market_type(self, market_id: str) -> MarketType:
        """Detect market type from market ID."""
        market_id_lower = market_id.lower()
        
        # Check for crypto patterns
        crypto_prefixes = ["crypto_", "btc", "eth", "sol", "btc_", "eth_"]
        if any(market_id_lower.startswith(prefix) for prefix in crypto_prefixes):
            return MarketType.CRYPTO
        
        # Check for stock patterns
        stock_prefixes = ["stock_", "aapl", "msft", "goog", "amzn", "meta"]
        if any(market_id_lower.startswith(prefix) for prefix in stock_prefixes):
            return MarketType.STOCK
        
        # Check for prediction market patterns (long alphanumeric IDs)
        if len(market_id) > 30 and "-" in market_id:
            return MarketType.PREDICTION
        
        # Default to crypto for short IDs
        if len(market_id) <= 10:
            return MarketType.CRYPTO
        
        return MarketType.PREDICTION
    
    def _convert_crypto_data(
        self,
        data: CryptoMarketData,
        market_id: str,
    ) -> UnifiedMarketData:
        """Convert CoinGecko data to unified format."""
        unified_id = f"crypto_{data.symbol}"
        return UnifiedMarketData(
            market_id=unified_id,
            market_type=MarketType.CRYPTO,
            current_price=data.current_price,
            previous_price=data.current_price / (1 + data.price_change_percentage_24h / 100),
            mid_price=(data.high_24h + data.low_24h) / 2,
            best_bid=data.current_price * 0.9999,
            best_ask=data.current_price * 1.0001,
            spread=(data.high_24h - data.low_24h) / data.current_price,
            volume_24h=data.volume_24h,
            liquidity=data.volume_24h * 0.1,
            volatility=abs(data.price_change_percentage_24h) / 100,
            name=data.name,
            last_updated=datetime.utcnow(),
        )
    
    def _convert_stock_data(
        self,
        data: StockMarketData,
        market_id: str,
    ) -> UnifiedMarketData:
        """Convert Yahoo Finance data to unified format."""
        unified_id = f"stock_{data.symbol}"
        return UnifiedMarketData(
            market_id=unified_id,
            market_type=MarketType.STOCK,
            current_price=data.current_price,
            previous_price=data.previous_close,
            mid_price=(data.high_price + data.low_price) / 2,
            best_bid=data.current_price * 0.9999,
            best_ask=data.current_price * 1.0001,
            spread=(data.high_price - data.low_price) / data.current_price,
            volume_24h=data.volume,
            liquidity=data.volume * data.current_price * 0.01,
            volatility=abs(data.price_change_percent) / 100,
            name=data.name,
            last_updated=datetime.utcnow(),
        )
    
    def _convert_prediction_data(
        self,
        data: PredictionMarketData,
    ) -> UnifiedMarketData:
        """Convert Polymarket data to unified format."""
        return UnifiedMarketData(
            market_id=data.market_id,
            market_type=MarketType.PREDICTION,
            current_price=data.current_price,
            previous_price=data.previous_price,
            mid_price=data.mid_price,
            best_bid=data.best_bid,
            best_ask=data.best_ask,
            spread=data.spread,
            volume_24h=data.volume_24h,
            liquidity=data.liquidity,
            volatility=data.volatility,
            name=data.question,
            last_updated=datetime.utcnow(),
        )
    
    def _get_fallback_data(
        self,
        market_id: str,
        market_type: MarketType,
    ) -> UnifiedMarketData:
        """Get fallback simulated data when API is unavailable."""
        return UnifiedMarketData(
            market_id=market_id,
            market_type=market_type,
            current_price=0.5,
            previous_price=0.5,
            mid_price=0.5,
            best_bid=0.49,
            best_ask=0.51,
            spread=0.02,
            volume_24h=1000000,
            liquidity=100000,
            volatility=0.02,
            name=market_id,
            last_updated=datetime.utcnow(),
        )
    
    def get_sandbox_market_data(self, market_id: str) -> Dict[str, Any]:
        """Get market data in sandbox-compatible format."""
        market_type = self._detect_market_type(market_id)
        data = self._get_fallback_data(market_id, market_type)
        return data.to_sandbox_format()
    
    def add_crypto_symbol(self, symbol: str):
        """Add a cryptocurrency symbol to track."""
        if symbol not in self.config.crypto_symbols:
            self.config.crypto_symbols.append(symbol)
            if MarketType.CRYPTO in self._providers:
                self._providers[MarketType.CRYPTO].symbols.append(symbol)
    
    def add_stock_symbol(self, symbol: str):
        """Add a stock symbol to track."""
        if symbol not in self.config.stock_symbols:
            self.config.stock_symbols.append(symbol)
            if MarketType.STOCK in self._providers:
                self._providers[MarketType.STOCK].symbols.append(symbol)
    
    def clear_all_caches(self):
        """Clear all provider caches."""
        for provider in self._providers.values():
            if hasattr(provider, 'clear_cache'):
                provider.clear_cache()


async def test_unified_provider():
    """Test the unified data provider."""
    config = DataProviderConfig(
        enable_crypto=True,
        enable_stocks=True,
        enable_prediction=True,
        crypto_symbols=["bitcoin", "ethereum"],
        stock_symbols=["AAPL", "MSFT"],
    )
    
    provider = UnifiedDataProvider(config)
    
    try:
        # Get specific market data
        btc_data = await provider.get_market_data("bitcoin")
        if btc_data:
            print(f"Bitcoin: ${btc_data.current_price:,.2f}")
        
        aapl_data = await provider.get_market_data("AAPL")
        if aapl_data:
            print(f"AAPL: ${aapl_data.current_price:,.2f}")
        
        # Get all market data
        all_data = await provider.get_all_market_data()
        print(f"\nTotal markets tracked: {len(all_data)}")
        for data in all_data[:5]:
            print(f"  {data.market_id}: ${data.current_price:.4f}")
        
        # Test fallback
        unknown_data = await provider.get_market_data("unknown_market_123")
        if unknown_data:
            print(f"\nFallback data: {unknown_data.market_id} - ${unknown_data.current_price:.4f}")
        
    finally:
        await provider.close()


if __name__ == "__main__":
    import sys
    asyncio.run(test_unified_provider())
