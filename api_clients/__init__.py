"""
API Clients module for CopyCat trading bot.
Provides unified interfaces for Polymarket and Kalshi APIs.
"""

from .base import (
    MarketAPIClient,
    MarketData,
    OrderBook,
    Trade,
    Trader,
    TraderPerformance,
    Position,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
)
from .polymarket import PolymarketAPIClient
from .kalshi import KalshiAPIClient
from .mock import MockMarketAPIClient, create_mock_client

__all__ = [
    "MarketAPIClient",
    "MarketData",
    "OrderBook",
    "Trade",
    "Trader",
    "TraderPerformance",
    "Position",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PolymarketAPIClient",
    "KalshiAPIClient",
    "MockMarketAPIClient",
    "create_mock_client",
]
