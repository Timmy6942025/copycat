"""
Base classes and interfaces for market API clients.
Defines the contract that all API clients must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class MarketData:
    """Market data snapshot."""
    market_id: str
    current_price: float
    previous_price: float = 0.0
    mid_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    volatility: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def price_change_pct(self) -> float:
        """Calculate price change percentage."""
        if self.previous_price == 0:
            return 0.0
        return (self.current_price - self.previous_price) / self.previous_price


@dataclass
class OrderBook:
    """Order book snapshot."""
    market_id: str
    bids: List[Dict[str, float]] = field(default_factory=list)
    asks: List[Dict[str, float]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if not self.bids:
            return None
        return max(b["price"] for b in self.bids)

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if not self.asks:
            return None
        return min(a["price"] for a in self.asks)

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2

    def get_spread(self) -> Optional[float]:
        """Get spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid


@dataclass
class Trade:
    """Trade record."""
    trade_id: str
    market_id: str
    trader_address: str
    side: OrderSide
    quantity: float
    price: float
    total_value: float
    fees: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    outcome: str = "YES"
    transaction_hash: Optional[str] = None


@dataclass
class Position:
    """Position record."""
    market_id: str
    outcome: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Order:
    """Order record."""
    order_id: str
    market_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fees: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    outcome: str = "YES"
    trader_address: Optional[str] = None

    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED or self.filled_quantity >= self.quantity

    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return max(0.0, self.quantity - self.filled_quantity)


@dataclass
class Trader:
    """Trader profile information."""
    address: str
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    is_verified: bool = False
    is_market_maker: bool = False
    total_trades: int = 0
    total_volume: float = 0.0
    first_trade_timestamp: Optional[datetime] = None
    last_trade_timestamp: Optional[datetime] = None


@dataclass
class TraderPerformance:
    """Trader performance metrics."""
    trader_address: str
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_hold_time_hours: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_volume: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # Insider detection metrics
    early_position_score: float = 0.0
    event_correlation_score: float = 0.0
    category_expertise_score: float = 0.0

    # Bot detection metrics
    is_bot_likely: bool = False
    hft_score: float = 0.0
    arbitrage_score: float = 0.0

    # Reputation scoring
    reputation_score: float = 0.0
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trader_address": self.trader_address,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "avg_hold_time_hours": self.avg_hold_time_hours,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_volume": self.total_volume,
            "early_position_score": self.early_position_score,
            "event_correlation_score": self.event_correlation_score,
            "category_expertise_score": self.category_expertise_score,
            "is_bot_likely": self.is_bot_likely,
            "hft_score": self.hft_score,
            "arbitrage_score": self.arbitrage_score,
            "reputation_score": self.reputation_score,
            "confidence_score": self.confidence_score,
            "last_updated": self.last_updated.isoformat(),
        }


class MarketAPIClient(ABC):
    """
    Abstract base class for market API clients.
    All platform-specific clients must implement these methods.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize API client."""
        self.api_key = api_key
        self._rate_limit_delay = 0.1  # Default delay between requests

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Get platform name."""
        pass

    @abstractmethod
    async def get_markets(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get list of markets."""
        pass

    @abstractmethod
    async def get_market_data(self, market_id: str) -> Optional[MarketData]:
        """Get market data for a specific market."""
        pass

    @abstractmethod
    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get order book for a specific market."""
        pass

    @abstractmethod
    async def get_trades(
        self,
        market_id: Optional[str] = None,
        trader_address: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """Get trades with optional filters."""
        pass

    @abstractmethod
    async def get_trader_info(self, trader_address: str) -> Optional[Trader]:
        """Get trader profile information."""
        pass

    @abstractmethod
    async def get_trader_positions(
        self, trader_address: str, market_id: Optional[str] = None
    ) -> List[Position]:
        """Get trader's current positions."""
        pass

    @abstractmethod
    async def get_trader_orders(
        self,
        trader_address: str,
        market_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> List[Order]:
        """Get trader's orders."""
        pass

    @abstractmethod
    async def create_order(
        self,
        market_id: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        limit_price: Optional[float] = None,
        outcome: str = "YES",
    ) -> Order:
        """Create a new order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    async def get_balance(self, trader_address: str) -> float:
        """Get trader's balance."""
        pass

    async def rate_limited_request(self, coro):
        """Execute a request with rate limiting."""
        await asyncio.sleep(self._rate_limit_delay)
        return await coro

    def set_rate_limit(self, requests_per_second: float):
        """Set rate limit delay between requests."""
        self._rate_limit_delay = 1.0 / requests_per_second if requests_per_second > 0 else 0
