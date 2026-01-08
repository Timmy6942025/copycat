"""
Unit tests for API clients module.
Tests Polymarket and Kalshi API client implementations.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

# Test imports
import sys
sys.path.insert(0, '/home/timmy/copycat')

from api_clients.base import (
    MarketData,
    OrderBook,
    Trade,
    Position,
    Order,
    Trader,
    TraderPerformance,
    OrderSide,
    OrderType,
    OrderStatus,
    MarketAPIClient,
)


class TestMarketData:
    """Test MarketData dataclass."""

    def test_price_change_pct_positive(self):
        """Test positive price change calculation."""
        data = MarketData(
            market_id="test-1",
            current_price=0.6,
            previous_price=0.5,
        )
        assert data.price_change_pct() == pytest.approx(0.2, rel=0.01)

    def test_price_change_pct_negative(self):
        """Test negative price change calculation."""
        data = MarketData(
            market_id="test-1",
            current_price=0.4,
            previous_price=0.5,
        )
        assert data.price_change_pct() == pytest.approx(-0.2, rel=0.01)

    def test_price_change_pct_zero_previous(self):
        """Test price change with zero previous price."""
        data = MarketData(
            market_id="test-1",
            current_price=0.5,
            previous_price=0.0,
        )
        assert data.price_change_pct() == 0.0


class TestOrderBook:
    """Test OrderBook dataclass."""

    def test_get_best_bid(self):
        """Test getting best bid from orderbook."""
        book = OrderBook(
            market_id="test-1",
            bids=[
                {"price": 0.49, "size": 100},
                {"price": 0.48, "size": 200},
                {"price": 0.47, "size": 300},
            ],
        )
        assert book.get_best_bid() == pytest.approx(0.49)

    def test_get_best_ask(self):
        """Test getting best ask from orderbook."""
        book = OrderBook(
            market_id="test-1",
            asks=[
                {"price": 0.51, "size": 100},
                {"price": 0.52, "size": 200},
                {"price": 0.53, "size": 300},
            ],
        )
        assert book.get_best_ask() == pytest.approx(0.51)

    def test_get_mid_price(self):
        """Test calculating mid price."""
        book = OrderBook(
            market_id="test-1",
            bids=[{"price": 0.49, "size": 100}],
            asks=[{"price": 0.51, "size": 100}],
        )
        assert book.get_mid_price() == pytest.approx(0.50)

    def test_get_spread(self):
        """Test calculating spread."""
        book = OrderBook(
            market_id="test-1",
            bids=[{"price": 0.49, "size": 100}],
            asks=[{"price": 0.51, "size": 100}],
        )
        assert book.get_spread() == pytest.approx(0.02)

    def test_empty_orderbook(self):
        """Test empty orderbook handling."""
        book = OrderBook(market_id="test-1")
        assert book.get_best_bid() is None
        assert book.get_best_ask() is None
        assert book.get_mid_price() is None
        assert book.get_spread() is None


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.55,
            total_value=55.0,
        )
        assert trade.trade_id == "t1"
        assert trade.side == OrderSide.BUY
        assert trade.price == pytest.approx(0.55)


class TestOrder:
    """Test Order dataclass."""

    def test_is_filled(self):
        """Test order filled status."""
        order = Order(
            order_id="o1",
            market_id="m1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            filled_quantity=100.0,
        )
        assert order.is_filled() is True

    def test_not_filled(self):
        """Test order not filled status."""
        order = Order(
            order_id="o1",
            market_id="m1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            filled_quantity=50.0,
        )
        assert order.is_filled() is False

    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = Order(
            order_id="o1",
            market_id="m1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            filled_quantity=30.0,
        )
        assert order.remaining_quantity() == pytest.approx(70.0)


class TestTraderPerformance:
    """Test TraderPerformance dataclass."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=100.0,
            win_rate=0.65,
            total_trades=50,
        )
        data = perf.to_dict()
        assert data["trader_address"] == "0x123"
        assert data["total_pnl"] == 100.0
        assert data["win_rate"] == 0.65


class TestMarketAPIClient:
    """Test MarketAPIClient abstract class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Verify MarketAPIClient is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            MarketAPIClient(api_key="test-key")

    def test_set_rate_limit_method_exists(self):
        """Test that set_rate_limit method exists on abstract class."""
        # Verify the method exists but class cannot be instantiated
        assert hasattr(MarketAPIClient, 'set_rate_limit')


class TestPolymarketClient:
    """Test Polymarket API client."""

    @pytest.fixture
    def polymarket_client(self):
        """Create a Polymarket client for testing."""
        from api_clients.polymarket import PolymarketAPIClient
        return PolymarketAPIClient(api_key="test-key", use_gamma=False, use_clob=False)

    def test_platform_name(self, polymarket_client):
        """Test platform name."""
        assert polymarket_client.platform_name == "polymarket"

    def test_client_initialization(self, polymarket_client):
        """Test client initialization."""
        assert polymarket_client.api_key == "test-key"
        assert polymarket_client.gamma is None
        assert polymarket_client.clob is None


class TestKalshiClient:
    """Test Kalshi API client."""

    @pytest.fixture
    def kalshi_client(self):
        """Create a Kalshi client for testing."""
        from api_clients.kalshi import KalshiAPIClient
        return KalshiAPIClient(api_key="test-key")

    def test_platform_name(self, kalshi_client):
        """Test platform name."""
        assert kalshi_client.platform_name == "kalshi"

    def test_client_initialization(self, kalshi_client):
        """Test client initialization."""
        assert kalshi_client.api_key == "test-key"
        assert kalshi_client.demo is False

    def test_rate_limit_set(self, kalshi_client):
        """Test rate limit is set on initialization."""
        # Basic tier should be 20 req/s
        assert kalshi_client._rate_limit_delay == pytest.approx(0.05, abs=0.01)


# Test fixtures for mock data
@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    base_time = datetime.utcnow()
    return [
        Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.55,
            total_value=55.0,
            fees=1.0,
            timestamp=base_time,
        ),
        Trade(
            trade_id="t2",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.SELL,
            quantity=100.0,
            price=0.60,
            total_value=60.0,
            fees=1.2,
            timestamp=base_time + timedelta(hours=1),
        ),
        Trade(
            trade_id="t3",
            market_id="m2",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=50.0,
            price=0.45,
            total_value=22.5,
            fees=0.5,
            timestamp=base_time + timedelta(hours=2),
        ),
    ]


@pytest.fixture
def sample_performance():
    """Create sample performance data."""
    return TraderPerformance(
        trader_address="0x123",
        total_pnl=250.0,
        total_pnl_pct=0.25,
        win_rate=0.7,
        profit_factor=2.5,
        sharpe_ratio=1.5,
        max_drawdown=0.1,
        total_trades=100,
        winning_trades=70,
        losing_trades=30,
        avg_win=10.0,
        avg_loss=5.0,
        total_volume=1000.0,
    )
