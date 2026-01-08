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


# =============================================================================
# Mock API Client Tests
# =============================================================================

class TestMockMarketAPIClient:
    """Tests for MockMarketAPIClient."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        from api_clients.mock import MockMarketAPIClient
        return MockMarketAPIClient(
            platform="polymarket",
            seed=12345,
            initial_balance=10000.0,
            num_markets=20,
            num_traders=30,
        )

    def test_platform_name(self, mock_client):
        """Test platform name property."""
        assert mock_client.platform_name == "polymarket"

    def test_initial_markets_generated(self, mock_client):
        """Test that markets are generated on init."""
        assert len(mock_client._markets) == 20

    def test_initial_traders_generated(self, mock_client):
        """Test that traders are generated on init."""
        assert len(mock_client._traders) == 30

    def test_initial_trades_generated(self, mock_client):
        """Test that trades are generated on init."""
        assert len(mock_client._trades) > 0

    @pytest.mark.asyncio
    async def test_get_markets_default(self, mock_client):
        """Test getting markets with default parameters."""
        markets = await mock_client.get_markets()
        
        assert isinstance(markets, list)
        assert len(markets) <= 50  # Default limit

    @pytest.mark.asyncio
    async def test_get_markets_with_limit(self, mock_client):
        """Test getting markets with custom limit."""
        markets = await mock_client.get_markets(limit=5)
        
        assert len(markets) == 5

    @pytest.mark.asyncio
    async def test_get_markets_with_offset(self, mock_client):
        """Test getting markets with offset pagination."""
        all_markets = await mock_client.get_markets(limit=100)
        first_five = await mock_client.get_markets(limit=5, offset=0)
        next_five = await mock_client.get_markets(limit=5, offset=5)
        
        assert first_five[0] != next_five[0]

    @pytest.mark.asyncio
    async def test_get_markets_by_category(self, mock_client):
        """Test filtering markets by category."""
        markets = await mock_client.get_markets(category="Crypto")
        
        for market in markets:
            assert market.get("category") == "Crypto"

    @pytest.mark.asyncio
    async def test_get_market_data_exists(self, mock_client):
        """Test getting data for existing market."""
        # Get first market ID
        markets = await mock_client.get_markets(limit=1)
        market_id = markets[0]["id"]
        
        data = await mock_client.get_market_data(market_id)
        
        assert data is not None
        assert data.market_id == market_id
        assert 0 < data.current_price < 1

    @pytest.mark.asyncio
    async def test_get_market_data_not_exists(self, mock_client):
        """Test getting data for non-existent market."""
        data = await mock_client.get_market_data("non_existent_market")
        
        assert data is None

    @pytest.mark.asyncio
    async def test_get_orderbook_exists(self, mock_client):
        """Test getting orderbook for existing market."""
        markets = await mock_client.get_markets(limit=1)
        market_id = markets[0]["id"]
        
        orderbook = await mock_client.get_orderbook(market_id)
        
        assert orderbook is not None
        assert orderbook.market_id == market_id
        assert orderbook.bids is not None
        assert orderbook.asks is not None

    @pytest.mark.asyncio
    async def test_get_orderbook_not_exists(self, mock_client):
        """Test getting orderbook for non-existent market."""
        orderbook = await mock_client.get_orderbook("non_existent_market")
        
        assert orderbook is None

    @pytest.mark.asyncio
    async def test_get_trades_no_filters(self, mock_client):
        """Test getting trades with no filters."""
        trades = await mock_client.get_trades()
        
        assert isinstance(trades, list)
        assert len(trades) > 0

    @pytest.mark.asyncio
    async def test_get_trades_by_market(self, mock_client):
        """Test filtering trades by market."""
        # First get a market ID
        markets = await mock_client.get_markets(limit=1)
        market_id = markets[0]["id"]
        
        trades = await mock_client.get_trades(market_id=market_id)
        
        for trade in trades:
            assert trade.market_id == market_id

    @pytest.mark.asyncio
    async def test_get_trades_by_trader(self, mock_client):
        """Test filtering trades by trader."""
        # Get first trader's address
        trader = mock_client._traders[0]
        address = trader["address"]
        
        trades = await mock_client.get_trades(trader_address=address)
        
        for trade in trades:
            assert trade.trader_address == address

    @pytest.mark.asyncio
    async def test_get_trades_with_limit(self, mock_client):
        """Test limiting trades."""
        trades = await mock_client.get_trades(limit=5)
        
        assert len(trades) == 5

    @pytest.mark.asyncio
    async def test_get_trades_by_time_range(self, mock_client):
        """Test filtering trades by time range."""
        now = datetime.utcnow()
        start_time = now - timedelta(days=1)
        end_time = now
        
        trades = await mock_client.get_trades(
            start_time=start_time,
            end_time=end_time,
        )
        
        for trade in trades:
            assert start_time <= trade.timestamp <= end_time

    @pytest.mark.asyncio
    async def test_get_trader_info_exists(self, mock_client):
        """Test getting info for existing trader."""
        trader = mock_client._traders[0]
        
        info = await mock_client.get_trader_info(trader["address"])
        
        assert info is not None
        assert info.address == trader["address"]

    @pytest.mark.asyncio
    async def test_get_trader_info_not_exists(self, mock_client):
        """Test getting info for non-existent trader."""
        info = await mock_client.get_trader_info("0xnon_existent_address")
        
        assert info is None

    @pytest.mark.asyncio
    async def test_get_trader_positions(self, mock_client):
        """Test getting trader positions."""
        trader = mock_client._traders[0]
        
        positions = await mock_client.get_trader_positions(trader["address"])
        
        assert isinstance(positions, list)

    @pytest.mark.asyncio
    async def test_get_trader_orders(self, mock_client):
        """Test getting trader orders."""
        trader = mock_client._traders[0]
        
        orders = await mock_client.get_trader_orders(trader["address"])
        
        assert isinstance(orders, list)

    @pytest.mark.asyncio
    async def test_get_balance(self, mock_client):
        """Test getting trader balance."""
        trader = mock_client._traders[0]
        
        balance = await mock_client.get_balance(trader["address"])
        
        assert balance == 10000.0  # Initial balance

    @pytest.mark.asyncio
    async def test_get_balance_new_trader(self, mock_client):
        """Test getting balance for new trader."""
        balance = await mock_client.get_balance("0xnew_trader_address")
        
        assert balance == 10000.0  # Initial balance

    @pytest.mark.asyncio
    async def test_create_market_order(self, mock_client):
        """Test creating a market order."""
        markets = await mock_client.get_markets(limit=1)
        
        order = await mock_client.create_order(
            market_id=markets[0]["id"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )
        
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100.0

    @pytest.mark.asyncio
    async def test_create_limit_order(self, mock_client):
        """Test creating a limit order."""
        markets = await mock_client.get_markets(limit=1)
        
        order = await mock_client.create_order(
            market_id=markets[0]["id"],
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=0.5,
        )
        
        assert order is not None
        assert order.status == OrderStatus.OPEN
        assert order.filled_quantity == 0.0

    @pytest.mark.asyncio
    async def test_cancel_order_exists(self, mock_client):
        """Test cancelling an existing order."""
        markets = await mock_client.get_markets(limit=1)
        
        # Create a limit order
        order = await mock_client.create_order(
            market_id=markets[0]["id"],
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=0.5,
        )
        
        # Cancel it
        result = await mock_client.cancel_order(order.order_id)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_exists(self, mock_client):
        """Test cancelling non-existent order."""
        result = await mock_client.cancel_order("non_existent_order_id")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_trader_performance(self, mock_client):
        """Test getting trader performance."""
        trader = mock_client._traders[0]
        
        perf = await mock_client.get_trader_performance(trader["address"])
        
        assert perf is not None
        assert perf.trader_address == trader["address"]
        assert 0 <= perf.win_rate <= 1

    @pytest.mark.asyncio
    async def test_get_trader_performance_no_trades(self, mock_client):
        """Test getting performance for trader with no trades."""
        # Create a new trader with no trades
        perf = await mock_client.get_trader_performance("0xnew_trader_no_trades")
        
        assert perf is None


class TestMockClientDataGeneration:
    """Tests for mock client data generation."""

    def test_deterministic_seed(self):
        """Test that same seed produces same data."""
        from api_clients.mock import MockMarketAPIClient
        
        client1 = MockMarketAPIClient(seed=42, num_markets=10, num_traders=10)
        client2 = MockMarketAPIClient(seed=42, num_markets=10, num_traders=10)
        
        # Same markets
        assert len(client1._markets) == len(client2._markets)
        assert client1._markets[0]["id"] == client2._markets[0]["id"]
        
        # Same traders
        assert len(client1._traders) == len(client2._traders)
        assert client1._traders[0]["address"] == client2._traders[0]["address"]

    def test_different_seeds_different_data(self):
        """Test that different seeds produce different data."""
        from api_clients.mock import MockMarketAPIClient
        
        client1 = MockMarketAPIClient(seed=111, num_markets=10, num_traders=10)
        client2 = MockMarketAPIClient(seed=222, num_markets=10, num_traders=10)
        
        # Different seeds should produce different market data (prices, questions)
        # Market IDs are deterministic, but the content should differ
        assert client1._markets[0]["question"] != client2._markets[0]["question"]
        assert client1._markets[0]["outcome_price"]["yes"] != client2._markets[0]["outcome_price"]["yes"]


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.fixture
    def client_with_rate_limit(self):
        """Create client with custom rate limit."""
        from api_clients.polymarket import PolymarketAPIClient
        return PolymarketAPIClient(api_key="test", use_gamma=False, use_clob=False)

    def test_default_rate_limit(self, client_with_rate_limit):
        """Test default rate limit is set."""
        assert client_with_rate_limit._rate_limit_delay == 0.1  # 10 req/s

    def test_set_rate_limit_high(self, client_with_rate_limit):
        """Test setting high rate limit."""
        client_with_rate_limit.set_rate_limit(100.0)
        assert client_with_rate_limit._rate_limit_delay == 0.01

    def test_set_rate_limit_low(self, client_with_rate_limit):
        """Test setting low rate limit."""
        client_with_rate_limit.set_rate_limit(1.0)
        assert client_with_rate_limit._rate_limit_delay == 1.0

    def test_set_rate_limit_zero(self, client_with_rate_limit):
        """Test setting zero rate limit."""
        client_with_rate_limit.set_rate_limit(0.0)
        assert client_with_rate_limit._rate_limit_delay == 0.0


class TestOrderBook:
    """Additional tests for OrderBook functionality."""

    def test_orderbook_best_bid_multiple(self):
        """Test best bid with multiple levels."""
        book = OrderBook(
            market_id="test-1",
            bids=[
                {"price": 0.49, "size": 100},
                {"price": 0.48, "size": 200},
                {"price": 0.47, "size": 300},
                {"price": 0.46, "size": 400},
            ],
        )
        assert book.get_best_bid() == pytest.approx(0.49)

    def test_orderbook_best_ask_multiple(self):
        """Test best ask with multiple levels."""
        book = OrderBook(
            market_id="test-1",
            asks=[
                {"price": 0.51, "size": 100},
                {"price": 0.52, "size": 200},
                {"price": 0.53, "size": 300},
                {"price": 0.54, "size": 400},
            ],
        )
        assert book.get_best_ask() == pytest.approx(0.51)

    def test_orderbook_with_empty_bids_only(self):
        """Test orderbook with only asks."""
        book = OrderBook(
            market_id="test-1",
            asks=[{"price": 0.51, "size": 100}],
        )
        assert book.get_best_bid() is None
        assert book.get_best_ask() == pytest.approx(0.51)

    def test_orderbook_with_empty_asks_only(self):
        """Test orderbook with only bids."""
        book = OrderBook(
            market_id="test-1",
            bids=[{"price": 0.49, "size": 100}],
        )
        assert book.get_best_bid() == pytest.approx(0.49)
        assert book.get_best_ask() is None


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        position = Position(
            market_id="m1",
            outcome="YES",
            quantity=100.0,
            avg_price=0.55,
        )
        assert position.market_id == "m1"
        assert position.outcome == "YES"
        assert position.quantity == 100.0

    def test_position_with_pnl(self):
        """Test position with P&L values."""
        position = Position(
            market_id="m1",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.60,
            unrealized_pnl=10.0,
            realized_pnl=5.0,
        )
        assert position.unrealized_pnl == 10.0
        assert position.realized_pnl == 5.0


class TestOrderEdgeCases:
    """Tests for edge cases in Order handling."""

    def test_order_with_zero_quantity(self):
        """Test order with zero quantity."""
        order = Order(
            order_id="o1",
            market_id="m1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.0,
        )
        assert order.is_filled() is True
        assert order.remaining_quantity() == 0.0

    def test_order_already_filled(self):
        """Test order that is already filled."""
        order = Order(
            order_id="o1",
            market_id="m1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            filled_quantity=100.0,
            status=OrderStatus.FILLED,
        )
        assert order.is_filled() is True

    def test_order_partial_fill(self):
        """Test order with partial fill."""
        order = Order(
            order_id="o1",
            market_id="m1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            filled_quantity=50.0,
            status=OrderStatus.PARTIAL_FILL,
        )
        assert order.is_filled() is False
        assert order.remaining_quantity() == pytest.approx(50.0)


class TestTraderProfile:
    """Tests for Trader profile."""

    def test_trader_with_all_fields(self):
        """Test creating a trader with all fields."""
        now = datetime.utcnow()
        trader = Trader(
            address="0x1234",
            username="TestTrader",
            avatar_url="https://example.com/avatar.png",
            bio="Test bio",
            is_verified=True,
            is_market_maker=False,
            total_trades=100,
            total_volume=50000.0,
            first_trade_timestamp=now - timedelta(days=30),
            last_trade_timestamp=now,
        )
        assert trader.is_verified is True
        assert trader.total_trades == 100

    def test_trader_minimal(self):
        """Test creating a trader with minimal fields."""
        trader = Trader(address="0x5678")
        assert trader.address == "0x5678"
        assert trader.username is None
        assert trader.is_verified is False


class TestTradeValidation:
    """Tests for trade data validation."""

    def test_trade_with_all_fields(self):
        """Test creating a trade with all fields."""
        now = datetime.utcnow()
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.55,
            total_value=55.0,
            fees=1.1,
            timestamp=now,
            outcome="YES",
            transaction_hash="0xabc123",
        )
        assert trade.fees == 1.1
        assert trade.transaction_hash is not None

    def test_trade_minimal(self):
        """Test creating a trade with minimal fields."""
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.55,
            total_value=55.0,
        )
        assert trade.fees == 0.0
        assert trade.outcome == "YES"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
