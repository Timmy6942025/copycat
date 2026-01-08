"""
Unit tests for the Live Trading Runner.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.runner import (
    LiveTradingRunner,
    LiveTradingConfig,
    LiveTradingState,
    LiveOrderResult,
    LiveOrderStatus,
    create_live_runner,
)


class TestLiveTradingConfig:
    """Tests for LiveTradingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LiveTradingConfig()
        
        assert config.initial_balance == 10000.0
        assert config.max_position_size_pct == 0.10
        assert config.max_total_exposure_pct == 0.50
        assert config.max_orders_per_day == 50
        assert config.min_order_size == 1.0
        assert config.require_order_confirmation is True
        assert config.max_slippage_pct == 0.05
        assert config.enable_price_protection is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LiveTradingConfig(
            initial_balance=50000.0,
            max_position_size_pct=0.20,
            max_orders_per_day=100,
        )
        
        assert config.initial_balance == 50000.0
        assert config.max_position_size_pct == 0.20
        assert config.max_orders_per_day == 100


class TestLiveTradingState:
    """Tests for LiveTradingState."""

    def test_default_state(self):
        """Test default state values."""
        state = LiveTradingState()
        
        assert state.balance == 0.0
        assert state.total_value == 0.0
        assert state.orders_placed == 0
        assert state.orders_filled == 0
        assert state.orders_failed == 0
        assert state.last_order_time is None

    def test_custom_state(self):
        """Test custom state values."""
        state = LiveTradingState(
            balance=25000.0,
            total_value=30000.0,
            orders_placed=10,
            orders_filled=8,
            orders_failed=2,
            last_order_time=datetime.utcnow(),
        )
        
        assert state.balance == 25000.0
        assert state.orders_placed == 10
        assert state.orders_filled == 8


class TestLiveOrderResult:
    """Tests for LiveOrderResult."""

    def test_order_result_filled(self):
        """Test filled order result."""
        result = LiveOrderResult(
            order_id="order_123",
            status=LiveOrderStatus.FILLED,
            filled_quantity=100.0,
            average_price=0.55,
            fees=1.10,
            message="Order executed successfully",
            trade_id="source_trade_456",
        )
        
        assert result.order_id == "order_123"
        assert result.status == LiveOrderStatus.FILLED
        assert result.filled_quantity == 100.0
        assert result.average_price == 0.55
        assert result.trade_id == "source_trade_456"

    def test_order_result_rejected(self):
        """Test rejected order result."""
        result = LiveOrderResult(
            order_id="",
            status=LiveOrderStatus.REJECTED,
            message="Order value below minimum",
        )
        
        assert result.status == LiveOrderStatus.REJECTED
        assert result.filled_quantity == 0.0


class TestLiveTradingRunner:
    """Tests for LiveTradingRunner."""

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client."""
        client = Mock()
        client.get_balance = AsyncMock(return_value=10000.0)
        client.get_market_data = AsyncMock(return_value=None)
        client.create_order = AsyncMock()
        client.cancel_order = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def runner(self, mock_api_client):
        """Create live trading runner for testing."""
        config = LiveTradingConfig(initial_balance=10000.0)
        return LiveTradingRunner(
            config=config,
            api_client=mock_api_client,
            wallet_address="0x1234567890abcdef",
        )

    def test_initialization(self, runner):
        """Test runner initialization."""
        assert runner.is_running is False
        assert runner.state.balance == 10000.0
        assert len(runner.orders) == 0
        assert len(runner.positions) == 0

    def test_set_callbacks(self, runner):
        """Test setting order and trade callbacks."""
        order_callback = Mock()
        trade_callback = Mock()
        
        runner.set_order_callback(order_callback)
        runner.set_trade_callback(trade_callback)
        
        assert runner.order_callback == order_callback
        assert runner.trade_callback == trade_callback

    @pytest.mark.asyncio
    async def test_start_success(self, runner, mock_api_client):
        """Test starting the runner successfully."""
        result = await runner.start()
        
        assert result is True
        assert runner.is_running is True
        mock_api_client.get_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_no_api_client(self, runner):
        """Test starting without API client."""
        runner.api_client = None
        result = await runner.start()
        
        assert result is False
        assert runner.is_running is False

    @pytest.mark.asyncio
    async def test_stop(self, runner):
        """Test stopping the runner."""
        await runner.start()
        result = await runner.stop()
        
        assert result is True
        assert runner.is_running is False

    def test_check_daily_limits(self, runner):
        """Test daily order limits."""
        # Should allow orders
        can_trade = runner._check_daily_limits()
        assert can_trade is True
        
        # Fill up daily limit
        runner._orders_today = runner.config.max_orders_per_day
        can_trade = runner._check_daily_limits()
        assert can_trade is False

    def test_check_position_limits_valid(self, runner):
        """Test valid position limits."""
        can_trade, message = runner._check_position_limits(quantity=100, price=0.50)
        
        assert can_trade is True
        assert message == ""

    def test_check_position_limits_below_minimum(self, runner):
        """Test position below minimum size."""
        can_trade, message = runner._check_position_limits(quantity=1, price=0.50)
        
        assert can_trade is False
        assert "minimum" in message.lower()

    def test_check_position_limits_exceeds_max(self, runner):
        """Test position exceeds max size."""
        # 20% of balance should exceed 10% max
        can_trade, message = runner._check_position_limits(quantity=4000, price=0.50)
        
        assert can_trade is False
        assert "exceeds" in message.lower() or "max" in message.lower()

    @pytest.mark.asyncio
    async def test_execute_order_not_running(self, runner):
        """Test executing order when runner is not running."""
        result = await runner.execute_order(
            market_id="market_1",
            side="buy",
            quantity=100.0,
        )
        
        assert result.status == LiveOrderStatus.REJECTED
        assert "not running" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_order_daily_limit_reached(self, runner):
        """Test executing order when daily limit is reached."""
        await runner.start()
        runner._orders_today = runner.config.max_orders_per_day
        
        result = await runner.execute_order(
            market_id="market_1",
            side="buy",
            quantity=100.0,
        )
        
        assert result.status == LiveOrderStatus.REJECTED
        assert "limit" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_order_success(self, runner, mock_api_client):
        """Test successful order execution."""
        from api_clients.base import Order, OrderStatus
        
        mock_order = Mock()
        mock_order.order_id = "real_order_123"
        mock_order.status = OrderStatus.FILLED
        mock_order.filled_quantity = 100.0
        mock_order.average_price = 0.55
        mock_order.fees = 1.10
        mock_api_client.create_order = AsyncMock(return_value=mock_order)
        
        await runner.start()
        
        result = await runner.execute_order(
            market_id="market_1",
            side="buy",
            quantity=100.0,
            order_type="market",
            outcome="YES",
            source_trade_id="source_123",
            source_trader="0xabc",
        )
        
        assert result.status == LiveOrderStatus.FILLED
        assert result.order_id == "real_order_123"
        assert result.filled_quantity == 100.0
        mock_api_client.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_order_api_failure(self, runner, mock_api_client):
        """Test order execution with API failure."""
        mock_api_client.create_order = AsyncMock(side_effect=Exception("API error"))
        
        await runner.start()
        
        result = await runner.execute_order(
            market_id="market_1",
            side="buy",
            quantity=100.0,
        )
        
        assert result.status == LiveOrderStatus.REJECTED
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, runner, mock_api_client):
        """Test cancelling an order."""
        await runner.start()
        
        result = await runner.cancel_order("order_123")
        
        assert result is True
        mock_api_client.cancel_order.assert_called_once_with("order_123")

    def test_get_state(self, runner):
        """Test getting trading state."""
        runner.state.balance = 9500.0
        runner.state.orders_placed = 5
        runner.state.orders_filled = 4
        
        state = runner.get_state()
        
        assert state.balance == 9500.0
        assert state.orders_placed == 5

    def test_get_portfolio_summary(self, runner):
        """Test getting portfolio summary."""
        runner.state.balance = 9500.0
        runner._orders_today = 3
        
        summary = runner.get_portfolio_summary()
        
        assert summary["balance"] == 9500.0
        assert summary["orders_today"] == 3
        assert "positions_count" in summary

    def test_get_performance_metrics(self, runner):
        """Test getting performance metrics."""
        runner.state.balance = 11000.0
        runner.state.orders_placed = 10
        runner.state.orders_filled = 8
        
        metrics = runner.get_performance_metrics()
        
        assert metrics["total_pnl"] == 1000.0  # 11000 - 10000
        assert metrics["total_pnl_pct"] == 0.10
        assert metrics["win_rate"] == 0.8
        assert metrics["trades_executed"] == 8


class TestCreateLiveRunner:
    """Tests for create_live_runner factory function."""

    def test_create_polymarket_runner(self):
        """Test creating Polymarket live runner."""
        runner = create_live_runner(
            platform="polymarket",
            wallet_address="0x123",
            initial_balance=25000.0,
        )
        
        assert isinstance(runner, LiveTradingRunner)
        assert runner.config.initial_balance == 25000.0
        assert runner.wallet_address == "0x123"

    def test_create_kalshi_runner(self):
        """Test creating Kalshi live runner."""
        runner = create_live_runner(
            platform="kalshi",
            wallet_address="0x456",
        )
        
        assert isinstance(runner, LiveTradingRunner)

    def test_invalid_platform(self):
        """Test creating runner with invalid platform."""
        with pytest.raises(ValueError):
            create_live_runner(platform="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
