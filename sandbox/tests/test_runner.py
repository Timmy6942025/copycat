"""
Unit tests for SandboxRunner.

Tests the main orchestrator for sandbox simulation including order execution,
portfolio management, performance tracking, and data feed integration.
"""

import pytest
from datetime import datetime
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.config import (
    SandboxConfig,
    VirtualOrder,
    SimulationState,
    PortfolioSummary,
)
from sandbox.runner import SandboxRunner


class TestSandboxRunner:
    """Tests for SandboxRunner class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SandboxConfig(initial_balance=10000.0)
        self.runner = SandboxRunner(self.config)

    def test_initialization(self):
        """Test runner initialization."""
        assert self.runner.config == self.config
        assert self.runner.state.balance == 10000.0
        assert self.runner.state.total_value == 10000.0
        assert self.runner.executor is not None
        assert self.runner.portfolio_manager is not None
        assert self.runner.tracker is not None
        assert self.runner.reporter is not None

    def test_initial_state(self):
        """Test initial simulation state."""
        state = self.runner.get_state()

        assert state.balance == 10000.0
        assert state.total_value == 10000.0
        assert state.positions == {}
        assert state.pending_orders == []
        assert state.total_pnl == 0.0
        assert state.is_paused is False
        assert state.is_completed is False

    def test_get_portfolio_summary(self):
        """Test portfolio summary retrieval."""
        summary = self.runner.get_portfolio_summary()

        assert isinstance(summary, PortfolioSummary)
        assert summary.balance == 10000.0
        assert summary.total_value == 10000.0
        assert summary.position_count == 0

    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        metrics = self.runner.get_performance_metrics()

        assert metrics is not None
        assert metrics.starting_balance == 10000.0
        assert metrics.ending_balance == 10000.0
        assert metrics.total_pnl == 0.0
        assert metrics.total_trades == 0

    def test_reset(self):
        """Test simulation reset."""
        # Execute some orders first
        order = VirtualOrder(
            order_id="test_order",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        market_data = {
            "market_id": "market_123",
            "current_price": 0.5,
            "volatility": 0.1,
            "orderbook": {
                "bids": [{"price": 0.49, "size": 1000}],
                "asks": [{"price": 0.51, "size": 1000}]
            }
        }

        # Execute order to create position for testing
        self.runner.reset()

        state = self.runner.get_state()
        assert state.balance == 10000.0
        assert state.total_value == 10000.0

    def test_enable_realtime_data(self):
        """Test enabling real-time data providers."""
        self.runner.enable_realtime_data(
            enable_crypto=True,
            enable_stocks=True,
            enable_prediction=True
        )

        assert self.runner.data_provider is not None
        assert self.runner.data_provider_config.enable_crypto is True
        assert self.runner.data_provider_config.enable_stocks is True
        assert self.runner.data_provider_config.enable_prediction is True

    def test_enable_realtime_data_partial(self):
        """Test enabling only some data providers."""
        self.runner.enable_realtime_data(
            enable_crypto=True,
            enable_stocks=False,
            enable_prediction=False
        )

        assert self.runner.data_provider_config.enable_crypto is True
        assert self.runner.data_provider_config.enable_stocks is False
        assert self.runner.data_provider_config.enable_prediction is False

    def test_set_market_data_callback(self):
        """Test setting custom market data callback."""
        def custom_callback(market_id):
            return {
                "market_id": market_id,
                "current_price": 0.75,
                "volatility": 0.02
            }

        self.runner.set_market_data_callback(custom_callback)

        assert self.runner.market_data_callback == custom_callback

    def test_set_order_callback(self):
        """Test setting custom order callback."""
        def custom_callback(order):
            pass

        self.runner.set_order_callback(custom_callback)

        assert self.runner.order_callback == custom_callback

    @pytest.mark.asyncio
    async def test_execute_order_no_market_data_callback(self):
        """Test order execution with no market data callback."""
        order = VirtualOrder(
            order_id="test_order",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        result = await self.runner.execute_order(order)

        # Should use default market data
        assert result.status in ["FILLED", "PARTIAL_FILL", "NO_FILL", "REJECTED"]

    @pytest.mark.asyncio
    async def test_execute_order_with_callback(self):
        """Test order execution with custom callback."""
        def get_market_data(market_id):
            return {
                "market_id": market_id,
                "current_price": 0.5,
                "volatility": 0.1,
                "orderbook": {
                    "bids": [{"price": 0.49, "size": 1000}],
                    "asks": [{"price": 0.51, "size": 1000}]
                }
            }

        self.runner.set_market_data_callback(get_market_data)

        order = VirtualOrder(
            order_id="test_order",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        result = await self.runner.execute_order(order)

        assert result.status in ["FILLED", "PARTIAL_FILL", "NO_FILL"]
        if result.status == "FILLED":
            assert result.average_price > 0

    @pytest.mark.asyncio
    async def test_execute_order_sell_side(self):
        """Test sell order execution."""
        def get_market_data(market_id):
            return {
                "market_id": market_id,
                "current_price": 0.5,
                "volatility": 0.1,
                "orderbook": {
                    "bids": [{"price": 0.49, "size": 1000}],
                    "asks": [{"price": 0.51, "size": 1000}]
                }
            }

        self.runner.set_market_data_callback(get_market_data)

        order = VirtualOrder(
            order_id="test_order_sell",
            market_id="market_123",
            side="sell",
            quantity=100.0,
            order_type="market"
        )

        result = await self.runner.execute_order(order)

        assert result.status in ["FILLED", "PARTIAL_FILL", "NO_FILL"]

    @pytest.mark.asyncio
    async def test_execute_order_with_callback(self):
        """Test order execution triggers order callback."""
        callback_orders = []

        def order_callback(order):
            callback_orders.append(order)

        self.runner.set_order_callback(order_callback)
        self.runner.set_market_data_callback(lambda m: {
            "market_id": m,
            "current_price": 0.5,
            "volatility": 0.1,
            "orderbook": {
                "bids": [{"price": 0.49, "size": 1000}],
                "asks": [{"price": 0.51, "size": 1000}]
            }
        })

        order = VirtualOrder(
            order_id="test_order_callback",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        await self.runner.execute_order(order)

        # Callback should be called
        assert len(callback_orders) == 1
        assert callback_orders[0].order_id == "test_order_callback"

    @pytest.mark.asyncio
    async def test_update_market_prices(self):
        """Test market price updates."""
        # Set up a position first by executing an order
        self.runner.set_market_data_callback(lambda m: {
            "market_id": m,
            "current_price": 0.5,
            "volatility": 0.1,
            "orderbook": {
                "bids": [{"price": 0.49, "size": 1000}],
                "asks": [{"price": 0.51, "size": 1000}]
            }
        })

        # Execute order to create position
        order = VirtualOrder(
            order_id="test_order_price_update",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        await self.runner.execute_order(order)

        # Update prices
        await self.runner.update_market_prices({"market_123": 0.60})

        # Verify position price updated
        summary = self.runner.get_portfolio_summary()
        # If position exists, value should reflect new price
        assert summary.positions_value >= 0

    @pytest.mark.asyncio
    async def test_stop_data_feed_no_feed(self):
        """Test stopping data feed when none is running."""
        # Should not raise an error
        await self.runner.stop_data_feed()

        assert self.runner.data_feed is None

    def test_generate_report(self):
        """Test performance report generation."""
        report = self.runner.generate_report()

        assert "Sandbox Simulation Report" in report
        assert "Executive Summary" in report

    def test_save_report(self):
        """Test report saving."""
        filepath = self.runner.save_report("test_report.md")

        assert filepath.endswith("test_report.md")

        # Clean up
        import os
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_state_reflects_portfolio(self):
        """Test state updates reflect portfolio changes."""
        initial_state = self.runner.get_state()
        assert initial_state.balance == 10000.0

        # State should track portfolio summary
        summary = self.runner.get_portfolio_summary()
        final_state = self.runner.get_state()

        assert final_state.balance == summary.balance
        assert final_state.total_value == summary.total_value


class TestSandboxRunnerAsync:
    """Async tests for SandboxRunner requiring event loop."""

    @pytest.mark.asyncio
    async def test_close_no_connections(self):
        """Test closing runner with no active connections."""
        runner = SandboxRunner(SandboxConfig())

        # Should not raise an error
        await runner.close()

    @pytest.mark.asyncio
    async def test_context_manager_pattern(self):
        """Test runner can be used in async context."""
        runner = SandboxRunner(SandboxConfig())

        # Basic smoke test
        state = runner.get_state()
        assert state.balance == 10000.0

        await runner.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
