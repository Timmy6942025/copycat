"""
Unit tests for sandbox simulation module.
Tests core functionality including order execution, portfolio management, and performance tracking.
"""

import unittest
import asyncio
from datetime import datetime
from sandbox.models import (
    SandboxConfig, VirtualOrder, VirtualOrderResult, OrderStatus, OrderSide,
    MarketData, OrderType
)
from sandbox.executors import VirtualOrderExecutor
from sandbox.managers import VirtualPortfolioManager
from sandbox.trackers import PerformanceTracker


class TestVirtualOrderExecutor(unittest.TestCase):
    """Tests for VirtualOrderExecutor."""

    def setUp(self):
        self.config = SandboxConfig()
        self.executor = VirtualOrderExecutor(self.config)

    def test_execute_market_order(self):
        """Test executing a market order."""
        order = VirtualOrder(
            market_id="test_market",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            outcome="YES"
        )

        market_data = MarketData(
            market_id="test_market",
            current_price=0.50,
            previous_price=0.50
        )

        # Run async test
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.executor.execute_virtual_order(order, market_data)
            )
            self.assertIn(result.status, [OrderStatus.FILLED, OrderStatus.NO_FILL])
        finally:
            loop.close()

    def test_order_validation(self):
        """Test order validation."""
        # Test minimum order size
        order = VirtualOrder(
            market_id="test_market",
            side=OrderSide.BUY,
            quantity=0.1,  # Below minimum
            outcome="YES"
        )

        market_data = MarketData(market_id="test_market", current_price=0.50)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.executor.execute_virtual_order(order, market_data)
            )
            self.assertEqual(result.status, OrderStatus.REJECTED)
        finally:
            loop.close()


class TestVirtualPortfolioManager(unittest.TestCase):
    """Tests for VirtualPortfolioManager."""

    def setUp(self):
        self.config = SandboxConfig()
        self.manager = VirtualPortfolioManager(self.config)
        self.manager.initialize_portfolio(10000.0)

    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.manager.get_balance(), 10000.0)
        self.assertEqual(self.manager.get_total_value(), 10000.0)

    def test_get_portfolio_summary(self):
        """Test portfolio summary generation."""
        summary = self.manager.get_portfolio_summary()
        self.assertEqual(summary.balance, 10000.0)
        self.assertEqual(summary.total_value, 10000.0)
        self.assertEqual(summary.position_count, 0)


class TestPerformanceTracker(unittest.TestCase):
    """Tests for PerformanceTracker."""

    def setUp(self):
        self.tracker = PerformanceTracker(initial_balance=10000.0)

    def test_record_trade(self):
        """Test recording a trade."""
        from sandbox.models import VirtualTrade

        trade = VirtualTrade(
            trade_id="test_trade_1",
            order_id="order_1",
            market_id="market_1",
            side=OrderSide.BUY,
            quantity=100.0,
            entry_price=0.50,
            profit=10.0,  # Winning trade
            fees=0.01
        )

        self.tracker.record_trade(trade)
        self.assertEqual(len(self.tracker.trades), 1)

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = self.tracker.calculate_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.starting_balance, 10000.0)


class TestSandboxConfig(unittest.TestCase):
    """Tests for SandboxConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()
        self.assertEqual(config.initial_balance, 10000.0)
        self.assertEqual(config.max_position_size_pct, 0.10)
        self.assertEqual(config.simulate_slippage, True)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SandboxConfig(
            initial_balance=50000.0,
            max_position_size_pct=0.20,
            simulate_fees=False
        )
        self.assertEqual(config.initial_balance, 50000.0)
        self.assertEqual(config.max_position_size_pct, 0.20)
        self.assertEqual(config.simulate_fees, False)


if __name__ == '__main__':
    unittest.main()
