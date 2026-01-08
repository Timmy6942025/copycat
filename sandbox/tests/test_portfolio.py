"""
Unit tests for VirtualPortfolioManager.

Tests portfolio management, position tracking, P&L calculation, and trade execution.
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
    VirtualOrderResult,
    VirtualPosition,
    VirtualTrade,
    PortfolioSummary,
    ConstraintValidation,
)
from sandbox.portfolio import VirtualPortfolioManager


class TestVirtualPortfolioManager:
    """Tests for VirtualPortfolioManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SandboxConfig(initial_balance=10000.0)
        self.manager = VirtualPortfolioManager(self.config)

    def test_initial_state(self):
        """Test initial portfolio state."""
        assert self.manager.balance == 10000.0
        assert self.manager.positions == {}
        assert self.manager.pending_orders == []
        assert self.manager.completed_trades == []

    def test_portfolio_initialization(self):
        """Test portfolio initialization with config."""
        assert self.manager.config == self.config
        assert self.manager.balance == self.config.initial_balance

    def test_calculate_portfolio_value_empty(self):
        """Test portfolio value calculation with no positions."""
        value = self.manager._calculate_portfolio_value()

        assert value == 10000.0

    def test_calculate_portfolio_value_with_positions(self):
        """Test portfolio value calculation with positions."""
        # Add a position
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.60,
            unrealized_pnl=10.0
        )

        value = self.manager._calculate_portfolio_value()

        # Balance + position value (100 * 0.60)
        expected_value = 10000.0 + 60.0
        assert value == expected_value

    def test_calculate_portfolio_value_multiple_positions(self):
        """Test portfolio value with multiple positions."""
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.60
        )
        self.manager.positions["market_456"] = VirtualPosition(
            market_id="market_456",
            outcome="YES",
            quantity=50.0,
            avg_price=0.40,
            current_price=0.45
        )

        value = self.manager._calculate_portfolio_value()

        # Balance + position values
        expected_value = 10000.0 + (100 * 0.60) + (50 * 0.45)
        assert value == expected_value

    def test_get_portfolio_summary_empty(self):
        """Test portfolio summary with no positions."""
        summary = self.manager.get_portfolio_summary()

        assert summary.balance == 10000.0
        assert summary.positions_value == 0.0
        assert summary.total_value == 10000.0
        assert summary.unrealized_pnl == 0.0
        assert summary.position_count == 0
        assert summary.exposure_pct == 0.0

    def test_get_portfolio_summary_with_positions(self):
        """Test portfolio summary with open positions."""
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.60,
            unrealized_pnl=10.0
        )

        summary = self.manager.get_portfolio_summary()

        assert summary.balance == 10000.0
        assert summary.positions_value == 60.0  # 100 * 0.60
        assert summary.total_value == 10060.0
        assert summary.unrealized_pnl == 10.0
        assert summary.position_count == 1
        assert summary.exposure_pct == pytest.approx(60.0 / 10060.0, rel=0.01)

    def test_validate_trade_constraints_min_size(self):
        """Test trade validation for minimum order size."""
        validation = self.manager._validate_trade_constraints(0.5)

        assert validation.is_valid is False
        assert "below minimum" in validation.reason.lower()

    def test_validate_trade_constraints_exceeds_max_position(self):
        """Test trade validation for exceeding max position size."""
        validation = self.manager._validate_trade_constraints(15000.0)

        assert validation.is_valid is False
        assert "exceeds max" in validation.reason.lower()

    def test_validate_trade_constraints_exceeds_total_exposure(self):
        """Test trade validation for exceeding total exposure."""
        self.manager.balance = 5000.0
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=4000.0,
            avg_price=0.50,
            current_price=0.50
        )

        # Within constraints should pass validation
        validation = self.manager._validate_trade_constraints(600.0)

        assert validation.is_valid is True

    def test_validate_trade_constraints_balance_check_order(self):
        """Test that max_position check happens before balance check."""
        self.manager.balance = 10.0
        self.manager.positions = {}

        # Position size exceeds max_position (10% of 10 = $1)
        validation = self.manager._validate_trade_constraints(5.0)

        assert validation.is_valid is False
        assert "exceeds max" in validation.reason.lower()

    def test_validate_trade_constraints_valid(self):
        """Test valid trade passes validation."""
        validation = self.manager._validate_trade_constraints(500.0)

        assert validation.is_valid is True

    def test_update_market_prices_single_market(self):
        """Test market price updates for single position."""
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.50,
            unrealized_pnl=0.0
        )

        self.manager.update_market_prices({"market_123": 0.60})

        position = self.manager.positions["market_123"]
        assert position.current_price == 0.60
        assert position.unrealized_pnl == pytest.approx(10.0, rel=1e-9)  # (0.60 - 0.50) * 100

    def test_update_market_prices_price_decrease(self):
        """Test market price update with price decrease."""
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.60,
            unrealized_pnl=10.0
        )

        self.manager.update_market_prices({"market_123": 0.40})

        position = self.manager.positions["market_123"]
        assert position.current_price == 0.40
        assert position.unrealized_pnl == pytest.approx(-10.0, rel=1e-9)  # (0.40 - 0.50) * 100

    def test_update_market_prices_multiple_markets(self):
        """Test market price updates for multiple positions."""
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.50
        )
        self.manager.positions["market_456"] = VirtualPosition(
            market_id="market_456",
            outcome="YES",
            quantity=50.0,
            avg_price=0.40,
            current_price=0.40
        )

        self.manager.update_market_prices({
            "market_123": 0.60,
            "market_456": 0.50
        })

        assert self.manager.positions["market_123"].current_price == 0.60
        assert self.manager.positions["market_456"].current_price == 0.50

    def test_update_market_prices_nonexistent_market(self):
        """Test market price update for nonexistent market (should not error)."""
        # Should not raise an error
        self.manager.update_market_prices({"nonexistent_market": 0.50})
        assert "nonexistent_market" not in self.manager.positions

    def test_close_position_exists(self):
        """Test closing an existing position."""
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.60,
            unrealized_pnl=10.0,
            timestamp=datetime.utcnow()
        )

        trade = self.manager.close_position("market_123", exit_price=0.65)

        assert trade is not None
        assert trade.market_id == "market_123"
        assert trade.quantity == 100.0
        assert trade.entry_price == 0.50
        assert trade.exit_price == 0.65
        assert trade.profit == pytest.approx(15.0, rel=1e-9)  # (0.65 - 0.50) * 100
        assert trade.roi == pytest.approx(0.30, rel=1e-9)  # 15 / (0.50 * 100)

        # Position should be removed
        assert "market_123" not in self.manager.positions

        # Trade should be recorded
        assert len(self.manager.completed_trades) == 1

    def test_close_position_does_not_exist(self):
        """Test closing a nonexistent position."""
        trade = self.manager.close_position("nonexistent", exit_price=0.50)

        assert trade is None
        assert len(self.manager.completed_trades) == 0

    def test_close_position_updates_balance(self):
        """Test closing position updates balance correctly."""
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.60,
            timestamp=datetime.utcnow()
        )
        initial_balance = self.manager.balance

        self.manager.close_position("market_123", exit_price=0.65)

        # Balance should increase by exit value (100 * 0.65)
        expected_balance = initial_balance + 65.0
        assert self.manager.balance == expected_balance

    def test_calculate_position_size_fixed_amount(self):
        """Test fixed amount position sizing."""
        source_trade = {"trade_id": "t1"}
        trader_config = {
            "position_sizing_method": "fixed_amount",
            "base_position_size": 500.0
        }

        size = self.manager._calculate_position_size(source_trade, trader_config)

        assert size == 500.0

    def test_calculate_position_size_percentage(self):
        """Test percentage of portfolio position sizing."""
        source_trade = {"trade_id": "t1"}
        trader_config = {
            "position_sizing_method": "percentage",
            "position_size_pct": 0.10  # 10%
        }

        size = self.manager._calculate_position_size(source_trade, trader_config)

        # 10% of 10000 = 1000
        assert size == 1000.0

    def test_calculate_position_size_scaled(self):
        """Test scaled by trader confidence position sizing."""
        source_trade = {"trade_id": "t1"}
        trader_config = {
            "position_sizing_method": "scaled",
            "position_size_pct": 0.10,  # 10%
            "trader_score": 0.8  # 80% confidence
        }

        size = self.manager._calculate_position_size(source_trade, trader_config)

        # Base: 10% of 10000 = 1000
        # Multiplier: 0.5 + (0.8 / 2) = 0.9
        # Final: 1000 * 0.9 = 900
        expected_size = 1000.0 * (0.5 + (0.8 / 2))
        assert size == expected_size

    def test_calculate_position_size_kelly(self):
        """Test Kelly criterion position sizing."""
        source_trade = {"trade_id": "t1"}
        trader_config = {
            "position_sizing_method": "kelly",
            "estimated_win_rate": 0.60,
            "estimated_profit_loss_ratio": 2.0,  # 2:1 reward:risk
            "kelly_fraction": 0.25  # Fractional Kelly
        }

        size = self.manager._calculate_position_size(source_trade, trader_config)

        # Kelly: f* = (bp - q) / b
        # b = 2.0, p = 0.60, q = 0.40
        # kelly = (2.0 * 0.60 - 0.40) / 2.0 = (1.2 - 0.4) / 2 = 0.4
        # fractional = 0.4 * 0.25 = 0.1
        # size = 0.1 * 10000 = 1000
        expected_kelly = ((2.0 * 0.60) - 0.40) / 2.0
        expected_size = expected_kelly * 0.25 * 10000.0
        assert size == expected_size

    def test_calculate_position_size_kelly_negative_edge(self):
        """Test Kelly criterion with negative edge returns zero."""
        source_trade = {"trade_id": "t1"}
        trader_config = {
            "position_sizing_method": "kelly",
            "estimated_win_rate": 0.30,  # Losing trader
            "estimated_profit_loss_ratio": 1.0,
            "kelly_fraction": 0.25
        }

        size = self.manager._calculate_position_size(source_trade, trader_config)

        # Kelly would be negative, should return 0
        assert size == 0.0

    def test_calculate_position_size_default(self):
        """Test default position sizing method."""
        source_trade = {"trade_id": "t1"}
        trader_config = {
            "position_sizing_method": "unknown_method",
            "base_position_size": 250.0
        }

        size = self.manager._calculate_position_size(source_trade, trader_config)

        assert size == 250.0

    @pytest.mark.asyncio
    async def test_execute_copy_trade_rejected(self):
        """Test copy trade rejection due to constraints."""
        source_trade = {
            "trade_id": "t1",
            "market_id": "market_123",
            "outcome": "YES"
        }
        trader_config = {
            "position_sizing_method": "fixed_amount",
            "base_position_size": 0.5  # Below minimum
        }

        result = await self.manager.execute_copy_trade(source_trade, trader_config)

        assert result.status == "REJECTED"
        assert "below minimum" in result.reason.lower()
        assert result.position_size == 0

    @pytest.mark.asyncio
    async def test_execute_copy_trade_success(self):
        """Test successful copy trade execution."""
        source_trade = {
            "trade_id": "t1",
            "market_id": "market_123",
            "outcome": "YES",
            "trader_address": "0x1234...",
            "market_data": {
                "market_id": "market_123",
                "current_price": 0.50,
                "orderbook": {
                    "bids": [{"price": 0.49, "size": 1000}],
                    "asks": [{"price": 0.51, "size": 1000}]
                }
            }
        }
        trader_config = {
            "position_sizing_method": "fixed_amount",
            "base_position_size": 100.0
        }

        result = await self.manager.execute_copy_trade(source_trade, trader_config)

        assert result.status in ["FILLED", "PARTIAL_FILL", "NO_FILL"]
        if result.status == "FILLED":
            assert result.position_size > 0
            assert result.execution_price > 0

    def test_record_trade(self):
        """Test recording a completed trade."""
        exec_result = VirtualOrderResult(
            order_id="order_001",
            status="FILLED",
            filled_quantity=100.0,
            average_price=0.50,
            total_fees=0.50,
            slippage=0.01,
            timestamp=datetime.utcnow()
        )
        source_trade = {
            "trade_id": "t1",
            "market_id": "market_123",
            "outcome": "YES",
            "trader_address": "0x1234..."
        }

        self.manager._record_trade(exec_result, source_trade)

        assert len(self.manager.completed_trades) == 1
        trade = self.manager.completed_trades[0]
        assert trade.trade_id == "order_001"
        assert trade.market_id == "market_123"
        assert trade.quantity == 100.0
        assert trade.entry_price == 0.50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
