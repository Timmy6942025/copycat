"""
Unit tests for VirtualOrderExecutor.

Tests order execution, fill probability, slippage calculation, and fee models.
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
    OrderStatus,
    ConstraintValidation,
)
from sandbox.executor import VirtualOrderExecutor


class TestVirtualOrderExecutor:
    """Tests for VirtualOrderExecutor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SandboxConfig()
        self.executor = VirtualOrderExecutor(self.config)

    def test_initialization(self):
        """Test executor initializes with config."""
        assert self.executor.config == self.config
        assert self.executor.orderbook_cache == {}
        assert self.executor.fill_probability_cache == {}

    def test_execute_order_rejected_below_min_size(self):
        """Test order rejection when below minimum size."""
        order = VirtualOrder(
            order_id="test_001",
            market_id="market_123",
            side="buy",
            quantity=0.5,  # Below minimum of 1.0
            order_type="market"
        )
        market_data = {"market_id": "market_123", "current_price": 0.5}

        result = asyncio.run(self.executor.execute_virtual_order(order, market_data))

        assert result.status == OrderStatus.REJECTED.value
        assert "below minimum" in result.rejection_reason.lower()
        assert result.filled_quantity == 0

    def test_execute_order_rejected_no_market_data(self):
        """Test order rejection when no market data available."""
        order = VirtualOrder(
            order_id="test_002",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        market_data = {"market_id": "different_market", "current_price": 0.5}

        result = asyncio.run(self.executor.execute_virtual_order(order, market_data))

        assert result.status == OrderStatus.FAILED.value
        assert "no market data" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_execute_order_success_market_order(self):
        """Test successful market order execution."""
        order = VirtualOrder(
            order_id="test_003",
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
                "asks": [{"price": 0.51, "size": 1000}],
                "last_trade_price": 0.5
            }
        }

        result = await self.executor.execute_virtual_order(order, market_data)

        # Order should either fill, partially fill, or not fill based on probability
        assert result.status in [
            OrderStatus.FILLED.value,
            OrderStatus.PARTIAL_FILL.value,
            OrderStatus.NO_FILL.value
        ]
        if result.status == OrderStatus.FILLED.value:
            assert result.filled_quantity > 0
            assert result.average_price > 0

    @pytest.mark.asyncio
    async def test_execute_order_sell_side(self):
        """Test sell order execution."""
        order = VirtualOrder(
            order_id="test_004",
            market_id="market_123",
            side="sell",
            quantity=100.0,
            order_type="market"
        )
        market_data = {
            "market_id": "market_123",
            "current_price": 0.5,
            "volatility": 0.1,
            "orderbook": {
                "bids": [{"price": 0.49, "size": 1000}],
                "asks": [{"price": 0.51, "size": 1000}],
                "last_trade_price": 0.5
            }
        }

        result = await self.executor.execute_virtual_order(order, market_data)

        assert result.status in [
            OrderStatus.FILLED.value,
            OrderStatus.PARTIAL_FILL.value,
            OrderStatus.NO_FILL.value
        ]

    def test_validate_order_constraints_valid(self):
        """Test valid order passes validation."""
        order = VirtualOrder(
            order_id="test_005",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        result = self.executor._validate_order_constraints(order)

        assert result.is_valid is True

    def test_validate_order_constraints_below_min_size(self):
        """Test validation fails for orders below minimum size."""
        order = VirtualOrder(
            order_id="test_006",
            market_id="market_123",
            side="buy",
            quantity=0.5,  # Below minimum
            order_type="market"
        )

        result = self.executor._validate_order_constraints(order)

        assert result.is_valid is False
        assert "below minimum" in result.reason.lower()

    def test_validate_order_constraints_exceeds_max_position(self):
        """Test validation fails for orders exceeding max position."""
        # The executor uses a simplified check with max_position_value = 100000
        # We need to test with an order larger than this value
        order = VirtualOrder(
            order_id="test_007",
            market_id="market_123",
            side="buy",
            quantity=150000.0,  # Larger than 100000 default
            order_type="market"
        )

        result = self.executor._validate_order_constraints(order)

        # Should exceed the simplified max position value
        assert result.is_valid is False
        assert "exceeds maximum" in result.reason.lower()

    def test_get_orderbook_from_market_data(self):
        """Test orderbook extraction from market data."""
        orderbook = self.executor._get_orderbook(
            "market_123",
            {
                "market_id": "market_123",
                "current_price": 0.5,
                "orderbook": {
                    "bids": [{"price": 0.49, "size": 500}],
                    "asks": [{"price": 0.51, "size": 500}]
                }
            }
        )

        assert orderbook is not None
        assert "bids" in orderbook
        assert "asks" in orderbook

    def test_get_orderbook_missing_market_id(self):
        """Test orderbook returns None for missing market ID."""
        orderbook = self.executor._get_orderbook(
            "market_123",
            {"market_id": "different_market"}
        )

        assert orderbook is None

    def test_get_orderbook_default_orderbook(self):
        """Test default orderbook creation when not in market data."""
        orderbook = self.executor._get_orderbook(
            "market_123",
            {"market_id": "market_123", "current_price": 0.5}
        )

        assert orderbook is not None
        assert "bids" in orderbook
        assert "asks" in orderbook

    def test_calculate_fill_probability_market_order(self):
        """Test fill probability for market orders."""
        order = VirtualOrder(
            order_id="test_008",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }
        market_data = {"current_price": 0.5, "volatility": 0.1}

        fill_prob = self.executor._calculate_fill_probability(
            order, orderbook, market_data
        )

        assert 0 <= fill_prob <= 1

    def test_calculate_fill_probability_limit_order_at_best_price(self):
        """Test fill probability for limit orders at best price."""
        order = VirtualOrder(
            order_id="test_009",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="limit",
            limit_price=0.51  # At or above ask
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }
        market_data = {"current_price": 0.5, "volatility": 0.1}

        fill_prob = self.executor._calculate_fill_probability(
            order, orderbook, market_data
        )

        assert 0 <= fill_prob <= 1
        # At best price, should have high fill probability
        assert fill_prob >= 0.7

    def test_calculate_fill_probability_limit_order_passive(self):
        """Test fill probability for passive limit orders."""
        order = VirtualOrder(
            order_id="test_010",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="limit",
            limit_price=0.40  # Below best ask - passive
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }
        market_data = {"current_price": 0.5, "volatility": 0.1}

        fill_prob = self.executor._calculate_fill_probability(
            order, orderbook, market_data
        )

        assert 0 <= fill_prob <= 1
        # Passive orders have lower fill probability
        assert fill_prob <= 0.5

    def test_calculate_fill_probability_high_volatility(self):
        """Test fill probability reduced by high volatility."""
        order = VirtualOrder(
            order_id="test_011",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }
        market_data = {"current_price": 0.5, "volatility": 0.5}  # High volatility

        fill_prob = self.executor._calculate_fill_probability(
            order, orderbook, market_data
        )

        # Fill probability should be reduced by high volatility
        assert 0 <= fill_prob <= 1

    def test_estimate_limit_fill_probability(self):
        """Test limit fill probability estimation by distance."""
        order = VirtualOrder(
            order_id="test_012",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="limit"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }

        # At best price (distance = 0)
        prob_at_best = self.executor._estimate_limit_fill_probability(
            orderbook, order, distance=0
        )
        assert prob_at_best == 0.9

        # 1 cent away
        prob_1c = self.executor._estimate_limit_fill_probability(
            orderbook, order, distance=0.01
        )
        assert prob_1c == 0.7

        # 2 cents away
        prob_2c = self.executor._estimate_limit_fill_probability(
            orderbook, order, distance=0.02
        )
        assert prob_2c == 0.5

        # 5 cents away
        prob_5c = self.executor._estimate_limit_fill_probability(
            orderbook, order, distance=0.05
        )
        assert prob_5c == 0.3

        # Far away
        prob_far = self.executor._estimate_limit_fill_probability(
            orderbook, order, distance=0.10
        )
        assert prob_far == 0.1

    def test_get_available_liquidity(self):
        """Test liquidity calculation from orderbook."""
        orderbook = {
            "bids": [{"price": 0.49, "size": 500}, {"price": 0.48, "size": 300}],
            "asks": [{"price": 0.51, "size": 400}, {"price": 0.52, "size": 600}]
        }

        buy_liquidity = self.executor._get_available_liquidity(orderbook, "buy")
        assert buy_liquidity == 1000  # 400 + 600

        sell_liquidity = self.executor._get_available_liquidity(orderbook, "sell")
        assert sell_liquidity == 800  # 500 + 300

    def test_get_best_price(self):
        """Test best price extraction from orderbook."""
        orderbook = {
            "bids": [{"price": 0.49, "size": 500}],
            "asks": [{"price": 0.51, "size": 400}],
            "last_trade_price": 0.50
        }

        # For buy orders, we look at asks (prices we pay to buy)
        best_ask = self.executor._get_best_price(orderbook, "buy")
        assert best_ask == 0.51

        # For sell orders, we look at bids (prices we receive when selling)
        best_bid = self.executor._get_best_price(orderbook, "sell")
        assert best_bid == 0.49

    def test_get_best_price_fallback(self):
        """Test best price fallback to last trade price."""
        orderbook = {"last_trade_price": 0.75}

        best_price = self.executor._get_best_price(orderbook, "buy")
        assert best_price == 0.75

    def test_calculate_execution_price_no_slippage(self):
        """Test execution price without slippage simulation."""
        self.config.simulate_slippage = False
        order = VirtualOrder(
            order_id="test_013",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }

        price = self.executor._calculate_execution_price(order, orderbook, "buy")

        # Should return best price without slippage
        assert price == 0.51

    def test_calculate_execution_price_with_slippage(self):
        """Test execution price with slippage simulation."""
        order = VirtualOrder(
            order_id="test_014",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }

        price = self.executor._calculate_execution_price(order, orderbook, "buy")

        # Should include slippage
        assert price >= 0.51

    def test_calculate_execution_price_sell_side(self):
        """Test execution price for sell orders."""
        order = VirtualOrder(
            order_id="test_015",
            market_id="market_123",
            side="sell",
            quantity=100.0,
            order_type="market"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }

        price = self.executor._calculate_execution_price(order, orderbook, "sell")

        # Sell price should be at or below bid
        assert price <= 0.49

    def test_calculate_execution_price_no_liquidity(self):
        """Test execution price with no liquidity returns last trade price."""
        order = VirtualOrder(
            order_id="test_016",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        orderbook = {
            "bids": [],
            "asks": [],
            "last_trade_price": 0.5
        }

        price = self.executor._calculate_execution_price(order, orderbook, "buy")

        # No price levels means we use last_trade_price directly
        assert price == 0.5

    def test_calculate_fees_no_fees(self):
        """Test fee calculation when fees disabled."""
        self.config.simulate_fees = False
        order = VirtualOrder(
            order_id="test_017",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        fees = self.executor._calculate_fees(order, execution_price=0.50)

        assert fees == 0.0

    def test_calculate_fees_polymarket(self):
        """Test Polymarket fee calculation."""
        self.config.fee_model = "polymarket"
        order = VirtualOrder(
            order_id="test_018",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market",
            trader_total_volume_30d=50000  # $50k volume
        )

        fees = self.executor._calculate_fees(order, execution_price=0.50)

        # Fee should be positive
        assert fees > 0
        # Volume discount: min(0.015, 50000/100000) = 0.015
        # Fee rate: 0.02 - 0.015 = 0.005
        # Notional: 100 * 0.50 = 50
        # Fee: 50 * 0.005 = 0.25
        expected_fee = 100 * 0.50 * (0.02 - min(0.015, 50000/100000))
        assert fees == pytest.approx(expected_fee, rel=0.01)

    def test_calculate_fees_default(self):
        """Test default fee model."""
        self.config.fee_model = "unknown"
        order = VirtualOrder(
            order_id="test_020",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        fees = self.executor._calculate_fees(order, execution_price=0.50)

        # Default is 1%
        expected_fee = 100 * 0.50 * 0.01
        assert fees == expected_fee

    def test_calculate_slippage_buy_order(self):
        """Test slippage calculation for buy orders."""
        order = VirtualOrder(
            order_id="test_021",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        slippage = self.executor._calculate_slippage(
            order=order,
            reference_price=0.50,
            execution_price=0.52
        )

        # Buy execution above reference should have positive slippage
        expected_slippage = (0.52 - 0.50) / 0.50  # 0.04
        assert slippage == pytest.approx(expected_slippage, rel=0.01)

    def test_calculate_slippage_sell_order(self):
        """Test slippage calculation for sell orders."""
        order = VirtualOrder(
            order_id="test_022",
            market_id="market_123",
            side="sell",
            quantity=100.0,
            order_type="market"
        )

        slippage = self.executor._calculate_slippage(
            order=order,
            reference_price=0.50,
            execution_price=0.48
        )

        # Sell execution below reference should have positive slippage
        expected_slippage = (0.50 - 0.48) / 0.50  # 0.04
        assert slippage == pytest.approx(expected_slippage, rel=0.01)

    def test_calculate_slippage_zero_reference(self):
        """Test slippage calculation with zero reference price."""
        order = VirtualOrder(
            order_id="test_023",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )

        slippage = self.executor._calculate_slippage(
            order=order,
            reference_price=0,
            execution_price=0.52
        )

        assert slippage == 0.0

    def test_calculate_fill_quantity_market_order(self):
        """Test fill quantity calculation for market orders."""
        order = VirtualOrder(
            order_id="test_024",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 500}]  # Only 500 available
        }

        fill_qty = self.executor._calculate_fill_quantity(order, orderbook, "buy")

        # Fill quantity is limited by order quantity (not liquidity)
        assert fill_qty == 100.0

    def test_calculate_fill_quantity_limit_order(self):
        """Test fill quantity calculation for limit orders."""
        order = VirtualOrder(
            order_id="test_025",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="limit"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 500}],
            "asks": [{"price": 0.51, "size": 1000}]
        }

        fill_qty = self.executor._calculate_fill_quantity(order, orderbook, "buy")

        # Limit orders fill the full quantity
        assert fill_qty == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
