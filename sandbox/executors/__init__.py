"""
Virtual Order Executor - Simulates order execution with realistic slippage, fees, and fill probability.
Mimics actual exchange behavior without using real funds.
"""

import random
from datetime import datetime
from typing import Dict, Optional
from sandbox.models import (
    SandboxConfig, VirtualOrder, VirtualOrderResult, VirtualOrderResult,
    OrderBook, MarketData, ValidationResult, OrderStatus
)


class VirtualOrderExecutor:
    """Simulates order execution with realistic slippage, fees, and fill probability."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.orderbook_cache: Dict[str, OrderBook] = {}
        self.fill_probability_cache: Dict[str, float] = {}

    async def execute_virtual_order(
        self,
        order: VirtualOrder,
        current_market_data: MarketData
    ) -> VirtualOrderResult:
        """Execute a virtual order with realistic simulation."""
        # Step 1: Constraint Validation
        validation_result = self._validate_order_constraints(order)
        if not validation_result.is_valid:
            return VirtualOrderResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=validation_result.reason,
                filled_quantity=0,
                average_price=0,
                total_fees=0,
                slippage=0,
                timestamp=datetime.utcnow()
            )

        # Step 2: Get Market Data
        orderbook = self._get_orderbook(order.market_id, current_market_data)
        if orderbook is None:
            return VirtualOrderResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                rejection_reason="No market data available",
                filled_quantity=0,
                average_price=0,
                total_fees=0,
                slippage=0,
                timestamp=datetime.utcnow()
            )

        # Step 3: Calculate Fill Probability
        fill_probability = self._calculate_fill_probability(
            order=order,
            orderbook=orderbook,
            market_data=current_market_data
        )

        # Step 4: Determine Execution Outcome
        if random.random() > fill_probability:
            status = OrderStatus.PARTIAL_FILL if random.random() > 0.5 else OrderStatus.NO_FILL
            return VirtualOrderResult(
                order_id=order.order_id,
                status=status,
                rejection_reason="Order did not fill (simulated)",
                filled_quantity=0,
                average_price=0,
                total_fees=0,
                slippage=0,
                timestamp=datetime.utcnow()
            )

        # Step 5: Calculate Execution Price with Slippage
        execution_price = self._calculate_execution_price(
            order=order,
            orderbook=orderbook,
            side=order.side
        )

        # Step 6: Apply Fees
        fees = self._calculate_fees(
            order=order,
            execution_price=execution_price
        )

        # Step 7: Calculate Slippage
        slippage = self._calculate_slippage(
            order=order,
            reference_price=current_market_data.current_price,
            execution_price=execution_price
        )

        # Step 8: Determine Final Fill Quantity
        fill_quantity = self._calculate_fill_quantity(
            order=order,
            orderbook=orderbook,
            side=order.side
        )

        return VirtualOrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=fill_quantity,
            average_price=execution_price,
            total_fees=fees,
            slippage=slippage,
            timestamp=datetime.utcnow()
        )

    def _validate_order_constraints(self, order: VirtualOrder) -> ValidationResult:
        """Validate order against sandbox constraints."""
        # Check minimum order size
        if order.quantity < self.config.min_order_size:
            return ValidationResult(
                is_valid=False,
                reason=f"Order size ${order.quantity:.2f} below minimum ${self.config.min_order_size:.2f}"
            )

        # Check position size limit
        max_position_size = self.config.initial_balance * self.config.max_position_size_pct
        if order.quantity > max_position_size:
            return ValidationResult(
                is_valid=False,
                reason=f"Order size ${order.quantity:.2f} exceeds maximum ${max_position_size:.2f}"
            )

        return ValidationResult(is_valid=True)

    def _get_orderbook(self, market_id: str, market_data: MarketData) -> Optional[OrderBook]:
        """Get or create order book for market."""
        if market_id in self.orderbook_cache:
            return self.orderbook_cache[market_id]

        # Create synthetic orderbook based on market data
        mid_price = market_data.current_price
        spread = mid_price * 0.02  # 2% spread
        bid_depth = self.config.initial_balance * 0.1  # 10% of portfolio as depth
        ask_depth = self.config.initial_balance * 0.1

        orderbook = OrderBook(
            market_id=market_id,
            best_bid=mid_price - spread / 2,
            best_ask=mid_price + spread / 2,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            mid_price=mid_price,
            spread=spread,
            last_updated=datetime.utcnow()
        )

        self.orderbook_cache[market_id] = orderbook
        return orderbook

    def _calculate_fill_probability(
        self,
        order: VirtualOrder,
        orderbook: OrderBook,
        market_data: MarketData
    ) -> float:
        """Calculate probability of order filling based on market conditions."""
        if not self.config.simulate_fill_probability:
            return 1.0

        # Get liquidity at or better than limit price
        if order.order_type.value == "market":
            liquidity = self._get_available_liquidity(orderbook, order.side.value)
            fill_prob = min(1.0, liquidity / order.quantity)
        else:
            limit_price = order.limit_price
            best_price = self._get_best_price(orderbook, order.side.value)

            if order.side.value == "buy":
                if limit_price >= best_price:
                    fill_prob = self._estimate_limit_fill_probability(
                        orderbook, order, distance_from_best=limit_price - best_price
                    )
                else:
                    fill_prob = 0.3
            else:
                if limit_price <= best_price:
                    fill_prob = self._estimate_limit_fill_probability(
                        orderbook, order, distance_from_best=best_price - limit_price
                    )
                else:
                    fill_prob = 0.3

        return fill_prob

    def _get_available_liquidity(self, orderbook: OrderBook, side: str) -> float:
        """Get available liquidity at best price."""
        if side == "buy":
            return orderbook.ask_depth
        else:
            return orderbook.bid_depth

    def _get_best_price(self, orderbook: OrderBook, side: str) -> float:
        """Get best price for side."""
        if side == "buy":
            return orderbook.best_ask
        else:
            return orderbook.best_bid

    def _estimate_limit_fill_probability(
        self,
        orderbook: OrderBook,
        order: VirtualOrder,
        distance_from_best: float
    ) -> float:
        """Estimate fill probability for limit orders."""
        base_prob = 0.9

        # Reduce probability for limit orders further from market
        if distance_from_best > 0.05:  # More than 5% away
            base_prob = 0.5
        elif distance_from_best > 0.02:  # More than 2% away
            base_prob = 0.7
        elif distance_from_best > 0.01:  # More than 1% away
            base_prob = 0.8

        # Adjust based on volatility
        volatility_factor = max(0.5, 1.0 - abs(distance_from_best) * 5)

        return base_prob * volatility_factor

    def _calculate_execution_price(
        self,
        order: VirtualOrder,
        orderbook: OrderBook,
        side: str
    ) -> float:
        """Calculate execution price with slippage."""
        if order.order_type.value == "market":
            # Market orders execute at best available price
            if side == "buy":
                return orderbook.best_ask
            else:
                return orderbook.best_bid
        else:
            # Limit orders execute at limit price or better
            if side == "buy":
                return min(order.limit_price, orderbook.best_ask)
            else:
                return max(order.limit_price, orderbook.best_bid)

    def _calculate_fees(self, order: VirtualOrder, execution_price: float) -> float:
        """Calculate fees for order."""
        if not self.config.simulate_fees:
            return 0.0

        # Polymarket-style fees: 0% on order flow, small flat fee
        fee_rate = 0.0
        flat_fee = 0.0

        if self.config.fee_model == "polymarket":
            fee_rate = 0.0
            flat_fee = 0.01  # 1 cent per trade
        elif self.config.fee_model == "kalshi":
            fee_rate = 0.0
            flat_fee = 0.02  # 2 cents per trade

        total_value = order.quantity
        fees = total_value * fee_rate + flat_fee

        return min(fees, total_value * 0.01)  # Cap at 1%

    def _calculate_slippage(
        self,
        order: VirtualOrder,
        reference_price: float,
        execution_price: float
    ) -> float:
        """Calculate slippage for order."""
        if not self.config.simulate_slippage or reference_price == 0:
            return 0.0

        if reference_price > 0:
            return abs(execution_price - reference_price) / reference_price
        return 0.0

    def _calculate_fill_quantity(
        self,
        order: VirtualOrder,
        orderbook: OrderBook,
        side: str
    ) -> float:
        """Calculate fill quantity based on order size and liquidity."""
        if side == "buy":
            available = orderbook.ask_depth
        else:
            available = orderbook.bid_depth

        # Fill up to available liquidity
        return min(order.quantity, available)

    def update_orderbook(self, market_id: str, orderbook: OrderBook):
        """Update orderbook cache."""
        self.orderbook_cache[market_id] = orderbook
