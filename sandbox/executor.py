"""
Virtual Order Executor for Sandbox Simulation.

Simulates order execution with realistic slippage, fees, and fill probability.
"""

import random
from datetime import datetime
from typing import Dict, Optional

from sandbox.config import (
    SandboxConfig,
    VirtualOrder,
    VirtualOrderResult,
    OrderStatus,
    ConstraintValidation,
)


class VirtualOrderExecutor:
    """
    Simulates order execution with realistic slippage, fees, and fill probability.
    Mimics actual exchange behavior without using real funds.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.orderbook_cache: Dict[str, dict] = {}
        self.fill_probability_cache: Dict[str, float] = {}

    async def execute_virtual_order(
        self,
        order: VirtualOrder,
        current_market_data: dict
    ) -> VirtualOrderResult:
        """
        Execute a virtual order with realistic simulation.

        Process:
        1. Validate order against sandbox constraints
        2. Calculate realistic fill probability
        3. Simulate slippage based on order size and market liquidity
        4. Apply platform fees
        5. Determine fill price and quantity
        6. Record execution details
        """
        # Step 1: Constraint Validation
        validation_result = self._validate_order_constraints(order)
        if not validation_result.is_valid:
            return VirtualOrderResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED.value,
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
                status=OrderStatus.FAILED.value,
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
            status = OrderStatus.PARTIAL_FILL.value if random.random() > 0.5 else OrderStatus.NO_FILL.value
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
            reference_price=current_market_data.get("current_price", execution_price),
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
            status=OrderStatus.FILLED.value,
            filled_quantity=fill_quantity,
            average_price=execution_price,
            total_fees=fees,
            slippage=slippage,
            timestamp=datetime.utcnow()
        )

    def _validate_order_constraints(self, order: VirtualOrder) -> ConstraintValidation:
        """Validate order against sandbox constraints."""
        # Check minimum order size
        if order.quantity < self.config.min_order_size:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Order size ${order.quantity:.2f} below minimum ${self.config.min_order_size:.2f}"
            )

        # Check max position size (simplified check)
        max_position_value = 100000  # Simplified: using a large default
        if order.quantity > max_position_value:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Order size ${order.quantity:.2f} exceeds maximum"
            )

        return ConstraintValidation(is_valid=True)

    def _get_orderbook(self, market_id: str, market_data: dict) -> Optional[dict]:
        """Get orderbook data for market."""
        # Return simulated orderbook from market data
        if market_data and market_id == market_data.get("market_id"):
            return market_data.get("orderbook", {
                "bids": [{"price": market_data.get("current_price", 0.5), "size": 1000}],
                "asks": [{"price": market_data.get("current_price", 0.5) + 0.01, "size": 1000}],
                "last_trade_price": market_data.get("current_price", 0.5)
            })
        return None

    def _calculate_fill_probability(
        self,
        order: VirtualOrder,
        orderbook: dict,
        market_data: dict
    ) -> float:
        """
        Calculate probability of order filling based on market conditions.

        Factors:
        - Order size vs available liquidity at price levels
        - Market volatility
        - Time priority (simulated)
        - Order type (market vs limit)
        """
        # Get liquidity at or better than limit price
        if order.order_type == "market":
            # Market orders fill immediately but may suffer slippage
            liquidity = self._get_available_liquidity(orderbook, order.side)
            fill_prob = min(1.0, liquidity / order.quantity) if order.quantity > 0 else 1.0
        else:
            # Limit orders: fill probability based on how "aggressive" the price is
            limit_price = order.limit_price
            best_price = self._get_best_price(orderbook, order.side)

            if order.side == "buy":
                # Buy limit: price must be at or below limit to fill
                if limit_price is not None and limit_price >= best_price:
                    # Aggressive: high fill probability
                    distance = limit_price - best_price
                    fill_prob = self._estimate_limit_fill_probability(
                        orderbook, order, distance=distance
                    )
                else:
                    # Passive: lower fill probability
                    fill_prob = 0.3  # 30% chance of fill for passive limit
            else:
                # Sell limit: price must be at or above limit to fill
                if limit_price is not None and limit_price <= best_price:
                    distance = best_price - limit_price
                    fill_prob = self._estimate_limit_fill_probability(
                        orderbook, order, distance=distance
                    )
                else:
                    fill_prob = 0.3

        # Adjust for volatility
        volatility = market_data.get("volatility", 0.0)
        volatility_factor = min(1.0, volatility / 0.5) if volatility else 0.0  # Cap at 50% volatility
        fill_prob *= (1 - volatility_factor * 0.2)  # Reduce fill probability in high volatility

        return max(0.0, min(1.0, fill_prob))

    def _estimate_limit_fill_probability(
        self,
        orderbook: dict,
        order: VirtualOrder,
        distance: float
    ) -> float:
        """Estimate fill probability for limit orders based on price distance."""
        if distance <= 0:
            return 0.9  # At or better than best price: high fill probability
        elif distance <= 0.01:
            return 0.7  # 1 cent away
        elif distance <= 0.02:
            return 0.5  # 2 cents away
        elif distance <= 0.05:
            return 0.3  # 5 cents away
        else:
            return 0.1  # Far from best price

    def _get_available_liquidity(self, orderbook: dict, side: str) -> float:
        """Get available liquidity at best prices."""
        if side == "buy":
            asks = orderbook.get("asks", [])
            return sum(a.get("size", 0) for a in asks)
        else:
            bids = orderbook.get("bids", [])
            return sum(b.get("size", 0) for b in bids)

    def _get_best_price(self, orderbook: dict, side: str) -> float:
        """Get best bid/ask price from orderbook."""
        if side == "buy":
            asks = orderbook.get("asks", [])
            if asks:
                return asks[0].get("price", 0.5)
        else:
            bids = orderbook.get("bids", [])
            if bids:
                return bids[0].get("price", 0.5)
        return orderbook.get("last_trade_price", 0.5)

    def _calculate_execution_price(
        self,
        order: VirtualOrder,
        orderbook: dict,
        side: str
    ) -> float:
        """
        Calculate realistic execution price with slippage.

        Uses volume-weighted average price (VWAP) based on order size
        and available liquidity at different price levels.
        """
        if not self.config.simulate_slippage:
            return self._get_best_price(orderbook, side)

        # Get price levels from orderbook
        if side == "buy":
            asks = sorted(orderbook.get("asks", []), key=lambda x: x.get("price", 0))
            price_levels = [(a.get("price", 0), a.get("size", 0)) for a in asks]
        else:
            bids = sorted(orderbook.get("bids", []), key=lambda x: -x.get("price", 0))
            price_levels = [(b.get("price", 0), b.get("size", 0)) for b in bids]

        if not price_levels:
            return orderbook.get("last_trade_price", 0.5)

        # Calculate VWAP based on order size
        remaining_qty = order.quantity
        total_cost = 0
        filled_qty = 0

        for price, size in price_levels:
            if remaining_qty <= 0:
                break

            fill_qty = min(remaining_qty, size)
            total_cost += fill_qty * price
            filled_qty += fill_qty
            remaining_qty -= fill_qty

        # If order exceeds available liquidity, use last price with penalty
        if filled_qty > 0:
            vwap = total_cost / filled_qty

            # Apply additional slippage for large orders
            size_ratio = order.quantity / filled_qty if filled_qty > 0 else 1.0
            size_penalty = min(0.05, (size_ratio - 1) * 0.01)
            if side == "buy":
                vwap *= (1 + size_penalty)
            else:
                vwap *= (1 - size_penalty)

            return vwap
        else:
            # No liquidity: use last resort price with large slippage
            last_price = orderbook.get("last_trade_price", 0.5)
            if side == "buy":
                return last_price * 1.1
            else:
                return last_price * 0.9

    def _calculate_fees(
        self,
        order: VirtualOrder,
        execution_price: float
    ) -> float:
        """Calculate platform fees based on simulated fee model."""
        if not self.config.simulate_fees:
            return 0.0

        notional_value = order.quantity * execution_price

        if self.config.fee_model == "polymarket":
            # Polymarket: ~0.5% to 2% depending on volume and market
            base_fee_rate = 0.02  # 2% maximum
            volume_discount = min(0.015, order.trader_total_volume_30d / 100000)
            fee_rate = base_fee_rate - volume_discount
        elif self.config.fee_model == "kalshi":
            # Kalshi: ~2% flat fee
            fee_rate = 0.02
        else:
            fee_rate = 0.01  # Default 1%

        return notional_value * fee_rate

    def _calculate_slippage(
        self,
        order: VirtualOrder,
        reference_price: float,
        execution_price: float
    ) -> float:
        """Calculate slippage as percentage difference from reference price."""
        if reference_price == 0:
            return 0.0

        if order.side == "buy":
            return (execution_price - reference_price) / reference_price
        else:
            return (reference_price - execution_price) / reference_price

    def _calculate_fill_quantity(
        self,
        order: VirtualOrder,
        orderbook: dict,
        side: str
    ) -> float:
        """Calculate fill quantity based on orderbook liquidity."""
        liquidity = self._get_available_liquidity(orderbook, side)
        return min(order.quantity, liquidity) if order.order_type == "market" else order.quantity
