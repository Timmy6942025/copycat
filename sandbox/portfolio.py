"""
Virtual Portfolio Manager for Sandbox Simulation.

Manages virtual portfolio, positions, P&L, and risk metrics.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sandbox.config import (
    SandboxConfig,
    VirtualOrder,
    VirtualOrderResult,
    VirtualPosition,
    VirtualTrade,
    VirtualTradeResult,
    PortfolioSummary,
    ConstraintValidation,
)


class VirtualPortfolioManager:
    """
    Manages the virtual portfolio for sandbox simulation.
    Tracks positions, P&L, and risk metrics.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.positions: Dict[str, VirtualPosition] = {}
        self.balance = config.initial_balance
        self.pending_orders: List[VirtualOrder] = []
        self.completed_trades: List[VirtualTrade] = []

    async def execute_copy_trade(
        self,
        source_trade: dict,
        trader_config: dict
    ) -> VirtualTradeResult:
        """
        Execute a copy trade in virtual portfolio.

        Process:
        1. Calculate position size based on trader config
        2. Create virtual order
        3. Execute order through virtual executor
        4. Update portfolio state
        5. Record trade for analytics
        """
        # Step 1: Calculate Position Size
        position_size = self._calculate_position_size(
            source_trade=source_trade,
            trader_config=trader_config
        )

        # Step 2: Validate Against Constraints
        validation = self._validate_trade_constraints(position_size)
        if not validation.is_valid:
            return VirtualTradeResult(
                status="REJECTED",
                reason=validation.reason,
                source_trade_id=source_trade.get("trade_id"),
                position_size=0
            )

        # Step 3: Create Virtual Order
        virtual_order = VirtualOrder(
            order_id=f"sim_{source_trade.get('trade_id', uuid.uuid4().hex[:8])}_{uuid.uuid4().hex[:8]}",
            market_id=source_trade.get("market_id"),
            side="buy" if source_trade.get("outcome") == "YES" else "sell",
            quantity=position_size,
            order_type="market",  # Copy trades use market for immediate execution
            limit_price=None,
            timestamp=datetime.utcnow(),
            source_trade_id=source_trade.get("trade_id"),
            source_trader=source_trade.get("trader_address")
        )

        # Step 4: Execute Order (simplified synchronous execution for simulation)
        from sandbox.executor import VirtualOrderExecutor
        executor = VirtualOrderExecutor(self.config)
        market_data = source_trade.get("market_data", {})
        exec_result = await executor.execute_virtual_order(virtual_order, market_data)

        # Step 5: Update Portfolio
        if exec_result.status == "FILLED":
            self._update_portfolio(exec_result, source_trade)
            self._record_trade(exec_result, source_trade)

        return VirtualTradeResult(
            status=exec_result.status,
            reason=exec_result.rejection_reason if exec_result.status != "FILLED" else None,
            source_trade_id=source_trade.get("trade_id"),
            position_size=exec_result.filled_quantity,
            execution_price=exec_result.average_price,
            fees=exec_result.total_fees,
            slippage=exec_result.slippage,
            timestamp=exec_result.timestamp,
            execution_id=exec_result.order_id
        )

    def _calculate_position_size(
        self,
        source_trade: dict,
        trader_config: dict
    ) -> float:
        """
        Calculate virtual position size for copy trade.

        Methods:
        - Fixed Amount: Fixed $ amount per trade
        - Percentage of Portfolio: % of total portfolio value
        - Scaled by Trader Confidence: Size scales with trader score
        - Kelly Criterion: Optimal size based on edge and odds
        """
        position_sizing_method = trader_config.get("position_sizing_method", "fixed_amount")

        if position_sizing_method == "fixed_amount":
            return trader_config.get("base_position_size", 100.0)
        elif position_sizing_method == "percentage":
            portfolio_value = self._calculate_portfolio_value()
            return portfolio_value * trader_config.get("position_size_pct", 0.01)
        elif position_sizing_method == "scaled":
            portfolio_value = self._calculate_portfolio_value()
            base_size = portfolio_value * trader_config.get("position_size_pct", 0.01)
            # Scale by trader confidence (0.5 to 1.0)
            confidence_multiplier = 0.5 + (trader_config.get("trader_score", 0.5) / 2)
            return base_size * confidence_multiplier
        elif position_sizing_method == "kelly":
            # Kelly Criterion: f* = (bp - q) / b
            # Where: b = odds, p = win probability, q = loss probability
            win_rate = trader_config.get("estimated_win_rate", 0.5)
            profit_loss_ratio = trader_config.get("estimated_profit_loss_ratio", 1.0)
            kelly_fraction = trader_config.get("kelly_fraction", 0.25)  # Typically 0.25-0.5 for safety

            if profit_loss_ratio <= 0:
                return 0

            kelly = ((profit_loss_ratio * win_rate) - (1 - win_rate)) / profit_loss_ratio
            kelly = max(0, kelly) * kelly_fraction  # Apply fractional Kelly

            portfolio_value = self._calculate_portfolio_value()
            return portfolio_value * kelly
        else:
            return trader_config.get("base_position_size", 100.0)

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value (balance + positions)."""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return self.balance + positions_value

    def _validate_trade_constraints(
        self,
        position_size: float
    ) -> ConstraintValidation:
        """Validate trade against sandbox constraints."""
        portfolio_value = self._calculate_portfolio_value()

        # Check minimum order size
        if position_size < self.config.min_order_size:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Order size ${position_size:.2f} below minimum ${self.config.min_order_size:.2f}"
            )

        # Check max position size
        max_position_value = portfolio_value * self.config.max_position_size_pct
        if position_size > max_position_value:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Position size ${position_size:.2f} exceeds max ${max_position_value:.2f} ({self.config.max_position_size_pct:.0%})"
            )

        # Check total exposure
        current_exposure = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        total_exposure_after = current_exposure + position_size
        max_exposure = portfolio_value * self.config.max_total_exposure_pct
        if total_exposure_after > max_exposure:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Total exposure ${total_exposure_after:.2f} would exceed max ${max_exposure:.2f} ({self.config.max_total_exposure_pct:.0%})"
            )

        # Check balance
        if position_size > self.balance:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Insufficient balance: ${self.balance:.2f} < ${position_size:.2f}"
            )

        return ConstraintValidation(is_valid=True)

    def _update_portfolio(
        self,
        exec_result: VirtualOrderResult,
        source_trade: dict
    ):
        """Update portfolio with executed trade."""
        market_id = source_trade.get("market_id", "unknown")
        outcome = source_trade.get("outcome", "YES")

        # Deduct balance for buy or add for sell
        total_cost = exec_result.filled_quantity * exec_result.average_price + exec_result.total_fees

        if outcome == "YES":
            # Buying YES outcome
            self.balance -= total_cost

            if market_id in self.positions:
                position = self.positions[market_id]
                # Average in
                total_qty = position.quantity + exec_result.filled_quantity
                avg_price = (
                    (position.quantity * position.avg_price) +
                    (exec_result.filled_quantity * exec_result.average_price)
                ) / total_qty
                position.quantity = total_qty
                position.avg_price = avg_price
            else:
                self.positions[market_id] = VirtualPosition(
                    market_id=market_id,
                    outcome="YES",
                    quantity=exec_result.filled_quantity,
                    avg_price=exec_result.average_price,
                    current_price=exec_result.average_price,
                    unrealized_pnl=0,
                    timestamp=datetime.utcnow()
                )
        else:
            # Selling YES (buying NO)
            # For simplicity, track YES positions only; NO is inverse
            self.balance += total_cost  # Closing or reducing YES position


        self._update_position_prices()

    def _record_trade(
        self,
        exec_result: VirtualOrderResult,
        source_trade: dict
    ):
        """Record completed trade for analytics."""
        market_id = source_trade.get("market_id", "unknown")
        outcome = source_trade.get("outcome", "YES")

        trade = VirtualTrade(
            trade_id=exec_result.order_id,
            market_id=market_id,
            outcome=outcome,
            quantity=exec_result.filled_quantity,
            entry_price=exec_result.average_price,
            exit_price=None,
            profit=0,  # Will be calculated when position is closed
            roi=0,
            timestamp=exec_result.timestamp,
            source_trader=source_trade.get("trader_address"),
            fees=exec_result.total_fees,
            slippage=exec_result.slippage
        )
        self.completed_trades.append(trade)

    def update_market_prices(self, market_updates: Dict[str, float]):
        """Update current prices for all positions."""
        for market_id, new_price in market_updates.items():
            if market_id in self.positions:
                position = self.positions[market_id]
                position.current_price = new_price
                if position.outcome == "YES":
                    position.unrealized_pnl = (new_price - position.avg_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.avg_price - new_price) * position.quantity

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Generate portfolio summary with all metrics."""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        unrealized_pnl = sum(
            pos.unrealized_pnl
            for pos in self.positions.values()
        )

        total_value = self.balance + positions_value
        exposure_pct = positions_value / total_value if total_value > 0 else 0

        return PortfolioSummary(
            balance=self.balance,
            positions_value=positions_value,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            position_count=len(self.positions),
            exposure_pct=exposure_pct
        )

    def close_position(self, market_id: str, exit_price: float) -> Optional[VirtualTrade]:
        """Close a position and return the realized trade."""
        if market_id not in self.positions:
            return None

        position = self.positions[market_id]
        quantity = position.quantity

        # Calculate profit
        if position.outcome == "YES":
            profit = (exit_price - position.avg_price) * quantity
        else:
            profit = (position.avg_price - exit_price) * quantity

        roi = profit / (position.avg_price * quantity) if position.avg_price * quantity > 0 else 0

        # Create trade record
        trade = VirtualTrade(
            trade_id=f"close_{market_id}_{uuid.uuid4().hex[:8]}",
            market_id=market_id,
            outcome=position.outcome,
            quantity=quantity,
            entry_price=position.avg_price,
            exit_price=exit_price,
            profit=profit,
            roi=roi,
            timestamp=position.timestamp,
            exit_timestamp=datetime.utcnow(),
            hold_time_hours=None  # Could calculate if needed
        )


        if position.outcome == "YES":
            # Closing YES position
            self.balance += quantity * exit_price
        else:
            # Closing NO position (buying YES)
            self.balance -= quantity * exit_price


        del self.positions[market_id]
        self.completed_trades.append(trade)

        return trade
