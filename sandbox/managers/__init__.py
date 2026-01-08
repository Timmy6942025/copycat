"""
Virtual Portfolio Manager - Manages virtual portfolio state including positions, balance, and P&L.
Tracks all virtual positions and calculates portfolio metrics.
"""

from datetime import datetime
from typing import Dict, List, Optional
from sandbox.models import (
    SandboxConfig, SimulationState, VirtualOrder, VirtualPosition,
    VirtualTrade, PortfolioSummary, OrderStatus, OrderSide
)


class VirtualPortfolioManager:
    """Manages virtual portfolio state and position tracking."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.state = SimulationState()

    def initialize_portfolio(self, initial_balance: float = None):
        """Initialize portfolio with starting balance."""
        if initial_balance is not None:
            self.state.balance = initial_balance
            self.state.total_value = initial_balance

    async def process_order_result(
        self,
        order: VirtualOrder,
        result,
        current_market_data
    ) -> Optional[VirtualTrade]:
        """Process order result and update portfolio."""
        if result.status == OrderStatus.REJECTED or result.status == OrderStatus.NO_FILL:
            from sandbox.models import RejectedOrder
            self.state.rejected_orders.append(RejectedOrder(
                order_id=order.order_id,
                market_id=order.market_id,
                rejection_reason=result.rejection_reason
            ))
            return None

        trade = await self._create_trade_from_order(order, result, current_market_data)
        self._apply_trade_to_portfolio(trade, result)
        self.state.completed_trades.append(trade)

        return trade

    async def _create_trade_from_order(
        self,
        order: VirtualOrder,
        result,
        market_data
    ) -> VirtualTrade:
        """Create trade record from filled order."""
        trade = VirtualTrade(
            trade_id=f"trade_{order.order_id}",
            order_id=order.order_id,
            market_id=order.market_id,
            side=order.side,
            quantity=result.filled_quantity,
            entry_price=result.average_price,
            fees=result.total_fees,
            slippage=result.slippage,
            source_trader_id=order.source_trader_id,
            confidence_score=order.confidence_score,
            is_open=True
        )

        self._update_position(order, result, market_data)

        return trade

    def _update_position(self, order: VirtualOrder, result, market_data):
        """Update or create position from order."""
        market_id = order.market_id
        side = order.side
        filled_qty = result.filled_quantity
        exec_price = result.average_price

        if market_id not in self.state.positions:
            self.state.positions[market_id] = VirtualPosition(
                market_id=market_id,
                outcome=order.outcome,
                quantity=filled_qty,
                avg_price=exec_price,
                current_price=exec_price,
                source_trader_id=order.source_trader_id,
                order_id=order.order_id
            )
        else:
            position = self.state.positions[market_id]

            if side == OrderSide.BUY:
                total_cost = position.quantity * position.avg_price + filled_qty * exec_price
                new_quantity = position.quantity + filled_qty
                position.avg_price = total_cost / new_quantity if new_quantity > 0 else exec_price
                position.quantity = new_quantity
            else:
                if filled_qty >= position.quantity:
                    del self.state.positions[market_id]
                    return
                else:
                    position.quantity -= filled_qty
                    position.avg_price = exec_price

        self._update_position_prices({market_id: market_data.current_price})

    def _apply_trade_to_portfolio(self, trade: VirtualTrade, result):
        """Apply trade result to portfolio state."""
        if trade.side == OrderSide.BUY:
            cost = result.filled_quantity * result.average_price + result.total_fees
            self.state.balance -= cost
        else:
            proceeds = result.filled_quantity * result.average_price - result.total_fees
            self.state.balance += proceeds

        self._recalculate_total_value()

    def _recalculate_total_value(self):
        """Recalculate total portfolio value."""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.state.positions.values()
        )
        unrealized_pnl = sum(
            pos.unrealized_pnl
            for pos in self.state.positions.values()
        )

        self.state.total_value = self.state.balance + positions_value
        self.state.total_exposure = positions_value
        self.state.total_pnl = self.state.total_value - self.config.initial_balance
        self.state.total_pnl_pct = self.state.total_pnl / self.config.initial_balance if self.config.initial_balance > 0 else 0

    def _update_position_prices(self, market_updates: Dict[str, float]):
        """Update current prices for all positions."""
        for market_id, new_price in market_updates.items():
            if market_id in self.state.positions:
                position = self.state.positions[market_id]
                old_price = position.current_price
                position.current_price = new_price
                position.last_updated = datetime.utcnow()

                if position.quantity > 0:
                    if position.outcome == "YES":
                        position.unrealized_pnl = (new_price - position.avg_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.avg_price - new_price) * position.quantity

        self._recalculate_total_value()

    def close_position(self, market_id: str, exit_price: float) -> Optional[VirtualTrade]:
        """Close a position at specified price."""
        if market_id not in self.state.positions:
            return None

        position = self.state.positions[market_id]
        filled_qty = position.quantity
        avg_price = position.avg_price

        # Calculate realized P&L
        if position.outcome == "YES":
            realized_pnl = (exit_price - avg_price) * filled_qty
        else:
            realized_pnl = (avg_price - exit_price) * filled_qty

        trade = VirtualTrade(
            trade_id=f"close_{position.order_id}",
            order_id=position.order_id,
            market_id=market_id,
            side=OrderSide.SELL,
            quantity=filled_qty,
            entry_price=avg_price,
            exit_price=exit_price,
            profit=realized_pnl,
            fees=0,
            opened_at=position.opened_at,
            closed_at=datetime.utcnow(),
            hold_time_hours=(datetime.utcnow() - position.opened_at).total_seconds() / 3600,
            source_trader_id=position.source_trader_id,
            is_open=False
        )

        self.state.balance += filled_qty * exit_price

        del self.state.positions[market_id]

        self._recalculate_total_value()
        self.state.completed_trades.append(trade)

        return trade

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Generate portfolio summary with all metrics."""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.state.positions.values()
        )
        unrealized_pnl = sum(
            pos.unrealized_pnl
            for pos in self.state.positions.values()
        )

        return PortfolioSummary(
            balance=self.state.balance,
            positions_value=positions_value,
            total_value=self.state.balance + positions_value,
            unrealized_pnl=unrealized_pnl,
            position_count=len(self.state.positions),
            exposure_pct=positions_value / (self.state.balance + positions_value) if (self.state.balance + positions_value) > 0 else 0
        )

    def get_positions(self) -> Dict[str, VirtualPosition]:
        """Get all current positions."""
        return self.state.positions.copy()

    def get_balance(self) -> float:
        """Get current balance."""
        return self.state.balance

    def get_total_value(self) -> float:
        """Get total portfolio value."""
        return self.state.total_value

    def get_state(self) -> SimulationState:
        """Get full portfolio state."""
        return self.state

    def update_market_prices(self, market_updates: Dict[str, float]):
        """Update current prices for all positions."""
        self._update_position_prices(market_updates)
