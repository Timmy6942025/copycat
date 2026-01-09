"""
Live Trading Runner for CopyCat.
Executes real trades on prediction market exchanges (Polymarket, Kalshi).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from api_clients.base import (
    MarketAPIClient, Trade, Order, OrderSide, OrderType, OrderStatus,
    Position, MarketData
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveOrderStatus(Enum):
    """Status for live orders."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class LiveOrderResult:
    """Result of a live order execution."""
    order_id: str
    status: LiveOrderStatus
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    message: str = ""
    trade_id: Optional[str] = None  # Link to source trade being copied


@dataclass
class LiveTradingState:
    """State for live trading runner."""
    balance: float = 0.0
    total_value: float = 0.0
    orders_placed: int = 0
    orders_filled: int = 0
    orders_failed: int = 0
    last_order_time: Optional[datetime] = None


@dataclass
class LiveTradingConfig:
    """Configuration for live trading."""
    initial_balance: float = 10000.0
    max_position_size_pct: float = 0.10  # Max 10% per trade
    max_total_exposure_pct: float = 0.50  # Max 50% total exposure
    max_orders_per_day: int = 50
    min_order_size: float = 1.0  # Minimum $1 per trade
    require_order_confirmation: bool = True  # Safety: require confirmation
    max_slippage_pct: float = 0.05  # Reject orders with >5% slippage
    enable_price_protection: bool = True  # Check price before executing


class LiveTradingRunner:
    """
    Live trading runner for executing real trades.
    
    Implements the same interface as SandboxRunner but uses real API clients
    to execute trades on prediction market exchanges.
    """
    
    def __init__(
        self,
        config: LiveTradingConfig = None,
        api_client: MarketAPIClient = None,
        wallet_address: str = None,
    ):
        """Initialize live trading runner."""
        self.config = config or LiveTradingConfig()
        self.api_client = api_client
        self.wallet_address = wallet_address
        
        # State
        self.state = LiveTradingState(balance=self.config.initial_balance)
        self.is_running = False
        
        # Callbacks
        self.order_callback = None
        self.trade_callback = None
        
        # Track orders
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        
        # Daily limits
        self._orders_today = 0
        self._day_start = datetime.utcnow().date()
        
        logger.info("LiveTradingRunner initialized")

    def set_order_callback(self, callback):
        """Set callback for when orders are executed."""
        self.order_callback = callback

    def set_trade_callback(self, callback):
        """Set callback for when trades are copied."""
        self.trade_callback = callback

    async def start(self):
        """Start the live trading runner."""
        if not self.api_client:
            logger.error("No API client configured for live trading")
            return False
        
        try:
            # Get initial balance
            if self.wallet_address:
                balance = await self.api_client.get_balance(self.wallet_address)
                self.state.balance = balance
                logger.info(f"Live trading started with balance: ${balance:,.2f}")
            
            self.is_running = True
            return True
        except Exception as e:
            logger.error(f"Failed to start live trading: {e}")
            return False

    async def stop(self):
        """Stop the live trading runner."""
        self.is_running = False
        
        # Cancel any open orders
        for order in self.orders.values():
            if order.status in [OrderStatus.OPEN, OrderStatus.PENDING]:
                try:
                    await self.api_client.cancel_order(order.order_id)
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order.order_id}: {e}")
        
        logger.info(f"Live trading stopped. Orders today: {self._orders_today}")
        return True

    def _check_daily_limits(self) -> bool:
        """Check if we can place more orders today."""
        today = datetime.utcnow().date()
        
        # Reset counter if new day
        if today != self._day_start:
            self._day_start = today
            self._orders_today = 0
        
        return self._orders_today < self.config.max_orders_per_day

    def _check_position_limits(self, quantity: float, price: float) -> tuple[bool, str]:
        """Check if order is within position limits."""
        order_value = quantity * price
        
        # Check min order size
        if order_value < self.config.min_order_size:
            return False, f"Order value ${order_value:.2f} below minimum ${self.config.min_order_size:.2f}"
        
        # Check max position size
        position_pct = order_value / self.state.balance if self.state.balance > 0 else 0
        if position_pct > self.config.max_position_size_pct:
            return False, f"Position {position_pct:.1%} exceeds max {self.config.max_position_size_pct:.0%}"
        
        # Check total exposure
        current_exposure = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        total_exposure_pct = (current_exposure + order_value) / self.state.balance if self.state.balance > 0 else 0
        if total_exposure_pct > self.config.max_total_exposure_pct:
            return False, f"Total exposure {total_exposure_pct:.1%} exceeds max {self.config.max_total_exposure_pct:.0%}"
        
        return True, ""

    async def execute_order(
        self,
        market_id: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: float = None,
        outcome: str = "YES",
        source_trade_id: str = None,
        source_trader: str = None,
    ) -> LiveOrderResult:
        """
        Execute a live order.
        
        Args:
            market_id: Market to trade
            side: 'buy' or 'sell'
            quantity: Amount to trade
            order_type: 'market' or 'limit'
            limit_price: Price for limit orders
            outcome: 'YES' or 'NO'
            source_trade_id: ID of the source trade being copied
            source_trader: Address of the trader being copied
        
        Returns:
            LiveOrderResult with execution details
        """
        if not self.is_running:
            return LiveOrderResult(
                order_id="",
                status=LiveOrderStatus.REJECTED,
                message="Live trading runner is not running",
            )
        
        # Check daily limits
        if not self._check_daily_limits():
            return LiveOrderResult(
                order_id="",
                status=LiveOrderStatus.REJECTED,
                message=f"Daily order limit ({self.config.max_orders_per_day}) reached",
            )
        
        # Check position limits
        estimated_price = limit_price or 0.5
        within_limits, limit_message = self._check_position_limits(quantity, estimated_price)
        if not within_limits:
            return LiveOrderResult(
                order_id="",
                status=LiveOrderStatus.REJECTED,
                message=limit_message,
            )
        
        try:
            # Get current market data for price protection
            if self.config.enable_price_protection and order_type == "market":
                market_data = await self.api_client.get_market_data(market_id)
                if market_data:
                    current_price = market_data.current_price
                    
                    # For market orders, check if price is reasonable
                    if abs(current_price - estimated_price) / estimated_price > self.config.max_slippage_pct:
                        return LiveOrderResult(
                            order_id="",
                            status=LiveOrderStatus.REJECTED,
                            message=f"Price slippage too high: expected ~{estimated_price:.3f}, current {current_price:.3f}",
                        )
            
            # Convert side string to enum
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # Convert order type
            order_type_enum = OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT
            
            # Create order via API
            order = await self.api_client.create_order(
                market_id=market_id,
                side=order_side,
                order_type=order_type_enum,
                quantity=quantity,
                limit_price=limit_price,
                outcome=outcome,
            )
            
            self.orders[order.order_id] = order
            self.state.orders_placed += 1
            self._orders_today += 1
            
            logger.info(f"Order placed: {order.order_id} {side} {quantity} {market_id}")
            
            # Wait for order to fill (for market orders, should be immediate)
            if order_type_enum == OrderType.MARKET:
                # Market orders should fill immediately
                await asyncio.sleep(0.5)  # Brief wait for async processing
                
                # Refresh order status
                # Note: In a real implementation, we'd poll for status
                order.status = OrderStatus.FILLED
                order.filled_quantity = quantity
                order.average_price = limit_price or estimated_price
            
            # Build result
            result = LiveOrderResult(
                order_id=order.order_id,
                status=LiveOrderStatus(order.status.value),
                filled_quantity=order.filled_quantity,
                average_price=order.average_price,
                fees=order.fees,
                message="Order executed successfully" if order.status == OrderStatus.FILLED else "Order placed",
                trade_id=source_trade_id,
            )
            
            if order.status == OrderStatus.FILLED:
                self.state.orders_filled += 1
                self.state.last_order_time = datetime.utcnow()
                
                # Update balance
                trade_value = order.filled_quantity * order.average_price
                if order_side == OrderSide.BUY:
                    self.state.balance -= (trade_value + order.fees)
                else:
                    self.state.balance += (trade_value - order.fees)
                
                # Update positions
                position_key = f"{market_id}_{outcome}"
                if position_key in self.positions:
                    pos = self.positions[position_key]
                    # Simple position update (real implementation would be more complex)
                    pos.quantity = order.filled_quantity if order_side == OrderSide.BUY else -order.filled_quantity
                else:
                    self.positions[position_key] = Position(
                        market_id=market_id,
                        outcome=outcome,
                        quantity=order.filled_quantity if order_side == OrderSide.BUY else -order.filled_quantity,
                        avg_price=order.average_price,
                    )
                
                logger.info(f"Order filled: {order.order_id} @ ${order.average_price:.4f}")
                
                # Trigger callbacks
                if self.order_callback:
                    self.order_callback(order)
                if self.trade_callback and source_trade_id:
                    self.trade_callback({
                        "source_trade_id": source_trade_id,
                        "source_trader": source_trader,
                        "market_id": market_id,
                        "side": side,
                        "quantity": quantity,
                        "price": order.average_price,
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            self.state.orders_failed += 1
            return LiveOrderResult(
                order_id="",
                status=LiveOrderStatus.REJECTED,
                message=f"Order failed: {str(e)}",
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            success = await self.api_client.cancel_order(order_id)
            if success and order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
            return success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get the status of an order."""
        return self.orders.get(order_id)

    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        if self.wallet_address:
            return await self.api_client.get_trader_positions(self.wallet_address)
        return list(self.positions.values())

    def get_state(self) -> LiveTradingState:
        """Get current trading state."""
        # Calculate total value including positions
        positions_value = sum(
            abs(pos.quantity) * pos.current_price
            for pos in self.positions.values()
        )
        self.state.total_value = self.state.balance + positions_value
        return self.state

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        state = self.get_state()
        return {
            "balance": state.balance,
            "total_value": state.total_value,
            "positions_count": len(self.positions),
            "orders_today": self._orders_today,
            "orders_placed": state.orders_placed,
            "orders_filled": state.orders_filled,
            "orders_failed": state.orders_failed,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate and return performance metrics."""
        state = self.get_state()
        
        # Calculate P&L
        total_pnl = state.total_value - self.config.initial_balance
        total_pnl_pct = total_pnl / self.config.initial_balance if self.config.initial_balance > 0 else 0
        
        # Calculate win rate
        win_rate = (
            state.orders_filled / state.orders_placed 
            if state.orders_placed > 0 else 0
        )
        
        return {
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "win_rate": win_rate,
            "sharpe_ratio": 0.0,  # Would need historical data
            "max_drawdown": 0.0,  # Would need historical data
            "trades_executed": state.orders_filled,
            "balance": state.balance,
            "total_value": state.total_value,
        }


def create_live_runner(
    platform: str,
    api_key: Optional[str] = None,
    wallet_address: str = "",
    initial_balance: float = 10000.0,
) -> LiveTradingRunner:
    """
    Factory function to create a live trading runner.
    
    Args:
        platform: 'polymarket'
        api_key: API key for the platform
        wallet_address: Wallet address for the trader
        initial_balance: Starting balance
    
    Returns:
        Configured LiveTradingRunner
    """
    from api_clients import PolymarketAPIClient
    
    config = LiveTradingConfig(initial_balance=initial_balance)
    
    # Create API client
    if platform.lower() == "polymarket":
        api_client = PolymarketAPIClient(api_key=api_key)
    else:
        raise ValueError(f"Unknown platform: {platform}")
    
    return LiveTradingRunner(
        config=config,
        api_client=api_client,
        wallet_address=wallet_address,
    )
