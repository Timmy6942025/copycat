"""
Live Trading Integration with Micro Mode Settings.

Provides Micro Mode-specific configuration and integration for live trading.

Usage:
    from orchestrator.live_trading_micro import (
        MicroLiveTradingConfig,
        MicroLiveTradingRunner,
        create_micro_live_runner,
    )
    
    config = MicroLiveTradingConfig(
        initial_balance=10.0,
        micro_mode="nano",
        position_size_pct=0.75,
        kelly_fraction=0.75,
        drawdown_stop_threshold=0.30,
    )
    
    runner = MicroLiveTradingRunner(config=config, api_client=client)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from api_clients.base import (
    MarketAPIClient, Trade, Order, OrderSide, OrderType, OrderStatus,
    Position, MarketData
)
from live_trading.runner import (
    LiveTradingRunner, LiveTradingConfig, LiveOrderResult, LiveOrderStatus,
    LiveTradingState
)

from orchestrator.config_micro import MicroModeLevel
from orchestrator.mode_transition import TradingModeLevel
from orchestrator.circuit_breaker import MicroCircuitBreaker, CircuitState
from orchestrator.notification_service import NotificationService, NotificationEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveMicroMode(Enum):
    """Live trading micro mode levels."""
    NANO = "nano"       # Maximum aggression for $10-$15
    MICRO = "micro"     # High aggression for $15-$25
    MINI = "mini"       # Moderate aggression for $25-$50
    BALANCED = "balanced"  # Standard for $50+


@dataclass
class MicroLiveTradingConfig:
    """
    Micro Mode configuration for live trading.
    
    Extends LiveTradingConfig with micro-specific settings for small accounts.
    """
    # Base settings
    initial_balance: float = 10.0
    wallet_address: str = ""
    
    # Micro mode settings
    micro_mode: LiveMicroMode = LiveMicroMode.NANO
    position_size_pct: float = 0.75  # 75% for NANO
    kelly_fraction: float = 0.75  # Aggressive Kelly
    max_position_size_pct: float = 0.80  # Max 80% per trade
    max_total_exposure_pct: float = 0.95  # Up to 95% exposure for small accounts
    max_orders_per_day: int = 100  # More orders for micro trading
    min_order_size: float = 0.50  # Lower minimum for small accounts
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    drawdown_warning_threshold: float = 0.10  # 10% warning
    drawdown_stop_threshold: float = 0.25  # 25% stop
    emergency_stop_threshold: float = 0.30  # 30% emergency
    require_consecutive_cycles: int = 2
    
    # Safety settings
    require_order_confirmation: bool = True  # Always confirm in live mode
    max_slippage_pct: float = 0.05
    enable_price_protection: bool = True
    
    # Notification settings
    notify_on_trade: bool = True
    notify_on_error: bool = True
    notify_on_circuit_breaker: bool = True
    milestone_channels: List[float] = field(default_factory=lambda: [20, 50, 100])
    
    # Bootstrap settings
    enable_bootstrap: bool = True
    bootstrap_traders_count: int = 3
    bootstrap_max_position_pct: float = 0.40
    
    def get_mode_level(self) -> TradingModeLevel:
        """Get the corresponding TradingModeLevel."""
        mapping = {
            LiveMicroMode.NANO: TradingModeLevel.NANO,
            LiveMicroMode.MICRO: TradingModeLevel.MICRO,
            LiveMicroMode.MINI: TradingModeLevel.MINI,
            LiveMicroMode.BALANCED: TradingModeLevel.BALANCED,
        }
        return mapping.get(self.micro_mode, TradingModeLevel.NANO)
    
    def get_drawdown_thresholds(self) -> Dict[str, float]:
        """Get drawdown thresholds for the current mode."""
        return {
            "warning": self.drawdown_warning_threshold,
            "stop": self.drawdown_stop_threshold,
            "emergency": self.emergency_stop_threshold,
        }


class MicroLiveTradingRunner:
    """
    Live trading runner with Micro Mode support.
    
    Extends LiveTradingRunner with:
    - Micro Mode position sizing
    - Circuit breaker integration
    - Milestone notifications
    - Bootstrap trader copying
    """
    
    def __init__(
        self,
        config: Optional[MicroLiveTradingConfig] = None,
        api_client: MarketAPIClient = None,
        wallet_address: str = None,
        notification_service: Optional[NotificationService] = None,
        on_trade_callback: Optional[Callable] = None,
        on_circuit_breaker_callback: Optional[Callable] = None,
    ):
        """Initialize the micro live trading runner."""
        self.config = config or MicroLiveTradingConfig()
        self.api_client = api_client
        self.wallet_address = wallet_address or self.config.wallet_address
        self.notification_service = notification_service
        self.on_trade_callback = on_trade_callback
        self.on_circuit_breaker_callback = on_circuit_breaker_callback
        
        # Initialize base runner config
        base_config = LiveTradingConfig(
            initial_balance=self.config.initial_balance,
            max_position_size_pct=self.config.max_position_size_pct,
            max_total_exposure_pct=self.config.max_total_exposure_pct,
            max_orders_per_day=self.config.max_orders_per_day,
            min_order_size=self.config.min_order_size,
            require_order_confirmation=self.config.require_order_confirmation,
            max_slippage_pct=self.config.max_slippage_pct,
            enable_price_protection=self.config.enable_price_protection,
        )
        
        # Base runner
        self.runner = LiveTradingRunner(
            config=base_config,
            api_client=api_client,
            wallet_address=self.wallet_address,
        )
        
        # State
        self.state = self.runner.state
        self.is_running = False
        
        # Circuit breaker
        self.circuit_breaker: Optional[MicroCircuitBreaker] = None
        if self.config.circuit_breaker_enabled:
            self._init_circuit_breaker()
        
        # Milestone tracking
        self._reached_milestones: set = set()
        
        # Bootstrap tracking
        self._bootstrap_complete: bool = False
        self._bootstrap_traders_copied: List[str] = []
        
        logger.info(
            f"MicroLiveTradingRunner initialized in {self.config.micro_mode.value} mode "
            f"(position: {self.config.position_size_pct:.0%}, kelly: {self.config.kelly_fraction:.2f})"
        )
    
    def _init_circuit_breaker(self):
        """Initialize the circuit breaker."""
        mode_level = self.config.get_mode_level()
        
        self.circuit_breaker = MicroCircuitBreaker(
            config=None,  # Use defaults
            orchestrator=None,  # Will update manually
            mode=mode_level,
            on_state_change_callback=self._on_circuit_breaker_state_change,
        )
        
        logger.info(f"Circuit breaker initialized for {mode_level.value} mode")
    
    async def _on_circuit_breaker_state_change(self, record):
        """Handle circuit breaker state changes."""
        # Notify via callback
        if self.on_circuit_breaker_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_circuit_breaker_callback):
                    await self.on_circuit_breaker_callback(record)
                else:
                    self.on_circuit_breaker_callback(record)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}")
        
        # Notify via notification service
        if self.notification_service and self.config.notify_on_circuit_breaker:
            if record.new_state == CircuitState.OPEN:
                await self.notification_service.notify_circuit_breaker_opened(
                    reason=record.reason,
                    consecutive_failures=int(record.drawdown_pct * 100),
                )
            elif record.new_state == CircuitState.CLOSED:
                await self.notification_service.notify_circuit_breaker_closed()
    
    def set_notification_service(self, service: NotificationService):
        """Set the notification service."""
        self.notification_service = service
    
    def set_trade_callback(self, callback: Callable):
        """Set callback for trade execution."""
        self.on_trade_callback = callback
        self.runner.set_trade_callback(callback)
    
    def set_order_callback(self, callback: Callable):
        """Set callback for order execution (compatibility with sandbox)."""
        self.runner.set_order_callback(callback)
    
    def set_market_data_callback(self, callback: Callable[[str], Dict[str, Any]]):
        """Set callback for market data (no-op for live - real API provides data)."""
        logger.debug("set_market_data_callback called - not used in live trading (real API provides data)")
    
    def set_circuit_breaker_callback(self, callback: Callable):
        """Set callback for circuit breaker state changes."""
        self.on_circuit_breaker_callback = callback
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        return await self.runner.get_positions()
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        return await self.runner.cancel_order(order_id)
    
    async def update_market_prices(self, updates: Dict[str, float]):
        """Update market prices (no-op for live - real API provides data)."""
        logger.debug("update_market_prices called - not used in live trading")
    
    async def start(self) -> bool:
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
            
            # Start base runner
            success = await self.runner.start()
            if not success:
                return False
            
            self.is_running = True
            
            # Execute bootstrap if enabled
            if self.config.enable_bootstrap and not self._bootstrap_complete:
                await self._execute_bootstrap()
            
            logger.info(f"MicroLiveTradingRunner started in {self.config.micro_mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start live trading: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the live trading runner."""
        self.is_running = False
        
        # Stop base runner
        await self.runner.stop()
        
        logger.info(f"MicroLiveTradingRunner stopped")
        return True
    
    async def _execute_bootstrap(self):
        """Execute bootstrap trading - copy multiple traders immediately."""
        if not self.config.enable_bootstrap:
            return
        
        logger.info("Executing bootstrap trading...")
        
        # Bootstrap trading copies proven traders immediately
        # This happens in parallel with normal discovery
        
        self._bootstrap_complete = True
        logger.info(f"Bootstrap complete - copied {len(self._bootstrap_traders_copied)} initial traders")
    
    def calculate_position_size(
        self,
        base_size: float,
        trader_confidence: float = 1.0,
        growth_score: float = 1.0,
    ) -> float:
        """
        Calculate position size using micro mode settings.
        
        Args:
            base_size: Base position size
            trader_confidence: Trader confidence score (0-1)
            growth_score: Growth score (0-1)
            
        Returns:
            Position size adjusted for micro mode
        """
        # Apply Kelly fraction
        kelly = self.config.kelly_fraction
        
        # Apply confidence scaling
        confidence_multiplier = 0.5 + (trader_confidence * 0.5)
        
        # Apply growth scaling
        growth_multiplier = 0.5 + (growth_score * 0.5)
        
        # Calculate position
        position = base_size * kelly * confidence_multiplier * growth_multiplier
        
        # Apply max position cap
        portfolio_value = self.state.balance
        max_position = portfolio_value * self.config.max_position_size_pct
        position = min(position, max_position)
        
        # Ensure minimum order size
        if position < self.config.min_order_size:
            position = self.config.min_order_size
        
        return position
    
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
        trader_confidence: float = 1.0,
        growth_score: float = 1.0,
    ) -> LiveOrderResult:
        """
        Execute a live order with micro mode adjustments.
        
        Args:
            market_id: Market to trade
            side: 'buy' or 'sell'
            quantity: Amount to trade
            order_type: 'market' or 'limit'
            limit_price: Price for limit orders
            outcome: 'YES' or 'NO'
            source_trade_id: ID of the source trade being copied
            source_trader: Address of the trader being copied
            trader_confidence: Trader confidence score
            growth_score: Growth score
            
        Returns:
            LiveOrderResult with execution details
        """
        if not self.is_running:
            return LiveOrderResult(
                order_id="",
                status=LiveOrderStatus.REJECTED,
                message="Live trading runner is not running",
            )
        
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.current_state == CircuitState.OPEN:
            return LiveOrderResult(
                order_id="",
                status=LiveOrderStatus.REJECTED,
                message="Circuit breaker is open - trading paused",
            )
        
        # Calculate micro mode adjusted position size
        adjusted_quantity = self.calculate_position_size(
            base_size=quantity,
            trader_confidence=trader_confidence,
            growth_score=growth_score,
        )
        
        # Execute order via base runner
        result = await self.runner.execute_order(
            market_id=market_id,
            side=side,
            quantity=adjusted_quantity,
            order_type=order_type,
            limit_price=limit_price,
            outcome=outcome,
            source_trade_id=source_trade_id,
            source_trader=source_trader,
        )
        
        # Check milestones after trade
        if result.status == LiveOrderStatus.FILLED:
            await self._check_milestones()
        
        # Update circuit breaker
        if self.circuit_breaker:
            await self.circuit_breaker.check()
        
        return result
    
    async def _check_milestones(self):
        """Check and notify on balance milestones."""
        if not self.notification_service or not self.config.milestone_channels:
            return
        
        current_balance = self.state.balance
        
        for milestone in self.config.milestone_channels:
            if current_balance >= milestone and milestone not in self._reached_milestones:
                self._reached_milestones.add(milestone)
                
                # Calculate previous milestone
                previous = max([m for m in self._reached_milestones if m < milestone], default=0)
                growth_pct = ((current_balance - previous) / previous * 100) if previous > 0 else 0
                
                # Create milestone record
                from orchestrator.notification_service import MilestoneRecord
                record = MilestoneRecord(
                    balance=current_balance,
                    milestone=milestone,
                    previous_balance=previous,
                    growth_pct=growth_pct,
                )
                
                await self.notification_service.notify_milestone(record)
    
    def get_state(self) -> LiveTradingState:
        """Get current trading state."""
        return self.runner.get_state()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        state = self.get_state()
        summary = self.runner.get_portfolio_summary()
        
        # Add micro-specific info
        summary.update({
            "micro_mode": self.config.micro_mode.value,
            "position_size_pct": self.config.position_size_pct,
            "kelly_fraction": self.config.kelly_fraction,
            "circuit_breaker_state": (
                self.circuit_breaker.current_state.value 
                if self.circuit_breaker else "disabled"
            ),
            "drawdown_pct": (
                self.circuit_breaker.current_drawdown 
                if self.circuit_breaker else 0
            ),
            "reached_milestones": sorted(list(self._reached_milestones)),
        })
        
        return summary
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate and return performance metrics."""
        metrics = self.runner.get_performance_metrics()
        
        # Add circuit breaker info
        if self.circuit_breaker:
            metrics["circuit_breaker_state"] = self.circuit_breaker.current_state.value
            metrics["current_drawdown"] = self.circuit_breaker.current_drawdown
            metrics["position_multiplier"] = self.circuit_breaker.position_multiplier
        
        return metrics
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        if not self.circuit_breaker:
            return {"enabled": False}
        
        return self.circuit_breaker.get_status()
    
    def reset_circuit_breaker(self):
        """Reset the circuit breaker."""
        if self.circuit_breaker:
            self.circuit_breaker.reset()
            logger.info("Circuit breaker reset")

    def generate_report(self) -> str:
        """Generate a performance report for live trading."""
        state = self.get_state()
        metrics = self.get_performance_metrics()
        circuit_status = self.get_circuit_breaker_status()
        
        report = []
        report.append("=" * 60)
        report.append("CopyCat Live Trading Report")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append(f"Mode: {self.config.micro_mode.value}")
        report.append("=" * 60)
        
        report.append("\n--- Portfolio Summary ---")
        report.append(f"Balance: ${state.balance:,.2f}")
        report.append(f"Total Value: ${state.total_value:,.2f}")
        report.append(f"Positions: {len(self.runner.positions)}")
        report.append(f"Orders Today: {self.runner._orders_today}")
        
        report.append("\n--- Performance Metrics ---")
        report.append(f"Total P&L: ${metrics.get('total_pnl', 0):+,.2f}")
        report.append(f"P&L %: {metrics.get('total_pnl_pct', 0)*100:+.2f}%")
        report.append(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        report.append(f"Trades Executed: {metrics.get('trades_executed', 0)}")
        
        report.append("\n--- Circuit Breaker ---")
        report.append(f"State: {circuit_status.get('state', 'N/A')}")
        report.append(f"Drawdown: {circuit_status.get('drawdown_pct', 0)*100:.1f}%")
        report.append(f"Position Multiplier: {circuit_status.get('position_multiplier', 1.0):.2f}")
        
        report.append("\n--- Milestones ---")
        reached = sorted(list(self._reached_milestones))
        if reached:
            for m in reached:
                report.append(f"  ${m:,.0f}")
        else:
            report.append("  No milestones reached yet")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = None) -> str:
        """Generate and save a performance report."""
        report = self.generate_report()
        
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"live_trading_report_{timestamp}.md"
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return ""

    def reset(self):
        """Reset the trading runner to initial state."""
        self.runner.reset()
        self._reached_milestones.clear()
        self._bootstrap_complete = False
        self._bootstrap_traders_copied.clear()
        
        if self.circuit_breaker:
            self.circuit_breaker.reset()
        
        logger.info("Live trading runner reset")

    async def close(self):
        """Close and cleanup the trading runner."""
        await self.stop()
        logger.info("Live trading runner closed")


# =============================================================================
# Factory Functions
# =============================================================================

def create_micro_live_config(
    initial_balance: float = 10.0,
    wallet_address: str = "",
    micro_mode: str = "nano",
    enable_notifications: bool = True,
    discord_webhook_url: str = None,
) -> MicroLiveTradingConfig:
    """
    Create MicroLiveTradingConfig based on mode.
    
    Args:
        initial_balance: Starting balance
        wallet_address: Wallet address for live trading
        micro_mode: Mode - "nano", "micro", "mini", or "balanced"
        enable_notifications: Whether to enable notifications
        discord_webhook_url: Discord webhook URL for notifications
        
    Returns:
        Configured MicroLiveTradingConfig
    """
    mode_map = {
        "nano": (0.75, 0.75, 0.30),
        "micro": (0.60, 0.75, 0.25),
        "mini": (0.50, 0.50, 0.20),
        "balanced": (0.40, 0.40, 0.18),
    }
    
    position_pct, kelly, max_dd = mode_map.get(micro_mode, mode_map["nano"])
    
    return MicroLiveTradingConfig(
        initial_balance=initial_balance,
        wallet_address=wallet_address,
        micro_mode=LiveMicroMode(micro_mode),
        position_size_pct=position_pct,
        kelly_fraction=kelly,
        drawdown_stop_threshold=max_dd,
        emergency_stop_threshold=0.30,
        notify_on_trade=enable_notifications,
        notify_on_error=enable_notifications,
        notify_on_circuit_breaker=enable_notifications,
        milestone_channels=[20, 50, 100, 200, 500, 1000],
    )


async def create_micro_live_runner(
    config: MicroLiveTradingConfig,
    api_client: MarketAPIClient,
    notification_service: NotificationService = None,
) -> MicroLiveTradingRunner:
    """
    Create and initialize a MicroLiveTradingRunner.
    
    Args:
        config: MicroLiveTradingConfig
        api_client: API client for Polymarket
        notification_service: Optional notification service
        
    Returns:
        Initialized MicroLiveTradingRunner
    """
    runner = MicroLiveTradingRunner(
        config=config,
        api_client=api_client,
        notification_service=notification_service,
    )
    
    return runner


# =============================================================================
# Integration with Orchestrator
# =============================================================================

def create_live_runner_from_orchestrator_config(
    orchestrator_config,
    api_client: MarketAPIClient,
    notification_service: NotificationService = None,
) -> MicroLiveTradingRunner:
    """
    Create MicroLiveTradingRunner from OrchestratorConfig.
    
    This is the main integration point for live trading.
    """
    from orchestrator.config_micro import MicroConfig, MicroModeLevel
    
    # Determine micro mode from config
    micro_mode_str = "nano"
    
    # Check if using micro config
    if hasattr(orchestrator_config, 'micro_mode'):
        micro_config: MicroConfig = orchestrator_config.micro_mode
        if hasattr(micro_config, 'micro_mode'):
            level = micro_config.micro_mode
            if level == MicroModeLevel.NANO:
                micro_mode_str = "nano"
            elif level == MicroModeLevel.MICRO:
                micro_mode_str = "micro"
            elif level == MicroModeLevel.MINI:
                micro_mode_str = "mini"
    
    # Get position sizing from copy_trading config
    position_pct = orchestrator_config.copy_trading.position_size_pct
    kelly = orchestrator_config.copy_trading.kelly_fraction
    
    # Create config
    config = MicroLiveTradingConfig(
        initial_balance=orchestrator_config.live.initial_balance,
        wallet_address=orchestrator_config.live.wallet_address,
        micro_mode=LiveMicroMode(micro_mode_str),
        position_size_pct=position_pct,
        kelly_fraction=kelly,
        max_position_size_pct=orchestrator_config.copy_trading.max_position_size_pct,
        max_total_exposure_pct=orchestrator_config.copy_trading.max_total_exposure_pct,
        max_orders_per_day=orchestrator_config.live.max_orders_per_day,
        min_order_size=orchestrator_config.live.min_order_size,
        require_order_confirmation=orchestrator_config.live.require_order_confirmation,
        max_slippage_pct=orchestrator_config.live.max_slippage_pct,
        enable_price_protection=orchestrator_config.live.enable_price_protection,
        circuit_breaker_enabled=True,
        drawdown_warning_threshold=0.10,
        drawdown_stop_threshold=0.20,
        emergency_stop_threshold=0.30,
        notify_on_trade=True,
        notify_on_error=True,
        notify_on_circuit_breaker=True,
        milestone_channels=[20, 50, 100, 200, 500, 1000],
        enable_bootstrap=True,
    )
    
    # Create runner
    runner = MicroLiveTradingRunner(
        config=config,
        api_client=api_client,
        notification_service=notification_service,
    )
    
    return runner


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Micro Live Trading Configuration")
    print("=" * 60)
    
    for mode in ["nano", "micro", "mini", "balanced"]:
        config = create_micro_live_config(
            initial_balance=10.0,
            micro_mode=mode,
        )
        
        print(f"\n{mode.upper()} Mode:")
        print(f"  Position Size: {config.position_size_pct:.0%}")
        print(f"  Kelly Fraction: {config.kelly_fraction:.2f}")
        print(f"  Max Drawdown: {config.drawdown_stop_threshold:.0%}")
        print(f"  Min Order: ${config.min_order_size}")
    
    print("\n" + "=" * 60)
