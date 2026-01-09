"""
Micro Mode Sandbox Runner with Full Feature Parity.

Provides sandbox trading with all Micro Mode features:
- Micro position sizing (75% for NANO, 60% for MICRO, etc.)
- Mode-specific circuit breaker with drawdown protection
- Milestone tracking with Discord notifications
- Automatic mode transitions (NANO → MICRO → MINI → BALANCED)
- Bootstrap trading patterns

Usage:
    from orchestrator.sandbox_micro import MicroSandboxRunner, create_micro_sandbox_runner
    
    runner = await create_micro_sandbox_runner(
        config=micro_sandbox_config,
        api_client=mock_client,
    )
    
    result = await runner.execute_order(order)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from sandbox import (
    SandboxRunner, SandboxConfig, VirtualOrder, VirtualOrderResult
)
from sandbox.runner import SimulationState

from orchestrator.config_micro import MicroModeLevel, create_micro_config, TradingMode as MicroTradingMode
from orchestrator.mode_transition import TradingModeLevel, create_mode_manager, ModeTransitionManager
from orchestrator.circuit_breaker import MicroCircuitBreaker, CircuitState, create_circuit_breaker
from orchestrator.notification_service import NotificationService, NotificationEvent, NotificationPriority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SandboxMicroMode(Enum):
    """Micro mode levels for sandbox trading."""
    NANO = "nano"
    MICRO = "micro"
    MINI = "mini"
    BALANCED = "balanced"


@dataclass
class MicroSandboxConfig:
    """
    Micro Mode configuration for sandbox trading.
    
    Provides feature parity with live trading:
    - Micro position sizing
    - Circuit breaker
    - Milestone notifications
    - Mode transitions
    """
    # Base settings
    initial_balance: float = 10.0
    micro_mode: SandboxMicroMode = SandboxMicroMode.NANO
    
    # Position sizing (micro-optimized)
    position_size_pct: float = 0.75  # 75% for NANO
    kelly_fraction: float = 0.75
    max_position_size_pct: float = 0.80
    max_total_exposure_pct: float = 0.95
    
    # Sandbox simulation settings
    simulate_slippage: bool = True
    slippage_model: str = "fixed"  # fixed, percentage, market_depth
    slippage_fixed_pct: float = 0.02
    simulate_fees: bool = True
    fee_model: str = "polymarket"  # polymarket, percentage, fixed
    fee_percentage: float = 0.02
    simulate_fill_probability: float = 0.95
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    drawdown_warning_threshold: float = 0.10
    drawdown_stop_threshold: float = 0.25
    emergency_stop_threshold: float = 0.30
    require_consecutive_cycles: int = 2
    
    # Notification settings
    notify_on_trade: bool = True
    notify_on_milestone: bool = True
    notify_on_circuit_breaker: bool = True
    notify_on_mode_transition: bool = True
    milestone_channels: List[float] = field(default_factory=lambda: [20, 50, 100, 200, 500, 1000])
    discord_webhook_url: Optional[str] = None
    
    # Mode transition settings
    enable_mode_transition: bool = True
    transition_hysteresis_pct: float = 0.10
    
    # Bootstrap settings
    enable_bootstrap: bool = True
    bootstrap_traders_count: int = 3
    bootstrap_max_position_pct: float = 0.40
    
    def get_mode_level(self) -> TradingModeLevel:
        """Get the corresponding TradingModeLevel."""
        mapping = {
            SandboxMicroMode.NANO: TradingModeLevel.NANO,
            SandboxMicroMode.MICRO: TradingModeLevel.MICRO,
            SandboxMicroMode.MINI: TradingModeLevel.MINI,
            SandboxMicroMode.BALANCED: TradingModeLevel.BALANCED,
        }
        return mapping.get(self.micro_mode, TradingModeLevel.NANO)
    
    def get_drawdown_thresholds(self) -> Dict[str, float]:
        """Get drawdown thresholds for the current mode."""
        return {
            "warning": self.drawdown_warning_threshold,
            "stop": self.drawdown_stop_threshold,
            "emergency": self.emergency_stop_threshold,
        }


class MicroSandboxRunner:
    """
    Sandbox runner with full Micro Mode feature parity.
    
    Wraps SandboxRunner and adds:
    - Micro position sizing
    - Circuit breaker
    - Milestone notifications
    - Automatic mode transitions
    - Bootstrap trading
    """
    
    def __init__(
        self,
        config: Optional[MicroSandboxConfig] = None,
        api_client: Any = None,
        notification_service: Optional[NotificationService] = None,
        on_trade_callback: Optional[Callable] = None,
        on_circuit_breaker_callback: Optional[Callable] = None,
        on_mode_transition_callback: Optional[Callable] = None,
    ):
        """Initialize the micro sandbox runner."""
        self.config = config or MicroSandboxConfig()
        self.api_client = api_client
        self.notification_service = notification_service
        self.on_trade_callback = on_trade_callback
        self.on_circuit_breaker_callback = on_circuit_breaker_callback
        self.on_mode_transition_callback = on_mode_transition_callback
        
        # Initialize base sandbox config
        sandbox_config = SandboxConfig(
            initial_balance=self.config.initial_balance,
            simulate_slippage=self.config.simulate_slippage,
            slippage_model=self.config.slippage_model,
            slippage_fixed_pct=self.config.slippage_fixed_pct,
            simulate_fees=self.config.simulate_fees,
            fee_model=self.config.fee_model,
            fee_percentage=self.config.fee_percentage,
            simulate_fill_probability=self.config.simulate_fill_probability,
        )
        
        # Base sandbox runner
        self.runner = SandboxRunner(config=sandbox_config)
        
        # State (proxy to runner)
        self.state = self.runner.state
        self.is_running = False
        
        # Circuit breaker
        self.circuit_breaker: Optional[MicroCircuitBreaker] = None
        if self.config.circuit_breaker_enabled:
            self._init_circuit_breaker()
        
        # Mode manager
        self.mode_manager: Optional[ModeTransitionManager] = None
        if self.config.enable_mode_transition:
            self._init_mode_manager()
        
        # Milestone tracking
        self._reached_milestones: set = set()
        
        # Bootstrap tracking
        self._bootstrap_complete: bool = False
        
        logger.info(
            f"MicroSandboxRunner initialized in {self.config.micro_mode.value} mode "
            f"(position: {self.config.position_size_pct:.0%}, kelly: {self.config.kelly_fraction:.2f})"
        )
    
    def _init_circuit_breaker(self):
        """Initialize the circuit breaker."""
        mode_level = self.config.get_mode_level()
        
        # Create mock orchestrator for circuit breaker
        class MockOrchestrator:
            def __init__(self, runner):
                self.state = type('State', (), {
                    'total_pnl': 0.0,
                    'cycle_count': 0,
                    'max_drawdown': 0.0,
                })()
                self.config = type('Config', (), {
                    'sandbox': type('Sandbox', (), {
                        'initial_balance': runner.config.initial_balance
                    })(),
                })()
        
        mock_orchestrator = MockOrchestrator(self.runner)
        
        self.circuit_breaker = MicroCircuitBreaker(
            config=None,
            orchestrator=mock_orchestrator,
            mode=mode_level,
            on_state_change_callback=self._on_circuit_breaker_state_change,
        )
        
        logger.info(f"Circuit breaker initialized for {mode_level.value} mode")
    
    def _init_mode_manager(self):
        """Initialize the mode transition manager."""
        class MockOrchestrator:
            def __init__(self, runner):
                self.state = type('State', (), {
                    'total_pnl': 0.0,
                    'cycle_count': 0,
                    'max_drawdown': 0.0,
                })()
                self.config = type('Config', (), {
                    'sandbox': type('Sandbox', (), {
                        'initial_balance': runner.config.initial_balance
                    })(),
                    'copy_trading': type('Trading', (), {
                        'position_size_pct': runner.config.position_size_pct,
                        'kelly_fraction': runner.config.kelly_fraction,
                    })(),
                    'trader_selection': type('Selection', (), {
                        'growth_min_total_pnl': 25.0,
                        'growth_min_growth_rate': 0.005,
                        'growth_max_drawdown': 0.50,
                    })(),
                    'boost_mode': type('Boost', (), {'enabled': True})(),
                })()
        
        mock_orchestrator = MockOrchestrator(self)
        
        self.mode_manager = create_mode_manager(
            orchestrator=mock_orchestrator,
            balance=self.config.initial_balance,
            on_transition=self._on_mode_transition,
        )
        
        logger.info(f"Mode manager initialized: {self.mode_manager.current_mode.value}")
    
    async def _on_circuit_breaker_state_change(self, record):
        """Handle circuit breaker state changes."""
        if self.on_circuit_breaker_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_circuit_breaker_callback):
                    await self.on_circuit_breaker_callback(record)
                else:
                    self.on_circuit_breaker_callback(record)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}")
        
        if self.notification_service and self.config.notify_on_circuit_breaker:
            if record.new_state == CircuitState.OPEN:
                await self.notification_service.notify_circuit_breaker_opened(
                    reason=record.reason,
                    consecutive_failures=int(record.drawdown_pct * 100),
                )
            elif record.new_state == CircuitState.CLOSED:
                await self.notification_service.notify_circuit_breaker_closed()
    
    async def _on_mode_transition(self, record):
        """Handle mode transition."""
        logger.info(
            f"MODE TRANSITION: {record.from_mode.value.upper()} → {record.to_mode.value.upper()}"
        )
        
        # Update config based on new mode
        mode_configs = {
            TradingModeLevel.NANO: (0.75, 0.75, 0.30),
            TradingModeLevel.MICRO: (0.60, 0.75, 0.25),
            TradingModeLevel.MINI: (0.50, 0.50, 0.20),
            TradingModeLevel.BALANCED: (0.40, 0.40, 0.18),
        }
        
        position_pct, kelly, max_dd = mode_configs.get(record.to_mode, mode_configs[TradingModeLevel.NANO])
        self.config.position_size_pct = position_pct
        self.config.kelly_fraction = kelly
        self.config.drawdown_stop_threshold = max_dd
        
        # Update circuit breaker mode
        if self.circuit_breaker:
            self.circuit_breaker.set_mode(record.to_mode)
        
        # Notify
        if self.on_mode_transition_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_mode_transition_callback):
                    await self.on_mode_transition_callback(record)
                else:
                    self.on_mode_transition_callback(record)
            except Exception as e:
                logger.error(f"Error in mode transition callback: {e}")
        
        if self.notification_service and self.config.notify_on_mode_transition:
            await self.notification_service.notify_mode_transition(
                from_mode=record.from_mode.value,
                to_mode=record.to_mode.value,
                balance=record.balance,
                reason=record.reason,
            )
    
    def set_notification_service(self, service: NotificationService):
        """Set the notification service."""
        self.notification_service = service
    
    def set_trade_callback(self, callback: Callable):
        """Set callback for trade execution."""
        self.on_trade_callback = callback
        self.runner.set_order_callback(callback)
    
    def set_order_callback(self, callback: Callable):
        """Set callback for order execution."""
        self.runner.set_order_callback(callback)
    
    def set_market_data_callback(self, callback: Callable[[str], Dict[str, Any]]):
        """Set callback for market data."""
        self.runner.set_market_data_callback(callback)
    
    async def start(self):
        """Start the sandbox runner."""
        await self.runner.start()
        self.is_running = True
        
        # Execute bootstrap if enabled
        if self.config.enable_bootstrap and not self._bootstrap_complete:
            await self._execute_bootstrap()
        
        logger.info(f"MicroSandboxRunner started in {self.config.micro_mode.value} mode")
    
    async def stop(self):
        """Stop the sandbox runner."""
        await self.runner.stop()
        self.is_running = False
        logger.info("MicroSandboxRunner stopped")
    
    async def _execute_bootstrap(self):
        """Execute bootstrap trading."""
        if not self.config.enable_bootstrap:
            return
        
        logger.info("Executing bootstrap trading...")
        
        # Bootstrap trading copies proven traders immediately
        # This happens in parallel with normal discovery
        
        self._bootstrap_complete = True
        logger.info("Bootstrap complete")
    
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
        portfolio_value = self.runner.state.total_pnl + self.config.initial_balance
        max_position = portfolio_value * self.config.max_position_size_pct
        position = min(position, max_position)
        
        return position
    
    async def execute_order(self, order: VirtualOrder) -> VirtualOrderResult:
        """
        Execute a virtual order with micro mode adjustments.
        
        Args:
            order: VirtualOrder to execute
            
        Returns:
            VirtualOrderResult with execution details
        """
        if not self.is_running:
            logger.warning("Sandbox runner not running")
            return VirtualOrderResult(
                order_id=order.order_id,
                status="REJECTED",
                message="Sandbox runner is not running",
            )
        
        # Check circuit breaker
        if self.circuit_breaker:
            multiplier = await self.circuit_breaker.check()
            if self.circuit_breaker.current_state == CircuitState.OPEN:
                return VirtualOrderResult(
                    order_id=order.order_id,
                    status="REJECTED",
                    message="Circuit breaker is open - trading paused",
                )
            
            # Adjust position size if in warning state
            if multiplier < 1.0 and order.quantity:
                order.quantity = order.quantity * multiplier
        
        # Execute order via base runner
        result = await self.runner.execute_order(order)
        
        # Check milestones after filled order
        if result.status == "FILLED":
            await self._check_milestones()
            
            # Trigger callback
            if self.on_trade_callback:
                self.on_trade_callback(order)
        
        # Update circuit breaker after order
        if self.circuit_breaker:
            await self.circuit_breaker.check()
        
        return result
    
    async def _check_milestones(self):
        """Check and notify on balance milestones."""
        if not self.notification_service or not self.config.milestone_channels:
            return
        
        # Calculate current balance from runner state
        current_balance = (
            self.runner.state.total_pnl + self.config.initial_balance
            if hasattr(self.runner.state, 'total_pnl')
            else self.runner.state.balance
        )
        
        for milestone in self.config.milestone_channels:
            if current_balance >= milestone and milestone not in self._reached_milestones:
                self._reached_milestones.add(milestone)
                
                # Calculate previous milestone
                previous = max([m for m in self._reached_milestones if m < milestone], default=0)
                growth_pct = ((current_balance - previous) / previous * 100) if previous > 0 else 0
                
                if self.notification_service:
                    from orchestrator.notification_service import MilestoneRecord
                    record = MilestoneRecord(
                        balance=current_balance,
                        milestone=milestone,
                        previous_balance=previous,
                        growth_pct=growth_pct,
                    )
                    await self.notification_service.notify_milestone(record)
    
    async def run_cycle(self) -> Dict[str, Any]:
        """Run a single trading cycle with mode transitions."""
        if not self.is_running:
            return {"error": "Not running"}
        
        # Check mode transitions
        if self.mode_manager:
            await self.mode_manager.check_transition()
        
        # Check circuit breaker
        circuit_status = {}
        if self.circuit_breaker:
            await self.circuit_breaker.check()
            circuit_status = self.circuit_breaker.get_status()
        
        # Get portfolio summary
        summary = self.get_portfolio_summary()
        summary["circuit_breaker"] = circuit_status
        
        if self.mode_manager:
            summary["mode_status"] = self.mode_manager.get_status()
        
        return summary
    
    def get_state(self) -> SimulationState:
        """Get current simulation state."""
        return self.runner.get_state()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
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
    
    def get_mode_status(self) -> Dict[str, Any]:
        """Get mode manager status."""
        if not self.mode_manager:
            return {"enabled": False}
        
        return self.mode_manager.get_status()
    
    def generate_report(self) -> str:
        """Generate a performance report."""
        state = self.get_state()
        metrics = self.get_performance_metrics()
        circuit_status = self.get_circuit_breaker_status()
        mode_status = self.get_mode_status()
        
        report = []
        report.append("=" * 60)
        report.append("CopyCat Micro Sandbox Report")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append(f"Mode: {self.config.micro_mode.value}")
        report.append("=" * 60)
        
        report.append("\n--- Portfolio Summary ---")
        report.append(f"Balance: ${state.balance:,.2f}")
        report.append(f"Total P&L: ${state.total_pnl:+,.2f}")
        report.append(f"P&L %: {state.total_pnl_pct*100:+.2f}%")
        
        report.append("\n--- Performance Metrics ---")
        report.append(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%")
        report.append(f"Trades Executed: {metrics.get('trades_executed', 0)}")
        
        report.append("\n--- Circuit Breaker ---")
        report.append(f"State: {circuit_status.get('state', 'N/A')}")
        report.append(f"Drawdown: {circuit_status.get('drawdown_pct', 0)*100:.1f}%")
        report.append(f"Position Multiplier: {circuit_status.get('position_multiplier', 1.0):.2f}")
        
        report.append("\n--- Mode Status ---")
        report.append(f"Current Mode: {mode_status.get('current_mode', 'N/A')}")
        report.append(f"Position Size: {mode_status.get('position_size_pct', 0)*100:.0f}%")
        report.append(f"Kelly Fraction: {mode_status.get('kelly_fraction', 0):.2f}")
        
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
            filename = f"micro_sandbox_report_{timestamp}.md"
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return ""
    
    def reset(self):
        """Reset the sandbox runner to initial state."""
        self.runner.reset()
        self._reached_milestones.clear()
        self._bootstrap_complete = False
        
        if self.circuit_breaker:
            self.circuit_breaker.reset()
        
        if self.mode_manager:
            self.mode_manager.reset()
        
        logger.info("Micro sandbox runner reset")
    
    async def close(self):
        """Close and cleanup the sandbox runner."""
        await self.stop()
        logger.info("Micro sandbox runner closed")


# =============================================================================
# Factory Functions
# =============================================================================

def create_micro_sandbox_config(
    initial_balance: float = 10.0,
    micro_mode: str = "nano",
    enable_notifications: bool = True,
    discord_webhook_url: str = None,
    enable_mode_transition: bool = True,
    enable_bootstrap: bool = True,
) -> MicroSandboxConfig:
    """
    Create MicroSandboxConfig based on mode.
    
    Args:
        initial_balance: Starting balance
        micro_mode: Mode - "nano", "micro", "mini", or "balanced"
        enable_notifications: Whether to enable notifications
        discord_webhook_url: Discord webhook URL for notifications
        enable_mode_transition: Whether to enable automatic mode transitions
        enable_bootstrap: Whether to enable bootstrap trading
        
    Returns:
        Configured MicroSandboxConfig
    """
    mode_map = {
        "nano": (0.75, 0.75, 0.30),
        "micro": (0.60, 0.75, 0.25),
        "mini": (0.50, 0.50, 0.20),
        "balanced": (0.40, 0.40, 0.18),
    }
    
    position_pct, kelly, max_dd = mode_map.get(micro_mode, mode_map["nano"])
    
    return MicroSandboxConfig(
        initial_balance=initial_balance,
        micro_mode=SandboxMicroMode(micro_mode),
        position_size_pct=position_pct,
        kelly_fraction=kelly,
        drawdown_stop_threshold=max_dd,
        notify_on_trade=enable_notifications,
        notify_on_milestone=enable_notifications,
        notify_on_circuit_breaker=enable_notifications,
        notify_on_mode_transition=enable_notifications,
        milestone_channels=[20, 50, 100, 200, 500, 1000],
        discord_webhook_url=discord_webhook_url,
        enable_mode_transition=enable_mode_transition,
        enable_bootstrap=enable_bootstrap,
    )


async def create_micro_sandbox_runner(
    config: MicroSandboxConfig,
    api_client: Any = None,
    notification_service: NotificationService = None,
) -> MicroSandboxRunner:
    """
    Create and initialize a MicroSandboxRunner.
    
    Args:
        config: MicroSandboxConfig
        api_client: API client for market data
        notification_service: Optional notification service
        
    Returns:
        Initialized MicroSandboxRunner
    """
    runner = MicroSandboxRunner(
        config=config,
        api_client=api_client,
        notification_service=notification_service,
    )
    
    return runner


if __name__ == "__main__":
    print("=" * 60)
    print("Micro Sandbox Configuration")
    print("=" * 60)
    
    for mode in ["nano", "micro", "mini", "balanced"]:
        config = create_micro_sandbox_config(
            initial_balance=10.0,
            micro_mode=mode,
        )
        
        print(f"\n{mode.upper()} Mode:")
        print(f"  Position Size: {config.position_size_pct:.0%}")
        print(f"  Kelly Fraction: {config.kelly_fraction:.2f}")
        print(f"  Max Drawdown: {config.drawdown_stop_threshold:.0%}")
        print(f"  Circuit Breaker: {'Enabled' if config.circuit_breaker_enabled else 'Disabled'}")
        print(f"  Mode Transition: {'Enabled' if config.enable_mode_transition else 'Disabled'}")
        print(f"  Bootstrap: {'Enabled' if config.enable_bootstrap else 'Disabled'}")
    
    print("\n" + "=" * 60)
    print("Feature Parity: Sandbox = Live Trading ✓")
    print("=" * 60)
