"""
Auto Micro Mode Integration for CopyCat Orchestrator.

Automatically enables Micro Mode when account balance is ≤ $50, providing
aggressive growth strategies optimized for small accounts.

Features:
- Automatic mode detection based on balance
- Seamless integration with existing orchestrator
- Configurable threshold and behavior
- Full feature parity between sandbox and live modes

Usage:
    from orchestrator.auto_micro_mode import AutoMicroModeManager, create_auto_micro_config

    config = create_auto_micro_config(
        micro_mode_threshold=50.0,  # Enable for balances ≤ $50
        default_micro_mode="balanced",  # Default micro mode
    )
    
    manager = AutoMicroModeManager(orchestrator, config)
    
    # Check if micro mode should be active
    if manager.should_enable_micro_mode():
        manager.enable_micro_mode()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from orchestrator.config import OrchestratorConfig, TradingMode
from orchestrator.config_micro import MicroModeLevel, MicroModeConfig, MicroModeEngine
from orchestrator.mode_transition import ModeTransitionManager, TradingModeLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicroModeStrategy(Enum):
    """Strategy for micro mode behavior."""
    NANO_ONLY = "nano_only"  # Only NANO mode for balances ≤ $15
    AGGRESSIVE = "aggressive"  # NANO/MICRO/MINI based on balance
    BALANCED = "balanced"  # MINI/BALANCED based on balance
    AUTO = "auto"  # Automatically select best mode


@dataclass
class AutoMicroModeConfig:
    """Configuration for automatic micro mode."""
    # Threshold for enabling micro mode
    micro_mode_threshold: float = 50.0  # Enable when balance ≤ $50
    micro_mode_enabled: bool = True  # Master toggle
    
    # Strategy
    strategy: MicroModeStrategy = MicroModeStrategy.AUTO
    
    # Default mode when no balance info
    default_micro_mode: str = "balanced"
    
    # Transition settings
    auto_transition_on_growth: bool = True  # Transition up as balance grows
    growth_transition_threshold_pct: float = 0.10  # 10% above next threshold
    transition_cooldown_hours: float = 1.0  # Min time between transitions
    
    # Sandbox/Live parity
    sandbox_micro_enabled: bool = True
    live_micro_enabled: bool = True
    
    # Notification
    notify_on_mode_change: bool = True
    notify_on_milestone: bool = True  # Notify on $20, $50, $100, etc.
    milestone_thresholds: List[float] = field(default_factory=lambda: [20, 50, 100, 200, 500, 1000])
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_drawdown_pct: float = 0.25  # Stop at 25% drawdown
    
    # Bootstrap trading (instant diversification)
    bootstrap_enabled: bool = True
    bootstrap_traders_count: int = 3  # Number of traders to copy instantly
    bootstrap_capital_pct: float = 0.30  # 30% of capital for bootstrap


@dataclass
class MicroModeState:
    """State for auto micro mode."""
    is_enabled: bool = False
    current_mode: str = "standard"
    enabled_at: Optional[datetime] = None
    last_transition_at: Optional[datetime] = None
    last_milestone: float = 0.0
    milestone_reached_at: Optional[datetime] = None
    circuit_breaker_triggered: bool = False
    circuit_breaker_triggered_at: Optional[datetime] = None
    transition_count: int = 0
    bootstrap_traders: List[str] = field(default_factory=list)


class AutoMicroModeManager:
    """
    Manages automatic Micro Mode activation for small accounts.
    
    Automatically switches to Micro Mode when balance is ≤ $50 and provides:
    - Aggressive position sizing (50-75%)
    - Short-term trading focus
    - Bootstrap diversification
    - Circuit breaker protection
    - Milestone notifications
    
    As balance grows, gradually transitions to standard modes.
    """
    
    # Mode thresholds (balance ranges)
    MODE_THRESHOLDS = [
        (0, 15, "nano"),
        (15, 25, "micro"),
        (25, 50, "mini"),
        (50, 100, "balanced"),
        (100, 500, "moderate"),
        (500, float('inf'), "standard"),
    ]
    
    def __init__(
        self,
        orchestrator,
        config: Optional[AutoMicroModeConfig] = None,
        on_mode_change_callback: Optional[Callable] = None,
        on_milestone_callback: Optional[Callable] = None,
        on_circuit_breaker_callback: Optional[Callable] = None,
    ):
        self.orchestrator = orchestrator
        self.config = config or AutoMicroModeConfig()
        self.on_mode_change_callback = on_mode_change_callback
        self.on_milestone_callback = on_milestone_callback
        self.on_circuit_breaker_callback = on_circuit_breaker_callback
        
        self.state = MicroModeState()
        self._micro_engine: Optional[MicroModeEngine] = None
        self._transition_manager: Optional[ModeTransitionManager] = None
        
        # Initialize
        self._check_initial_mode()
        
        logger.info(f"AutoMicroModeManager initialized: enabled={self.state.is_enabled}, "
                   f"strategy={self.config.strategy.value}, threshold=${self.config.micro_mode_threshold}")
    
    def _check_initial_mode(self):
        """Check and set initial mode based on balance."""
        balance = self._get_balance()
        
        if self.config.micro_mode_enabled and balance <= self.config.micro_mode_threshold:
            self._enable_micro_mode()
        else:
            self.state.current_mode = self._get_mode_for_balance(balance)
    
    def _get_balance(self) -> float:
        """Get current account balance."""
        return self.orchestrator.state.total_pnl + self.orchestrator.config.sandbox.initial_balance
    
    def _get_mode_for_balance(self, balance: float) -> str:
        """Get the appropriate mode for a given balance."""
        for min_bal, max_bal, mode in self.MODE_THRESHOLDS:
            if min_bal <= balance < max_bal:
                return mode
        return "standard"
    
    def should_enable_micro_mode(self) -> bool:
        """Check if micro mode should be enabled."""
        if not self.config.micro_mode_enabled:
            return False
        
        balance = self._get_balance()
        return balance <= self.config.micro_mode_threshold
    
    def should_disable_micro_mode(self) -> bool:
        """Check if micro mode should be disabled (balance grew too large)."""
        if not self.state.is_enabled:
            return False
        
        balance = self._get_balance()
        threshold = self.config.micro_mode_threshold * (1 + self.config.growth_transition_threshold_pct)
        return balance > threshold
    
    def _enable_micro_mode(self):
        """Enable micro mode."""
        if self.state.is_enabled:
            return
        
        logger.info(f"Enabling Micro Mode (balance: ${self._get_balance():.2f})")
        
        self.state.is_enabled = True
        self.state.enabled_at = datetime.utcnow()
        self.state.current_mode = self._get_mode_for_balance(self._get_balance())
        
        # Initialize micro mode engine
        self._init_micro_engine()
        
        # Initialize transition manager
        self._init_transition_manager()
        
        # Apply micro mode configuration to orchestrator
        self._apply_micro_config()
        
        # Bootstrap traders if enabled
        if self.config.bootstrap_enabled:
            self._bootstrap_traders()
        
        logger.info(f"Micro Mode enabled: {self.state.current_mode}")
    
    def _disable_micro_mode(self, reason: str = "balance_growth"):
        """Disable micro mode and return to standard mode."""
        if not self.state.is_enabled:
            return
        
        logger.info(f"Disabling Micro Mode: {reason} (balance: ${self._get_balance():.2f})")
        
        old_mode = self.state.current_mode
        self.state.is_enabled = False
        self.state.current_mode = "standard"
        self.state.last_transition_at = datetime.utcnow()
        self.state.transition_count += 1
        
        # Revert to standard configuration
        self._revert_to_standard_config()
        
        logger.info(f"Micro Mode disabled: transitioned from {old_mode} to standard")
    
    def _init_micro_engine(self):
        """Initialize the micro mode engine."""
        balance = self._get_balance()
        
        # Determine micro mode level
        micro_mode = self.state.current_mode
        if micro_mode == "nano":
            micro_level = MicroModeLevel.NANO
        elif micro_mode == "micro":
            micro_level = MicroModeLevel.MICRO
        elif micro_mode == "mini":
            micro_level = MicroModeLevel.MINI
        else:
            micro_level = MicroModeLevel.MICRO  # Default
        
        # Create micro config
        micro_config = MicroModeConfig(
            micro_mode=micro_level,
            initial_balance=balance,
            mode=self.orchestrator.config.mode,
            enable_boost_mode=True,
            enable_hedging=False,
            enable_momentum_filter=False,
            enable_bootstrap=self.config.bootstrap_enabled,
        )
        
        self._micro_engine = MicroModeEngine(micro_config)
        logger.info(f"MicroModeEngine initialized: {micro_level.value}")
    
    def _init_transition_manager(self):
        """Initialize the mode transition manager."""
        self._transition_manager = ModeTransitionManager(
            orchestrator=self.orchestrator,
        )
        logger.info("ModeTransitionManager initialized for micro mode")
    
    def _apply_micro_config(self):
        """Apply micro mode configuration to the orchestrator."""
        if not self._micro_engine:
            return
        
        # Get micro orchestrator config
        micro_orchestrator_config = self._micro_engine.create_orchestrator_config()
        
        # Apply to main orchestrator config
        self.orchestrator.config.copy_trading.position_size_pct = (
            micro_orchestrator_config.copy_trading.position_size_pct
        )
        self.orchestrator.config.copy_trading.kelly_fraction = (
            micro_orchestrator_config.copy_trading.kelly_fraction
        )
        self.orchestrator.config.trader_data_refresh_interval_seconds = 10  # Fast cycles
        
        logger.info(f"Applied micro config: position={self.orchestrator.config.copy_trading.position_size_pct:.0%}")
    
    def _revert_to_standard_config(self):
        """Revert to standard orchestrator configuration."""
        # Reset to default values
        self.orchestrator.config.copy_trading.position_size_pct = 0.10  # 10%
        self.orchestrator.config.copy_trading.kelly_fraction = 0.25
        self.orchestrator.config.trader_data_refresh_interval_seconds = 60
        
        logger.info("Reverted to standard configuration")
    
    def _bootstrap_traders(self):
        """Bootstrap with proven traders for instant diversification."""
        if not self.config.bootstrap_enabled:
            return
        
        # Get bootstrap traders from engine
        bootstrap_traders = self._get_bootstrap_traders()
        
        if bootstrap_traders:
            self.state.bootstrap_traders = bootstrap_traders
            logger.info(f"Bootstrapped {len(bootstrap_traders)} traders: {bootstrap_traders[:3]}...")
    
    def _get_bootstrap_traders(self) -> List[str]:
        """Get list of bootstrap traders."""
        # Use known profitable traders (from config or hardcoded)
        # These would be real Polymarket trader addresses
        from orchestrator.engine import KNOWN_PROFITABLE_TRADERS
        
        # Return top N traders
        return KNOWN_PROFITABLE_TRADERS[:self.config.bootstrap_traders_count]
    
    async def check_and_update(self) -> Dict[str, Any]:
        """
        Check mode status and update if needed.
        
        Called at the end of each trading cycle.
        
        Returns:
            Dict with update details
        """
        result = {
            "mode_changed": False,
            "reason": None,
            "current_mode": self.state.current_mode,
            "balance": self._get_balance(),
        }
        
        # Check for circuit breaker
        if self.config.circuit_breaker_enabled:
            if self._check_circuit_breaker():
                result["circuit_breaker_triggered"] = True
                return result
        
        # Check if should enable micro mode
        if not self.state.is_enabled and self.should_enable_micro_mode():
            self._enable_micro_mode()
            result["mode_changed"] = True
            result["reason"] = "balance_below_threshold"
            await self._notify_mode_change(result)
            return result
        
        # Check if should disable micro mode
        if self.state.is_enabled and self.should_disable_micro_mode():
            self._disable_micro_mode("balance_above_threshold")
            result["mode_changed"] = True
            result["reason"] = "balance_above_threshold"
            await self._notify_mode_change(result)
            return result
        
        # Check for mode transitions within micro mode
        if self.state.is_enabled and self.config.auto_transition_on_growth:
            transition_result = await self._check_transition()
            if transition_result:
                result["mode_changed"] = True
                result["reason"] = transition_result.reason
                result["from_mode"] = transition_result.from_mode.value
                result["to_mode"] = transition_result.to_mode.value
                await self._notify_mode_change(result)
                return result
        
        # Check for milestones
        milestone_result = self._check_milestones()
        if milestone_result:
            result["milestone_reached"] = milestone_result
            await self._notify_milestone(milestone_result)
        
        return result
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be triggered."""
        current_drawdown = self.orchestrator.state.max_drawdown
        
        if current_drawdown >= self.config.circuit_breaker_drawdown_pct:
            logger.warning(
                f"CIRCUIT BREAKER TRIGGERED: Drawdown {current_drawdown:.1%} "
                f"exceeds threshold {self.config.circuit_breaker_drawdown_pct:.1%}"
            )
            
            self.state.circuit_breaker_triggered = True
            self.state.circuit_breaker_triggered_at = datetime.utcnow()
            
            # Call callback if registered
            if self.on_circuit_breaker_callback:
                try:
                    if asyncio.iscoroutinefunction(self.on_circuit_breaker_callback):
                        await self.on_circuit_breaker_callback(current_drawdown)
                    else:
                        self.on_circuit_breaker_callback(current_drawdown)
                except Exception as e:
                    logger.error(f"Error in circuit breaker callback: {e}")
            
            return True
        
        return False
    
    async def _check_transition(self) -> Optional[Any]:
        """Check for mode transitions within micro mode."""
        if not self._transition_manager:
            return None
        
        # Check cooldown
        if self.state.last_transition_at:
            time_since_transition = (datetime.utcnow() - self.state.last_transition_at).total_seconds()
            cooldown_seconds = self.config.transition_cooldown_hours * 3600
            if time_since_transition < cooldown_seconds:
                return None
        
        # Check for transition
        transition = await self._transition_manager.check_transition()
        
        if transition:
            # Update state
            old_mode = self.state.current_mode
            self.state.current_mode = transition.to_mode.value
            self.state.last_transition_at = datetime.utcnow()
            self.state.transition_count += 1
            
            # Apply new configuration
            self._transition_manager._apply_mode_config(
                self._transition_manager.get_threshold_for_mode(transition.to_mode)
            )
            
            logger.info(f"Micro mode transition: {old_mode} → {transition.to_mode.value}")
        
        return transition
    
    def _check_milestones(self) -> Optional[float]:
        """Check if a milestone has been reached."""
        if not self.config.notify_on_milestone:
            return None
        
        balance = self._get_balance()
        
        for milestone in sorted(self.config.milestone_thresholds):
            if balance >= milestone and self.state.last_milestone < milestone:
                self.state.last_milestone = milestone
                self.state.milestone_reached_at = datetime.utcnow()
                logger.info(f"MILESTONE REACHED: ${milestone} (balance: ${balance:.2f})")
                return milestone
        
        return None
    
    async def _notify_mode_change(self, result: Dict[str, Any]):
        """Notify about mode change."""
        if not self.config.notify_on_mode_change:
            return
        
        logger.info(f"Mode change notification: {result}")
        
        # Call callback if registered
        if self.on_mode_change_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_mode_change_callback):
                    await self.on_mode_change_callback(result)
                else:
                    self.on_mode_change_callback(result)
            except Exception as e:
                logger.error(f"Error in mode change callback: {e}")
    
    async def _notify_milestone(self, milestone: float):
        """Notify about milestone reached."""
        if not self.config.notify_on_milestone:
            return
        
        logger.info(f"Milestone notification: ${milestone}")
        
        # Call callback if registered
        if self.on_milestone_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_milestone_callback):
                    await self.on_milestone_callback(milestone)
                else:
                    self.on_milestone_callback(milestone)
            except Exception as e:
                logger.error(f"Error in milestone callback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current micro mode status."""
        balance = self._get_balance()
        
        return {
            "is_enabled": self.state.is_enabled,
            "current_mode": self.state.current_mode,
            "balance": balance,
            "strategy": self.config.strategy.value,
            "threshold": self.config.micro_mode_threshold,
            "position_size_pct": self.orchestrator.config.copy_trading.position_size_pct,
            "kelly_fraction": self.orchestrator.config.copy_trading.kelly_fraction,
            "cycle_interval_seconds": self.orchestrator.config.trader_data_refresh_interval_seconds,
            "circuit_breaker": {
                "enabled": self.config.circuit_breaker_enabled,
                "threshold": self.config.circuit_breaker_drawdown_pct,
                "triggered": self.state.circuit_breaker_triggered,
            },
            "bootstrap": {
                "enabled": self.config.bootstrap_enabled,
                "traders_count": self.config.bootstrap_traders_count,
                "traders": self.state.bootstrap_traders[:3],  # First 3
            },
            "milestones": {
                "next_threshold": self._get_next_milestone(),
                "last_reached": self.state.last_milestone,
                "thresholds": self.config.milestone_thresholds,
            },
            "statistics": {
                "enabled_at": self.state.enabled_at.isoformat() if self.state.enabled_at else None,
                "transition_count": self.state.transition_count,
                "last_transition": self.state.last_transition_at.isoformat() if self.state.last_transition_at else None,
            },
        }
    
    def _get_next_milestone(self) -> Optional[float]:
        """Get the next milestone threshold."""
        balance = self._get_balance()
        
        for milestone in sorted(self.config.milestone_thresholds):
            if balance < milestone:
                return milestone
        
        return None
    
    def get_recommended_mode(self, balance: float) -> Dict[str, Any]:
        """Get recommended mode for a given balance."""
        mode = self._get_mode_for_balance(balance)
        
        # Get mode-specific settings
        if mode == "nano":
            position_pct, kelly, drawdown = 0.75, 0.75, 0.30
        elif mode == "micro":
            position_pct, kelly, drawdown = 0.60, 0.75, 0.25
        elif mode == "mini":
            position_pct, kelly, drawdown = 0.50, 0.50, 0.20
        elif mode == "balanced":
            position_pct, kelly, drawdown = 0.40, 0.40, 0.18
        elif mode == "moderate":
            position_pct, kelly, drawdown = 0.35, 0.35, 0.15
        else:
            position_pct, kelly, drawdown = 0.10, 0.25, 0.12
        
        return {
            "mode": mode,
            "position_size_pct": position_pct,
            "kelly_fraction": kelly,
            "max_drawdown_threshold": drawdown,
            "is_micro_mode": balance <= self.config.micro_mode_threshold,
        }
    
    def reset(self):
        """Reset micro mode manager to initial state."""
        self.state = MicroModeState()
        self._micro_engine = None
        self._transition_manager = None
        
        self._check_initial_mode()
        
        logger.info("AutoMicroModeManager reset")


# =============================================================================
# Factory Functions
# =============================================================================

def create_auto_micro_config(
    micro_mode_threshold: float = 50.0,
    strategy: str = "auto",
    enable_bootstrap: bool = True,
    bootstrap_traders_count: int = 3,
    circuit_breaker_drawdown: float = 0.25,
    notify_on_milestone: bool = True,
) -> AutoMicroModeConfig:
    """Factory function to create AutoMicroModeConfig."""
    
    strategy_map = {
        "nano_only": MicroModeStrategy.NANO_ONLY,
        "aggressive": MicroModeStrategy.AGGRESSIVE,
        "balanced": MicroModeStrategy.BALANCED,
        "auto": MicroModeStrategy.AUTO,
    }
    
    return AutoMicroModeConfig(
        micro_mode_threshold=micro_mode_threshold,
        strategy=strategy_map.get(strategy, MicroModeStrategy.AUTO),
        bootstrap_enabled=enable_bootstrap,
        bootstrap_traders_count=bootstrap_traders_count,
        circuit_breaker_drawdown_pct=circuit_breaker_drawdown,
        notify_on_milestone=notify_on_milestone,
    )


def create_auto_micro_manager(
    orchestrator,
    micro_mode_threshold: float = 50.0,
    on_mode_change: Callable = None,
    on_milestone: Callable = None,
) -> AutoMicroModeManager:
    """Factory function to create AutoMicroModeManager."""
    config = create_auto_micro_config(
        micro_mode_threshold=micro_mode_threshold,
    )
    
    return AutoMicroModeManager(
        orchestrator=orchestrator,
        config=config,
        on_mode_change_callback=on_mode_change,
        on_milestone_callback=on_milestone,
    )


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AUTO MICRO MODE INTEGRATION - DEMO")
    print("=" * 70)
    
    # Create a mock orchestrator
    class MockOrchestrator:
        def __init__(self, initial_balance=10.0):
            self.config = type('Config', (), {
                'sandbox': type('Sandbox', (), {'initial_balance': initial_balance})(),
                'copy_trading': type('Trading', (), {
                    'position_size_pct': 0.10,
                    'kelly_fraction': 0.25,
                })(),
                'trader_selection': type('Selection', (), {
                    'growth_min_total_pnl': 25.0,
                    'growth_min_growth_rate': 0.005,
                    'growth_max_drawdown': 0.50,
                })(),
                'boost_mode': type('Boost', (), {'enabled': False})(),
                'mode': type('Mode', (), {'value': 'sandbox'})(),
            })()
            self.state = type('State', (), {
                'total_pnl': 0.0,
                'cycle_count': 0,
                'max_drawdown': 0.0,
            })()
    
    # Test with different balances
    for balance in [10, 25, 50, 75, 100, 200]:
        orchestrator = MockOrchestrator(initial_balance=balance)
        manager = create_auto_micro_manager(orchestrator, micro_mode_threshold=50.0)
        
        status = manager.get_status()
        recommended = manager.get_recommended_mode(balance)
        
        print(f"\n{'='*70}")
        print(f"BALANCE: ${balance}")
        print(f"{'='*70}")
        print(f"Micro Mode Enabled: {status['is_enabled']}")
        print(f"Current Mode: {status['current_mode']}")
        print(f"Position Size: {status['position_size_pct']:.0%}")
        print(f"Kelly Fraction: {status['kelly_fraction']:.2f}")
        print(f"\nRecommended:")
        print(f"  Mode: {recommended['mode']}")
        print(f"  Position: {recommended['position_size_pct']:.0%}")
        print(f"  Kelly: {recommended['kelly_fraction']:.2f}")
    
    print("\n" + "=" * 70)
    print("MODE THRESHOLDS:")
    print("=" * 70)
    print(f"{'Mode':<12} | {'Balance Range':<20} | {'Position':<10} | {'Kelly':<6} | {'Drawdown':<10}")
    print("-" * 70)
    print(f"{'NANO':<12} | $0 - $15{'':<11} | 75%{'':<5} | 0.75  | 30%")
    print(f"{'MICRO':<12} | $15 - $25{'':<10} | 60%{'':<5} | 0.75  | 25%")
    print(f"{'MINI':<12} | $25 - $50{'':<10} | 50%{'':<5} | 0.50  | 20%")
    print(f"{'BALANCED':<12} | $50 - $100{'':<9} | 40%{'':<5} | 0.40  | 18%")
    print(f"{'MODERATE':<12} | $100 - $500{'':<8} | 35%{'':<5} | 0.35  | 15%")
    print(f"{'STANDARD':<12} | $500+{'':<14} | 10%{'':<5} | 0.25  | 12%")
    print("=" * 70)
