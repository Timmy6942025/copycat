"""
Mode Transition Manager for CopyCat.

Automatically transitions between trading modes as account balance grows:
NANO → MICRO → MINI → BALANCED → AGGRESSIVE

Usage:
    from orchestrator.mode_transition import ModeTransitionManager, TransitionConfig
    
    config = TransitionConfig()
    manager = ModeTransitionManager(orchestrator, config)
    
    # Check for transitions after each cycle
    await manager.check_transition()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from orchestrator.config import OrchestratorConfig, TradingMode
from orchestrator.config_micro import MicroModeLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingModeLevel(Enum):
    NANO = "nano"
    MICRO = "micro"
    MINI = "mini"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


@dataclass
class ModeThreshold:
    """Threshold for mode transition."""
    min_balance: float
    max_balance: float
    target_mode: TradingModeLevel
    min_duration_cycles: int = 10  # Must maintain balance for N cycles
    position_size_pct: float = 0.50
    kelly_fraction: float = 0.50
    max_drawdown_threshold: float = 0.25
    enable_boost_mode: bool = True
    enable_hedging: bool = False
    enable_momentum_filter: bool = False


@dataclass
class TransitionConfig:
    """Configuration for mode transitions."""
    # Mode thresholds (balance ranges)
    thresholds: List[ModeThreshold] = field(default_factory=list)
    
    # Hysteresis to prevent oscillation
    hysteresis_pct: float = 0.10  # Must exceed threshold by 10% to upgrade
    downgrade_hysteresis_pct: float = 0.20  # Must fall 20% below to downgrade
    
    # Transition behavior
    require_profitable_before_upgrade: bool = True
    min_profitable_cycles: int = 5  # Cycles with positive P&L before upgrade
    require_stable_drawdown: bool = True
    max_drawdown_for_upgrade: float = 0.15  # Must be below this to upgrade
    
    # Notification
    notify_on_transition: bool = True
    min_balance_change_pct_for_notification: float = 0.05
    
    # Safety
    max_transitions_per_day: int = 4
    emergency_stop_threshold: float = 0.30  # Stop if drawdown exceeds 30%


def create_default_thresholds() -> List[ModeThreshold]:
    """Create default mode thresholds."""
    return [
        ModeThreshold(
            min_balance=0,
            max_balance=15,
            target_mode=TradingModeLevel.NANO,
            min_duration_cycles=5,
            position_size_pct=0.75,
            kelly_fraction=0.75,
            max_drawdown_threshold=0.30,
            enable_boost_mode=True,
            enable_hedging=False,
            enable_momentum_filter=False,
        ),
        ModeThreshold(
            min_balance=15,
            max_balance=25,
            target_mode=TradingModeLevel.MICRO,
            min_duration_cycles=10,
            position_size_pct=0.60,
            kelly_fraction=0.75,
            max_drawdown_threshold=0.25,
            enable_boost_mode=True,
            enable_hedging=False,
            enable_momentum_filter=False,
        ),
        ModeThreshold(
            min_balance=25,
            max_balance=50,
            target_mode=TradingModeLevel.MINI,
            min_duration_cycles=15,
            position_size_pct=0.50,
            kelly_fraction=0.50,
            max_drawdown_threshold=0.20,
            enable_boost_mode=True,
            enable_hedging=False,
            enable_momentum_filter=False,
        ),
        ModeThreshold(
            min_balance=50,
            max_balance=200,
            target_mode=TradingModeLevel.BALANCED,
            min_duration_cycles=20,
            position_size_pct=0.40,
            kelly_fraction=0.40,
            max_drawdown_threshold=0.18,
            enable_boost_mode=True,
            enable_hedging=False,
            enable_momentum_filter=True,
        ),
        ModeThreshold(
            min_balance=200,
            max_balance=1000,
            target_mode=TradingModeLevel.AGGRESSIVE,
            min_duration_cycles=30,
            position_size_pct=0.35,
            kelly_fraction=0.35,
            max_drawdown_threshold=0.15,
            enable_boost_mode=False,
            enable_hedging=True,
            enable_momentum_filter=True,
        ),
        ModeThreshold(
            min_balance=1000,
            max_balance=float('inf'),
            target_mode=TradingModeLevel.CONSERVATIVE,
            min_duration_cycles=50,
            position_size_pct=0.25,
            kelly_fraction=0.25,
            max_drawdown_threshold=0.12,
            enable_boost_mode=False,
            enable_hedging=True,
            enable_momentum_filter=True,
        ),
    ]


@dataclass
class TransitionRecord:
    """Record of a mode transition."""
    from_mode: TradingModeLevel
    to_mode: TradingModeLevel
    balance: float
    previous_balance: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_in_previous_mode_cycles: int = 0


class ModeTransitionManager:
    """
    Manages automatic transitions between trading modes based on account balance.
    
    Key features:
    - Balance-based mode selection with configurable thresholds
    - Hysteresis to prevent rapid oscillation between modes
    - Profitability requirements for upgrades
    - Drawdown protection
    - Transition history tracking
    """
    
    def __init__(
        self,
        orchestrator,
        config: Optional[TransitionConfig] = None,
        thresholds: Optional[List[ModeThreshold]] = None,
        on_transition_callback: Optional[Callable] = None,
    ):
        self.orchestrator = orchestrator
        self.config = config or TransitionConfig()
        self.thresholds = thresholds or create_default_thresholds()
        self.on_transition_callback = on_transition_callback
        
        # Current state
        self.current_mode: TradingModeLevel = TradingModeLevel.NANO
        self._mode_start_cycle: int = 0
        self._consecutive_profitable_cycles: int = 0
        self._consecutive_unprofitable_cycles: int = 0
        self._transitions_today: int = 0
        self._last_transition_date = datetime.utcnow().date()
        
        # Transition history
        self._transition_history: List[TransitionRecord] = []
        
        # Balance tracking for hysteresis
        self._peak_balance: float = 0.0
        self._trough_balance: float = float('inf')
        
        # Initialize with current balance
        self._update_balance_tracking()
        
        # Determine initial mode
        self._determine_initial_mode()
        
        logger.info(f"ModeTransitionManager initialized in {self.current_mode.value} mode")
    
    def _update_balance_tracking(self):
        """Update peak and trough balance tracking."""
        portfolio = self.orchestrator.state.total_pnl + self.orchestrator.config.sandbox.initial_balance
        
        if portfolio > self._peak_balance:
            self._peak_balance = portfolio
        if portfolio < self._trough_balance:
            self._trough_balance = portfolio
    
    def _determine_initial_mode(self):
        """Determine the initial mode based on current balance."""
        portfolio = self.orchestrator.state.total_pnl + self.orchestrator.config.sandbox.initial_balance
        
        for threshold in self.thresholds:
            if threshold.min_balance <= portfolio < threshold.max_balance:
                self.current_mode = threshold.target_mode
                self._mode_start_cycle = self.orchestrator.state.cycle_count
                logger.info(f"Initial mode set to {self.current_mode.value} (balance: ${portfolio:.2f})")
                return
        
        # Default to NANO if no threshold matches
        self.current_mode = TradingModeLevel.NANO
    
    def get_current_threshold(self) -> ModeThreshold:
        """Get the threshold for the current mode."""
        for threshold in self.thresholds:
            if threshold.target_mode == self.current_mode:
                return threshold
        return self.thresholds[0]  # Return NANO threshold as default
    
    def get_threshold_for_mode(self, mode: TradingModeLevel) -> Optional[ModeThreshold]:
        """Get threshold for a specific mode."""
        for threshold in self.thresholds:
            if threshold.target_mode == mode:
                return threshold
        return None
    
    async def check_transition(self) -> Optional[TransitionRecord]:
        """
        Check if a mode transition is needed.
        
        Called at the end of each trading cycle.
        
        Returns:
            TransitionRecord if transition occurred, None otherwise
        """
        # Reset daily counter if new day
        today = datetime.utcnow().date()
        if today != self._last_transition_date:
            self._transitions_today = 0
            self._last_transition_date = today
        
        # Check for emergency stop
        if self._check_emergency_stop():
            return None
        
        # Get current state
        current_balance = self.orchestrator.state.total_pnl + self.orchestrator.config.sandbox.initial_balance
        current_cycle = self.orchestrator.state.cycle_count
        current_threshold = self.get_current_threshold()
        
        # Update balance tracking
        self._update_balance_tracking()
        
        # Check profitability
        if self.orchestrator.state.total_pnl > 0:
            self._consecutive_profitable_cycles += 1
            self._consecutive_unprofitable_cycles = 0
        else:
            self._consecutive_unprofitable_cycles += 1
            self._consecutive_profitable_cycles = 0
        
        # Check for upgrade
        upgrade = self._check_for_upgrade(current_balance, current_cycle)
        if upgrade:
            return await self._execute_transition(upgrade, current_balance)
        
        # Check for downgrade
        downgrade = self._check_for_downgrade(current_balance, current_cycle)
        if downgrade:
            return await self._execute_transition(downgrade, current_balance)
        
        return None
    
    def _check_for_upgrade(
        self,
        current_balance: float,
        current_cycle: int
    ) -> Optional[TradingModeLevel]:
        """Check if we should upgrade to a higher mode."""
        current_threshold = self.get_current_threshold()
        
        for threshold in self.thresholds:
            # Skip if not a higher mode
            if threshold.target_mode.value <= self.current_mode.value:
                continue
            
            # Check if balance meets upgrade threshold
            upgrade_threshold = threshold.min_balance * (1 + self.config.hysteresis_pct)
            
            if current_balance >= upgrade_threshold:
                # Check duration requirement
                cycles_in_mode = current_cycle - self._mode_start_cycle
                if cycles_in_mode < threshold.min_duration_cycles:
                    continue
                
                # Check profitability requirement
                if self.config.require_profitable_before_upgrade:
                    if self._consecutive_profitable_cycles < self.config.min_profitable_cycles:
                        continue
                
                # Check drawdown requirement
                if self.config.require_stable_drawdown:
                    if self.orchestrator.state.max_drawdown > self.config.max_drawdown_for_upgrade:
                        continue
                
                # Check daily transition limit
                if self._transitions_today >= self.config.max_transitions_per_day:
                    continue
                
                return threshold.target_mode
        
        return None
    
    def _check_for_downgrade(
        self,
        current_balance: float,
        current_cycle: int
    ) -> Optional[TradingModeLevel]:
        """Check if we should downgrade to a lower mode."""
        current_threshold = self.get_current_threshold()
        
        for threshold in reversed(self.thresholds):
            # Skip if not a lower mode
            if threshold.target_mode.value >= self.current_mode.value:
                continue
            
            # Check if balance has fallen below downgrade threshold
            downgrade_threshold = current_threshold.min_balance * (1 - self.config.downgrade_hysteresis_pct)
            
            if current_balance < downgrade_threshold:
                return threshold.target_mode
        
        return None
    
    def _check_emergency_stop(self) -> bool:
        """Check if emergency stop is needed."""
        # Emergency stop if drawdown exceeds threshold
        if self.orchestrator.state.max_drawdown >= self.config.emergency_stop_threshold:
            logger.warning(
                f"EMERGENCY STOP: Drawdown {self.orchestrator.state.max_drawdown:.1%} "
                f"exceeds threshold {self.config.emergency_stop_threshold:.1%}"
            )
            return True
        return False
    
    async def _execute_transition(
        self,
        new_mode: TradingModeLevel,
        current_balance: float
    ) -> TransitionRecord:
        """Execute a mode transition."""
        old_mode = self.current_mode
        old_threshold = self.get_current_threshold()
        new_threshold = self.get_threshold_for_mode(new_mode)
        
        # Calculate reason
        balance_change = (current_balance - self._peak_balance) / self._peak_balance * 100 if self._peak_balance > 0 else 0
        
        if new_mode.value > old_mode.value:
            reason = f"Balance grew to ${current_balance:.2f} (+{balance_change:.1f}%)"
        else:
            reason = f"Balance dropped to ${current_balance:.2f} ({balance_change:.1f}%)"
        
        # Create transition record
        record = TransitionRecord(
            from_mode=old_mode,
            to_mode=new_mode,
            balance=current_balance,
            previous_balance=self._peak_balance,
            reason=reason,
            duration_in_previous_mode_cycles=self.orchestrator.state.cycle_count - self._mode_start_cycle,
        )
        
        # Execute transition
        self.current_mode = new_mode
        self._mode_start_cycle = self.orchestrator.state.cycle_count
        self._transitions_today += 1
        self._transition_history.append(record)
        
        # Reset balance tracking for new mode
        self._peak_balance = current_balance
        self._trough_balance = current_balance
        
        # Apply new configuration
        self._apply_mode_config(new_threshold)
        
        logger.info(
            f"MODE TRANSITION: {old_mode.value.upper()} → {new_mode.value.upper()} "
            f"(balance: ${current_balance:.2f}, reason: {reason})"
        )
        
        # Call callback if registered
        if self.on_transition_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_transition_callback):
                    await self.on_transition_callback(record)
                else:
                    self.on_transition_callback(record)
            except Exception as e:
                logger.error(f"Error in transition callback: {e}")
        
        return record
    
    def _apply_mode_config(self, threshold: ModeThreshold):
        """Apply configuration changes for the new mode."""
        config = self.orchestrator.config
        
        # Update position sizing
        config.copy_trading.position_size_pct = threshold.position_size_pct
        config.copy_trading.kelly_fraction = threshold.kelly_fraction
        
        # Update boost mode
        config.boost_mode.enabled = threshold.enable_boost_mode
        
        # Update trader selection based on mode
        if threshold.target_mode in [TradingModeLevel.NANO, TradingModeLevel.MICRO]:
            # Lenient selection for small accounts
            config.trader_selection.growth_min_total_pnl = 10.0
            config.trader_selection.growth_min_growth_rate = 0.002
            config.trader_selection.growth_max_drawdown = 0.75
        elif threshold.target_mode in [TradingModeLevel.MINI, TradingModeLevel.BALANCED]:
            # Standard selection
            config.trader_selection.growth_min_total_pnl = 25.0
            config.trader_selection.growth_min_growth_rate = 0.005
            config.trader_selection.growth_max_drawdown = 0.50
        else:
            # Conservative selection for larger accounts
            config.trader_selection.growth_min_total_pnl = 100.0
            config.trader_selection.growth_min_growth_rate = 0.010
            config.trader_selection.growth_max_drawdown = 0.30
        
        # Update max drawdown threshold
        config.trader_selection.growth_max_drawdown = threshold.max_drawdown_threshold
        
        logger.debug(f"Applied mode config: position_size={threshold.position_size_pct:.0%}, "
                    f"kelly={threshold.kelly_fraction:.2f}, boost={threshold.enable_boost_mode}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current mode status."""
        current_balance = self.orchestrator.state.total_pnl + self.orchestrator.config.sandbox.initial_balance
        current_threshold = self.get_current_threshold()
        
        return {
            "current_mode": self.current_mode.value,
            "position_size_pct": current_threshold.position_size_pct,
            "kelly_fraction": current_threshold.kelly_fraction,
            "max_drawdown_threshold": current_threshold.max_drawdown_threshold,
            "enable_boost_mode": current_threshold.enable_boost_mode,
            "enable_hedging": current_threshold.enable_hedging,
            "enable_momentum_filter": current_threshold.enable_momentum_filter,
            "cycles_in_mode": self.orchestrator.state.cycle_count - self._mode_start_cycle,
            "profitable_cycles_streak": self._consecutive_profitable_cycles,
            "balance": current_balance,
            "peak_balance": self._peak_balance,
            "trough_balance": self._trough_balance,
            "transitions_today": self._transitions_today,
            "total_transitions": len(self._transition_history),
            "next_upgrade_threshold": self._get_next_upgrade_threshold(),
            "next_downgrade_threshold": self._get_next_downgrade_threshold(),
        }
    
    def _get_next_upgrade_threshold(self) -> Optional[float]:
        """Get the balance threshold for the next upgrade."""
        for threshold in self.thresholds:
            if threshold.target_mode.value > self.current_mode.value:
                return threshold.min_balance * (1 + self.config.hysteresis_pct)
        return None
    
    def _get_next_downgrade_threshold(self) -> Optional[float]:
        """Get the balance threshold for the next downgrade."""
        current_threshold = self.get_current_threshold()
        return current_threshold.min_balance * (1 - self.config.downgrade_hysteresis_pct)
    
    def get_transition_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent transition history."""
        recent = self._transition_history[-limit:]
        return [
            {
                "from_mode": t.from_mode.value,
                "to_mode": t.to_mode.value,
                "balance": t.balance,
                "reason": t.reason,
                "timestamp": t.timestamp.isoformat(),
                "duration_cycles": t.duration_in_previous_mode_cycles,
            }
            for t in recent
        ]
    
    def get_recommended_mode(self, balance: float) -> Dict[str, Any]:
        """Get recommended mode for a given balance (no state change)."""
        for threshold in self.thresholds:
            if threshold.min_balance <= balance < threshold.max_balance:
                return {
                    "recommended_mode": threshold.target_mode.value,
                    "position_size_pct": threshold.position_size_pct,
                    "kelly_fraction": threshold.kelly_fraction,
                    "max_drawdown_threshold": threshold.max_drawdown_threshold,
                    "expected_monthly_return": self._estimate_monthly_return(threshold),
                    "risk_level": self._get_risk_level(threshold),
                }
        
        # Default to CONSERVATIVE for very large balances
        return {
            "recommended_mode": TradingModeLevel.CONSERVATIVE.value,
            "position_size_pct": 0.25,
            "kelly_fraction": 0.25,
            "max_drawdown_threshold": 0.12,
            "expected_monthly_return": "5-8%",
            "risk_level": "low",
        }
    
    def _estimate_monthly_return(self, threshold: ModeThreshold) -> str:
        """Estimate monthly return based on mode."""
        if threshold.position_size_pct >= 0.70:
            return "25-50%"
        elif threshold.position_size_pct >= 0.50:
            return "15-30%"
        elif threshold.position_size_pct >= 0.35:
            return "10-20%"
        else:
            return "5-12%"
    
    def _get_risk_level(self, threshold: ModeThreshold) -> str:
        """Get risk level description."""
        if threshold.position_size_pct >= 0.70:
            return "very_high"
        elif threshold.position_size_pct >= 0.50:
            return "high"
        elif threshold.position_size_pct >= 0.35:
            return "medium_high"
        elif threshold.position_size_pct >= 0.25:
            return "medium"
        else:
            return "low"
    
    def reset(self):
        """Reset mode manager to initial state."""
        self.current_mode = TradingModeLevel.NANO
        self._mode_start_cycle = 0
        self._consecutive_profitable_cycles = 0
        self._consecutive_unprofitable_cycles = 0
        self._transitions_today = 0
        self._peak_balance = 0.0
        self._trough_balance = float('inf')
        self._transition_history.clear()
        
        self._determine_initial_mode()
        logger.info("ModeTransitionManager reset")


# =============================================================================
# Factory Functions
# =============================================================================

def create_transition_config(
    upgrade_hysteresis: float = 0.10,
    downgrade_hysteresis: float = 0.20,
    require_profitable: bool = True,
    min_profitable_cycles: int = 5,
    max_drawdown_for_upgrade: float = 0.15,
    emergency_stop_threshold: float = 0.30,
) -> TransitionConfig:
    """Factory function to create TransitionConfig."""
    return TransitionConfig(
        hysteresis_pct=upgrade_hysteresis,
        downgrade_hysteresis_pct=downgrade_hysteresis,
        require_profitable_before_upgrade=require_profitable,
        min_profitable_cycles=min_profitable_cycles,
        require_stable_drawdown=True,
        max_drawdown_for_upgrade=max_drawdown_for_upgrade,
        emergency_stop_threshold=emergency_stop_threshold,
    )


def create_mode_manager(
    orchestrator,
    balance: float = None,
    on_transition: Callable = None,
) -> ModeTransitionManager:
    """Factory function to create ModeTransitionManager."""
    config = create_transition_config()
    
    manager = ModeTransitionManager(
        orchestrator=orchestrator,
        config=config,
        on_transition_callback=on_transition,
    )
    
    return manager


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("MODE TRANSITION MANAGER - EXAMPLE OUTPUT")
    print("=" * 60)
    
    # Show default thresholds
    print("\nDefault Mode Thresholds:")
    print("-" * 60)
    for threshold in create_default_thresholds():
        print(f"{threshold.target_mode.value.upper():15} | "
              f"${threshold.min_balance:>6.0f}-${threshold.max_balance:<6.0f} | "
              f"Position: {threshold.position_size_pct:.0%} | "
              f"Kelly: {threshold.kelly_fraction:.2f} | "
              f"Drawdown: {threshold.max_drawdown_threshold:.0%}")
    
    print("\n" + "=" * 60)
    print("Recommendations for Different Balances:")
    print("-" * 60)
    
    # Create a mock orchestrator for testing
    class MockOrchestrator:
        def __init__(self):
            self.config = type('Config', (), {
                'sandbox': type('Sandbox', (), {'initial_balance': 10.0})(),
                'copy_trading': type('Trading', (), {
                    'position_size_pct': 0.50,
                    'kelly_fraction': 0.50,
                })(),
                'trader_selection': type('Selection', (), {
                    'growth_min_total_pnl': 25.0,
                    'growth_min_growth_rate': 0.005,
                    'growth_max_drawdown': 0.50,
                })(),
                'boost_mode': type('Boost', (), {'enabled': True})(),
            })()
            self.state = type('State', (), {
                'total_pnl': 0.0,
                'cycle_count': 0,
                'max_drawdown': 0.0,
            })()
    
    mock_orchestrator = MockOrchestrator()
    manager = create_mode_manager(mock_orchestrator)
    
    for balance in [10, 15, 25, 50, 100, 250, 500, 1000, 2000]:
        recommendation = manager.get_recommended_mode(balance)
        print(f"${balance:>5}: {recommendation['recommended_mode'].upper():12} | "
              f"Position: {recommendation['position_size_pct']:.0%} | "
              f"Risk: {recommendation['risk_level']}")
    
    print("\n" + "=" * 60)
