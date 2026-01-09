"""
Micro-Specific Circuit Breaker for CopyCat.

Provides drawdown protection and position sizing reduction for small accounts.

Usage:
    from orchestrator.circuit_breaker import MicroCircuitBreaker, MicroCircuitBreakerConfig
    
    config = MicroCircuitBreakerConfig()
    breaker = MicroCircuitBreaker(config, orchestrator)
    
    # Check status after each cycle
    position_multiplier = await breaker.check()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.mode_transition import TradingModeLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    WARNING = "warning"    # Reduced position sizing
    OPEN = "open"          # Trading paused
    EMERGENCY = "emergency"  # Emergency stop


@dataclass
class DrawdownThreshold:
    """Drawdown threshold configuration."""
    drawdown_pct: float
    position_reduction_pct: float  # Reduce positions by this %
    state: CircuitState
    message: str


@dataclass
class MicroCircuitBreakerConfig:
    """Configuration for micro-specific circuit breaker."""
    # Mode-specific thresholds
    nano_thresholds: List[DrawdownThreshold] = field(default_factory=list)
    micro_thresholds: List[DrawdownThreshold] = field(default_factory=list)
    mini_thresholds: List[DrawdownThreshold] = field(default_factory=list)
    balanced_thresholds: List[DrawdownThreshold] = field(default_factory=list)
    
    # General settings
    check_interval_seconds: float = 10.0
    require_consecutive_cycles: int = 2  # Must hit threshold N consecutive cycles
    
    # Recovery
    auto_recovery_enabled: bool = True
    recovery_check_interval_cycles: int = 10  # Check for recovery every N cycles
    recovery_drawdown_reduction: float = 0.10  # Reduce drawdown by 10% for recovery
    
    # Emergency stop
    emergency_stop_threshold: float = 0.30  # 30% drawdown = emergency
    require_manual_reset: bool = False
    
    # Notifications
    notify_on_state_change: bool = True
    notify_on_warning: bool = True


def create_default_thresholds() -> Dict[TradingModeLevel, List[DrawdownThreshold]]:
    """Create default drawdown thresholds for each mode."""
    return {
        TradingModeLevel.NANO: [
            DrawdownThreshold(
                drawdown_pct=0.10,  # 10%
                position_reduction_pct=0.50,  # Reduce positions by 50%
                state=CircuitState.WARNING,
                message="Drawdown exceeded 10% - reducing position sizes"
            ),
            DrawdownThreshold(
                drawdown_pct=0.15,  # 15%
                position_reduction_pct=0.70,  # Reduce positions by 70%
                state=CircuitState.WARNING,
                message="Drawdown exceeded 15% - further position reduction"
            ),
            DrawdownThreshold(
                drawdown_pct=0.20,  # 20%
                position_reduction_pct=0.85,  # Reduce positions by 85%
                state=CircuitState.WARNING,
                message="Drawdown exceeded 20% - extreme position reduction"
            ),
            DrawdownThreshold(
                drawdown_pct=0.25,  # 25%
                position_reduction_pct=1.00,  # Stop trading
                state=CircuitState.OPEN,
                message="Drawdown exceeded 25% - trading paused"
            ),
        ],
        TradingModeLevel.MICRO: [
            DrawdownThreshold(
                drawdown_pct=0.08,  # 8%
                position_reduction_pct=0.40,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 8% - reducing position sizes"
            ),
            DrawdownThreshold(
                drawdown_pct=0.12,  # 12%
                position_reduction_pct=0.60,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 12% - further position reduction"
            ),
            DrawdownThreshold(
                drawdown_pct=0.18,  # 18%
                position_reduction_pct=0.80,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 18% - extreme position reduction"
            ),
            DrawdownThreshold(
                drawdown_pct=0.22,  # 22%
                position_reduction_pct=1.00,
                state=CircuitState.OPEN,
                message="Drawdown exceeded 22% - trading paused"
            ),
        ],
        TradingModeLevel.MINI: [
            DrawdownThreshold(
                drawdown_pct=0.06,  # 6%
                position_reduction_pct=0.30,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 6% - reducing position sizes"
            ),
            DrawdownThreshold(
                drawdown_pct=0.10,  # 10%
                position_reduction_pct=0.50,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 10% - further position reduction"
            ),
            DrawdownThreshold(
                drawdown_pct=0.15,  # 15%
                position_reduction_pct=0.70,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 15% - extreme position reduction"
            ),
            DrawdownThreshold(
                drawdown_pct=0.20,  # 20%
                position_reduction_pct=1.00,
                state=CircuitState.OPEN,
                message="Drawdown exceeded 20% - trading paused"
            ),
        ],
        TradingModeLevel.BALANCED: [
            DrawdownThreshold(
                drawdown_pct=0.05,  # 5%
                position_reduction_pct=0.25,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 5% - reducing position sizes"
            ),
            DrawdownThreshold(
                drawdown_pct=0.08,  # 8%
                position_reduction_pct=0.40,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 8% - further position reduction"
            ),
            DrawdownThreshold(
                drawdown_pct=0.12,  # 12%
                position_reduction_pct=0.60,
                state=CircuitState.WARNING,
                message="Drawdown exceeded 12% - extreme position reduction"
            ),
            DrawdownThreshold(
                drawdown_pct=0.15,  # 15%
                position_reduction_pct=1.00,
                state=CircuitState.OPEN,
                message="Drawdown exceeded 15% - trading paused"
            ),
        ],
    }


@dataclass
class CircuitBreakerRecord:
    """Record of circuit breaker state change."""
    previous_state: CircuitState
    new_state: CircuitState
    drawdown_pct: float
    position_multiplier: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MicroCircuitBreaker:
    """
    Circuit breaker with mode-specific drawdown protection.
    
    Features:
    - Mode-specific drawdown thresholds (NANO, MICRO, MINI, BALANCED)
    - Graduated response (warning → reduce → stop)
    - Position sizing multiplier based on drawdown
    - Automatic recovery when drawdown reduces
    - Emergency stop for extreme drawdown
    """
    
    def __init__(
        self,
        config: Optional[MicroCircuitBreakerConfig] = None,
        orchestrator=None,
        mode: TradingModeLevel = TradingModeLevel.NANO,
        on_state_change_callback=None,
    ):
        self.config = config or MicroCircuitBreakerConfig()
        self.orchestrator = orchestrator
        self.mode = mode
        self.on_state_change_callback = on_state_change_callback
        
        # State
        self.thresholds = create_default_thresholds()
        self.current_state: CircuitState = CircuitState.CLOSED
        self.current_drawdown: float = 0.0
        self.position_multiplier: float = 1.0
        
        # Tracking
        self._consecutive_threshold_hits: Dict[float, int] = {}
        self._state_history: List[CircuitBreakerRecord] = []
        self._last_recovery_check: int = 0
        
        # Peak drawdown tracking
        self._peak_drawdown: float = 0.0
        
        logger.info(f"MicroCircuitBreaker initialized in {self.mode.value} mode, state: {self.current_state.value}")
    
    def set_mode(self, mode: TradingModeLevel):
        """Change the mode (triggers re-evaluation)."""
        old_mode = self.mode
        self.mode = mode
        
        # Re-evaluate with new thresholds
        if self.orchestrator:
            self._update_drawdown_tracking()
            self._evaluate_thresholds()
        
        logger.info(f"Circuit breaker mode changed: {old_mode.value} → {self.mode.value}")
    
    async def check(self) -> float:
        """
        Check circuit breaker status and return position multiplier.
        
        Returns:
            Position multiplier (0.0 to 1.0) to apply to position sizes
            0.0 = stop trading, 1.0 = normal sizing
        """
        if not self.orchestrator:
            return 1.0
        
        # Update drawdown tracking
        self._update_drawdown_tracking()
        
        # Check for emergency stop
        if self._check_emergency_stop():
            return 0.0
        
        # Evaluate thresholds
        new_multiplier, new_state, threshold = self._evaluate_thresholds()
        
        # Check for state change
        if new_state != self.current_state:
            await self._transition_state(new_state, threshold)
        
        # Update position multiplier
        self.position_multiplier = new_multiplier
        
        return new_multiplier
    
    def _update_drawdown_tracking(self):
        """Update peak drawdown tracking."""
        current_drawdown = self.orchestrator.state.max_drawdown
        
        if current_drawdown > self._peak_drawdown:
            self._peak_drawdown = current_drawdown
        
        self.current_drawdown = current_drawdown
    
    def _evaluate_thresholds(self) -> Tuple[float, CircuitState, Optional[DrawdownThreshold]]:
        """Evaluate current drawdown against thresholds."""
        mode_thresholds = self.thresholds.get(self.mode, [])
        
        # Find the most severe threshold that's been exceeded
        most_severe: Optional[DrawdownThreshold] = None
        for threshold in sorted(mode_thresholds, key=lambda t: t.drawdown_pct, reverse=True):
            if self.current_drawdown >= threshold.drawdown_pct:
                most_severe = threshold
                break
        
        if not most_severe:
            # No thresholds exceeded
            return 1.0, CircuitState.CLOSED, None
        
        # Check consecutive cycle requirement
        consecutive = self._consecutive_threshold_hits.get(most_severe.drawdown_pct, 0) + 1
        self._consecutive_threshold_hits[most_severe.drawdown_pct] = consecutive
        
        if consecutive < self.config.require_consecutive_cycles:
            # Not yet triggered
            return 1.0, self.current_state, None
        
        # Calculate position multiplier
        position_multiplier = 1.0 - most_severe.position_reduction_pct
        
        return max(0.0, position_multiplier), most_severe.state, most_severe
    
    def _check_emergency_stop(self) -> bool:
        """Check if emergency stop is triggered."""
        if self.current_drawdown >= self.config.emergency_stop_threshold:
            logger.warning(
                f"EMERGENCY STOP triggered: drawdown {self.current_drawdown:.1%} "
                f"exceeds threshold {self.config.emergency_stop_threshold:.1%}"
            )
            return True
        return False
    
    async def _transition_state(
        self,
        new_state: CircuitState,
        threshold: Optional[DrawdownThreshold] = None
    ):
        """Handle state transition."""
        previous_state = self.current_state
        self.current_state = new_state
        
        # Create record
        record = CircuitBreakerRecord(
            previous_state=previous_state,
            new_state=new_state,
            drawdown_pct=self.current_drawdown,
            position_multiplier=self.position_multiplier,
            reason=threshold.message if threshold else "State changed",
        )
        self._state_history.append(record)
        
        # Log transition
        logger.info(
            f"CIRCUIT BREAKER: {previous_state.value.upper()} → {new_state.value.upper()} "
            f"(drawdown: {self.current_drawdown:.1%}, multiplier: {self.position_multiplier:.2f})"
        )
        
        # Call callback if registered
        if self.on_state_change_callback:
            try:
                if hasattr(self.on_state_change_callback, '__call__'):
                    if asyncio.iscoroutinefunction(self.on_state_change_callback):
                        await self.on_state_change_callback(record)
                    else:
                        self.on_state_change_callback(record)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}")
    
    def check_recovery(self, cycle_count: int) -> bool:
        """Check if recovery conditions are met."""
        if not self.config.auto_recovery_enabled:
            return False
        
        if self.current_state == CircuitState.CLOSED:
            return True
        
        # Check interval
        if cycle_count - self._last_recovery_check < self.config.recovery_check_interval_cycles:
            return False
        
        # Check if drawdown has reduced
        drawdown_reduction = self._peak_drawdown - self.current_drawdown
        if drawdown_reduction >= self.config.recovery_drawdown_reduction:
            self._last_recovery_check = cycle_count
            return True
        
        return False
    
    async def attempt_recovery(self) -> bool:
        """Attempt to recover from open/warning state."""
        if self.check_recovery(self.orchestrator.state.cycle_count):
            await self._transition_state(CircuitState.CLOSED)
            logger.info(
                f"RECOVERY: Circuit breaker recovered (drawdown: {self.current_drawdown:.1%})"
            )
            return True
        return False
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        self.current_state = CircuitState.CLOSED
        self.current_drawdown = 0.0
        self.position_multiplier = 1.0
        self._consecutive_threshold_hits.clear()
        self._peak_drawdown = 0.0
        self._last_recovery_check = 0
        
        logger.info("Circuit breaker reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.current_state.value,
            "drawdown_pct": self.current_drawdown,
            "position_multiplier": self.position_multiplier,
            "mode": self.mode.value,
            "peak_drawdown": self._peak_drawdown,
            "emergency_threshold": self.config.emergency_stop_threshold,
            "is_trading_paused": self.current_state == CircuitState.OPEN,
            "is_emergency": self.current_drawdown >= self.config.emergency_stop_threshold,
            "consecutive_hits": dict(self._consecutive_threshold_hits),
            "state_changes": len(self._state_history),
        }
    
    def get_state_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent state change history."""
        recent = self._state_history[-limit:]
        return [
            {
                "from_state": r.previous_state.value,
                "to_state": r.new_state.value,
                "drawdown_pct": r.drawdown_pct,
                "position_multiplier": r.position_multiplier,
                "reason": r.reason,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in recent
        ]
    
    def apply_to_config(self, config) -> None:
        """Apply circuit breaker state to orchestrator config."""
        # Update position sizes based on multiplier
        if hasattr(config, 'copy_trading'):
            original_pct = getattr(config.copy_trading, '_original_position_size_pct', None)
            if original_pct is None:
                original_pct = config.copy_trading.position_size_pct
                config.copy_trading._original_position_size_pct = original_pct
            
            config.copy_trading.position_size_pct = original_pct * self.position_multiplier
        
        # Update boost mode
        if hasattr(config, 'boost_mode'):
            if self.current_state == CircuitState.OPEN:
                config.boost_mode.enabled = False
            elif hasattr(config.boost_mode, '_original_enabled'):
                config.boost_mode.enabled = config.boost_mode._original_enabled


# =============================================================================
# Factory Functions
# =============================================================================

def create_circuit_breaker_config(
    mode: str = "nano",
    emergency_threshold: float = 0.30,
    auto_recovery: bool = True,
    consecutive_cycles: int = 2,
) -> MicroCircuitBreakerConfig:
    """Factory function to create MicroCircuitBreakerConfig."""
    return MicroCircuitBreakerConfig(
        emergency_stop_threshold=emergency_threshold,
        auto_recovery_enabled=auto_recovery,
        require_consecutive_cycles=consecutive_cycles,
    )


def create_circuit_breaker(
    orchestrator,
    mode: TradingModeLevel = TradingModeLevel.NANO,
    on_state_change=None,
) -> MicroCircuitBreaker:
    """Factory function to create MicroCircuitBreaker."""
    config = create_circuit_breaker_config(mode.value)
    
    return MicroCircuitBreaker(
        config=config,
        orchestrator=orchestrator,
        mode=mode,
        on_state_change_callback=on_state_change,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("MICRO CIRCUIT BREAKER - Configuration")
    print("=" * 60)
    
    thresholds = create_default_thresholds()
    
    for mode, mode_thresholds in thresholds.items():
        print(f"\n{mode.value.upper()} Mode Thresholds:")
        print("-" * 40)
        for threshold in mode_thresholds:
            print(f"  {threshold.drawdown_pct:>5.0%} drawdown → "
                  f"{threshold.position_reduction_pct:>5.0%} position reduction "
                  f"({threshold.state.value})")
    
    print("\n" + "=" * 60)
    print("Example Usage:")
    print("-" * 60)
    print("""
    from orchestrator.circuit_breaker import MicroCircuitBreaker, create_circuit_breaker
    
    # Create circuit breaker
    breaker = create_circuit_breaker(orchestrator, mode=TradingModeLevel.NANO)
    
    # Check after each cycle
    position_multiplier = await breaker.check()
    
    # Apply to config
    breaker.apply_to_config(config)
    
    # Get status
    status = breaker.get_status()
    print(f"State: {status['state']}, Drawdown: {status['drawdown_pct']:.1%}")
    """)
