"""
Risk Management Module for CopyCat Trading System.

Provides stop-loss, take-profit, and position risk management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import logging


logger = logging.getLogger(__name__)


@dataclass
class StopLossConfig:
    """Configuration for stop-loss settings."""

    enabled: bool = True
    percentage: float = 0.10  # 10% stop loss by default
    trailing: bool = False
    trailing_distance: float = 0.05  # 5% trailing stop
    timeout_seconds: int = 300  # Max time in a losing position


@dataclass
class TakeProfitConfig:
    """Configuration for take-profit settings."""

    enabled: bool = True
    percentage: float = 0.20  # 20% take profit by default
    partial_exit_pct: float = 0.50  # Exit 50% at target


@dataclass
class RiskMetrics:
    """Risk metrics for a position."""

    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss_triggered: bool = False
    take_profit_triggered: bool = False
    max_loss_hit: bool = False
    max_profit_hit: bool = False
    risk_reward_ratio: float = 0.0
    time_in_position_seconds: int = 0
    should_close: bool = False
    close_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionRiskConfig:
    """Complete risk configuration for a position."""

    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)
    max_position_size_pct: float = 0.10  # Max 10% of portfolio per position
    max_total_exposure_pct: float = 0.50  # Max 50% total exposure
    max_drawdown_pct: float = 0.25  # Max 25% drawdown from peak
    min_liquidity_required: float = 1000.0  # Min $1000 liquidity


class RiskManager:
    """
    Manages risk for trading positions.

    Provides:
    - Stop-loss execution
    - Take-profit execution
    - Position sizing limits
    - Exposure limits
    - Time-based risk checks
    """

    def __init__(self, config: Optional[PositionRiskConfig] = None):
        """Initialize risk manager."""
        self.config = config or PositionRiskConfig()
        self._peak_balance: float = 0.0
        self._positions_at_risk: Dict[str, datetime] = {}

    def analyze_position(
        self,
        entry_price: float,
        current_price: float,
        quantity: float,
        entry_time: datetime,
        portfolio_value: float,
        current_balance: float,
    ) -> RiskMetrics:
        """
        Analyze a position and determine risk metrics.

        Args:
            entry_price: Price at which position was opened
            current_price: Current market price
            quantity: Position quantity
            entry_time: When the position was opened
            portfolio_value: Total portfolio value
            current_balance: Current available balance

        Returns:
            RiskMetrics with risk assessment
        """
        cost = entry_price * quantity
        if cost == 0:
            logger.warning("Zero cost position detected")
            metrics.should_close = True
            metrics.close_reason = "Zero cost position"
            return metrics

        raw_pnl = (current_price - entry_price) * quantity
        metrics.current_pnl = raw_pnl
        metrics.current_pnl_pct = raw_pnl / cost

        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        time_delta = datetime.utcnow() - entry_time
        metrics.time_in_position_seconds = int(time_delta.total_seconds())

        if self.config.stop_loss.enabled:
            stop_result = self._check_stop_loss(
                metrics.current_pnl_pct,
                metrics.time_in_position_seconds,
            )
            metrics.stop_loss_triggered = stop_result.triggered
            metrics.should_close = metrics.should_close or stop_result.triggered
            if stop_result.triggered:
                metrics.close_reason = stop_result.reason

        # Check take-profit
        if self.config.take_profit.enabled and not metrics.should_close:
            tp_result = self._check_take_profit(metrics.current_pnl_pct)
            metrics.take_profit_triggered = tp_result.triggered
            if tp_result.triggered:
                metrics.close_reason = tp_result.reason

        # Check max drawdown
        drawdown = (
            (self._peak_balance - current_balance) / self._peak_balance
            if self._peak_balance > 0
            else 0
        )
        if drawdown >= self.config.max_drawdown_pct:
            metrics.max_loss_hit = True
            metrics.should_close = True
            metrics.close_reason = f"Max drawdown reached: {drawdown:.2%}"

        # Calculate risk/reward ratio
        if metrics.current_pnl_pct != 0:
            metrics.risk_reward_ratio = abs(
                metrics.current_pnl_pct / (-self.config.stop_loss.percentage)
            )

        # Record position at risk
        position_key = f"{entry_price}_{quantity}"
        if metrics.current_pnl_pct < 0:
            self._positions_at_risk[position_key] = entry_time

        # Log warning for positions held too long
        if metrics.time_in_position_seconds > self.config.stop_loss.timeout_seconds:
            logger.warning(
                f"Position held for {metrics.time_in_position_seconds}s "
                f"(max: {self.config.stop_loss.timeout_seconds}s)"
            )

        return metrics

    def _check_stop_loss(
        self,
        current_pnl_pct: float,
        time_in_position_seconds: int,
    ) -> Dict[str, Any]:
        """Check if stop-loss should be triggered."""
        result = {"triggered": False, "reason": None}

        # Fixed stop-loss
        if current_pnl_pct <= -self.config.stop_loss.percentage:
            result["triggered"] = True
            result["reason"] = f"Stop-loss triggered at {current_pnl_pct:.2%} loss"
            return result

        # Trailing stop-loss
        if (
            self.config.stop_loss.trailing
            and current_pnl_pct < -self.config.stop_loss.trailing_distance
        ):
            result["triggered"] = True
            result["reason"] = (
                f"Trailing stop-loss triggered at {current_pnl_pct:.2%} loss"
            )
            return result

        # Timeout stop-loss
        if (
            time_in_position_seconds >= self.config.stop_loss.timeout_seconds
            and current_pnl_pct < 0
        ):
            result["triggered"] = True
            result["reason"] = (
                f"Timeout stop-loss after {time_in_position_seconds}s in losing position"
            )
            return result

        return result

    def _check_take_profit(self, current_pnl_pct: float) -> Dict[str, Any]:
        """Check if take-profit should be triggered."""
        result = {"triggered": False, "reason": None}

        if current_pnl_pct >= self.config.take_profit.percentage:
            result["triggered"] = True
            result["reason"] = f"Take-profit triggered at {current_pnl_pct:.2%} gain"
            return result

        return result

    def validate_position_size(
        self,
        position_size: float,
        portfolio_value: float,
        current_exposure: float,
    ) -> tuple[bool, str]:
        """
        Validate if a position size is acceptable.

        Args:
            position_size: Proposed position size in dollars
            portfolio_value: Total portfolio value
            current_exposure: Current exposure from other positions

        Returns:
            Tuple of (is_valid, error_message)
        """
        if position_size <= 0:
            return False, "Position size must be greater than zero"

        if portfolio_value <= 0:
            return False, "Portfolio value must be greater than zero"

        max_position = portfolio_value * self.config.max_position_size_pct
        if position_size > max_position:
            return (
                False,
                f"Position size ${position_size:.2f} exceeds max ${max_position:.2f} ({self.config.max_position_size_pct:.0%} of portfolio)",
            )

        total_exposure = current_exposure + position_size
        max_exposure = portfolio_value * self.config.max_total_exposure_pct
        if total_exposure > max_exposure:
            return (
                False,
                f"Total exposure ${total_exposure:.2f} would exceed max ${max_exposure:.2f}",
            )

        return True, ""

    def reset(self):
        """Reset risk manager state."""
        self._peak_balance = 0.0
        self._positions_at_risk.clear()
        logger.info("Risk manager state reset")
