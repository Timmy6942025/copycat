"""
Consolidated Configuration System for CopyCat Trading System.

This module consolidates all configuration options into a single validated structure,
replacing the 12+ fragmented config files.

Usage:
    from orchestrator.config import CopyCatConfig

    config = CopyCatConfig(
        mode=TradingMode.SANDBOX,
        position_sizing=PositionSizingConfig(
            method=PositionSizingMethod.KELLY,
            kelly_fraction=0.25,
        ),
        risk=RiskConfig(
            stop_loss_pct=0.10,
            take_profit_pct=0.20,
            max_position_pct=0.10,
        ),
        trader_selection=TraderSelectionConfig(
            min_win_rate=0.55,
            min_trades=10,
        ),
    )
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode selection."""

    SANDBOX = "sandbox"
    LIVE = "live"


class MarketPlatform(Enum):
    """Supported market platforms."""

    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class PositionSizingMethod(Enum):
    """Position sizing calculation methods."""

    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE = "percentage"
    SCALED = "scaled"
    KELLY = "kelly"


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""

    method: PositionSizingMethod = PositionSizingMethod.KELLY
    base_position_size: float = 100.0
    position_size_pct: float = 0.05
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.10

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if self.position_size_pct <= 0 or self.position_size_pct > 1.0:
            return (
                False,
                f"position_size_pct must be between 0 and 1, got {self.position_size_pct}",
            )
        if self.kelly_fraction <= 0 or self.kelly_fraction > 1.0:
            return (
                False,
                f"kelly_fraction must be between 0 and 1, got {self.kelly_fraction}",
            )
        if self.max_position_pct <= 0 or self.max_position_pct > 1.0:
            return (
                False,
                f"max_position_pct must be between 0 and 1, got {self.max_position_pct}",
            )
        return True, ""


@dataclass
class RiskConfig:
    """Configuration for risk management."""

    stop_loss_pct: float = 0.10
    take_profit_pct: float = 0.20
    max_position_pct: float = 0.10
    max_total_exposure_pct: float = 0.50
    max_drawdown_pct: float = 0.25
    enable_trailing_stop: bool = False
    trailing_stop_distance: float = 0.05

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 1.0:
            return (
                False,
                f"stop_loss_pct must be between 0 and 1, got {self.stop_loss_pct}",
            )
        if self.take_profit_pct <= 0 or self.take_profit_pct > 1.0:
            return (
                False,
                f"take_profit_pct must be between 0 and 1, got {self.take_profit_pct}",
            )
        if self.max_position_pct <= 0 or self.max_position_pct > 1.0:
            return (
                False,
                f"max_position_pct must be between 0 and 1, got {self.max_position_pct}",
            )
        if self.max_total_exposure_pct <= 0 or self.max_total_exposure_pct > 1.0:
            return (
                False,
                f"max_total_exposure_pct must be between 0 and 1, got {self.max_total_exposure_pct}",
            )
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 1.0:
            return (
                False,
                f"max_drawdown_pct must be between 0 and 1, got {self.max_drawdown_pct}",
            )
        return True, ""


@dataclass
class TraderSelectionConfig:
    """Configuration for trader selection criteria."""

    min_win_rate: float = 0.55
    min_trades: int = 10
    max_drawdown: float = 0.25
    min_sharpe_ratio: float = 0.5
    min_profit_factor: float = 1.0
    min_total_pnl: float = 0.0
    min_reputation_score: float = 0.5

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if self.min_win_rate < 0 or self.min_win_rate > 1.0:
            return (
                False,
                f"min_win_rate must be between 0 and 1, got {self.min_win_rate}",
            )
        if self.min_trades <= 0:
            return False, f"min_trades must be positive, got {self.min_trades}"
        if self.max_drawdown < 0 or self.max_drawdown > 1.0:
            return (
                False,
                f"max_drawdown must be between 0 and 1, got {self.max_drawdown}",
            )
        return True, ""


@dataclass
class BotFilterConfig:
    """Configuration for bot filtering."""

    hft_max_hold_time_seconds: float = 1.0
    hft_min_trades_per_minute: int = 5
    arbitrage_max_profit_pct: float = 0.5
    min_hft_score_to_exclude: float = 0.7
    min_arbitrage_score_to_exclude: float = 0.7

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if self.hft_max_hold_time_seconds < 0:
            return (
                False,
                f"hft_max_hold_time_seconds must be non-negative, got {self.hft_max_hold_time_seconds}",
            )
        if self.min_hft_score_to_exclude < 0 or self.min_hft_score_to_exclude > 1.0:
            return (
                False,
                f"min_hft_score_to_exclude must be between 0 and 1, got {self.min_hft_score_to_exclude}",
            )
        return True, ""


@dataclass
class SpeedModeConfig:
    """Configuration for Speed Mode optimizations."""

    enabled: bool = False
    cycle_refresh_seconds: int = 10
    position_size_pct: float = 0.35
    max_orders_per_day: int = 500
    kelly_fraction: float = 0.50
    enable_hedging: bool = False
    min_growth_rate: float = 0.005

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if self.cycle_refresh_seconds < 1:
            return (
                False,
                f"cycle_refresh_seconds must be at least 1, got {self.cycle_refresh_seconds}",
            )
        if self.max_orders_per_day <= 0:
            return (
                False,
                f"max_orders_per_day must be positive, got {self.max_orders_per_day}",
            )
        return True, ""


@dataclass
class MicroModeConfig:
    """Configuration for Micro Mode (small accounts)."""

    enabled: bool = False
    nano_max_balance: float = 15.0
    micro_max_balance: float = 25.0
    mini_max_balance: float = 50.0
    nano_position_pct: float = 0.75
    micro_position_pct: float = 0.60
    mini_position_pct: float = 0.50
    balanced_position_pct: float = 0.40

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if self.nano_max_balance <= 0:
            return (
                False,
                f"nano_max_balance must be positive, got {self.nano_max_balance}",
            )
        if self.nano_position_pct > 1.0 or self.nano_position_pct <= 0:
            return (
                False,
                f"nano_position_pct must be between 0 and 1, got {self.nano_position_pct}",
            )
        return True, ""


@dataclass
class SandboxConfig:
    """Configuration for sandbox mode."""

    initial_balance: float = 10000.0
    simulate_slippage: bool = True
    simulate_fees: bool = True

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if self.initial_balance <= 0:
            return (
                False,
                f"initial_balance must be positive, got {self.initial_balance}",
            )
        return True, ""


@dataclass
class NotificationsConfig:
    """Configuration for notifications."""

    discord_webhook_url: str = ""
    slack_webhook_url: str = ""
    email_enabled: bool = False
    milestone_notifications: bool = True
    trade_notifications: bool = False
    error_notifications: bool = True

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if self.discord_webhook_url and not self.discord_webhook_url.startswith(
            "https://discord.com/api/webhooks/"
        ):
            logger.warning("Discord webhook URL may be invalid")
        return True, ""


@dataclass
class CopyCatConfig:
    """
    Consolidated configuration for CopyCat trading system.

    This is the main configuration class that combines all settings into
    a single validated structure.
    """

    # Core settings
    mode: TradingMode = TradingMode.SANDBOX
    platform: MarketPlatform = MarketPlatform.POLYMARKET

    # Feature configurations
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trader_selection: TraderSelectionConfig = field(
        default_factory=TraderSelectionConfig
    )
    bot_filter: BotFilterConfig = field(default_factory=BotFilterConfig)
    speed_mode: SpeedModeConfig = field(default_factory=SpeedModeConfig)
    micro_mode: MicroModeConfig = field(default_factory=MicroModeConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)

    # Constraints
    max_traders_to_copy: int = 10
    max_traders_to_analyze_per_cycle: int = 100
    trader_data_refresh_interval_seconds: int = 300

    def validate(self) -> tuple[bool, List[str]]:
        """Validate entire configuration."""
        errors = []

        # Validate all sub-configs
        for config_attr in [
            "position_sizing",
            "risk",
            "trader_selection",
            "bot_filter",
            "speed_mode",
            "micro_mode",
            "sandbox",
            "notifications",
        ]:
            config = getattr(self, config_attr)
            is_valid, error = config.validate()
            if not is_valid:
                errors.append(f"{config_attr}: {error}")

        # Validate constraints
        if self.max_traders_to_copy <= 0:
            errors.append("max_traders_to_copy must be positive")
        if self.max_traders_to_analyze_per_cycle <= 0:
            errors.append("max_traders_to_analyze_per_cycle must be positive")

        return len(errors) == 0, errors

    @classmethod
    def from_environment(cls) -> "CopyCatConfig":
        """Create configuration from environment variables."""
        # Map environment variables to config
        env_mappings = {
            "COPYCAT_MODE": ("mode", lambda v: TradingMode(v.lower())),
            "COPYCAT_PLATFORM": ("platform", lambda v: MarketPlatform(v.lower())),
            "COPYCAT_INITIAL_BALANCE": (
                "sandbox",
                lambda v: SandboxConfig(initial_balance=float(v)),
            ),
            "COPYCAT_MAX_TRADERS": ("max_traders_to_copy", int),
        }

        config = cls()

        for env_var, (attr, transform) in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                try:
                    if callable(transform) and not isinstance(transform, type):
                        new_value = transform(value)
                    else:
                        new_value = transform(value)
                    setattr(config, attr, new_value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse {env_var}: {e}")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "mode": self.mode.value,
            "platform": self.platform.value,
            "position_sizing": {
                "method": self.position_sizing.method.value,
                "base_position_size": self.position_sizing.base_position_size,
                "position_size_pct": self.position_sizing.position_size_pct,
                "kelly_fraction": self.position_sizing.kelly_fraction,
            },
            "risk": {
                "stop_loss_pct": self.risk.stop_loss_pct,
                "take_profit_pct": self.risk.take_profit_pct,
                "max_position_pct": self.risk.max_position_pct,
                "max_total_exposure_pct": self.risk.max_total_exposure_pct,
            },
            "trader_selection": {
                "min_win_rate": self.trader_selection.min_win_rate,
                "min_trades": self.trader_selection.min_trades,
                "max_drawdown": self.trader_selection.max_drawdown,
            },
            "constraints": {
                "max_traders_to_copy": self.max_traders_to_copy,
                "max_traders_to_analyze_per_cycle": self.max_traders_to_analyze_per_cycle,
            },
        }


# Factory functions for common configurations
def create_speed_config(
    initial_balance: float = 100.0, speed_mode: str = "balanced"
) -> CopyCatConfig:
    """Create a Speed Mode configuration."""
    config = CopyCatConfig()
    config.sandbox.initial_balance = initial_balance

    speed_configs = {
        "conservative": (0.25, 0.25),
        "balanced": (0.35, 0.50),
        "aggressive": (0.50, 0.75),
    }

    position_pct, kelly = speed_configs.get(speed_mode, speed_configs["balanced"])
    config.position_sizing.position_size_pct = position_pct
    config.position_sizing.kelly_fraction = kelly
    config.speed_mode.enabled = True

    return config


def create_micro_config(
    initial_balance: float = 10.0, micro_mode: str = "nano"
) -> CopyCatConfig:
    """Create a Micro Mode configuration for small accounts."""
    config = CopyCatConfig()
    config.sandbox.initial_balance = initial_balance

    mode_configs = {
        "nano": (15.0, 0.75, 0.75),
        "micro": (25.0, 0.60, 0.75),
        "mini": (50.0, 0.50, 0.50),
        "balanced": (200.0, 0.40, 0.40),
    }

    max_balance, position_pct, kelly = mode_configs.get(
        micro_mode, mode_configs["nano"]
    )
    config.micro_mode.enabled = True
    config.micro_mode.nano_max_balance = max_balance
    config.position_sizing.position_size_pct = position_pct
    config.position_sizing.kelly_fraction = kelly

    return config


def create_conservative_config() -> CopyCatConfig:
    """Create a conservative risk configuration."""
    config = CopyCatConfig()
    config.position_sizing.position_size_pct = 0.25
    config.risk.stop_loss_pct = 0.05
    config.risk.take_profit_pct = 0.10
    config.trader_selection.min_win_rate = 0.60
    return config


def create_aggressive_config() -> CopyCatConfig:
    """Create an aggressive growth configuration."""
    config = CopyCatConfig()
    config.position_sizing.position_size_pct = 0.50
    config.risk.stop_loss_pct = 0.15
    config.risk.take_profit_pct = 0.30
    config.trader_selection.min_win_rate = 0.50
    return config
