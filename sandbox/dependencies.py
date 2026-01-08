"""
Sandbox Module Dependencies.

Provides dependency injection and factory functions for the sandbox module.
This enables clean separation between sandbox and live trading modes.

Usage:
    # Get sandbox runner
    from sandbox.dependencies import get_sandbox_runner

    # Check current mode
    from sandbox.dependencies import get_trading_mode, is_sandbox_mode

    # Create configured runner
    runner = get_sandbox_runner(initial_balance=10000)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from sandbox.config import SandboxConfig, SandboxMode, get_sandbox_mode
from sandbox.runner import SandboxRunner
from sandbox.executor import VirtualOrderExecutor
from sandbox.portfolio import VirtualPortfolioManager
from sandbox.analytics import PerformanceTracker, PerformanceReporter


@dataclass
class TradingDependencies:
    """Container for all trading dependencies."""
    config: SandboxConfig
    runner: Optional[SandboxRunner] = None
    executor: Optional[VirtualOrderExecutor] = None
    portfolio_manager: Optional[VirtualPortfolioManager] = None
    tracker: Optional[PerformanceTracker] = None
    reporter: Optional[PerformanceReporter] = None

    @property
    def is_sandbox(self) -> bool:
        """Check if running in sandbox mode."""
        return self.config.mode.is_sandbox()

    @property
    def is_live(self) -> bool:
        """Check if running in live mode."""
        return self.config.mode.is_live()


def get_trading_mode() -> SandboxMode:
    """
    Get the current trading mode.

    Returns:
        SandboxMode.SANDBOX or SandboxMode.LIVE based on config/env
    """
    return get_sandbox_mode()


def is_sandbox_mode() -> bool:
    """
    Quick check if running in sandbox mode.

    Returns:
        True if in sandbox mode, False otherwise
    """
    return get_sandbox_mode().is_sandbox()


def is_live_mode() -> bool:
    """
    Quick check if running in live mode.

    Returns:
        True if in live mode, False otherwise
    """
    return get_sandbox_mode().is_live()


def create_sandbox_config(
    mode: Optional[SandboxMode] = None,
    initial_balance: float = 10000.0,
    **kwargs
) -> SandboxConfig:
    """
    Create a SandboxConfig with the specified settings.

    Args:
        mode: Trading mode (sandbox or live). Uses env var if not specified.
        initial_balance: Starting virtual balance
        **kwargs: Additional config parameters

    Returns:
        Configured SandboxConfig instance
    """
    effective_mode = mode or get_sandbox_mode()

    return SandboxConfig(
        mode=effective_mode,
        initial_balance=initial_balance,
        **kwargs
    )


def get_sandbox_runner(
    config: Optional[SandboxConfig] = None,
    initial_balance: float = 10000.0,
    enable_data_feed: bool = False,
    market_ids: Optional[List[str]] = None,
) -> SandboxRunner:
    """
    Get a configured SandboxRunner instance.

    This is the main entry point for getting a sandbox runner with
    proper configuration and optional real-time data.

    Args:
        config: Optional pre-configured SandboxConfig
        initial_balance: Starting balance if config not provided
        enable_data_feed: Whether to enable real-time data feed
        market_ids: Market IDs to track if data feed enabled

    Returns:
        Configured SandboxRunner instance

    Example:
        # Basic sandbox runner
        runner = get_sandbox_runner(initial_balance=5000)

        # With real-time data
        runner = get_sandbox_runner(
            initial_balance=10000,
            enable_data_feed=True,
            market_ids=["bitcoin", "AAPL"]
        )

        # Live trading mode
        runner = get_sandbox_runner(
            config=SandboxConfig(mode=SandboxMode.LIVE)
        )
    """
    sandbox_config = config or create_sandbox_config(initial_balance=initial_balance)

    runner = SandboxRunner(config=sandbox_config)

    if enable_data_feed and market_ids:
        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                runner.start_data_feed(market_ids)
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to start data feed: {e}"
            )

    return runner


def get_trading_dependencies(
    config: Optional[SandboxConfig] = None,
) -> TradingDependencies:
    """
    Get all trading dependencies as a single container.

    This is useful for dependency injection frameworks or when
    you need access to multiple trading components.

    Args:
        config: Optional SandboxConfig instance

    Returns:
        TradingDependencies container with all components
    """
    sandbox_config = config or create_sandbox_config()

    return TradingDependencies(
        config=sandbox_config,
        runner=SandboxRunner(config=sandbox_config),
        executor=VirtualOrderExecutor(sandbox_config),
        portfolio_manager=VirtualPortfolioManager(sandbox_config),
        tracker=PerformanceTracker(sandbox_config.initial_balance),
        reporter=PerformanceReporter(sandbox_config.results_storage_path),
    )


def switch_mode(
    config: SandboxConfig,
    new_mode: SandboxMode
) -> SandboxConfig:
    """
    Create a new config with a different mode.

    Args:
        config: Original config
        new_mode: Mode to switch to

    Returns:
        New SandboxConfig with updated mode

    Example:
        # Switch from sandbox to live
        live_config = switch_mode(sandbox_config, SandboxMode.LIVE)
    """
    import copy
    new_config = copy.copy(config)
    new_config.mode = new_mode
    return new_config


def validate_config_for_mode(
    config: SandboxConfig,
    expected_mode: SandboxMode
) -> tuple[bool, str]:
    """
    Validate that a config matches the expected mode.

    Args:
        config: Config to validate
        expected_mode: Expected mode (sandbox or live)

    Returns:
        Tuple of (is_valid, message)
    """
    if config.mode == expected_mode:
        return True, f"Config is valid for {expected_mode.value} mode"

    return False, (
        f"Config mode '{config.mode.value}' does not match "
        f"expected mode '{expected_mode.value}'"
    )


# Alias functions for backward compatibility
_sandbox_runner = get_sandbox_runner
_sandbox_config = create_sandbox_config

__all__ = [
    # Main exports
    "get_sandbox_runner",
    "create_sandbox_config",
    "get_trading_dependencies",
    # Mode utilities
    "get_trading_mode",
    "is_sandbox_mode",
    "is_live_mode",
    "switch_mode",
    "validate_config_for_mode",
    # Config
    "SandboxConfig",
    "SandboxMode",
    # Containers
    "TradingDependencies",
]
