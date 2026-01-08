"""
Sandbox/Paper Trading Simulation Module.

Provides risk-free environment for testing and validating trading strategies
using virtual money with real-time market data.

Modules:
- config: Configuration dataclasses and state management
- executor: Virtual order execution with realistic slippage/fees
- portfolio: Portfolio management and position tracking
- analytics: Performance metrics and report generation
- backtest: Historical backtesting engine
- cli: Command-line interface
"""

from sandbox.config import (
    SandboxConfig,
    SimulationState,
    SandboxMode,
    OrderStatus,
    VirtualPosition,
    VirtualOrder,
    VirtualOrderResult,
    VirtualTrade,
    RejectedOrder,
    OrderFillRecord,
    ConstraintValidation,
    PortfolioSummary,
    VirtualTradeResult,
)

from sandbox.executor import VirtualOrderExecutor

from sandbox.portfolio import VirtualPortfolioManager

from sandbox.analytics import (
    PerformanceTracker,
    PerformanceReporter,
    PerformanceMetrics,
    DailyReturn,
    EquityPoint,
)

from sandbox.backtest import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    HistoricalDataLoader,
    BacktestOptimizer,
)

from sandbox.cli import SandboxCLI, SandboxManager
from sandbox.runner import SandboxRunner

__all__ = [
    # Config
    "SandboxConfig",
    "SimulationState",
    "SandboxMode",
    "OrderStatus",
    "VirtualPosition",
    "VirtualOrder",
    "VirtualOrderResult",
    "VirtualTrade",
    "RejectedOrder",
    "OrderFillRecord",
    "ConstraintValidation",
    "PortfolioSummary",
    "VirtualTradeResult",
    # Executor
    "VirtualOrderExecutor",
    # Portfolio
    "VirtualPortfolioManager",
    # Analytics
    "PerformanceTracker",
    "PerformanceReporter",
    "PerformanceMetrics",
    "DailyReturn",
    "EquityPoint",
    # Backtest
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "HistoricalDataLoader",
    "BacktestOptimizer",
    # CLI
    "SandboxCLI",
    "SandboxManager",
    # Runner
    "SandboxRunner",
]
