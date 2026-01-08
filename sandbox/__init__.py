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
    # Helper functions for mode switching
    get_sandbox_mode,
    is_sandbox_mode,
    is_live_mode,
)

from sandbox.success_criteria import (
    SuccessLevel,
    MetricType,
    MetricThresholds,
    ConsistencyThresholds,
    SuccessCriteriaResult,
    PerformanceSnapshot,
    CRITICAL_SUCCESS_FACTORS,
    METRIC_THRESHOLDS,
    CONSISTENCY_THRESHOLDS,
    evaluate_success_criteria,
    get_threshold_summary,
    format_evaluation_result,
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
from sandbox.dashboard import TUIDashboard, SimpleDashboard, DashboardConfig, run_dashboard
from sandbox.dependencies import (
    get_sandbox_runner,
    create_sandbox_config,
    get_trading_dependencies,
    get_trading_mode,
    is_sandbox_mode,
    is_live_mode,
    switch_mode,
    validate_config_for_mode,
    TradingDependencies,
)

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
    # Mode switching helpers
    "get_sandbox_mode",
    "is_sandbox_mode",
    "is_live_mode",
    # Success Criteria
    "SuccessLevel",
    "MetricType",
    "MetricThresholds",
    "ConsistencyThresholds",
    "SuccessCriteriaResult",
    "PerformanceSnapshot",
    "CRITICAL_SUCCESS_FACTORS",
    "METRIC_THRESHOLDS",
    "CONSISTENCY_THRESHOLDS",
    "evaluate_success_criteria",
    "get_threshold_summary",
    "format_evaluation_result",
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
    # Dashboard
    "TUIDashboard",
    "SimpleDashboard",
    "DashboardConfig",
    "run_dashboard",
    # Dependencies
    "get_sandbox_runner",
    "create_sandbox_config",
    "get_trading_dependencies",
    "get_trading_mode",
    "is_sandbox_mode",
    "is_live_mode",
    "switch_mode",
    "validate_config_for_mode",
    "TradingDependencies",
]
