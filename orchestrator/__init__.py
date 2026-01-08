"""
CopyCat Main Orchestrator.
Coordinates all trading modules: API clients, trader identification, bot filtering, and sandbox/live trading.
"""

from .config import (
    TradingMode,
    MarketPlatform,
    APIClientConfig,
    TraderSelectionConfig,
    BotFilterOrchestratorConfig,
    CopyTradingConfig,
    SandboxConfigOrchestrator,
    HealthCheckConfig,
    ErrorRecoveryConfig,
    NotificationConfig,
    OrchestratorConfig,
    TraderCopyConfig,
    OrchestratorState,
    OrchestrationResult,
    TraderAnalysisResult,
)

from .engine import (
    CopyCatOrchestrator,
)

__all__ = [
    # Configuration classes
    "TradingMode",
    "MarketPlatform",
    "APIClientConfig",
    "TraderSelectionConfig",
    "BotFilterOrchestratorConfig",
    "CopyTradingConfig",
    "SandboxConfigOrchestrator",
    "HealthCheckConfig",
    "ErrorRecoveryConfig",
    "NotificationConfig",
    "OrchestratorConfig",
    "TraderCopyConfig",
    "OrchestratorState",
    "OrchestrationResult",
    "TraderAnalysisResult",
    # Main engine
    "CopyCatOrchestrator",
]
