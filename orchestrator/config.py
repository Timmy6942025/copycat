"""
Orchestrator Configuration.
Defines all configuration classes for the CopyCat trading orchestrator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class TradingMode(Enum):
    """Trading mode enumeration."""
    SANDBOX = "sandbox"
    LIVE = "live"


class MarketPlatform(Enum):
    """Supported market platforms."""
    POLYMARKET = "polymarket"


@dataclass
class APIClientConfig:
    """Configuration for API clients."""
    platform: MarketPlatform
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limit_rps: float = 10.0
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class SelectionMode(Enum):
    """Trader selection mode."""
    WIN_RATE = "win_rate"  # Traditional win-rate based selection
    GROWTH = "growth"  # Growth-first selection (recommended for faster returns)


class TraderSelectionConfig:
    """Configuration for trader selection criteria."""
    
    def __init__(
        self,
        # Selection mode
        mode: SelectionMode = SelectionMode.GROWTH,  # Use growth by default
        
        # Win-rate mode settings (used when mode=WIN_RATE)
        min_win_rate: float = 0.55,
        min_trades: int = 10,
        max_drawdown: float = 0.25,
        min_sharpe_ratio: float = 0.5,
        min_profit_factor: float = 1.0,
        min_total_pnl: float = 0.0,
        max_avg_hold_time_hours: float = 168.0,  # 1 week
        min_reputation_score: float = 0.5,
        
        # Growth mode settings (used when mode=GROWTH)
        growth_min_total_pnl: float = 100.0,  # At least $100 profit
        growth_min_growth_rate: float = 0.01,  # At least 1% per trade
        growth_max_drawdown: float = 0.50,  # Allow up to 50% drawdown
        growth_min_equity_slope: float = 0.0005,  # Positive trend required
        growth_min_consistency: float = 0.25,  # Reasonable consistency
        growth_min_active_days: int = 3,  # At least 3 days of activity
        
        # Weights for growth scoring
        growth_weight: float = 0.50,
        consistency_weight: float = 0.25,
        stability_weight: float = 0.25,
    ):
        self.mode = mode
        
        # Win-rate mode settings
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.max_drawdown = max_drawdown
        self.min_sharpe_ratio = min_sharpe_ratio
        self.min_profit_factor = min_profit_factor
        self.min_total_pnl = min_total_pnl
        self.max_avg_hold_time_hours = max_avg_hold_time_hours
        self.min_reputation_score = min_reputation_score
        
        # Growth mode settings
        self.growth_min_total_pnl = growth_min_total_pnl
        self.growth_min_growth_rate = growth_min_growth_rate
        self.growth_max_drawdown = growth_max_drawdown
        self.growth_min_equity_slope = growth_min_equity_slope
        self.growth_min_consistency = growth_min_consistency
        self.growth_min_active_days = growth_min_active_days
        self.growth_weight = growth_weight
        self.consistency_weight = consistency_weight
        self.stability_weight = stability_weight


@dataclass
class BotFilterOrchestratorConfig:
    """Configuration for bot filtering in orchestrator context."""
    hft_max_hold_time_seconds: float = 1.0
    hft_min_trades_per_minute: int = 5
    arbitrage_max_profit_pct: float = 0.5
    arbitrage_min_trade_frequency: int = 100
    min_hft_score_to_exclude: float = 0.7
    min_arbitrage_score_to_exclude: float = 0.7
    min_pattern_score_to_exclude: float = 0.7


@dataclass
class CopyTradingConfig:
    """Configuration for copy trading behavior."""
    position_sizing_method: str = "scaled"  # fixed_amount, percentage, scaled, kelly
    base_position_size: float = 10.0  # Fixed amount or percentage of portfolio
    position_size_pct: float = 0.10  # 10% of portfolio per trade (optimized for growth)
    kelly_fraction: float = 0.25  # Fractional Kelly to reduce risk
    max_position_size_pct: float = 0.20  # Max 20% per single trade
    max_total_exposure_pct: float = 0.50  # Max 50% total exposure
    min_order_size: float = 0.01  # Minimum $0.01 per trade (no Polymarket minimums)
    max_orders_per_day: int = 100  # Allow more trades for compound growth


@dataclass
class SandboxConfigOrchestrator:
    """Configuration for sandbox simulation (when mode is SANDBOX)."""
    initial_balance: float = 10000.0
    simulate_slippage: bool = True
    slippage_model: str = "volume_weighted"
    simulate_fees: bool = True
    fee_model: str = "polymarket"
    simulate_fill_probability: bool = True
    persist_results: bool = True
    results_storage_path: str = "./orchestrator_results"


@dataclass
class LiveConfig:
    """Configuration for live trading (when mode is LIVE)."""
    wallet_address: str = ""
    initial_balance: float = 10000.0
    max_position_size_pct: float = 0.20
    max_total_exposure_pct: float = 0.50
    max_orders_per_day: int = 100
    min_order_size: float = 0.01  # No Polymarket minimums
    require_order_confirmation: bool = True
    max_slippage_pct: float = 0.05
    enable_price_protection: bool = True
    dry_run_before_execution: bool = True  # Simulate before executing
    confirm_large_orders: float = 1000.0  # Require confirmation for orders > $1000


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    enabled: bool = True
    check_interval_seconds: float = 30.0
    api_client_check: bool = True
    api_client_timeout_seconds: float = 5.0
    max_consecutive_failures: int = 3


@dataclass
class ErrorRecoveryConfig:
    """Configuration for error recovery."""
    enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_timeout_seconds: float = 60.0


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    enabled: bool = False
    alert_on_profit_threshold: float = 0.05  # Alert at 5% profit
    alert_on_loss_threshold: float = -0.05  # Alert at 5% loss
    alert_on_trader_added: bool = True
    alert_on_trader_removed: bool = True
    alert_on_trading_error: bool = True
    discord_webhook_url: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    notification_email: Optional[str] = None


@dataclass
class BoostModeConfig:
    """
    Configuration for Boost Mode - aggressive growth when account is small.

    When enabled and account balance is below the threshold:
    - Position sizes are increased to accelerate growth
    - Trades on markets resolving sooner can get larger allocations
    - Once balance exceeds threshold, returns to normal behavior

    This helps accounts grow faster from small starting amounts (e.g., < $500)
    to reach a sustainable trading size more quickly.
    """
    enabled: bool = False  # Toggle boost mode on/off
    balance_threshold: float = 500.0  # Max balance to apply boost mode
    position_multiplier: float = 2.0  # Multiply position sizes when in boost mode
    max_boost_position_pct: float = 0.25  # Max 25% per trade even in boost mode
    prefer_quick_resolve: bool = True  # Prioritize faster-resolving markets
    quick_resolve_threshold_hours: float = 168.0  # Consider "quick" if resolves within 7 days
    quick_resolve_multiplier: float = 1.5  # Additional multiplier for quick-resolving markets


@dataclass
class OrchestratorConfig:
    """Main orchestrator configuration."""
    # Mode settings
    mode: TradingMode = TradingMode.SANDBOX
    platform: MarketPlatform = MarketPlatform.POLYMARKET

    # API Configuration
    api_clients: List[APIClientConfig] = field(default_factory=list)

    # Trader Selection
    trader_selection: TraderSelectionConfig = field(default_factory=TraderSelectionConfig)

    # Bot Filtering
    bot_filter: BotFilterOrchestratorConfig = field(default_factory=BotFilterOrchestratorConfig)

    # Copy Trading
    copy_trading: CopyTradingConfig = field(default_factory=CopyTradingConfig)

    # Sandbox (when mode is SANDBOX)
    sandbox: SandboxConfigOrchestrator = field(default_factory=SandboxConfigOrchestrator)

    # Live Trading (when mode is LIVE)
    live: LiveConfig = field(default_factory=LiveConfig)

    # Health & Recovery
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    error_recovery: ErrorRecoveryConfig = field(default_factory=ErrorRecoveryConfig)

    # Boost Mode
    boost_mode: BoostModeConfig = field(default_factory=BoostModeConfig)

    # Trading Constraints
    max_traders_to_copy: int = 10
    min_trader_reanalysis_interval_hours: float = 24.0
    max_traders_to_analyze_per_cycle: int = 100

    # Data Refresh
    market_data_refresh_interval_seconds: float = 60.0
    trader_data_refresh_interval_seconds: float = 300.0  # 5 minutes
    trade_history_days: int = 30

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class TraderCopyConfig:
    """Configuration for copying a specific trader."""
    trader_address: str
    position_sizing_method: Optional[str] = None  # Override global setting
    base_position_size: Optional[float] = None
    position_size_pct: Optional[float] = None
    max_position_size_pct: Optional[float] = None
    enabled: bool = True
    auto_remove_if_performance_drops: bool = True
    min_performance_threshold: float = 0.3  # Remove if win rate drops below 30%


@dataclass
class OrchestratorState:
    """Current state of the orchestrator."""
    # Runtime state
    is_running: bool = False
    is_paused: bool = False
    start_time: Optional[datetime] = None
    last_cycle_time: Optional[datetime] = None
    cycle_count: int = 0

    # Trading state
    copied_traders: Dict[str, TraderCopyConfig] = field(default_factory=dict)
    active_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Performance metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Health state
    api_healthy: bool = True
    consecutive_failures: int = 0
    circuit_breaker_open: bool = False

    # Statistics
    trades_executed: int = 0
    traders_analyzed: int = 0
    traders_added: int = 0
    traders_removed: int = 0
    errors_encountered: int = 0


@dataclass
class OrchestrationResult:
    """Result of an orchestration operation."""
    success: bool = False
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TraderAnalysisResult:
    """Result of analyzing a trader for copy trading."""
    trader_address: str
    is_suitable: bool = False
    reputation_score: float = 0.0
    confidence_score: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    bot_filter_result: Dict[str, Any] = field(default_factory=dict)
    selection_reasons: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    recommended_position_size: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
