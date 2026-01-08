"""
Sandbox configuration and state management.

Provides dataclasses for simulation configuration, portfolio state,
and execution results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SandboxMode(Enum):
    """Sandbox simulation mode."""
    SANDBOX = "sandbox"
    LIVE = "live"

    @classmethod
    def from_string(cls, mode_str: str) -> "SandboxMode":
        """Convert string to SandboxMode enum."""
        mode_upper = mode_str.upper()
        for mode in cls:
            if mode.name == mode_upper:
                return mode
        return cls.SANDBOX  # Default to sandbox

    def is_sandbox(self) -> bool:
        """Check if running in sandbox mode."""
        return self == self.SANDBOX

    def is_live(self) -> bool:
        """Check if running in live mode."""
        return self == self.LIVE


def get_sandbox_mode() -> SandboxMode:
    """
    Get the current sandbox mode from environment variable.

    Returns:
        SandboxMode.SANDBOX if COPYCAT_SANDBOX=true or not set
        SandboxMode.LIVE if COPYCAT_SANDBOX=false
    """
    import os
    sandbox_enabled = os.environ.get("COPYCAT_SANDBOX", "true").lower()
    if sandbox_enabled in ("true", "1", "yes"):
        return SandboxMode.SANDBOX
    return SandboxMode.LIVE


def is_sandbox_mode() -> bool:
    """Quick check if running in sandbox mode."""
    return get_sandbox_mode().is_sandbox()


def is_live_mode() -> bool:
    """Quick check if running in live mode."""
    return get_sandbox_mode().is_live()


class OrderStatus(Enum):
    """Status of a virtual order."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL_FILL = "PARTIAL_FILL"
    NO_FILL = "NO_FILL"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass
class SandboxConfig:
    """Configuration for sandbox simulation mode.

    Use the 'mode' field to switch between sandbox and live trading:
    - SandboxConfig(mode=SandboxMode.SANDBOX): Virtual trading with simulated execution
    - SandboxConfig(mode=SandboxMode.LIVE): Real trading (use with caution)

    Can also be initialized with a string:
    - SandboxConfig(mode="sandbox") or SandboxConfig(mode="live")

    Environment variable COPYCAT_SANDBOX can override the default mode.
    """

    # Mode Selection - uses SandboxMode enum for consistency with orchestrator
    mode: SandboxMode = SandboxMode.SANDBOX

    # Virtual Portfolio Settings
    initial_balance: float = 10000.0  # Starting USD balance
    max_portfolio_value: Optional[float] = None  # Cap at specific value

    # Simulation Constraints (mimic real trading limits)
    max_orders_per_day: int = 50
    max_position_size_pct: float = 0.10  # 10% max per single trade
    max_total_exposure_pct: float = 0.50  # 50% max total exposure
    min_order_size: float = 1.0  # Minimum $1 per trade

    # Execution Simulation
    simulate_slippage: bool = True
    slippage_model: str = "volume_weighted"  # "fixed", "volume_weighted", "realistic"
    simulate_fees: bool = True
    fee_model: str = "polymarket"  # Platform-specific fee structures
    simulate_fill_probability: bool = True
    fill_probability_model: str = "orderbook_depth"

    # Data Configuration
    use_historical_data: bool = False  # If True, run backtesting instead
    historical_data_range: Optional[Tuple[datetime, datetime]] = None
    replay_speed: float = 1.0  # 1.0 = real-time, >1 = faster playback

    # Persistence
    persist_results: bool = True
    results_storage_path: str = "./sandbox_results"
    auto_save_interval_seconds: int = 60

    # Logging & Analytics
    log_all_trades: bool = True
    generate_performance_reports: bool = True
    report_interval_hours: int = 24
    alert_on_thresholds: bool = True
    profit_alert_threshold: float = 0.05  # Alert at 5% profit
    loss_alert_threshold: float = -0.05  # Alert at 5% loss


@dataclass
class VirtualPosition:
    """Represents a virtual position in the sandbox portfolio."""
    market_id: str
    outcome: str  # "YES" or "NO"
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VirtualOrder:
    """Represents a virtual order for simulation."""
    order_id: str
    market_id: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str  # "market" or "limit"
    limit_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_trade_id: Optional[str] = None
    source_trader: Optional[str] = None
    outcome: str = "YES"
    trader_total_volume_30d: float = 0.0


@dataclass
class VirtualOrderResult:
    """Result of a virtual order execution."""
    order_id: str
    status: str
    rejection_reason: Optional[str] = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    total_fees: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VirtualTrade:
    """Represents a completed virtual trade."""
    trade_id: str
    market_id: str
    outcome: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    profit: float = 0.0
    roi: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    exit_timestamp: Optional[datetime] = None
    source_trader: Optional[str] = None
    fees: float = 0.0
    slippage: float = 0.0
    hold_time_hours: Optional[float] = None


@dataclass
class RejectedOrder:
    """Record of a rejected order."""
    order_id: str
    market_id: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderFillRecord:
    """Record of order fill details."""
    order_id: str
    market_id: str
    filled_quantity: float
    average_price: float
    fees: float
    slippage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConstraintValidation:
    """Result of constraint validation."""
    is_valid: bool
    reason: Optional[str] = None


@dataclass
class PortfolioSummary:
    """Summary of portfolio state."""
    balance: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    position_count: int
    exposure_pct: float


@dataclass
class SimulationState:
    """Current state of sandbox simulation."""

    # Portfolio State
    balance: float
    positions: Dict[str, VirtualPosition] = field(default_factory=dict)
    pending_orders: List[VirtualOrder] = field(default_factory=list)
    total_exposure: float = 0.0
    total_value: float = 0.0

    # Performance Metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Trade History
    completed_trades: List[VirtualTrade] = field(default_factory=list)
    rejected_orders: List[RejectedOrder] = field(default_factory=list)
    order_fill_history: List[OrderFillRecord] = field(default_factory=list)

    # Runtime State
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)
    simulation_days: int = 0
    is_paused: bool = False
    is_completed: bool = False


@dataclass
class VirtualTradeResult:
    """Result of a virtual copy trade execution."""
    status: str
    reason: Optional[str] = None
    source_trade_id: Optional[str] = None
    position_size: float = 0.0
    execution_price: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    timestamp: Optional[datetime] = None
    execution_id: Optional[str] = None
    profit: float = 0.0
    roi: float = 0.0
