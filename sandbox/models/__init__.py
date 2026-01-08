"""
Core data models for the sandbox simulation module.
Contains all dataclasses for configuration, state, orders, positions, and performance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uuid


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    NO_FILL = "no_fill"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class SimulationMode(Enum):
    SANDBOX = "sandbox"
    LIVE = "live"


@dataclass
class SandboxConfig:
    mode: str = "sandbox"
    initial_balance: float = 10000.0
    max_portfolio_value: Optional[float] = None
    max_orders_per_day: int = 50
    max_position_size_pct: float = 0.10
    max_total_exposure_pct: float = 0.50
    min_order_size: float = 1.0
    simulate_slippage: bool = True
    slippage_model: str = "volume_weighted"
    simulate_fees: bool = True
    fee_model: str = "polymarket"
    simulate_fill_probability: bool = True
    fill_probability_model: str = "orderbook_depth"
    use_historical_data: bool = False
    historical_data_range: Optional[Tuple[datetime, datetime]] = None
    replay_speed: float = 1.0
    persist_results: bool = True
    results_storage_path: str = "./sandbox_results"
    auto_save_interval_seconds: int = 60
    log_all_trades: bool = True
    generate_performance_reports: bool = True
    report_interval_hours: int = 24
    alert_on_thresholds: bool = True
    profit_alert_threshold: float = 0.05
    loss_alert_threshold: float = -0.05


@dataclass
class SimulationState:
    balance: float = 10000.0
    positions: Dict[str, 'VirtualPosition'] = field(default_factory=dict)
    pending_orders: List['VirtualOrder'] = field(default_factory=list)
    total_exposure: float = 0.0
    total_value: float = 10000.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    completed_trades: List['VirtualTrade'] = field(default_factory=list)
    rejected_orders: List['RejectedOrder'] = field(default_factory=list)
    order_fill_history: List['OrderFillRecord'] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)
    simulation_days: int = 0
    is_paused: bool = False
    is_completed: bool = False


@dataclass
class VirtualOrder:
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_id: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: Optional[float] = None
    outcome: str = "YES"
    source_trader_id: Optional[str] = None
    confidence_score: float = 1.0
    copy_ratio: float = 0.1
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    total_fees: float = 0.0

    def __post_init__(self):
        if self.limit_price is None and self.order_type == OrderType.LIMIT:
            raise ValueError("Limit price required for limit orders")


@dataclass
class VirtualPosition:
    market_id: str = ""
    outcome: str = "YES"
    quantity: float = 0.0
    avg_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    source_trader_id: Optional[str] = None
    order_id: Optional[str] = None

    def calculate_value(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.quantity * self.current_price

    def calculate_realized_pnl(self, exit_price: float) -> float:
        if self.quantity == 0:
            return 0.0
        if self.outcome == "YES":
            return (exit_price - self.avg_price) * self.quantity
        else:
            return (self.avg_price - exit_price) * self.quantity


@dataclass
class VirtualTrade:
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    market_id: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    profit: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    hold_time_hours: float = 0.0
    source_trader_id: Optional[str] = None
    confidence_score: float = 1.0
    is_open: bool = True


@dataclass
class OrderFillRecord:
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    market_id: str = ""
    filled_quantity: float = 0.0
    average_price: float = 0.0
    total_fees: float = 0.0
    slippage: float = 0.0
    fill_probability: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RejectedOrder:
    order_id: str = ""
    market_id: str = ""
    rejection_reason: str = ""
    order_details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VirtualOrderResult:
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    rejection_reason: str = ""
    filled_quantity: float = 0.0
    average_price: float = 0.0
    total_fees: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    is_valid: bool = True
    reason: str = ""
    adjustments_made: List[str] = field(default_factory=list)


@dataclass
class OrderBook:
    market_id: str = ""
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketData:
    market_id: str = ""
    current_price: float = 0.5
    previous_price: float = 0.5
    volume_24h: float = 0.0
    volatility: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioSummary:
    balance: float = 0.0
    positions_value: float = 0.0
    total_value: float = 0.0
    unrealized_pnl: float = 0.0
    position_count: int = 0
    exposure_pct: float = 0.0


@dataclass
class EquityPoint:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    value: float = 0.0
    return_pct: float = 0.0


@dataclass
class DailyReturn:
    date: datetime = field(default_factory=datetime.utcnow)
    return_pct: float = 0.0
    portfolio_value: float = 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for sandbox simulation."""
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime = field(default_factory=datetime.utcnow)
    trading_days: int = 0
    starting_balance: float = 0.0
    ending_balance: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    annualized_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    profit_factor: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    calmar_ratio: float = 0.0
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    avg_hold_time_hours: float = 0.0
    max_hold_time_hours: float = 0.0
    min_hold_time_hours: float = 0.0
    traders_copied: int = 0
    profitable_traders: int = 0
    top_performing_trader: str = ""
    worst_performing_trader: str = ""
    trader_specific_returns: Dict = field(default_factory=dict)
    avg_slippage: float = 0.0
    total_fees_paid: float = 0.0
    fill_rate: float = 0.0
    partial_fill_rate: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    markets: List[str] = field(default_factory=list)
    initial_balance: float = 10000.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    reinvest_profits: bool = True


@dataclass
class BacktestResult:
    """Results from backtest."""
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime = field(default_factory=datetime.utcnow)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    equity_curve: List = field(default_factory=list)


@dataclass
class TraderSelectionConfig:
    """Configuration for trader selection."""
    min_win_rate: float = 0.55
    min_trades: int = 10
    max_drawdown: float = 0.25
    min_sharpe_ratio: float = 0.5


@dataclass
class TraderCopyConfig:
    """Configuration for copy trading."""
    copy_ratio: float = 0.10
    max_position_pct: float = 0.10
    min_order_size: float = 10.0
    max_orders_per_day: int = 5
