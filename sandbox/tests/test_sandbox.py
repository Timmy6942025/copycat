"""
Unit tests for Sandbox/Paper Trading Simulation Module.

Tests configuration, order execution, portfolio management,
and performance tracking.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class TestSandboxConfig:
    """Tests for SandboxConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()

        assert config.mode.value == "sandbox"  # Enum value is lowercase
        assert config.initial_balance == 10000.0
        assert config.max_orders_per_day == 50
        assert config.max_position_size_pct == 0.10
        assert config.max_total_exposure_pct == 0.50
        assert config.min_order_size == 1.0
        assert config.simulate_slippage is True
        assert config.simulate_fees is True
        assert config.fee_model == "polymarket"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SandboxConfig(
            mode=SandboxMode.LIVE,
            initial_balance=50000.0,
            max_position_size_pct=0.20,
            simulate_slippage=False
        )

        assert config.mode.value == "live"
        assert config.initial_balance == 50000.0
        assert config.max_position_size_pct == 0.20
        assert config.simulate_slippage is False


class TestSimulationState:
    """Tests for SimulationState dataclass."""

    def test_default_state(self):
        """Test default simulation state."""
        state = SimulationState(balance=10000.0)

        assert state.balance == 10000.0
        assert state.positions == {}
        assert state.pending_orders == []
        assert state.total_pnl == 0.0
        assert state.is_paused is False
        assert state.is_completed is False


class TestVirtualOrderExecutor:
    """Tests for VirtualOrderExecutor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SandboxConfig()
        self.executor = VirtualOrderExecutor(self.config)

    def test_execute_order_rejected_below_min_size(self):
        """Test order rejection when below minimum size."""
        order = VirtualOrder(
            order_id="test_001",
            market_id="market_123",
            side="buy",
            quantity=0.5,  # Below minimum of 1.0
            order_type="market"
        )
        market_data = {"market_id": "market_123", "current_price": 0.5}

        result = asyncio.run(self.executor.execute_virtual_order(order, market_data))

        assert result.status == OrderStatus.REJECTED.value
        assert "below minimum" in result.rejection_reason.lower()
        assert result.filled_quantity == 0

    @pytest.mark.asyncio
    async def test_execute_order_success(self):
        """Test successful order execution."""
        order = VirtualOrder(
            order_id="test_002",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        market_data = {
            "market_id": "market_123",
            "current_price": 0.5,
            "volatility": 0.1,
            "orderbook": {
                "bids": [{"price": 0.49, "size": 1000}],
                "asks": [{"price": 0.51, "size": 1000}],
                "last_trade_price": 0.5
            }
        }

        result = await self.executor.execute_virtual_order(order, market_data)

        assert result.status in [OrderStatus.FILLED.value, OrderStatus.PARTIAL_FILL.value, OrderStatus.NO_FILL.value]
        if result.status == OrderStatus.FILLED.value:
            assert result.filled_quantity > 0
            assert result.average_price > 0

    def test_fill_probability_market_order(self):
        """Test fill probability for market orders."""
        order = VirtualOrder(
            order_id="test_003",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market"
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }
        market_data = {"current_price": 0.5, "volatility": 0.1}

        fill_prob = self.executor._calculate_fill_probability(order, orderbook, market_data)

        assert 0 <= fill_prob <= 1

    def test_fill_probability_limit_order_aggressive(self):
        """Test fill probability for aggressive limit orders."""
        order = VirtualOrder(
            order_id="test_004",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="limit",
            limit_price=0.52
        )
        orderbook = {
            "bids": [{"price": 0.49, "size": 1000}],
            "asks": [{"price": 0.51, "size": 1000}],
            "last_trade_price": 0.5
        }
        market_data = {"current_price": 0.5, "volatility": 0.1}

        fill_prob = self.executor._calculate_fill_probability(order, orderbook, market_data)

        assert 0 <= fill_prob <= 1
        # Limit order at price should have some fill probability
        assert fill_prob > 0

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        slippage = self.executor._calculate_slippage(
            order=VirtualOrder(
                order_id="test_005",
                market_id="market_123",
                side="buy",
                quantity=100.0,
                order_type="market"
            ),
            reference_price=0.50,
            execution_price=0.52
        )

        # Buy execution above reference should have positive slippage
        assert slippage == pytest.approx(0.04, rel=0.01)

    def test_fee_calculation_polymarket(self):
        """Test Polymarket fee calculation."""
        self.config.fee_model = "polymarket"
        order = VirtualOrder(
            order_id="test_006",
            market_id="market_123",
            side="buy",
            quantity=100.0,
            order_type="market",
            trader_total_volume_30d=50000
        )

        fees = self.executor._calculate_fees(order, execution_price=0.50)

        # Fee should be positive and reasonable
        assert fees > 0
        assert fees < 10  # Less than 10% of notional


class TestVirtualPortfolioManager:
    """Tests for VirtualPortfolioManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SandboxConfig(initial_balance=10000.0)
        self.manager = VirtualPortfolioManager(self.config)

    def test_initial_state(self):
        """Test initial portfolio state."""
        summary = self.manager.get_portfolio_summary()

        assert summary.balance == 10000.0
        assert summary.positions_value == 0.0
        assert summary.total_value == 10000.0
        assert summary.position_count == 0

    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation."""
        value = self.manager._calculate_portfolio_value()

        assert value == 10000.0

    def test_validate_trade_constraints_min_size(self):
        """Test trade validation for minimum order size."""
        validation = self.manager._validate_trade_constraints(0.5)

        assert validation.is_valid is False
        assert "below minimum" in validation.reason.lower()

    def test_validate_trade_constraints_exceeds_max_position(self):
        """Test trade validation for exceeding max position size."""
        validation = self.manager._validate_trade_constraints(15000.0)

        assert validation.is_valid is False
        assert "exceeds max" in validation.reason.lower()

    def test_validate_trade_constraints_valid(self):
        """Test valid trade passes validation."""
        validation = self.manager._validate_trade_constraints(500.0)

        assert validation.is_valid is True

    def test_update_market_prices(self):
        """Test market price updates for positions."""
        # Add a position
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.50,
            unrealized_pnl=0.0
        )

        # Update prices
        self.manager.update_market_prices({"market_123": 0.60})

        position = self.manager.positions["market_123"]
        assert position.current_price == 0.60
        assert position.unrealized_pnl > 0  # Price went up

    def test_get_portfolio_summary_with_positions(self):
        """Test portfolio summary with open positions."""
        # Add a position
        self.manager.positions["market_123"] = VirtualPosition(
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            avg_price=0.50,
            current_price=0.60
        )

        summary = self.manager.get_portfolio_summary()

        assert summary.balance == 10000.0
        assert summary.positions_value == 60.0  # 100 * 0.60
        assert summary.total_value == 10060.0
        assert summary.position_count == 1


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = PerformanceTracker(initial_balance=10000.0)

    def test_initial_state(self):
        """Test initial tracker state."""
        assert self.tracker.initial_balance == 10000.0
        assert self.tracker.trades == []
        assert self.tracker.equity_curve == []

    def test_record_trade(self):
        """Test recording a trade."""
        trade = VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=10.0,
            roi=0.02
        )

        self.tracker.record_trade(trade)

        assert len(self.tracker.trades) == 1
        assert len(self.tracker.equity_curve) == 1

    def test_calculate_metrics_no_trades(self):
        """Test metrics calculation with no trades."""
        metrics = self.tracker.calculate_metrics()

        assert metrics.starting_balance == 10000.0
        assert metrics.ending_balance == 10000.0
        assert metrics.total_pnl == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_calculate_metrics_with_trades(self):
        """Test metrics calculation with trades."""
        # Add winning trade
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            exit_price=0.60,
            profit=10.0,
            roi=0.20
        ))

        # Add losing trade
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=50.0,
            entry_price=0.50,
            exit_price=0.40,
            profit=-5.0,
            roi=-0.10
        ))

        metrics = self.tracker.calculate_metrics()

        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(0.5, rel=0.01)
        assert metrics.profit_factor == 2.0  # 10 profit / 5 loss = 2.0

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Add daily returns
        self.tracker.daily_returns = [
            DailyReturn(date=datetime.utcnow(), return_pct=0.01, value=10100),
            DailyReturn(date=datetime.utcnow(), return_pct=-0.005, value=10050),
            DailyReturn(date=datetime.utcnow(), return_pct=0.02, value=10251),
        ]

        metrics = self.tracker.calculate_metrics()

        assert isinstance(metrics.sharpe_ratio, float)

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Add equity curve with drawdown
        self.tracker.equity_curve = [
            EquityPoint(timestamp=datetime.utcnow(), value=10000, return_pct=0),
            EquityPoint(timestamp=datetime.utcnow(), value=11000, return_pct=0.10),
            EquityPoint(timestamp=datetime.utcnow(), value=10500, return_pct=0.05),
            EquityPoint(timestamp=datetime.utcnow(), value=10000, return_pct=0),
        ]

        metrics = self.tracker.calculate_metrics()

        assert metrics.max_drawdown >= 0


class TestPerformanceReporter:
    """Tests for PerformanceReporter."""

    def test_generate_report(self):
        """Test report generation."""
        reporter = PerformanceReporter(output_path="./test_results")

        metrics = PerformanceMetrics(
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow(),
            trading_days=30,
            starting_balance=10000.0,
            ending_balance=11000.0,
            total_pnl=1000.0,
            total_pnl_pct=0.10,
            annualized_return=0.15,
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate=0.70,
            avg_win=200.0,
            avg_loss=-100.0,
            win_loss_ratio=2.0,
            profit_factor=2.33,
            volatility=0.02,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.05,
            max_drawdown_duration_days=5,
            calmar_ratio=3.0,
            avg_position_size=1000.0,
            max_position_size=2000.0,
            avg_hold_time_hours=24.0,
            max_hold_time_hours=72.0,
            min_hold_time_hours=1.0,
            traders_copied=5,
            profitable_traders=4,
            top_performing_trader="0x1234...",
            worst_performing_trader="0x5678...",
            avg_slippage=0.002,
            total_fees_paid=50.0,
            fill_rate=0.95,
            partial_fill_rate=0.05
        )

        report = reporter.generate_report(metrics, [], [])

        assert "Sandbox Simulation Report" in report
        assert "Executive Summary" in report
        assert "Performance Overview" in report
        assert "Risk Analysis" in report
        assert "$1,000.00" in report or "$1,000.00" in report or "1,000.00" in report


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_backtest_config(self):
        """Test backtest configuration."""
        config = BacktestConfig(starting_balance=50000.0)

        assert config.starting_balance == 50000.0
        assert config.output_path == "./backtest_results"

    def test_backtest_result(self):
        """Test backtest result structure."""
        result = BacktestResult(
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow(),
            starting_balance=10000.0,
            ending_balance=11000.0,
            total_pnl=1000.0,
            total_pnl_pct=0.10
        )

        assert result.starting_balance == 10000.0
        assert result.ending_balance == 11000.0
        assert result.total_pnl == 1000.0


class TestSandboxCLI:
    """Tests for Sandbox CLI."""

    def test_sandbox_manager_status(self):
        """Test sandbox manager status."""
        manager = SandboxManager()
        status = manager.get_status()

        assert status["status"] == "idle"
        assert status["balance"] == 10000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
