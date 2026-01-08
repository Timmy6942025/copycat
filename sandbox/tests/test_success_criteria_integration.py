"""
Integration tests for sandbox success criteria workflow.

Tests the complete flow from sandbox simulation execution
through success criteria evaluation and reporting.
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from sandbox import (
    SandboxConfig,
    SandboxMode,
    VirtualOrderExecutor,
    VirtualPortfolioManager,
    PerformanceTracker,
    SuccessLevel,
    PerformanceSnapshot,
    evaluate_success_criteria,
    format_evaluation_result,
    get_threshold_summary,
)


@dataclass
class MockPerformanceMetrics:
    """Mock performance metrics for testing."""
    start_date: datetime
    end_date: datetime
    trading_days: int
    total_pnl_pct: float
    annualized_return: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    daily_returns: list = field(default_factory=list)


class TestSandboxSuccessCriteriaIntegration:
    """Integration tests for sandbox to success criteria workflow."""

    def test_full_workflow_minimum_approval(self):
        """
        Test complete workflow: simulate trading → get metrics → evaluate criteria.
        
        Scenario: A bot that meets all MINIMUM requirements for live trading.
        """
        # Step 1: Configure sandbox
        config = SandboxConfig(
            mode=SandboxMode.SANDBOX,
            initial_balance=10000.0,
            max_orders_per_day=10,
        )
        assert config.initial_balance == 10000.0
        
        # Step 2: Create components
        executor = VirtualOrderExecutor(config)
        portfolio = VirtualPortfolioManager(config)
        tracker = PerformanceTracker(initial_balance=config.initial_balance)
        
        # Step 3: Simulate trades that produce minimum-level metrics
        # Monthly return: ~4% (>3% minimum)
        # Win rate: 55% (>52% minimum)
        # Sharpe ratio: 0.9 (>0.8 minimum)
        # Max drawdown: 22% (<25% minimum)
        # Profit factor: 1.4 (>1.2 minimum)
        # Simulation: 180 days (>90 minimum) with 6 months of data
        
        # Step 4: Create performance snapshot with 6 months of data
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.04,  # > 3% minimum
            total_return_pct=0.24,
            annualized_return_pct=0.35,
            win_rate=0.55,  # > 52% minimum
            total_trades=60,
            winning_trades=33,
            losing_trades=27,
            sharpe_ratio=0.9,  # > 0.8 minimum
            max_drawdown=0.22,  # < 25% minimum
            profit_factor=1.4,  # > 1.2 minimum
            simulation_days=180,  # > 90 minimum
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 7, 1),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.04),
                (datetime(2024, 2, 1), 0.04),
                (datetime(2024, 3, 1), 0.03),
                (datetime(2024, 4, 1), 0.05),
                (datetime(2024, 5, 1), 0.04),
                (datetime(2024, 6, 1), 0.04),
            ]
        )
        
        # Step 5: Evaluate against success criteria
        result = evaluate_success_criteria(snapshot)
        
        # Verify approval status
        assert result.is_approved_for_live is True
        assert result.overall_level == SuccessLevel.MINIMUM
        
        # Verify individual metrics
        assert result.monthly_return_met is True
        assert result.win_rate_met is True
        assert result.sharpe_ratio_met is True
        assert result.max_drawdown_met is True
        assert result.profit_factor_met is True
        assert result.simulation_duration_met is True
        assert result.consistency_met is True

    def test_full_workflow_target_approval(self):
        """
        Test complete workflow for a bot that meets TARGET requirements.
        
        Scenario: A better performing bot that exceeds minimums.
        """
        # Create snapshot with TARGET-level metrics
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.07,  # > 5% target
            total_return_pct=0.14,
            annualized_return_pct=0.25,
            win_rate=0.62,  # > 58% target
            total_trades=50,
            winning_trades=31,
            losing_trades=19,
            sharpe_ratio=1.4,  # > 1.2 target
            max_drawdown=0.12,  # < 15% target
            profit_factor=1.8,  # > 1.5 target
            simulation_days=200,  # > 180 target
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 7, 19),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.02),
                (datetime(2024, 4, 1), 0.03),
                (datetime(2024, 5, 1), 0.02),
                (datetime(2024, 6, 1), 0.03),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is True
        assert result.overall_level == SuccessLevel.TARGET
        assert result.monthly_return_level == SuccessLevel.TARGET
        assert result.win_rate_level == SuccessLevel.TARGET

    def test_full_workflow_rejection(self):
        """
        Test complete workflow for a bot that fails to meet requirements.
        
        Scenario: A poorly performing bot that doesn't qualify for live trading.
        """
        # Create snapshot with failing metrics
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.01,  # < 3% minimum - FAIL
            total_return_pct=0.02,
            annualized_return_pct=0.03,
            win_rate=0.40,  # < 52% minimum - FAIL
            total_trades=30,
            winning_trades=12,
            losing_trades=18,
            sharpe_ratio=0.3,  # < 0.8 minimum - FAIL
            max_drawdown=0.35,  # > 25% maximum - FAIL
            profit_factor=0.8,  # < 1.2 minimum - FAIL
            simulation_days=60,  # < 90 minimum - FAIL
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.01),
                (datetime(2024, 2, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is False
        assert result.overall_level == SuccessLevel.BELOW_MINIMUM
        assert "monthly_return" in result.failed_metrics
        assert "win_rate" in result.failed_metrics
        assert "sharpe_ratio" in result.failed_metrics
        assert "max_drawdown" in result.failed_metrics
        assert "profit_factor" in result.failed_metrics
        assert "simulation_duration" in result.failed_metrics

    def test_consistency_requirement(self):
        """Test that consistency requirement is properly evaluated."""
        # 6 months with 4 positive - should meet minimum
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.04,
            total_return_pct=0.08,
            annualized_return_pct=0.12,
            win_rate=0.55,
            total_trades=60,
            winning_trades=33,
            losing_trades=27,
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            profit_factor=1.4,
            simulation_days=180,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 7, 1),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),   # positive
                (datetime(2024, 2, 1), -0.01),  # negative
                (datetime(2024, 3, 1), 0.03),   # positive
                (datetime(2024, 4, 1), 0.01),   # positive
                (datetime(2024, 5, 1), -0.02),  # negative
                (datetime(2024, 6, 1), 0.02),   # positive
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.positive_months == 4
        assert result.total_months == 6
        assert result.consistency_met is True

    def test_formatted_output_includes_all_metrics(self):
        """Test that formatted evaluation result includes all relevant metrics."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.06,
            total_return_pct=0.12,
            annualized_return_pct=0.20,
            win_rate=0.60,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            sharpe_ratio=1.3,
            max_drawdown=0.15,
            profit_factor=1.7,
            simulation_days=180,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 7, 1),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
                (datetime(2024, 4, 1), 0.02),
                (datetime(2024, 5, 1), 0.03),
                (datetime(2024, 6, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        formatted = format_evaluation_result(result)
        
        # Verify formatted output contains key sections
        assert "APPROVED" in formatted or "NOT APPROVED" in formatted
        assert "Monthly Return" in formatted
        assert "Win Rate" in formatted
        assert "Sharpe Ratio" in formatted
        assert "Max Drawdown" in formatted
        assert "Profit Factor" in formatted

    def test_threshold_summary_contains_all_requirements(self):
        """Test that threshold summary displays all success criteria."""
        summary = get_threshold_summary()
        
        # Verify all metrics are included
        assert "Monthly Return" in summary
        assert "Win Rate" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary
        assert "Profit Factor" in summary
        assert "Simulation Duration" in summary
        assert "CONSISTENCY" in summary
        
        # Verify threshold values
        assert "3%" in summary  # Minimum monthly return
        assert "52%" in summary  # Minimum win rate
        assert "0.8" in summary  # Minimum Sharpe ratio
        assert "90 days" in summary  # Minimum simulation duration

    def test_sandbox_components_integration(self):
        """Test that sandbox components work together correctly."""
        # Configure sandbox
        config = SandboxConfig(
            mode=SandboxMode.SANDBOX,
            initial_balance=50000.0,
            max_position_size_pct=0.10,
            max_total_exposure_pct=0.50,
        )
        
        # Create components
        executor = VirtualOrderExecutor(config)
        portfolio = VirtualPortfolioManager(config)
        tracker = PerformanceTracker(initial_balance=config.initial_balance)
        
        # Verify initial states
        assert tracker.initial_balance == 50000.0
        assert len(tracker.trades) == 0
        assert len(tracker.equity_curve) == 0
        
        # Simulate a series of trades with proper VirtualTrade attributes
        from sandbox.config import VirtualTrade
        
        trades_data = [
            {"profit": 500, "quantity": 100, "entry_price": 0.5, "timestamp": datetime(2024, 1, 5)},
            {"profit": 300, "quantity": 75, "entry_price": 0.6, "timestamp": datetime(2024, 1, 15)},
            {"profit": -200, "quantity": 50, "entry_price": 0.7, "timestamp": datetime(2024, 2, 1)},
            {"profit": 400, "quantity": 80, "entry_price": 0.55, "timestamp": datetime(2024, 2, 10)},
            {"profit": 100, "quantity": 40, "entry_price": 0.65, "timestamp": datetime(2024, 2, 20)},
        ]
        
        for trade_data in trades_data:
            trade = VirtualTrade(
                trade_id=f"test_{trade_data['timestamp'].timestamp()}",
                market_id="test_market",
                outcome="YES",
                quantity=trade_data["quantity"],
                entry_price=trade_data["entry_price"],
                exit_price=trade_data["entry_price"] + (trade_data["profit"] / trade_data["quantity"]) * 0.01,
                profit=trade_data["profit"],
                roi=trade_data["profit"] / (trade_data["quantity"] * trade_data["entry_price"]),
                timestamp=trade_data["timestamp"],
            )
            tracker.record_trade(trade)
        
        # Verify trades were recorded
        assert len(tracker.trades) == 5
        
        # Calculate metrics
        metrics = tracker.calculate_metrics()
        
        # Verify metrics calculation
        assert metrics.total_trades == 5
        assert metrics.winning_trades == 4
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.8  # 4/5
        assert metrics.profit_factor > 0

    def test_stretch_requirements_approval(self):
        """Test that stretch requirements produce STRETCH overall level."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.12,  # > 8% stretch
            total_return_pct=0.25,
            annualized_return_pct=0.45,
            win_rate=0.75,  # > 65% stretch
            total_trades=200,
            winning_trades=150,
            losing_trades=50,
            sharpe_ratio=1.8,  # > 1.5 stretch
            max_drawdown=0.07,  # < 10% stretch
            profit_factor=2.5,  # > 2.0 stretch
            simulation_days=400,  # > 365 stretch
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 2, 6),
            monthly_returns=[
                (datetime(2023, 1, 1), 0.03),
                (datetime(2023, 2, 1), 0.04),
                (datetime(2023, 3, 1), 0.03),
                (datetime(2023, 4, 1), 0.04),
                (datetime(2023, 5, 1), 0.03),
                (datetime(2023, 6, 1), 0.04),
                (datetime(2023, 7, 1), 0.03),
                (datetime(2023, 8, 1), 0.04),
                (datetime(2023, 9, 1), 0.03),
                (datetime(2023, 10, 1), 0.04),
                (datetime(2023, 11, 1), 0.03),
                (datetime(2023, 12, 1), 0.04),
                (datetime(2024, 1, 1), 0.03),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is True
        assert result.overall_level == SuccessLevel.STRETCH


class TestSandboxConfigurationIntegration:
    """Test sandbox configuration integration with success criteria."""

    def test_different_initial_balances(self):
        """Test that different initial balances work with success criteria."""
        balances = [1000, 5000, 10000, 50000, 100000]
        
        for balance in balances:
            config = SandboxConfig(initial_balance=balance)
            tracker = PerformanceTracker(initial_balance=balance)
            
            # Simulate 5% return
            from sandbox.config import VirtualTrade
            virtual_trade = VirtualTrade(
                trade_id=f"test_{datetime.utcnow().timestamp()}",
                market_id="test_market",
                outcome="YES",
                quantity=100,
                entry_price=0.5,
                exit_price=0.55,
                profit=balance * 0.05,
                roi=0.1,
                timestamp=datetime.utcnow(),
            )
            tracker.record_trade(virtual_trade)
            
            # Create snapshot with 6 months of data for consistency
            snapshot = PerformanceSnapshot(
                monthly_return_pct=0.05,
                total_return_pct=0.30,
                annualized_return_pct=0.60,
                win_rate=1.0,
                total_trades=6,
                winning_trades=6,
                losing_trades=0,
                sharpe_ratio=1.5,
                max_drawdown=0.05,
                profit_factor=3.0,
                simulation_days=180,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 7, 1),
                monthly_returns=[
                    (datetime(2024, 1, 1), 0.05),
                    (datetime(2024, 2, 1), 0.05),
                    (datetime(2024, 3, 1), 0.05),
                    (datetime(2024, 4, 1), 0.05),
                    (datetime(2024, 5, 1), 0.05),
                    (datetime(2024, 6, 1), 0.05),
                ]
            )
            
            result = evaluate_success_criteria(snapshot)
            assert result.is_approved_for_live is True
