"""
Unit tests for PerformanceTracker and PerformanceReporter.

Tests metrics calculation, equity curve tracking, and report generation.
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.config import VirtualTrade
from sandbox.analytics import (
    PerformanceTracker,
    PerformanceReporter,
    PerformanceMetrics,
    DailyReturn,
    EquityPoint,
)


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = PerformanceTracker(initial_balance=10000.0)

    def test_initial_state(self):
        """Test initial tracker state."""
        assert self.tracker.initial_balance == 10000.0
        assert self.tracker.trades == []
        assert self.tracker.daily_returns == []
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

    def test_record_trade_updates_equity_curve(self):
        """Test equity curve update after recording trade with exit price."""
        trade = VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            exit_price=0.60,  # Add exit price to trigger P&L calculation
            profit=10.0,
            roi=0.02
        )

        self.tracker.record_trade(trade)

        assert len(self.tracker.equity_curve) == 1
        equity_point = self.tracker.equity_curve[0]
        assert equity_point.value == 10010.0  # 10000 + 10 profit
        assert equity_point.return_pct == 0.001  # 0.1%

    def test_calculate_current_value_with_trades(self):
        """Test current value calculation with trades that have exit prices."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            exit_price=0.60,
            profit=10.0,
            roi=0.02
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=50.0,
            entry_price=0.40,
            exit_price=0.30,
            profit=-5.0,
            roi=-0.10
        ))

        value = self.tracker._calculate_current_value()

        # 10000 + 10 - 5 = 10005
        assert value == 10005.0

    def test_calculate_metrics_no_trades(self):
        """Test metrics calculation with no trades."""
        metrics = self.tracker.calculate_metrics()

        assert metrics.starting_balance == 10000.0
        assert metrics.ending_balance == 10000.0
        assert metrics.total_pnl == 0.0
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        # With no trades, profit_factor is inf (no losses to divide by)
        assert metrics.profit_factor == float('inf')

    def test_calculate_metrics_single_trade(self):
        """Test metrics calculation with single winning trade."""
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

        metrics = self.tracker.calculate_metrics()

        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 1.0
        assert metrics.total_pnl == 10.0
        assert metrics.avg_win == 10.0
        assert metrics.avg_loss == 0.0

    def test_calculate_metrics_multiple_trades(self):
        """Test metrics calculation with multiple trades."""
        # Add winning trades
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            exit_price=0.60,
            profit=10.0,
            roi=0.20,
            source_trader="0x1111..."
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            exit_price=0.70,
            profit=20.0,
            roi=0.40,
            source_trader="0x1111..."
        ))

        # Add losing trade
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_003",
            market_id="market_789",
            outcome="YES",
            quantity=50.0,
            entry_price=0.50,
            exit_price=0.40,
            profit=-5.0,
            roi=-0.10,
            source_trader="0x2222..."
        ))

        metrics = self.tracker.calculate_metrics()

        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(2/3, rel=0.01)
        assert metrics.total_pnl == 25.0  # 10 + 20 - 5
        assert metrics.profit_factor == 30.0 / 5.0  # Gross profits / Gross losses

    def test_calculate_metrics_profit_factor_infinite(self):
        """Test profit factor when no losing trades."""
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

        metrics = self.tracker.calculate_metrics()

        assert metrics.profit_factor == float('inf')

    def test_calculate_volatility_no_returns(self):
        """Test volatility calculation with no returns."""
        volatility = self.tracker._calculate_volatility([])

        assert volatility == 0.0

    def test_calculate_volatility_single_return(self):
        """Test volatility calculation with single return."""
        volatility = self.tracker._calculate_volatility([0.01])

        assert volatility == 0.0  # Need at least 2 data points

    def test_calculate_volatility_multiple_returns(self):
        """Test volatility calculation with multiple returns."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.005]

        volatility = self.tracker._calculate_volatility(returns)

        assert volatility > 0

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.005]

        sharpe = self.tracker._calculate_sharpe_ratio(returns, annualized_return=0.10)

        assert isinstance(sharpe, float)

    def test_calculate_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        sharpe = self.tracker._calculate_sharpe_ratio([], annualized_return=0.10)

        assert sharpe == 0.0

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.005]

        sortino = self.tracker._calculate_sortino_ratio(returns)

        assert isinstance(sortino, float)

    def test_calculate_sortino_ratio_no_downside(self):
        """Test Sortino ratio with no downside returns."""
        returns = [0.01, 0.02, 0.005, 0.01]  # All positive

        sortino = self.tracker._calculate_sortino_ratio(returns)

        assert sortino == float('inf')

    def test_calculate_max_drawdown_no_equity(self):
        """Test max drawdown calculation with no equity curve."""
        max_dd, duration = self.tracker._calculate_max_drawdown()

        assert max_dd == 0.0
        assert duration == 0

    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with increasing equity."""
        self.tracker.equity_curve = [
            EquityPoint(timestamp=datetime.utcnow(), value=10000, return_pct=0),
            EquityPoint(timestamp=datetime.utcnow(), value=10500, return_pct=0.05),
            EquityPoint(timestamp=datetime.utcnow(), value=11000, return_pct=0.10),
        ]

        max_dd, duration = self.tracker._calculate_max_drawdown()

        assert max_dd == 0.0
        assert duration == 0

    def test_calculate_max_drawdown_with_drawdown(self):
        """Test max drawdown calculation with equity decline."""
        self.tracker.equity_curve = [
            EquityPoint(timestamp=datetime.utcnow(), value=10000, return_pct=0),
            EquityPoint(timestamp=datetime.utcnow(), value=11000, return_pct=0.10),
            EquityPoint(timestamp=datetime.utcnow(), value=10500, return_pct=0.05),
            EquityPoint(timestamp=datetime.utcnow(), value=10000, return_pct=0),
            EquityPoint(timestamp=datetime.utcnow(), value=11500, return_pct=0.15),
        ]

        max_dd, duration = self.tracker._calculate_max_drawdown()

        # Max drawdown from 11000 to 10000 = ~9.09%
        assert max_dd > 0
        assert max_dd == pytest.approx(0.0909, rel=0.01)

    def test_avg_position_size(self):
        """Test average position size calculation."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,  # $50 notional
            profit=5.0,
            roi=0.10
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=50.0,
            entry_price=1.00,  # $50 notional
            profit=5.0,
            roi=0.10
        ))

        avg_size = self.tracker._avg_position_size()

        # Both trades have $50 notional
        assert avg_size == 50.0

    def test_max_position_size(self):
        """Test maximum position size calculation."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=50.0,
            entry_price=0.50,  # $25 notional
            profit=2.5,
            roi=0.10
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=1.00,  # $100 notional
            profit=10.0,
            roi=0.10
        ))

        max_size = self.tracker._max_position_size()

        assert max_size == 100.0

    def test_avg_hold_time(self):
        """Test average hold time calculation."""
        base_time = datetime.utcnow()
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            hold_time_hours=24.0
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            hold_time_hours=48.0
        ))

        avg_time = self.tracker._avg_hold_time()

        assert avg_time == 36.0  # (24 + 48) / 2

    def test_unique_traders(self):
        """Test unique trader counting."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            source_trader="0x1111..."
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            source_trader="0x2222..."
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_003",
            market_id="market_789",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            source_trader="0x1111..."  # Same as first
        ))

        count = self.tracker._unique_traders()

        assert count == 2

    def test_profitable_traders(self):
        """Test profitable trader counting."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=10.0,  # Winning
            roi=0.20,
            source_trader="0x1111..."
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=-5.0,  # Losing
            roi=-0.10,
            source_trader="0x2222..."
        ))

        count = self.tracker._profitable_traders()

        assert count == 1  # Only 0x1111... is profitable

    def test_top_trader(self):
        """Test top performing trader identification."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=10.0,
            roi=0.20,
            source_trader="0x1111..."
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=20.0,  # Bigger profit
            roi=0.40,
            source_trader="0x2222..."
        ))

        top_trader = self.tracker._top_trader()

        assert top_trader == "0x2222..."

    def test_worst_trader(self):
        """Test worst performing trader identification."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=10.0,
            roi=0.20,
            source_trader="0x1111..."
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=-20.0,  # Bigger loss
            roi=-0.40,
            source_trader="0x2222..."
        ))

        worst_trader = self.tracker._worst_trader()

        assert worst_trader == "0x2222..."

    def test_trader_returns(self):
        """Test returns by trader calculation."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=10.0,
            roi=0.20,
            source_trader="0x1111..."
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=-5.0,
            roi=-0.10,
            source_trader="0x1111..."
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_003",
            market_id="market_789",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=15.0,
            roi=0.30,
            source_trader="0x2222..."
        ))

        returns = self.tracker._trader_returns()

        assert returns["0x1111..."] == 5.0  # 10 - 5
        assert returns["0x2222..."] == 15.0

    def test_avg_slippage(self):
        """Test average slippage calculation."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            slippage=0.01
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            slippage=0.03
        ))

        avg_slippage = self.tracker._avg_slippage()

        assert avg_slippage == 0.02  # (0.01 + 0.03) / 2

    def test_total_fees(self):
        """Test total fees calculation."""
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_001",
            market_id="market_123",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            fees=0.50
        ))
        self.tracker.record_trade(VirtualTrade(
            trade_id="trade_002",
            market_id="market_456",
            outcome="YES",
            quantity=100.0,
            entry_price=0.50,
            profit=5.0,
            roi=0.10,
            fees=0.75
        ))

        total_fees = self.tracker._total_fees()

        assert total_fees == 1.25  # 0.50 + 0.75

    def test_fill_rate(self):
        """Test fill rate (simplified)."""
        fill_rate = self.tracker._fill_rate()

        assert fill_rate == 0.95  # Simplified default

    def test_partial_fill_rate(self):
        """Test partial fill rate (simplified)."""
        partial_rate = self.tracker._partial_fill_rate()

        assert partial_rate == 0.05  # Simplified default


class TestPerformanceReporter:
    """Tests for PerformanceReporter class."""

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
        assert "Trade Analysis" in report
        assert "Trader Copy Performance" in report
        assert "Execution Quality" in report
        assert "Recommendations" in report

    def test_generate_equity_chart(self):
        """Test equity chart generation."""
        reporter = PerformanceReporter(output_path="./test_results")

        equity_curve = [
            EquityPoint(timestamp=datetime.utcnow(), value=10000, return_pct=0),
            EquityPoint(timestamp=datetime.utcnow(), value=10500, return_pct=0.05),
            EquityPoint(timestamp=datetime.utcnow(), value=10200, return_pct=0.02),
        ]

        chart = reporter._generate_equity_chart(equity_curve)

        assert "█" in chart  # Should contain ASCII bars
        assert "░" in chart  # Should contain empty bars

    def test_generate_equity_chart_empty(self):
        """Test equity chart with no data."""
        reporter = PerformanceReporter(output_path="./test_results")

        chart = reporter._generate_equity_chart([])

        assert "No equity data available" in chart

    def test_generate_recommendations_low_sharpe(self):
        """Test recommendations for low Sharpe ratio."""
        reporter = PerformanceReporter(output_path="./test_results")
        metrics = PerformanceMetrics(
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            trading_days=1,
            starting_balance=10000.0,
            ending_balance=10000.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            annualized_return=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            profit_factor=0.0,
            volatility=0.0,
            sharpe_ratio=0.5,  # Low Sharpe
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            calmar_ratio=0.0,
            avg_position_size=0.0,
            max_position_size=0.0,
            avg_hold_time_hours=0.0,
            max_hold_time_hours=0.0,
            min_hold_time_hours=0.0,
            traders_copied=0,
            profitable_traders=0,
            top_performing_trader="",
            worst_performing_trader=""
        )

        recommendations = reporter._generate_recommendations(metrics)

        assert "Consider reducing position sizes" in recommendations

    def test_generate_recommendations_high_drawdown(self):
        """Test recommendations for high drawdown."""
        reporter = PerformanceReporter(output_path="./test_results")
        metrics = PerformanceMetrics(
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            trading_days=1,
            starting_balance=10000.0,
            ending_balance=10000.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            annualized_return=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            profit_factor=0.0,
            volatility=0.0,
            sharpe_ratio=1.0,
            sortino_ratio=0.0,
            max_drawdown=0.3,  # High drawdown
            max_drawdown_duration_days=0,
            calmar_ratio=0.0,
            avg_position_size=0.0,
            max_position_size=0.0,
            avg_hold_time_hours=0.0,
            max_hold_time_hours=0.0,
            min_hold_time_hours=0.0,
            traders_copied=0,
            profitable_traders=0,
            top_performing_trader="",
            worst_performing_trader=""
        )

        recommendations = reporter._generate_recommendations(metrics)

        assert "stricter stop-losses" in recommendations

    def test_save_report(self):
        """Test report saving to file."""
        reporter = PerformanceReporter(output_path="./test_results")

        report = "# Test Report"
        filepath = reporter.save_report(report, "test_report.md")

        assert filepath == "./test_results/test_report.md"

        # Clean up
        import os
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_save_report_auto_filename(self):
        """Test report saving with auto-generated filename."""
        reporter = PerformanceReporter(output_path="./test_results")

        report = "# Test Report"
        filepath = reporter.save_report(report)

        assert "sandbox_report_" in filepath
        assert filepath.endswith(".md")

        # Clean up
        import os
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
