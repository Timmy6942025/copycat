"""
Unit tests for success criteria evaluation module.
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass

from sandbox.success_criteria import (
    SuccessLevel,
    MetricType,
    MetricThresholds,
    ConsistencyThresholds,
    SuccessCriteriaResult,
    PerformanceSnapshot,
    METRIC_THRESHOLDS,
    CONSISTENCY_THRESHOLDS,
    CRITICAL_SUCCESS_FACTORS,
    evaluate_success_criteria,
    get_threshold_summary,
    format_evaluation_result,
)


class TestMetricThresholds:
    """Tests for MetricThresholds class."""

    def test_evaluate_below_minimum(self):
        """Test evaluation returns BELOW_MINIMUM when value is below minimum."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        result = thresholds.evaluate(0.4)
        assert result == SuccessLevel.BELOW_MINIMUM

    def test_evaluate_at_minimum(self):
        """Test evaluation returns MINIMUM when value equals minimum."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        result = thresholds.evaluate(0.5)
        assert result == SuccessLevel.MINIMUM

    def test_evaluate_above_minimum_below_target(self):
        """Test evaluation returns MINIMUM when value is between minimum and target."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        result = thresholds.evaluate(0.6)
        assert result == SuccessLevel.MINIMUM

    def test_evaluate_at_target(self):
        """Test evaluation returns TARGET when value equals target."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        result = thresholds.evaluate(0.7)
        assert result == SuccessLevel.TARGET

    def test_evaluate_above_target_below_stretch(self):
        """Test evaluation returns TARGET when value is between target and stretch."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        result = thresholds.evaluate(0.8)
        assert result == SuccessLevel.TARGET

    def test_evaluate_at_stretch(self):
        """Test evaluation returns STRETCH when value equals stretch."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        result = thresholds.evaluate(0.9)
        assert result == SuccessLevel.STRETCH

    def test_evaluate_above_stretch(self):
        """Test evaluation returns STRETCH when value exceeds stretch."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        result = thresholds.evaluate(1.0)
        assert result == SuccessLevel.STRETCH

    def test_is_met_minimum_level(self):
        """Test is_met returns True when value meets minimum threshold."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        assert thresholds.is_met(0.5, SuccessLevel.MINIMUM) is True
        assert thresholds.is_met(0.6, SuccessLevel.MINIMUM) is True
        assert thresholds.is_met(0.4, SuccessLevel.MINIMUM) is False

    def test_is_met_target_level(self):
        """Test is_met returns True when value meets target threshold."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        assert thresholds.is_met(0.7, SuccessLevel.TARGET) is True
        assert thresholds.is_met(0.8, SuccessLevel.TARGET) is True
        assert thresholds.is_met(0.6, SuccessLevel.TARGET) is False

    def test_is_met_stretch_level(self):
        """Test is_met returns True when value meets stretch threshold."""
        thresholds = MetricThresholds(minimum=0.5, target=0.7, stretch=0.9)
        
        assert thresholds.is_met(0.9, SuccessLevel.STRETCH) is True
        assert thresholds.is_met(1.0, SuccessLevel.STRETCH) is True
        assert thresholds.is_met(0.8, SuccessLevel.STRETCH) is False


class TestConsistencyThresholds:
    """Tests for ConsistencyThresholds class."""

    def test_evaluate_below_minimum(self):
        """Test evaluation returns BELOW_MINIMUM when too few positive months."""
        thresholds = ConsistencyThresholds(minimum=4, target=5, stretch=6)
        
        result = thresholds.evaluate(2, 6)
        assert result == SuccessLevel.BELOW_MINIMUM

    def test_evaluate_at_minimum(self):
        """Test evaluation returns MINIMUM when exactly 4/6 positive months."""
        thresholds = ConsistencyThresholds(minimum=4, target=5, stretch=6)
        
        result = thresholds.evaluate(4, 6)
        assert result == SuccessLevel.MINIMUM

    def test_evaluate_above_minimum(self):
        """Test evaluation returns MINIMUM when between minimum and target."""
        thresholds = ConsistencyThresholds(minimum=4, target=5, stretch=6)
        
        result = thresholds.evaluate(4, 6)  # 4/6 = 0.667, should be MINIMUM
        assert result == SuccessLevel.MINIMUM

    def test_evaluate_at_target(self):
        """Test evaluation returns TARGET when 5/6 positive months."""
        thresholds = ConsistencyThresholds(minimum=4, target=5, stretch=6)
        
        result = thresholds.evaluate(5, 6)
        assert result == SuccessLevel.TARGET

    def test_evaluate_at_stretch(self):
        """Test evaluation returns STRETCH when all months positive."""
        thresholds = ConsistencyThresholds(minimum=4, target=5, stretch=6)
        
        result = thresholds.evaluate(6, 6)
        assert result == SuccessLevel.STRETCH

    def test_evaluate_insufficient_data(self):
        """Test evaluation with less than 6 months of data."""
        thresholds = ConsistencyThresholds(minimum=4, target=5, stretch=6)
        
        # 3 months with all positive (but less than minimum required)
        result = thresholds.evaluate(3, 3)
        assert result == SuccessLevel.BELOW_MINIMUM


class TestMetricThresholdsFromPlan:
    """Tests verifying thresholds match plan requirements."""

    def test_monthly_return_thresholds(self):
        """Verify monthly return thresholds match plan (>3%, >5%, >8%)."""
        thresholds = METRIC_THRESHOLDS[MetricType.MONTHLY_RETURN]
        
        assert thresholds.minimum == 0.03
        assert thresholds.target == 0.05
        assert thresholds.stretch == 0.08

    def test_win_rate_thresholds(self):
        """Verify win rate thresholds match plan (>52%, >58%, >65%)."""
        thresholds = METRIC_THRESHOLDS[MetricType.WIN_RATE]
        
        assert thresholds.minimum == 0.52
        assert thresholds.target == 0.58
        assert thresholds.stretch == 0.65

    def test_sharpe_ratio_thresholds(self):
        """Verify Sharpe ratio thresholds match plan (>0.8, >1.2, >1.5)."""
        thresholds = METRIC_THRESHOLDS[MetricType.SHARPE_RATIO]
        
        assert thresholds.minimum == 0.8
        assert thresholds.target == 1.2
        assert thresholds.stretch == 1.5

    def test_max_drawdown_thresholds(self):
        """Verify max drawdown thresholds match plan (<25%, <15%, <10%)."""
        thresholds = METRIC_THRESHOLDS[MetricType.MAX_DRAWDOWN]
        
        assert thresholds.minimum == 0.25
        assert thresholds.target == 0.15
        assert thresholds.stretch == 0.10

    def test_profit_factor_thresholds(self):
        """Verify profit factor thresholds match plan (>1.2, >1.5, >2.0)."""
        thresholds = METRIC_THRESHOLDS[MetricType.PROFIT_FACTOR]
        
        assert thresholds.minimum == 1.2
        assert thresholds.target == 1.5
        assert thresholds.stretch == 2.0

    def test_simulation_duration_thresholds(self):
        """Verify simulation duration thresholds match plan (90, 180, 365 days)."""
        thresholds = METRIC_THRESHOLDS[MetricType.SIMULATION_DURATION]
        
        assert thresholds.minimum == 90
        assert thresholds.target == 180
        assert thresholds.stretch == 365

    def test_consistency_thresholds(self):
        """Verify consistency thresholds match plan (4/6, 5/6, 6/6)."""
        assert CONSISTENCY_THRESHOLDS.minimum == 4
        assert CONSISTENCY_THRESHOLDS.target == 5
        assert CONSISTENCY_THRESHOLDS.stretch == 6


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot class."""

    def test_from_metrics(self):
        """Test creating PerformanceSnapshot from PerformanceMetrics."""
        @dataclass
        class MockMetrics:
            total_pnl_pct: float = 0.10
            annualized_return: float = 0.15
            win_rate: float = 0.60
            total_trades: int = 100
            winning_trades: int = 60
            losing_trades: int = 40
            sharpe_ratio: float = 1.0
            max_drawdown: float = 0.15
            profit_factor: float = 1.8
            trading_days: int = 180
            start_date: datetime = datetime(2024, 1, 1)
            end_date: datetime = datetime(2024, 7, 1)
            daily_returns: list = None
        
        # Mock daily returns
        MockMetrics.daily_returns = []
        
        metrics = MockMetrics()
        snapshot = PerformanceSnapshot.from_metrics(metrics)
        
        assert snapshot.monthly_return_pct == 0.10
        assert snapshot.total_return_pct == 0.10
        assert snapshot.annualized_return_pct == 0.15
        assert snapshot.win_rate == 0.60
        assert snapshot.sharpe_ratio == 1.0
        assert snapshot.max_drawdown == 0.15
        assert snapshot.profit_factor == 1.8
        assert snapshot.simulation_days == 180


class TestEvaluateSuccessCriteria:
    """Tests for evaluate_success_criteria function."""

    def test_meets_all_minimum_requirements(self):
        """Test evaluation when all minimum requirements are met."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            profit_factor=1.5,
            simulation_days=100,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 7, 1),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
                (datetime(2024, 4, 1), 0.02),
                (datetime(2024, 5, 1), 0.01),
                (datetime(2024, 6, 1), 0.02),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is True
        assert result.overall_level == SuccessLevel.MINIMUM

    def test_fails_monthly_return(self):
        """Test evaluation fails when monthly return is too low."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.01,  # Below 3% minimum
            total_return_pct=0.02,
            annualized_return_pct=0.03,
            win_rate=0.60,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            profit_factor=1.8,
            simulation_days=120,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 5, 1),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.01),
                (datetime(2024, 2, 1), 0.01),
                (datetime(2024, 3, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is False
        assert "monthly_return" in result.failed_metrics
        assert result.monthly_return_met is False

    def test_fails_win_rate(self):
        """Test evaluation fails when win rate is too low."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.45,  # Below 52% minimum
            total_trades=100,
            winning_trades=45,
            losing_trades=55,
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            profit_factor=1.4,
            simulation_days=100,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 11),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is False
        assert "win_rate" in result.failed_metrics

    def test_fails_sharpe_ratio(self):
        """Test evaluation fails when Sharpe ratio is too low."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            sharpe_ratio=0.5,  # Below 0.8 minimum
            max_drawdown=0.20,
            profit_factor=1.5,
            simulation_days=100,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 11),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is False
        assert "sharpe_ratio" in result.failed_metrics

    def test_fails_max_drawdown(self):
        """Test evaluation fails when max drawdown is too high."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            sharpe_ratio=1.0,
            max_drawdown=0.30,  # Above 25% maximum
            profit_factor=1.5,
            simulation_days=100,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 11),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is False
        assert "max_drawdown" in result.failed_metrics

    def test_fails_profit_factor(self):
        """Test evaluation fails when profit factor is too low."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            profit_factor=1.0,  # Below 1.2 minimum
            simulation_days=100,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 11),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is False
        assert "profit_factor" in result.failed_metrics

    def test_fails_simulation_duration(self):
        """Test evaluation fails when simulation duration is too short."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            profit_factor=1.5,
            simulation_days=30,  # Below 90 days minimum
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.05),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is False
        assert "simulation_duration" in result.failed_metrics

    def test_meets_target_requirements(self):
        """Test evaluation when all target requirements are met."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.06,  # > 5% target
            total_return_pct=0.12,
            annualized_return_pct=0.20,
            win_rate=0.62,  # > 58% target
            total_trades=150,
            winning_trades=93,
            losing_trades=57,
            sharpe_ratio=1.4,  # > 1.2 target
            max_drawdown=0.12,  # < 15% target
            profit_factor=1.8,  # > 1.5 target
            simulation_days=200,  # > 180 target
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 7, 19),
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
        
        assert result.is_approved_for_live is True
        assert result.overall_level == SuccessLevel.TARGET

    def test_meets_stretch_requirements(self):
        """Test evaluation when all stretch requirements are met."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.10,  # > 8% stretch
            total_return_pct=0.20,
            annualized_return_pct=0.35,
            win_rate=0.70,  # > 65% stretch
            total_trades=200,
            winning_trades=140,
            losing_trades=60,
            sharpe_ratio=1.8,  # > 1.5 stretch
            max_drawdown=0.08,  # < 10% stretch
            profit_factor=2.5,  # > 2.0 stretch
            simulation_days=400,  # > 365 stretch
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 2, 6),
            monthly_returns=[
                (datetime(2023, 1, 1), 0.02),
                (datetime(2023, 2, 1), 0.03),
                (datetime(2023, 3, 1), 0.02),
                (datetime(2023, 4, 1), 0.03),
                (datetime(2023, 5, 1), 0.02),
                (datetime(2023, 6, 1), 0.03),
                (datetime(2023, 7, 1), 0.02),
                (datetime(2023, 8, 1), 0.03),
                (datetime(2023, 9, 1), 0.02),
                (datetime(2023, 10, 1), 0.03),
                (datetime(2023, 11, 1), 0.02),
                (datetime(2023, 12, 1), 0.03),
                (datetime(2024, 1, 1), 0.02),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        
        assert result.is_approved_for_live is True
        assert result.overall_level == SuccessLevel.STRETCH


class TestGetThresholdSummary:
    """Tests for get_threshold_summary function."""

    def test_returns_non_empty_string(self):
        """Test that threshold summary returns a non-empty string."""
        summary = get_threshold_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_contains_metric_names(self):
        """Test that summary contains all metric names."""
        summary = get_threshold_summary()
        
        assert "Monthly Return" in summary
        assert "Win Rate" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary
        assert "Profit Factor" in summary
        assert "Simulation Duration" in summary

    def test_contains_thresholds(self):
        """Test that summary contains threshold values."""
        summary = get_threshold_summary()
        
        # Check for percentage values
        assert "3%" in summary
        assert "5%" in summary
        assert "8%" in summary
        assert "52%" in summary
        assert "58%" in summary
        assert "65%" in summary
        # Check for days
        assert "90 days" in summary
        assert "180 days" in summary
        assert "365 days" in summary

    def test_contains_critical_success_factors(self):
        """Test that summary contains critical success factors."""
        summary = get_threshold_summary()
        
        assert "CRITICAL SUCCESS FACTORS" in summary
        assert "Consistency" in summary
        assert "Risk Control" in summary


class TestFormatEvaluationResult:
    """Tests for format_evaluation_result function."""

    def test_returns_non_empty_string(self):
        """Test that formatted result returns a non-empty string."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            profit_factor=1.5,
            simulation_days=100,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 11),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        formatted = format_evaluation_result(result)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_contains_overall_status(self):
        """Test that formatted result contains approval status."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            profit_factor=1.5,
            simulation_days=100,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 11),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        formatted = format_evaluation_result(result)
        
        assert "APPROVED" in formatted or "NOT APPROVED" in formatted

    def test_contains_metric_values(self):
        """Test that formatted result contains metric values."""
        snapshot = PerformanceSnapshot(
            monthly_return_pct=0.05,
            total_return_pct=0.10,
            annualized_return_pct=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            profit_factor=1.5,
            simulation_days=100,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 11),
            monthly_returns=[
                (datetime(2024, 1, 1), 0.02),
                (datetime(2024, 2, 1), 0.03),
                (datetime(2024, 3, 1), 0.01),
            ]
        )
        
        result = evaluate_success_criteria(snapshot)
        formatted = format_evaluation_result(result)
        
        # Check for metric names
        assert "Monthly Return" in formatted
        assert "Win Rate" in formatted
        assert "Sharpe Ratio" in formatted


class TestCriticalSuccessFactors:
    """Tests for critical success factors."""

    def test_all_factors_present(self):
        """Test that all critical success factors from plan are present."""
        assert len(CRITICAL_SUCCESS_FACTORS) == 5
        
        # Check for key concepts
        factors_text = " ".join(CRITICAL_SUCCESS_FACTORS)
        assert "Consistency" in factors_text
        assert "Risk Control" in factors_text
        assert "Reproducibility" in factors_text
        assert "Transparency" in factors_text
        assert "Learning" in factors_text
