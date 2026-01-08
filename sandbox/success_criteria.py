"""
Success Criteria Evaluation for Sandbox Simulation.

Defines minimum requirements, targets, and stretch goals for live trading approval.
Provides evaluation functions to check if performance meets success criteria.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SuccessLevel(Enum):
    """Success level classification based on criteria thresholds."""
    BELOW_MINIMUM = "below_minimum"  # Does not meet minimum requirements
    MINIMUM = "minimum"  # Meets minimum requirements
    TARGET = "target"  # Meets target requirements
    STRETCH = "stretch"  # Exceeds target (stretch goal)


class MetricType(Enum):
    """Types of performance metrics for success criteria evaluation."""
    MONTHLY_RETURN = "monthly_return"
    WIN_RATE = "win_rate"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    SIMULATION_DURATION = "simulation_duration"
    CONSISTENCY = "consistency"


@dataclass
class MetricThresholds:
    """Thresholds for a single metric."""
    minimum: float
    target: float
    stretch: float

    def evaluate(self, value: float) -> SuccessLevel:
        """Evaluate a metric value against thresholds."""
        if value >= self.stretch:
            return SuccessLevel.STRETCH
        elif value >= self.target:
            return SuccessLevel.TARGET
        elif value >= self.minimum:
            return SuccessLevel.MINIMUM
        else:
            return SuccessLevel.BELOW_MINIMUM

    def is_met(self, value: float, level: SuccessLevel = SuccessLevel.MINIMUM) -> bool:
        """Check if a value meets the specified success level."""
        if level == SuccessLevel.STRETCH:
            return value >= self.stretch
        elif level == SuccessLevel.TARGET:
            return value >= self.target
        elif level == SuccessLevel.MINIMUM:
            return value >= self.minimum
        else:  # BELOW_MINIMUM - always "met" in the sense that it's below
            return True


# Metric threshold definitions based on plan requirements
METRIC_THRESHOLDS: Dict[MetricType, MetricThresholds] = {
    MetricType.MONTHLY_RETURN: MetricThresholds(
        minimum=0.03,    # > 3%
        target=0.05,     # > 5%
        stretch=0.08     # > 8%
    ),
    MetricType.WIN_RATE: MetricThresholds(
        minimum=0.52,    # > 52%
        target=0.58,     # > 58%
        stretch=0.65     # > 65%
    ),
    MetricType.SHARPE_RATIO: MetricThresholds(
        minimum=0.8,     # > 0.8
        target=1.2,      # > 1.2
        stretch=1.5      # > 1.5
    ),
    MetricType.MAX_DRAWDOWN: MetricThresholds(
        minimum=0.25,    # < 25%
        target=0.15,     # < 15%
        stretch=0.10     # < 10%
    ),
    MetricType.PROFIT_FACTOR: MetricThresholds(
        minimum=1.2,     # > 1.2
        target=1.5,      # > 1.5
        stretch=2.0      # > 2.0
    ),
    MetricType.SIMULATION_DURATION: MetricThresholds(
        minimum=90,      # 90 days minimum
        target=180,      # 180 days target
        stretch=365      # 365 days stretch
    ),
    # Consistency is evaluated differently (see ConsistencyThresholds below)
}


@dataclass
class ConsistencyThresholds:
    """Thresholds for consistency metric (evaluated as count of positive months)."""
    minimum: int      # 4/6 months positive (plan default)
    target: int       # 5/6 months positive
    stretch: int      # All months positive

    def evaluate(self, positive_months: int, total_months: int) -> SuccessLevel:
        """Evaluate consistency based on number of positive months."""
        required_ratio = positive_months / total_months if total_months > 0 else 0
        
        if total_months >= self.stretch and positive_months >= total_months:
            return SuccessLevel.STRETCH
        elif total_months >= self.target and positive_months >= (self.target * 0.83):  # 5/6 = 0.833
            return SuccessLevel.TARGET
        elif total_months >= self.minimum and positive_months >= (self.minimum * 0.67):  # 4/6 = 0.667
            return SuccessLevel.MINIMUM
        else:
            return SuccessLevel.BELOW_MINIMUM


# Default consistency thresholds (based on 6-month evaluation period)
CONSISTENCY_THRESHOLDS = ConsistencyThresholds(
    minimum=4,  # 4/6 months positive
    target=5,   # 5/6 months positive
    stretch=6   # All months positive
)


# Critical success factors from plan
CRITICAL_SUCCESS_FACTORS: List[str] = [
    "Consistency > Speed: Bot must generate steady returns, not volatile swings",
    "Risk Control: Never lose more than 20% of portfolio in worst case",
    "Reproducibility: Results must be consistent across multiple simulation runs",
    "Transparency: Every trade must be logged with full context",
    "Learning: Bot should improve over time as it 'learns' from more trader data"
]


@dataclass
class SuccessCriteriaResult:
    """Result of evaluating success criteria against performance metrics."""
    
    # Overall assessment
    overall_level: SuccessLevel
    is_approved_for_live: bool
    
    # Individual metric results
    monthly_return_level: SuccessLevel
    monthly_return_value: float
    monthly_return_met: bool
    
    win_rate_level: SuccessLevel
    win_rate_value: float
    win_rate_met: bool
    
    sharpe_ratio_level: SuccessLevel
    sharpe_ratio_value: float
    sharpe_ratio_met: bool
    
    max_drawdown_level: SuccessLevel
    max_drawdown_value: float
    max_drawdown_met: bool
    
    profit_factor_level: SuccessLevel
    profit_factor_value: float
    profit_factor_met: bool
    
    simulation_duration_level: SuccessLevel
    simulation_days: int
    simulation_duration_met: bool
    
    consistency_level: SuccessLevel
    positive_months: int
    total_months: int
    consistency_met: bool
    
    # Failure reasons (if any)
    failed_metrics: List[str]
    failure_reasons: Dict[str, str]
    
    # Timestamp
    evaluated_at: datetime


@dataclass  
class PerformanceSnapshot:
    """Snapshot of performance metrics for success criteria evaluation."""
    
    # Return metrics
    monthly_return_pct: float
    total_return_pct: float
    annualized_return_pct: float
    
    # Win/Loss metrics
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    
    # Duration
    simulation_days: int
    start_date: datetime
    end_date: datetime
    
    # Consistency (monthly returns)
    monthly_returns: List[Tuple[datetime, float]]  # (month_start, return_pct)
    
    @classmethod
    def from_metrics(cls, metrics) -> "PerformanceSnapshot":
        """Create snapshot from PerformanceMetrics dataclass."""
        # Calculate monthly returns from daily returns if available
        monthly_returns = []
        if hasattr(metrics, 'daily_returns') and metrics.daily_returns:
            from collections import defaultdict
            monthly_pnl = defaultdict(float)
            for daily in metrics.daily_returns:
                month_key = (daily.date.year, daily.date.month)
                monthly_pnl[month_key] += daily.return_pct
            
            # Convert to sorted list
            monthly_returns = [
                (datetime(year, month, 1), return_pct)
                for (year, month), return_pct in sorted(monthly_pnl.items())
            ]
        
        return cls(
            monthly_return_pct=metrics.total_pnl_pct if hasattr(metrics, 'total_pnl_pct') else 0,
            total_return_pct=metrics.total_pnl_pct,
            annualized_return_pct=metrics.annualized_return,
            win_rate=metrics.win_rate,
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            profit_factor=metrics.profit_factor,
            simulation_days=metrics.trading_days,
            start_date=metrics.start_date,
            end_date=metrics.end_date,
            monthly_returns=monthly_returns
        )


def evaluate_success_criteria(
    snapshot: PerformanceSnapshot,
    required_level: SuccessLevel = SuccessLevel.MINIMUM
) -> SuccessCriteriaResult:
    """
    Evaluate performance metrics against success criteria.
    
    Args:
        snapshot: PerformanceSnapshot containing current metrics
        required_level: Minimum success level required for live approval
        
    Returns:
        SuccessCriteriaResult with detailed evaluation
        
    The evaluation checks:
    1. Monthly Return > threshold (3% minimum, 5% target, 8% stretch)
    2. Win Rate > threshold (52% minimum, 58% target, 65% stretch)
    3. Sharpe Ratio > threshold (0.8 minimum, 1.2 target, 1.5 stretch)
    4. Max Drawdown < threshold (25% minimum, 15% target, 10% stretch)
    5. Profit Factor > threshold (1.2 minimum, 1.5 target, 2.0 stretch)
    6. Simulation Duration >= threshold (90 days minimum, 180 target, 365 stretch)
    7. Consistency: Positive months >= threshold (4/6 minimum, 5/6 target, 6/6 stretch)
    """
    
    failed_metrics = []
    failure_reasons = {}
    
    # Evaluate Monthly Return
    return_level = METRIC_THRESHOLDS[MetricType.MONTHLY_RETURN].evaluate(snapshot.monthly_return_pct)
    return_met = METRIC_THRESHOLDS[MetricType.MONTHLY_RETURN].is_met(snapshot.monthly_return_pct, required_level)
    if not return_met:
        failed_metrics.append("monthly_return")
        failure_reasons["monthly_return"] = (
            f"Monthly return {snapshot.monthly_return_pct:.2%} does not meet {required_level.value} "
            f"requirement of {METRIC_THRESHOLDS[MetricType.MONTHLY_RETURN].minimum:.1%}"
        )
    
    # Evaluate Win Rate
    win_rate_level = METRIC_THRESHOLDS[MetricType.WIN_RATE].evaluate(snapshot.win_rate)
    win_rate_met = METRIC_THRESHOLDS[MetricType.WIN_RATE].is_met(snapshot.win_rate, required_level)
    if not win_rate_met:
        failed_metrics.append("win_rate")
        failure_reasons["win_rate"] = (
            f"Win rate {snapshot.win_rate:.2%} does not meet {required_level.value} "
            f"requirement of {METRIC_THRESHOLDS[MetricType.WIN_RATE].minimum:.0%}"
        )
    
    # Evaluate Sharpe Ratio
    sharpe_level = METRIC_THRESHOLDS[MetricType.SHARPE_RATIO].evaluate(snapshot.sharpe_ratio)
    sharpe_met = METRIC_THRESHOLDS[MetricType.SHARPE_RATIO].is_met(snapshot.sharpe_ratio, required_level)
    if not sharpe_met:
        failed_metrics.append("sharpe_ratio")
        failure_reasons["sharpe_ratio"] = (
            f"Sharpe ratio {snapshot.sharpe_ratio:.2f} does not meet {required_level.value} "
            f"requirement of {METRIC_THRESHOLDS[MetricType.SHARPE_RATIO].minimum:.1f}"
        )
    
    # Evaluate Max Drawdown (inverted - lower is better)
    # For drawdown, we need inverted comparison: smaller value = better level
    dd_thresholds = METRIC_THRESHOLDS[MetricType.MAX_DRAWDOWN]
    dd_value = snapshot.max_drawdown
    
    # Determine level based on inverted comparison
    if dd_value <= dd_thresholds.stretch:
        dd_level = SuccessLevel.STRETCH
    elif dd_value <= dd_thresholds.target:
        dd_level = SuccessLevel.TARGET
    elif dd_value <= dd_thresholds.minimum:
        dd_level = SuccessLevel.MINIMUM
    else:
        dd_level = SuccessLevel.BELOW_MINIMUM
    
    # For drawdown, "meeting" means value <= threshold
    dd_met = dd_value <= dd_thresholds.minimum if required_level == SuccessLevel.MINIMUM else \
             dd_value <= dd_thresholds.target if required_level == SuccessLevel.TARGET else \
             dd_value <= dd_thresholds.stretch
    if not dd_met:
        failed_metrics.append("max_drawdown")
        failure_reasons["max_drawdown"] = (
            f"Max drawdown {snapshot.max_drawdown:.2%} does not meet {required_level.value} "
            f"requirement of <{METRIC_THRESHOLDS[MetricType.MAX_DRAWDOWN].minimum:.0%}"
        )
    
    # Evaluate Profit Factor
    pf_level = METRIC_THRESHOLDS[MetricType.PROFIT_FACTOR].evaluate(snapshot.profit_factor)
    pf_met = METRIC_THRESHOLDS[MetricType.PROFIT_FACTOR].is_met(snapshot.profit_factor, required_level)
    if not pf_met:
        failed_metrics.append("profit_factor")
        failure_reasons["profit_factor"] = (
            f"Profit factor {snapshot.profit_factor:.2f} does not meet {required_level.value} "
            f"requirement of {METRIC_THRESHOLDS[MetricType.PROFIT_FACTOR].minimum:.1f}"
        )
    
    # Evaluate Simulation Duration
    duration_level = METRIC_THRESHOLDS[MetricType.SIMULATION_DURATION].evaluate(snapshot.simulation_days)
    duration_met = METRIC_THRESHOLDS[MetricType.SIMULATION_DURATION].is_met(snapshot.simulation_days, required_level)
    if not duration_met:
        failed_metrics.append("simulation_duration")
        failure_reasons["simulation_duration"] = (
            f"Simulation duration {snapshot.simulation_days} days does not meet {required_level.value} "
            f"requirement of {int(METRIC_THRESHOLDS[MetricType.SIMULATION_DURATION].minimum)} days"
        )
    
    # Evaluate Consistency
    positive_months = sum(1 for _, ret in snapshot.monthly_returns if ret > 0)
    total_months = len(snapshot.monthly_returns)
    consistency_level = CONSISTENCY_THRESHOLDS.evaluate(positive_months, total_months)
    # For consistency, require at least 6 months of data for proper evaluation
    if total_months < 6:
        consistency_met = False
        failure_reasons["consistency"] = (
            f"Insufficient data for consistency evaluation: {total_months} months available, "
            f"minimum 6 months required"
        )
        failed_metrics.append("consistency")
    else:
        consistency_met = consistency_level != SuccessLevel.BELOW_MINIMUM
        if not consistency_met:
            failed_metrics.append("consistency")
            failure_reasons["consistency"] = (
                f"Consistency {positive_months}/{total_months} positive months does not meet "
                f"{required_level.value} requirement of {CONSISTENCY_THRESHOLDS.minimum}/6 positive months"
            )
    
    # Determine overall level and approval status
    # Must meet all minimum requirements for approval
    all_met = return_met and win_rate_met and sharpe_met and dd_met and pf_met and duration_met
    if total_months >= 6:
        all_met = all_met and consistency_met
    
    is_approved = all_met and len(failed_metrics) == 0
    
    # Determine overall level based on which level all metrics achieve
    levels_achieved = [
        return_level, win_rate_level, sharpe_level, dd_level, pf_level, duration_level
    ]
    if total_months >= 6:
        levels_achieved.append(consistency_level)
    
    # Overall level is the lowest level achieved (most conservative)
    level_order = [SuccessLevel.BELOW_MINIMUM, SuccessLevel.MINIMUM, SuccessLevel.TARGET, SuccessLevel.STRETCH]
    overall_level = min(levels_achieved, key=lambda x: level_order.index(x))
    
    return SuccessCriteriaResult(
        overall_level=overall_level,
        is_approved_for_live=is_approved,
        monthly_return_level=return_level,
        monthly_return_value=snapshot.monthly_return_pct,
        monthly_return_met=return_met,
        win_rate_level=win_rate_level,
        win_rate_value=snapshot.win_rate,
        win_rate_met=win_rate_met,
        sharpe_ratio_level=sharpe_level,
        sharpe_ratio_value=snapshot.sharpe_ratio,
        sharpe_ratio_met=sharpe_met,
        max_drawdown_level=dd_level,
        max_drawdown_value=snapshot.max_drawdown,
        max_drawdown_met=dd_met,
        profit_factor_level=pf_level,
        profit_factor_value=snapshot.profit_factor,
        profit_factor_met=pf_met,
        simulation_duration_level=duration_level,
        simulation_days=snapshot.simulation_days,
        simulation_duration_met=duration_met,
        consistency_level=consistency_level,
        positive_months=positive_months,
        total_months=total_months,
        consistency_met=consistency_met,
        failed_metrics=failed_metrics,
        failure_reasons=failure_reasons,
        evaluated_at=datetime.utcnow()
    )


def get_threshold_summary() -> str:
    """Get a formatted summary of all success criteria thresholds."""
    # Human-readable metric names
    metric_names = {
        MetricType.MONTHLY_RETURN: "Monthly Return",
        MetricType.WIN_RATE: "Win Rate",
        MetricType.SHARPE_RATIO: "Sharpe Ratio",
        MetricType.MAX_DRAWDOWN: "Max Drawdown",
        MetricType.PROFIT_FACTOR: "Profit Factor",
        MetricType.SIMULATION_DURATION: "Simulation Duration",
    }
    
    summary = "=" * 70 + "\n"
    summary += "COPY CAT SANDBOX SUCCESS CRITERIA SUMMARY\n"
    summary += "=" * 70 + "\n\n"
    
    summary += "MINIMUM REQUIREMENTS FOR LIVE TRADING APPROVAL\n"
    summary += "-" * 50 + "\n"
    
    for metric_type, thresholds in METRIC_THRESHOLDS.items():
        name = metric_names.get(metric_type, metric_type.value.replace("_", " ").title())
        
        if metric_type == MetricType.MAX_DRAWDOWN:
            # Drawdown: lower is better, use < symbol
            summary += f"\n{name}:\n"
            summary += f"  Minimum: < {thresholds.minimum:.0%}\n"
            summary += f"  Target:  < {thresholds.target:.0%}\n"
            summary += f"  Stretch: < {thresholds.stretch:.0%}\n"
        elif metric_type == MetricType.SIMULATION_DURATION:
            # Duration: integer days
            summary += f"\n{name}:\n"
            summary += f"  Minimum: {int(thresholds.minimum)} days\n"
            summary += f"  Target:  {int(thresholds.target)} days\n"
            summary += f"  Stretch: {int(thresholds.stretch)} days\n"
        elif metric_type == MetricType.SHARPE_RATIO:
            # Sharpe ratio: raw number
            summary += f"\n{name}:\n"
            summary += f"  Minimum: > {thresholds.minimum:.1f}\n"
            summary += f"  Target:  > {thresholds.target:.1f}\n"
            summary += f"  Stretch: > {thresholds.stretch:.1f}\n"
        elif metric_type == MetricType.PROFIT_FACTOR:
            # Profit factor: raw number
            summary += f"\n{name}:\n"
            summary += f"  Minimum: > {thresholds.minimum:.1f}\n"
            summary += f"  Target:  > {thresholds.target:.1f}\n"
            summary += f"  Stretch: > {thresholds.stretch:.1f}\n"
        else:
            # Percentages: return, win rate
            summary += f"\n{name}:\n"
            summary += f"  Minimum: > {thresholds.minimum:.0%}\n"
            summary += f"  Target:  > {thresholds.target:.0%}\n"
            summary += f"  Stretch: > {thresholds.stretch:.0%}\n"
    
    summary += f"\nCONSISTENCY:\n"
    summary += f"  Minimum: {CONSISTENCY_THRESHOLDS.minimum}/6 positive months\n"
    summary += f"  Target:  {CONSISTENCY_THRESHOLDS.target}/6 positive months\n"
    summary += f"  Stretch: {CONSISTENCY_THRESHOLDS.stretch}/6 positive months\n"
    
    summary += "\n" + "-" * 50 + "\n"
    summary += "CRITICAL SUCCESS FACTORS:\n"
    for factor in CRITICAL_SUCCESS_FACTORS:
        summary += f"  - {factor}\n"
    
    summary += "\n" + "=" * 70 + "\n"
    return summary


def format_evaluation_result(result: SuccessCriteriaResult) -> str:
    """Format a success criteria evaluation result as a readable string."""
    
    lines = []
    lines.append("=" * 60)
    lines.append("SUCCESS CRITERIA EVALUATION RESULT")
    lines.append("=" * 60)
    lines.append(f"\nEvaluated at: {result.evaluated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"\nOVERALL STATUS: {'✓ APPROVED' if result.is_approved_for_live else '✗ NOT APPROVED'}")
    lines.append(f"Overall Level: {result.overall_level.value.upper()}")
    
    lines.append("\n" + "-" * 60)
    lines.append("INDIVIDUAL METRICS:")
    lines.append("-" * 60)
    
    # Helper function to format metric result
    def format_metric(name: str, value: float, level: SuccessLevel, met: bool, format_str: str = "{:.2%}"):
        symbol = "✓" if met else "✗"
        lines.append(f"  {symbol} {name}: {format_str.format(value)} [{level.value.upper()}]")
    
    format_metric("Monthly Return", result.monthly_return_value, result.monthly_return_level, 
                  result.monthly_return_met)
    format_metric("Win Rate", result.win_rate_value, result.win_rate_level, result.win_rate_met)
    format_metric("Sharpe Ratio", result.sharpe_ratio_value, result.sharpe_ratio_level, 
                  result.sharpe_ratio_met, "{:.2f}")
    format_metric("Max Drawdown", result.max_drawdown_value, result.max_drawdown_level, 
                  result.max_drawdown_met)
    format_metric("Profit Factor", result.profit_factor_value, result.profit_factor_level, 
                  result.profit_factor_met, "{:.2f}")
    
    lines.append(f"  {'✓' if result.simulation_duration_met else '✗'} Simulation Duration: "
                 f"{result.simulation_days} days [{result.simulation_duration_level.value.upper()}]")
    
    if result.total_months > 0:
        lines.append(f"  {'✓' if result.consistency_met else '✗'} Consistency: "
                     f"{result.positive_months}/{result.total_months} positive months "
                     f"[{result.consistency_level.value.upper()}]")
    else:
        lines.append(f"  ⚪ Consistency: No monthly data available")
    
    if result.failed_metrics:
        lines.append("\n" + "-" * 60)
        lines.append("FAILURE DETAILS:")
        lines.append("-" * 60)
        for metric, reason in result.failure_reasons.items():
            lines.append(f"  ✗ {metric}: {reason}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)
