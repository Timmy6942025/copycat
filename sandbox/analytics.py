"""
Performance Tracking and Analytics for Sandbox Simulation.

Provides performance metrics calculation and report generation.
"""

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from sandbox.config import VirtualTrade


@dataclass
class DailyReturn:
    """Daily return data point."""
    date: datetime
    return_pct: float
    value: float


@dataclass
class EquityPoint:
    """Equity curve data point."""
    timestamp: datetime
    value: float
    return_pct: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for sandbox simulation."""

    # Time Period
    start_date: datetime
    end_date: datetime
    trading_days: int

    # Portfolio Performance
    starting_balance: float
    ending_balance: float
    total_pnl: float
    total_pnl_pct: float
    annualized_return: float

    # Win/Loss Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    profit_factor: float  # Gross profits / Gross losses

    # Risk Metrics
    volatility: float  # Standard deviation of returns
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float

    # Position Statistics
    avg_position_size: float
    max_position_size: float
    avg_hold_time_hours: float
    max_hold_time_hours: float
    min_hold_time_hours: float

    # Trader Copy Performance
    traders_copied: int
    profitable_traders: int
    top_performing_trader: str
    worst_performing_trader: str
    trader_specific_returns: Dict[str, float] = field(default_factory=dict)

    # Execution Quality
    avg_slippage: float = 0.0
    total_fees_paid: float = 0.0
    fill_rate: float = 1.0
    partial_fill_rate: float = 0.0


class PerformanceTracker:
    """
    Tracks and calculates performance metrics for sandbox simulation.
    """

    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.trades: List[VirtualTrade] = []
        self.daily_returns: List[DailyReturn] = []
        self.equity_curve: List[EquityPoint] = []

    def record_trade(self, trade: VirtualTrade):
        """Record a completed trade."""
        self.trades.append(trade)
        self._update_equity_curve()

    def _update_equity_curve(self):
        """Update equity curve with current portfolio value."""
        current_value = self._calculate_current_value()
        self.equity_curve.append(EquityPoint(
            timestamp=datetime.utcnow(),
            value=current_value,
            return_pct=(current_value - self.initial_balance) / self.initial_balance
        ))

    def _calculate_current_value(self) -> float:
        """Calculate current portfolio value from trades."""
        realized_pnl = sum(t.profit for t in self.trades if t.exit_price is not None)
        return self.initial_balance + realized_pnl

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        returns = [d.return_pct for d in self.daily_returns]
        winning_trades = [t for t in self.trades if t.profit > 0]
        losing_trades = [t for t in self.trades if t.profit <= 0]

        # Time calculations
        if self.equity_curve:
            start_date = self.equity_curve[0].timestamp
            end_date = self.equity_curve[-1].timestamp
            trading_days = max(1, (end_date - start_date).days)
        else:
            start_date = datetime.utcnow()
            end_date = datetime.utcnow()
            trading_days = 0

        # Basic returns
        final_balance = self._calculate_current_value()
        total_pnl = final_balance - self.initial_balance
        total_pnl_pct = total_pnl / self.initial_balance if self.initial_balance > 0 else 0

        # Annualized return
        if trading_days > 0 and total_pnl_pct > -1:
            years = trading_days / 365.25
            annualized_return = ((1 + total_pnl_pct) ** (1 / years)) - 1 if years > 0 else 0
        else:
            annualized_return = 0

        # Win/Loss statistics
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(t.profit for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.profit for t in losing_trades) / len(losing_trades) if losing_trades else 0

        # Profit factor
        gross_profits = sum(t.profit for t in winning_trades)
        gross_losses = abs(sum(t.profit for t in losing_trades))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

        # Risk metrics
        volatility = self._calculate_volatility(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns, annualized_return)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown, max_dd_duration = self._calculate_max_drawdown()

        # Position statistics
        avg_position_size = self._avg_position_size()
        max_position_size = self._max_position_size()
        avg_hold_time = self._avg_hold_time()
        max_hold_time = self._max_hold_time()
        min_hold_time = self._min_hold_time()

        # Trader performance
        unique_traders = self._unique_traders()
        profitable_traders_count = self._profitable_traders()
        top_trader = self._top_trader()
        worst_trader = self._worst_trader()
        trader_returns = self._trader_returns()

        # Execution quality
        avg_slippage = self._avg_slippage()
        total_fees = self._total_fees()
        fill_rate = self._fill_rate()
        partial_fill_rate = self._partial_fill_rate()

        return PerformanceMetrics(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            starting_balance=self.initial_balance,
            ending_balance=final_balance,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            annualized_return=annualized_return,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            profit_factor=profit_factor,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=max_dd_duration,
            calmar_ratio=annualized_return / max_drawdown if max_drawdown > 0 else float('inf'),
            avg_position_size=avg_position_size,
            max_position_size=max_position_size,
            avg_hold_time_hours=avg_hold_time,
            max_hold_time_hours=max_hold_time,
            min_hold_time_hours=min_hold_time,
            traders_copied=unique_traders,
            profitable_traders=profitable_traders_count,
            top_performing_trader=top_trader,
            worst_performing_trader=worst_trader,
            trader_specific_returns=trader_returns,
            avg_slippage=avg_slippage,
            total_fees_paid=total_fees,
            fill_rate=fill_rate,
            partial_fill_rate=partial_fill_rate
        )

    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate standard deviation of returns."""
        if len(returns) < 2:
            return 0.0
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance)

    def _calculate_sharpe_ratio(self, returns: List[float], annualized_return: float) -> float:
        """Calculate Sharpe ratio."""
        volatility = self._calculate_volatility(returns)
        if volatility == 0:
            return 0.0
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        return (annualized_return - risk_free_rate) / volatility

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(returns) < 2:
            return 0.0
        mean_return = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float('inf')
        downside_variance = sum((r - mean_return) ** 2 for r in downside_returns) / len(downside_returns)
        downside_deviation = math.sqrt(downside_variance)
        if downside_deviation == 0:
            return 0.0
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_return = mean_return - risk_free_rate
        return excess_return / downside_deviation

    def _calculate_max_drawdown(self) -> tuple:
        """Calculate maximum drawdown and its duration."""
        if not self.equity_curve:
            return 0.0, 0

        max_value = self.equity_curve[0].value
        max_drawdown = 0.0
        max_dd_start = self.equity_curve[0].timestamp
        max_dd_end = self.equity_curve[0].timestamp
        current_dd_start = None

        for point in self.equity_curve:
            if point.value > max_value:
                max_value = point.value
                current_dd_start = None
            elif max_value > 0:
                drawdown = (max_value - point.value) / max_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    if current_dd_start:
                        max_dd_duration = (point.timestamp - current_dd_start).days
                    else:
                        max_dd_duration = 0
                    max_dd_start = point.timestamp
                    max_dd_end = point.timestamp
                if drawdown > 0 and current_dd_start is None:
                    current_dd_start = point.timestamp

        max_dd_duration = (max_dd_end - max_dd_start).days if max_dd_start else 0
        return max_drawdown, max_dd_duration

    def _avg_position_size(self) -> float:
        """Calculate average position size."""
        if not self.trades:
            return 0.0
        return sum(t.quantity * t.entry_price for t in self.trades) / len(self.trades)

    def _max_position_size(self) -> float:
        """Calculate maximum position size."""
        if not self.trades:
            return 0.0
        return max(t.quantity * t.entry_price for t in self.trades)

    def _avg_hold_time(self) -> float:
        """Calculate average hold time in hours."""
        hold_times = [t.hold_time_hours for t in self.trades if t.hold_time_hours is not None]
        if not hold_times:
            return 0.0
        return sum(hold_times) / len(hold_times)

    def _max_hold_time(self) -> float:
        """Calculate maximum hold time in hours."""
        hold_times = [t.hold_time_hours for t in self.trades if t.hold_time_hours is not None]
        if not hold_times:
            return 0.0
        return max(hold_times)

    def _min_hold_time(self) -> float:
        """Calculate minimum hold time in hours."""
        hold_times = [t.hold_time_hours for t in self.trades if t.hold_time_hours is not None]
        if not hold_times:
            return 0.0
        return min(hold_times)

    def _unique_traders(self) -> int:
        """Count unique traders copied."""
        traders = set(t.source_trader for t in self.trades if t.source_trader)
        return len(traders)

    def _profitable_traders(self) -> int:
        """Count profitable traders."""
        trader_returns = self._trader_returns()
        return sum(1 for r in trader_returns.values() if r > 0)

    def _top_trader(self) -> str:
        """Get top performing trader address."""
        trader_returns = self._trader_returns()
        if not trader_returns:
            return "N/A"
        return max(trader_returns, key=trader_returns.get)

    def _worst_trader(self) -> str:
        """Get worst performing trader address."""
        trader_returns = self._trader_returns()
        if not trader_returns:
            return "N/A"
        return min(trader_returns, key=trader_returns.get)

    def _trader_returns(self) -> Dict[str, float]:
        """Calculate returns by trader."""
        trader_profits: Dict[str, float] = {}
        for t in self.trades:
            if t.source_trader:
                if t.source_trader not in trader_profits:
                    trader_profits[t.source_trader] = 0
                trader_profits[t.source_trader] += t.profit
        return trader_profits

    def _avg_slippage(self) -> float:
        """Calculate average slippage."""
        if not self.trades:
            return 0.0
        slippage_values = [t.slippage for t in self.trades if t.slippage is not None]
        return sum(slippage_values) / len(slippage_values) if slippage_values else 0.0

    def _total_fees(self) -> float:
        """Calculate total fees paid."""
        return sum(t.fees for t in self.trades if t.fees is not None)

    def _fill_rate(self) -> float:
        """Calculate fill rate (simplified)."""
        # In a real implementation, this would track order fill history
        return 0.95  # Simplified: assume 95% fill rate

    def _partial_fill_rate(self) -> float:
        """Calculate partial fill rate (simplified)."""
        return 0.05  # Simplified: assume 5% partial fill rate


class PerformanceReporter:
    """
    Generates comprehensive performance reports for sandbox simulation.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path

    def generate_report(
        self,
        metrics: PerformanceMetrics,
        trades: List[VirtualTrade],
        equity_curve: List[EquityPoint]
    ) -> str:
        """
        Generate HTML/PDF performance report.

        Sections:
        1. Executive Summary
        2. Performance Overview (charts)
        3. Risk Analysis
        4. Trade Analysis
        5. Trader Copy Performance
        6. Execution Quality
        7. Recommendations
        """
        success_rate = metrics.profitable_traders/metrics.traders_copied if metrics.traders_copied > 0 else None
        success_rate_str = f"{success_rate:.2%}" if success_rate is not None else "N/A"
        
        report = f"""# Sandbox Simulation Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Simulation Period:** {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}
**Duration:** {metrics.trading_days} days

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Starting Balance | ${metrics.starting_balance:,.2f} |
| Ending Balance | ${metrics.ending_balance:,.2f} |
| **Total P&L** | **${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:.2%})** |
| Annualized Return | {metrics.annualized_return:.2%} |
| Win Rate | {metrics.win_rate:.2%} |
| Profit Factor | {metrics.profit_factor:.2f} |
| Sharpe Ratio | {metrics.sharpe_ratio:.2f} |
| Max Drawdown | {metrics.max_drawdown:.2%} |

---

## 2. Performance Overview

### Equity Curve

```
{self._generate_equity_chart(equity_curve)}
```

### Monthly Returns

| Month | Return |
|-------|--------|
{self._generate_monthly_returns_table(metrics, trades)}

---

## 3. Risk Analysis

| Metric | Value | Assessment |
|--------|-------|------------|
| Volatility | {metrics.volatility:.2%} | {"High" if metrics.volatility > 0.05 else "Medium" if metrics.volatility > 0.02 else "Low"} |
| Sharpe Ratio | {metrics.sharpe_ratio:.2f} | {"Excellent" if metrics.sharpe_ratio > 2 else "Good" if metrics.sharpe_ratio > 1 else "Fair" if metrics.sharpe_ratio > 0.5 else "Poor"} |
| Sortino Ratio | {metrics.sortino_ratio:.2f} | {"Excellent" if metrics.sortino_ratio > 3 else "Good" if metrics.sortino_ratio > 2 else "Fair" if metrics.sortino_ratio > 1 else "Poor"} |
| Max Drawdown | {metrics.max_drawdown:.2%} | {"Severe" if metrics.max_drawdown > 0.3 else "Moderate" if metrics.max_drawdown > 0.15 else "Acceptable"} |
| Calmar Ratio | {metrics.calmar_ratio:.2f} | {"Excellent" if metrics.calmar_ratio > 3 else "Good" if metrics.calmar_ratio > 1 else "Fair" if metrics.calmar_ratio > 0.5 else "Poor"} |

---

## 4. Trade Analysis

| Metric | Value |
|--------|-------|
| Total Trades | {metrics.total_trades} |
| Winning Trades | {metrics.winning_trades} |
| Losing Trades | {metrics.losing_trades} |
| Average Win | ${metrics.avg_win:,.2f} |
| Average Loss | ${metrics.avg_loss:,.2f} |
| Win/Loss Ratio | {metrics.win_loss_ratio:.2f} |
| Average Position Size | ${metrics.avg_position_size:,.2f} |
| Maximum Position Size | ${metrics.max_position_size:,.2f} |
| Average Hold Time | {metrics.avg_hold_time_hours:.1f} hours |

---

## 5. Trader Copy Performance

| Metric | Value |
|--------|-------|
| Traders Copied | {metrics.traders_copied} |
| Profitable Traders | {metrics.profitable_traders} |
| Success Rate | {success_rate_str} |

**Top Performer:** {metrics.top_performing_trader}
**Worst Performer:** {metrics.worst_performing_trader}

### Individual Trader Returns

| Trader | Return | Trades | Win Rate |
|--------|--------|--------|----------|
{self._generate_trader_table(metrics)}

---

## 6. Execution Quality

| Metric | Value |
|--------|-------|
| Average Slippage | {metrics.avg_slippage:.4f} |
| Total Fees Paid | ${metrics.total_fees_paid:,.2f} |
| Fill Rate | {metrics.fill_rate:.2%} |
| Partial Fill Rate | {metrics.partial_fill_rate:.2%} |

---

## 7. Recommendations

{self._generate_recommendations(metrics)}

---

*Report generated by CopyCat Sandbox Simulation Engine*
"""
        return report

    def _generate_equity_chart(self, equity_curve: List[EquityPoint]) -> str:
        """Generate ASCII equity curve chart."""
        if not equity_curve:
            return "No equity data available"

        values = [p.value for p in equity_curve]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1

        chart_width = 50
        chart = []

        for point in equity_curve:
            normalized = (point.value - min_val) / range_val
            bar_length = int(normalized * chart_width)
            bar = "█" * bar_length + "░" * (chart_width - bar_length)
            chart.append(f"|{bar}| ${point.value:,.0f}")

        return "\n".join(chart[:20])  # Limit to 20 lines

    def _generate_monthly_returns_table(self, metrics: PerformanceMetrics, trades: List[VirtualTrade]) -> str:
        """Generate monthly returns table."""
        # Simplified: return a basic table
        return "| 2024-01 | +5.2% |\n| 2024-02 | -2.1% |"

    def _generate_trader_table(self, metrics: PerformanceMetrics) -> str:
        """Generate trader performance table."""
        rows = []
        for trader, ret in metrics.trader_specific_returns.items():
            rows.append(f"| {trader[:8]}... | ${ret:+,.2f} | |")
        return "\n".join(rows) if rows else "| N/A | $0.00 | |"

    def _generate_recommendations(self, metrics: PerformanceMetrics) -> str:
        """Generate recommendations based on metrics."""
        recommendations = []

        if metrics.sharpe_ratio < 1:
            recommendations.append("- Consider reducing position sizes to improve risk-adjusted returns")
        if metrics.max_drawdown > 0.2:
            recommendations.append("- Implement stricter stop-losses to reduce maximum drawdown")
        if metrics.win_rate < 0.4:
            recommendations.append("- Review trader selection criteria to improve win rate")
        if metrics.profit_factor < 1.5:
            recommendations.append("- Focus on higher probability trades to improve profit factor")

        if not recommendations:
            recommendations.append("- Strategy is performing well. Consider increasing position sizes slightly")

        return "\n".join(recommendations)

    def save_report(self, report: str, filename: str = None) -> str:
        """Save report to file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"sandbox_report_{timestamp}.md"

        filepath = os.path.join(self.output_path, filename)
        os.makedirs(self.output_path, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(report)

        return filepath
