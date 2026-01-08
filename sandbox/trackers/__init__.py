"""
Performance Tracker - Calculates and tracks comprehensive performance metrics for sandbox simulation.
Calculates ROI, win rate, Sharpe ratio, drawdown, and other risk metrics.
"""

import math
from datetime import datetime
from typing import Dict, List, Optional
from sandbox.models import (
    PerformanceMetrics, VirtualTrade, EquityPoint, DailyReturn,
    PortfolioSummary, SimulationState
)


class PerformanceTracker:
    """Tracks and calculates performance metrics for sandbox simulation."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.trades: List[VirtualTrade] = []
        self.daily_returns: List[DailyReturn] = []
        self.equity_curve: List[EquityPoint] = []
        self.trader_returns: Dict[str, Dict] = {}

    def record_trade(self, trade: VirtualTrade):
        """Record a completed trade."""
        self.trades.append(trade)
        self._update_equity_curve()

        # Track per-trader performance
        if trade.source_trader_id:
            if trade.source_trader_id not in self.trader_returns:
                self.trader_returns[trade.source_trader_id] = {
                    'trades': [],
                    'total_pnl': 0.0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }
            self.trader_returns[trade.source_trader_id]['trades'].append(trade)
            self.trader_returns[trade.source_trader_id]['total_pnl'] += trade.profit
            if trade.profit > 0:
                self.trader_returns[trade.source_trader_id]['winning_trades'] += 1
            else:
                self.trader_returns[trade.source_trader_id]['losing_trades'] += 1

    def record_daily_return(self, return_pct: float, portfolio_value: float):
        """Record daily return."""
        self.daily_returns.append(DailyReturn(
            date=datetime.utcnow(),
            return_pct=return_pct,
            portfolio_value=portfolio_value
        ))

    def _update_equity_curve(self):
        """Update equity curve with current portfolio value."""
        current_value = self._calculate_current_value()
        self.equity_curve.append(EquityPoint(
            timestamp=datetime.utcnow(),
            value=current_value,
            return_pct=(current_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        ))

    def _calculate_current_value(self) -> float:
        """Calculate current portfolio value from trades."""
        realized_pnl = sum(trade.profit for trade in self.trades if not trade.is_open)
        return self.initial_balance + realized_pnl

    def calculate_metrics(self, state: SimulationState = None) -> PerformanceMetrics:
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
            trading_days = 1

        # Basic returns
        final_balance = state.total_value if state else self._calculate_current_value()
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

        # Trader copy performance
        traders_copied = len(self.trader_returns)
        profitable_traders = sum(1 for data in self.trader_returns.values() if data['total_pnl'] > 0)
        top_trader, worst_trader = self._get_best_worst_trader()

        # Execution quality
        avg_slippage = self._avg_slippage()
        total_fees = self._total_fees()
        fill_rate = self._calculate_fill_rate()
        partial_fill_rate = self._calculate_partial_fill_rate()

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
            traders_copied=traders_copied,
            profitable_traders=profitable_traders,
            top_performing_trader=top_trader,
            worst_performing_trader=worst_trader,
            trader_specific_returns={k: v['total_pnl'] for k, v in self.trader_returns.items()},
            avg_slippage=avg_slippage,
            total_fees_paid=total_fees,
            fill_rate=fill_rate,
            partial_fill_rate=partial_fill_rate
        )

    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate standard deviation of returns."""
        if len(returns) < 2:
            return 0.0

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    def _calculate_sharpe_ratio(self, returns: List[float], annualized_return: float) -> float:
        """Calculate Sharpe ratio."""
        volatility = self._calculate_volatility(returns)
        if volatility == 0:
            return 0.0
        # Assuming risk-free rate of 2%
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_return = annualized_return - risk_free_rate * len(returns) / 252
        return excess_return / volatility if volatility > 0 else 0

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside risk only)."""
        if len(returns) < 2:
            return 0.0

        mean = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float('inf')

        downside_variance = sum((r - mean) ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_variance)

        if downside_std == 0:
            return 0.0

        excess_return = sum(r - 0.02 / 252 for r in returns) / len(returns)
        return excess_return / downside_std

    def _calculate_max_drawdown(self) -> tuple:
        """Calculate maximum drawdown and duration."""
        if not self.equity_curve:
            return 0.0, 0

        max_drawdown = 0.0
        max_dd_start = None
        max_dd_duration = 0
        peak = self.equity_curve[0].value

        for point in self.equity_curve:
            if point.value > peak:
                peak = point.value

            drawdown = (peak - point.value) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_dd_start = point.timestamp

            if max_dd_start and drawdown == max_drawdown:
                max_dd_duration = (point.timestamp - max_dd_start).days

        return max_drawdown, max_dd_duration

    def _avg_position_size(self) -> float:
        """Calculate average position size."""
        if not self.trades:
            return 0.0
        return sum(t.quantity for t in self.trades) / len(self.trades)

    def _max_position_size(self) -> float:
        """Calculate maximum position size."""
        if not self.trades:
            return 0.0
        return max(t.quantity for t in self.trades)

    def _avg_hold_time(self) -> float:
        """Calculate average hold time in hours."""
        closed_trades = [t for t in self.trades if not t.is_open and t.hold_time_hours > 0]
        if not closed_trades:
            return 0.0
        return sum(t.hold_time_hours for t in closed_trades) / len(closed_trades)

    def _max_hold_time(self) -> float:
        """Calculate maximum hold time in hours."""
        closed_trades = [t for t in self.trades if not t.is_open and t.hold_time_hours > 0]
        if not closed_trades:
            return 0.0
        return max(t.hold_time_hours for t in closed_trades)

    def _min_hold_time(self) -> float:
        """Calculate minimum hold time in hours."""
        closed_trades = [t for t in self.trades if not t.is_open and t.hold_time_hours > 0]
        if not closed_trades:
            return 0.0
        return min(t.hold_time_hours for t in closed_trades)

    def _get_best_worst_trader(self) -> tuple:
        """Get best and worst performing traders."""
        if not self.trader_returns:
            return "None", "None"

        sorted_traders = sorted(
            self.trader_returns.items(),
            key=lambda x: x[1]['total_pnl'],
            reverse=True
        )

        best = sorted_traders[0][0] if sorted_traders else "None"
        worst = sorted_traders[-1][0] if sorted_traders else "None"

        return best, worst

    def _avg_slippage(self) -> float:
        """Calculate average slippage."""
        if not self.trades:
            return 0.0
        return sum(t.slippage for t in self.trades) / len(self.trades)

    def _total_fees(self) -> float:
        """Calculate total fees paid."""
        return sum(t.fees for t in self.trades)

    def _calculate_fill_rate(self) -> float:
        """Calculate order fill rate."""
        total_trades = len(self.trades)
        if total_trades == 0:
            return 0.0
        filled_trades = sum(1 for t in self.trades if t.profit != 0 or t.fees > 0)
        return filled_trades / total_trades

    def _calculate_partial_fill_rate(self) -> float:
        """Calculate partial fill rate."""
        # Simplified: assume no partial fills for now
        return 0.0

    def get_equity_curve(self) -> List[EquityPoint]:
        """Get equity curve data."""
        return self.equity_curve.copy()

    def get_trade_history(self) -> List[VirtualTrade]:
        """Get trade history."""
        return self.trades.copy()
