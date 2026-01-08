"""
Growth-First Trader Selection Engine.

Selects traders based on ACCOUNT GROWTH instead of win rate.
More robust against cherry-picking and metric manipulation.

Key metrics:
- Equity curve slope (real growth rate)
- Total P&L accumulated
- Drawdown control
- Growth consistency
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_clients.base import Trade, OrderSide


@dataclass
class GrowthMetrics:
    """Metrics based on account growth, not win rate."""
    
    # Growth metrics
    total_pnl: float  # Total profit/loss
    growth_rate: float  # % growth per trade on average
    avg_daily_growth: float  # Expected daily growth %
    
    # Consistency
    equity_curve_slope: float  # Trend line slope (positive = good)
    growth_stddev: float  # Lower = more consistent
    positive_days_pct: float  # % of profitable periods
    
    # Risk control
    max_drawdown: float  # Maximum peak-to-trough drop
    recovery_time_hours: float  # Avg time to recover from loss
    
    # Activity
    total_trades: int
    active_days: int
    
    # Scores (0-1)
    growth_score: float  # Based on total growth
    consistency_score: float  # Based on low variance
    stability_score: float  # Based on drawdown control
    overall_score: float  # Weighted combination


class GrowthBasedSelector:
    """
    Selects traders based on ACCOUNT GROWTH, not win rate.
    
    Philosophy: Win rate can be gamed. Growth cannot.
    A trader who grew $100 to $1000 over 100 trades is good,
    regardless of whether they won 50% or 80%.
    """
    
    def __init__(
        self,
        min_total_pnl: float = 100.0,  # At least $100 profit
        min_growth_rate: float = 0.02,  # At least 2% per trade avg
        max_drawdown: float = 0.40,  # Allow up to 40% drawdown
        min_equity_slope: float = 0.001,  # Positive trend required
        min_consistency_score: float = 0.3,  # Reasonable consistency
        min_active_days: int = 5,  # At least 5 days of activity
    ):
        """
        Args:
            min_total_pnl: Minimum total profit required
            min_growth_rate: Minimum average growth per trade
            max_drawdown: Maximum allowed drawdown
            min_equity_slope: Minimum positive trend slope
            min_consistency_score: Minimum consistency score
            min_active_days: Minimum trading days
        """
        self.min_total_pnl = min_total_pnl
        self.min_growth_rate = min_growth_rate
        self.max_drawdown = max_drawdown
        self.min_equity_slope = min_equity_slope
        self.min_consistency_score = min_consistency_score
        self.min_active_days = min_active_days
    
    def calculate_growth_metrics(self, trades: List[Trade]) -> GrowthMetrics:
        """
        Calculate growth metrics from trade history.
        
        Builds a virtual equity curve and analyzes it.
        """
        if not trades or len(trades) < 2:
            return GrowthMetrics(
                total_pnl=0,
                growth_rate=0,
                avg_daily_growth=0,
                equity_curve_slope=0,
                growth_stddev=0,
                positive_days_pct=0,
                max_drawdown=1.0,
                recovery_time_hours=0,
                total_trades=len(trades),
                active_days=0,
                growth_score=0,
                consistency_score=0,
                stability_score=0,
                overall_score=0,
            )
        
        # Sort by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        # Build equity curve
        equity = 100.0  # Start with $100 baseline
        equity_curve = [equity]
        trade_results = []
        
        for trade in sorted_trades:
            # Calculate trade P&L
            if trade.side == OrderSide.BUY:
                # Long position - profit if price went up
                # Simplified: use total_value as proxy
                trade_pnl = trade.total_value * (trade.price - 0.5) * 2 if trade.total_value > 0 else 0
            else:
                # Short position
                trade_pnl = trade.total_value * (0.5 - trade.price) * 2 if trade.total_value > 0 else 0
            
            equity += trade_pnl
            equity_curve.append(equity)
            trade_results.append({
                'timestamp': trade.timestamp,
                'pnl': trade_pnl,
                'equity': equity,
            })
        
        # Total P&L
        total_pnl = equity - 100.0
        
        # Growth rate per trade
        growth_rate = (equity / 100.0) ** (1.0 / max(len(trades), 1)) - 1.0
        
        # Equity curve slope (linear regression)
        n = len(equity_curve)
        if n > 1:
            x = list(range(n))
            x_mean = (n - 1) / 2
            y_mean = sum(equity_curve) / n
            
            numerator = sum((x[i] - x_mean) * (equity_curve[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            equity_curve_slope = numerator / denominator if denominator > 0 else 0
        else:
            equity_curve_slope = 0
        
        # Growth consistency (stddev of daily returns)
        if len(trade_results) > 1:
            daily_returns = [
                (trade_results[i]['equity'] - trade_results[i-1]['equity']) / trade_results[i-1]['equity']
                for i in range(1, len(trade_results))
                if trade_results[i-1]['equity'] > 0
            ]
            growth_stddev = sum(abs(r) for r in daily_returns) / len(daily_returns)
            
            positive_days = sum(1 for r in daily_returns if r > 0)
            positive_days_pct = positive_days / len(daily_returns) if daily_returns else 0
        else:
            growth_stddev = 0
            positive_days_pct = 0
        
        # Max drawdown
        max_equity = 100.0
        max_drawdown = 0.0
        
        for equity_val in equity_curve:
            if equity_val > max_equity:
                max_equity = equity_val
            drawdown = (max_equity - equity_val) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Active days
        if sorted_trades:
            first_trade = sorted_trades[0].timestamp
            last_trade = sorted_trades[-1].timestamp
            active_days = (last_trade - first_trade).days + 1
        else:
            active_days = 0
        
        # Recovery time (avg time between loss and profit)
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for result in trade_results:
            if result['pnl'] < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = result['timestamp']
            elif result['pnl'] > 0 and in_drawdown:
                in_drawdown = False
                recovery_time = (result['timestamp'] - drawdown_start).total_seconds() / 3600
                if recovery_time > 0:
                    recovery_times.append(recovery_time)
        
        avg_recovery_hours = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        # Calculate scores (0-1)
        
        # Growth score: based on total profit
        growth_score = min(total_pnl / 1000.0, 1.0)  # $1000 profit = max score
        
        # Consistency score: inverse of stddev, normalized
        consistency_score = max(0, 1.0 - growth_stddev * 10)  # Lower variance = higher score
        
        # Stability score: based on drawdown control
        stability_score = max(0, 1.0 - max_drawdown)
        
        # Overall score: weighted combination
        overall_score = (
            growth_score * 0.5 +      # 50% based on actual growth
            consistency_score * 0.25 + # 25% on consistency
            stability_score * 0.25     # 25% on drawdown control
        )
        
        # Daily growth rate
        if active_days > 0:
            daily_growth = ((equity / 100.0) ** (1.0 / max(active_days, 1)) - 1.0)
        else:
            daily_growth = 0
        
        return GrowthMetrics(
            total_pnl=total_pnl,
            growth_rate=growth_rate,
            avg_daily_growth=daily_growth,
            equity_curve_slope=equity_curve_slope,
            growth_stddev=growth_stddev,
            positive_days_pct=positive_days_pct,
            max_drawdown=max_drawdown,
            recovery_time_hours=avg_recovery_hours,
            total_trades=len(trades),
            active_days=active_days,
            growth_score=growth_score,
            consistency_score=consistency_score,
            stability_score=stability_score,
            overall_score=overall_score,
        )
    
    def select_traders(
        self,
        traders_data: Dict[str, List[Trade]],
        top_n: int = 20,
    ) -> List[Tuple[str, GrowthMetrics]]:
        """
        Select top traders based on growth metrics.
        
        Args:
            traders_data: Dict mapping address -> list of trades
            top_n: Number of traders to return
        
        Returns:
            List of (address, GrowthMetrics) tuples, sorted by overall_score
        """
        results = []
        
        for address, trades in traders_data.items():
            metrics = self.calculate_growth_metrics(trades)
            
            # Apply filters
            if metrics.total_pnl < self.min_total_pnl:
                continue
            if metrics.max_drawdown > self.max_drawdown:
                continue
            if metrics.equity_curve_slope < self.min_equity_slope:
                continue
            if metrics.consistency_score < self.min_consistency_score:
                continue
            if metrics.active_days < self.min_active_days:
                continue
            
            results.append((address, metrics))
        
        # Sort by overall score
        results.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return results[:top_n]
    
    def analyze_trader_growth(
        self,
        address: str,
        trades: List[Trade],
    ) -> Tuple[GrowthMetrics, bool]:
        """
        Analyze a single trader's growth profile.
        
        Returns:
            Tuple of (metrics, is_suitable)
        """
        metrics = self.calculate_growth_metrics(trades)
        
        is_suitable = (
            metrics.total_pnl >= self.min_total_pnl and
            metrics.max_drawdown <= self.max_drawdown and
            metrics.equity_curve_slope >= self.min_equity_slope and
            metrics.overall_score >= 0.3  # Minimum overall score
        )
        
        return metrics, is_suitable


class GrowthSelectionConfig:
    """Configuration for growth-based selection."""
    
    def __init__(
        self,
        min_total_pnl: float = 100.0,
        min_growth_rate: float = 0.01,
        max_drawdown: float = 0.50,
        min_equity_slope: float = 0.0005,
        min_consistency_score: float = 0.25,
        min_active_days: int = 3,
        growth_weight: float = 0.50,
        consistency_weight: float = 0.25,
        stability_weight: float = 0.25,
    ):
        self.min_total_pnl = min_total_pnl
        self.min_growth_rate = min_growth_rate
        self.max_drawdown = max_drawdown
        self.min_equity_slope = min_equity_slope
        self.min_consistency_score = min_consistency_score
        self.min_active_days = min_active_days
        self.growth_weight = growth_weight
        self.consistency_weight = consistency_weight
        self.stability_weight = stability_weight


def create_growth_selector(config: Optional[GrowthSelectionConfig] = None) -> GrowthBasedSelector:
    """Factory function to create growth selector."""
    if config is None:
        config = GrowthSelectionConfig()
    
    return GrowthBasedSelector(
        min_total_pnl=config.min_total_pnl,
        min_growth_rate=config.min_growth_rate,
        max_drawdown=config.max_drawdown,
        min_equity_slope=config.min_equity_slope,
        min_consistency_score=config.min_consistency_score,
        min_active_days=config.min_active_days,
    )


if __name__ == "__main__":
    # Demo: Show how growth-based selection works
    print("=" * 60)
    print("GROWTH-FIRST TRADER SELECTION")
    print("=" * 60)
    print("\nPhilosophy: Win rate can be gamed. Growth cannot.")
    print("\nMetrics analyzed:")
    print("  - Total P&L (actual money made)")
    print("  - Equity curve slope (upward trend)")
    print("  - Drawdown control (risk management)")
    print("  - Growth consistency (low variance)")
    print("\nSettings:")
    
    selector = create_growth_selector()
    print(f"  Min Total P&L: ${selector.min_total_pnl}")
    print(f"  Max Drawdown: {selector.max_drawdown:.0%}")
    print(f"  Min Consistency: {selector.min_consistency_score:.0%}")
    print(f"  Min Active Days: {selector.min_active_days}")
    
    print("\n" + "=" * 60)
    print("Example: A trader with these results would qualify:")
    print("=" * 60)
    print("  Total Trades: 50")
    print("  Total P&L: $850 (+850%)")
    print("  Win Rate: 45% (low, but who cares?)")
    print("  Max Drawdown: 15%")
    print("  Growth Score: 0.85")
    print("  Overall Score: 0.78 ✓ QUALIFIES")
