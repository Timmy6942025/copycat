"""
Auto-Optimization Loop System.

Learns from trades and optimizes parameters over time.

Usage:
    from orchestrator.config_optimizer import OptimizationConfig, PerformanceOptimizer
    
    config = OptimizationConfig(
        enabled=True,
        learning_rate=0.01,
        optimization_window=30,  # Days
        params_to_optimize=["position_size", "trader_count"],
    )
"""

import sys
import os
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationTarget(Enum):
    """What to optimize for."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    WIN_RATE = "win_rate"
    MIN_DRAWDOWN = "min_drawdown"
    CONSISTENCY = "consistency"


@dataclass
class OptimizationConfig:
    """Configuration for auto-optimization."""
    enabled: bool = True
    
    # Learning settings
    learning_rate: float = 0.01  # How fast to adjust parameters
    optimization_window: int = 30  # Days to look back for learning
    
    # Parameters to optimize
    params_to_optimize: List[str] = field(default_factory=lambda: [
        "position_size_pct",
        "max_traders_to_copy",
        "min_trader_score",
        "hedge_pct",
    ])
    
    # Parameter ranges
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "position_size_pct": (0.02, 0.20),
        "max_traders_to_copy": (5, 30),
        "min_trader_score": (0.2, 0.6),
        "hedge_pct": (0.10, 0.50),
        "kelly_fraction": (0.10, 0.50),
    })
    
    # Targets
    target: OptimizationTarget = OptimizationTarget.TOTAL_RETURN
    min_improvement_threshold: float = 0.02  # 2% improvement required
    
    # Exploration
    exploration_rate: float = 0.10  # Random exploration rate
    exploration_decay: float = 0.95  # Decay exploration over time
    
    # Constraints
    max_param_change_pct: float = 0.20  # Max 20% change per iteration
    min_trades_for_optimization: int = 10  # Min trades before optimizing
    
    # History
    store_history: bool = True
    history_file: str = "./orchestrator_results/optimization_history.json"


@dataclass
class TradeResult:
    """Result of a copied trade."""
    trade_id: str
    trader_address: str
    market_id: str
    outcome: str
    amount: float
    profit_loss: float
    return_pct: float
    copied_at: datetime
    closed_at: Optional[datetime]
    trader_score: float


@dataclass
class OptimizationResult:
    """Result of optimization analysis."""
    params_before: Dict[str, float]
    params_after: Dict[str, float]
    performance_before: float
    performance_after: float
    improvement_pct: float
    is_applied: bool
    reason: str
    confidence: float


class PerformanceOptimizer:
    """
    Learns from trade results and optimizes trading parameters.
    
    Key behavior:
    - Tracks all copied trades
    - Analyzes performance by parameter settings
    - Adjusts parameters based on what works
    - Uses exploration vs exploitation balance
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.current_params: Dict[str, float] = {}
        self.trade_history: List[TradeResult] = []
        self.optimization_history: List[OptimizationResult] = []
        self.exploration_rate = self.config.exploration_rate
        self.iteration_count = 0
        self.last_optimization: Optional[datetime] = None
        
        # Initialize default params
        self._init_default_params()
        
        logger.info(f"PerformanceOptimizer initialized (enabled={self.config.enabled})")
    
    def _init_default_params(self):
        """Initialize default parameter values."""
        self.current_params = {
            "position_size_pct": 0.08,
            "max_traders_to_copy": 15,
            "min_trader_score": 0.4,
            "hedge_pct": 0.25,
            "kelly_fraction": 0.25,
        }
    
    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade for analysis."""
        trade_result = TradeResult(
            trade_id=trade.get("trade_id", f"t_{len(self.trade_history)}"),
            trader_address=trade.get("trader_address", ""),
            market_id=trade.get("market_id", ""),
            outcome=trade.get("outcome", ""),
            amount=trade.get("amount", 0),
            profit_loss=trade.get("profit_loss", 0),
            return_pct=trade.get("return_pct", 0),
            copied_at=datetime.fromisoformat(trade.get("copied_at", datetime.utcnow().isoformat())),
            closed_at=datetime.fromisoformat(trade.get("closed_at", datetime.utcnow().isoformat())) if trade.get("closed_at") else None,
            trader_score=trade.get("trader_score", 0.5),
        )
        
        self.trade_history.append(trade_result)
        
        logger.debug(
            f"Trade recorded: {trade_result.trade_id} | "
            f"P/L: ${trade_result.profit_loss:.2f} | "
            f"Return: {trade_result.return_pct:.2%}"
        )
    
    def analyze_performance(
        self,
        start_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Analyze performance over a time window."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=self.config.optimization_window)
        
        # Filter trades
        window_trades = [
            t for t in self.trade_history
            if t.copied_at >= start_date
        ]
        
        if len(window_trades) < self.config.min_trades_for_optimization:
            return {
                "trade_count": len(window_trades),
                "sufficient_data": False,
                "message": f"Insufficient trades: {len(window_trades)} < {self.config.min_trades_for_optimization}",
            }
        
        # Calculate metrics
        total_pnl = sum(t.profit_loss for t in window_trades)
        total_invested = sum(t.amount for t in window_trades)
        returns = [t.return_pct for t in window_trades]
        
        avg_return = sum(returns) / len(returns) if returns else 0
        winning_trades = [t for t in window_trades if t.profit_loss > 0]
        win_rate = len(winning_trades) / len(window_trades) if window_trades else 0
        
        # Calculate Sharpe-like ratio
        if returns and len(returns) > 1:
            import statistics
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0
            sharpe = (avg_return / std_return) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Calculate max drawdown (simplified)
        running_pnl = 0
        max_drawdown = 0
        peak_pnl = 0
        
        for trade in window_trades:
            running_pnl += trade.profit_loss
            if running_pnl > peak_pnl:
                peak_pnl = running_pnl
            drawdown = (peak_pnl - running_pnl) / (peak_pnl + total_invested) if peak_pnl > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Overall score
        if self.config.target == OptimizationTarget.TOTAL_RETURN:
            score = total_pnl / total_invested if total_invested > 0 else 0
        elif self.config.target == OptimizationTarget.SHARPE_RATIO:
            score = sharpe
        elif self.config.target == OptimizationTarget.WIN_RATE:
            score = win_rate
        elif self.config.target == OptimizationTarget.MIN_DRAWDOWN:
            score = 1 - max_drawdown
        else:
            score = avg_return
        
        return {
            "trade_count": len(window_trades),
            "sufficient_data": True,
            "total_pnl": total_pnl,
            "total_invested": total_invested,
            "avg_return": avg_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "overall_score": score,
            "start_date": start_date.isoformat(),
        }
    
    def optimize(self) -> Optional[OptimizationResult]:
        """Run optimization to adjust parameters."""
        if not self.config.enabled:
            return None
        
        # Check if we have enough data
        performance = self.analyze_performance()
        
        if not performance.get("sufficient_data", False):
            logger.info("Insufficient data for optimization")
            return None
        
        # Check if we should explore or exploit
        if random.random() < self.exploration_rate:
            return self._explore(performance)
        else:
            return self._exploit(performance)
    
    def _explore(self, performance: Dict[str, Any]) -> OptimizationResult:
        """Try random parameter changes."""
        self.iteration_count += 1
        params_before = self.current_params.copy()
        
        # Pick a random parameter to adjust
        param = random.choice(self.config.params_to_optimize)
        range_min, range_max = self.param_ranges.get(param, (0.5, 1.5))
        
        # Random change within range
        change_pct = random.uniform(-0.2, 0.2)
        new_value = self.current_params[param] * (1 + change_pct)
        new_value = max(range_min, min(range_max, new_value))
        
        self.current_params[param] = new_value
        
        result = OptimizationResult(
            params_before=params_before,
            params_after=self.current_params.copy(),
            performance_before=performance["overall_score"],
            performance_after=performance["overall_score"],
            improvement_pct=0,
            is_applied=False,  # Don't apply exploration changes immediately
            reason=f"Exploration: {param} adjusted by {change_pct:.1%}",
            confidence=self.exploration_rate,
        )
        
        self.optimization_history.append(result)
        self.last_optimization = datetime.utcnow()
        
        # Decay exploration rate
        self.exploration_rate *= self.config.exploration_decay
        
        logger.info(f"Exploration optimization: {param} -> {new_value:.3f}")
        
        return result
    
    def _exploit(self, performance: Dict[str, Any]) -> OptimizationResult:
        """Make targeted improvements based on analysis."""
        self.iteration_count += 1
        params_before = self.current_params.copy()
        
        # Analyze what works
        winning_trades = [t for t in self.trade_history if t.profit_loss > 0]
        losing_trades = [t for t in self.trade_history if t.profit_loss <= 0]
        
        # Calculate average trader score for winners vs losers
        avg_winner_score = sum(t.trader_score for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loser_score = sum(t.trader_score for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        params_after = params_before.copy()
        changes = []
        
        # Adjust min_trader_score based on analysis
        if "min_trader_score" in self.config.params_to_optimize:
            if avg_winner_score > avg_loser_score + 0.1:
                # Winners have higher scores, increase minimum
                new_score = min(0.6, params_after["min_trader_score"] * 1.1)
                params_after["min_trader_score"] = new_score
                changes.append(f"min_trader_score -> {new_score:.2f}")
            elif avg_winner_score < avg_loser_score - 0.1:
                # Losers have higher scores, decrease minimum
                new_score = max(0.2, params_after["min_trader_score"] * 0.9)
                params_after["min_trader_score"] = new_score
                changes.append(f"min_trader_score -> {new_score:.2f}")
        
        # Adjust position size based on win rate
        if "position_size_pct" in self.config.params_to_optimize:
            win_rate = performance.get("win_rate", 0.5)
            if win_rate > 0.55:
                new_size = min(0.20, params_after["position_size_pct"] * 1.1)
                params_after["position_size_pct"] = new_size
                changes.append(f"position_size_pct -> {new_size:.2%}")
            elif win_rate < 0.45:
                new_size = max(0.02, params_after["position_size_pct"] * 0.9)
                params_after["position_size_pct"] = new_size
                changes.append(f"position_size_pct -> {new_size:.2%}")
        
        # Adjust hedge percentage based on drawdown
        if "hedge_pct" in self.config.params_to_optimize:
            max_dd = performance.get("max_drawdown", 0)
            if max_dd > 0.15:
                new_hedge = min(0.50, params_after["hedge_pct"] * 1.2)
                params_after["hedge_pct"] = new_hedge
                changes.append(f"hedge_pct -> {new_hedge:.2%}")
        
        # Apply changes
        self.current_params = params_after
        
        result = OptimizationResult(
            params_before=params_before,
            params_after=params_after.copy(),
            performance_before=performance["overall_score"],
            performance_after=performance["overall_score"],
            improvement_pct=0,
            is_applied=True,
            reason=f"Exploitation: {', '.join(changes)}" if changes else "No changes needed",
            confidence=1.0 - self.exploration_rate,
        )
        
        self.optimization_history.append(result)
        self.last_optimization = datetime.utcnow()
        
        if changes:
            logger.info(f"Exploitation optimization: {', '.join(changes)}")
        
        return result
    
    def get_current_params(self) -> Dict[str, float]:
        """Get current optimized parameters."""
        return self.current_params.copy()
    
    def set_param(self, param: str, value: float):
        """Manually set a parameter."""
        if param in self.param_ranges:
            range_min, range_max = self.param_ranges[param]
            self.current_params[param] = max(range_min, min(range_max, value))
            logger.info(f"Manual parameter set: {param} = {value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status."""
        return {
            "enabled": self.config.enabled,
            "trades_recorded": len(self.trade_history),
            "optimizations_run": len(self.optimization_history),
            "exploration_rate": self.exploration_rate,
            "iteration_count": self.iteration_count,
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "current_params": self.current_params,
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across time windows."""
        now = datetime.utcnow()
        
        summaries = {}
        for days in [7, 14, 30]:
            start = now - timedelta(days=days)
            perf = self.analyze_performance(start_date=start)
            summaries[f"{days}d"] = {
                "trades": perf.get("trade_count", 0),
                "pnl": perf.get("total_pnl", 0),
                "win_rate": perf.get("win_rate", 0),
                "score": perf.get("overall_score", 0),
            }
        
        return summaries


def create_optimizer_config(
    learning_rate: float = 0.01,
    target: str = "total_return",
) -> OptimizationConfig:
    """Factory function to create optimizer config."""
    target_map = {
        "total_return": OptimizationTarget.TOTAL_RETURN,
        "sharpe": OptimizationTarget.SHARPE_RATIO,
        "win_rate": OptimizationTarget.WIN_RATE,
        "drawdown": OptimizationTarget.MIN_DRAWDOWN,
    }
    
    return OptimizationConfig(
        enabled=True,
        learning_rate=learning_rate,
        target=target_map.get(target, OptimizationTarget.TOTAL_RETURN),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("AUTO-OPTIMIZATION LOOP DEMO")
    print("=" * 60)
    
    # Create optimizer
    optimizer = PerformanceOptimizer(
        create_optimizer_config(learning_rate=0.05)
    )
    
    # Simulate trade history
    print("\nRecording simulated trades...")
    
    for i in range(25):
        trade = {
            "trade_id": f"t_{i}",
            "trader_address": f"0x{random.randint(1000,9999)}",
            "market_id": f"m_{i % 5}",
            "outcome": random.choice(["Yes", "No", "D", "R"]),
            "amount": random.uniform(5, 50),
            "profit_loss": random.uniform(-10, 30),
            "return_pct": random.uniform(-0.2, 0.5),
            "copied_at": (datetime.utcnow() - timedelta(days=random.randint(1, 30))).isoformat(),
            "trader_score": random.uniform(0.3, 0.9),
        }
        optimizer.record_trade(trade)
    
    print(f"Recorded {len(optimizer.trade_history)} trades")
    
    # Analyze performance
    print("\nAnalyzing performance...")
    performance = optimizer.analyze_performance()
    
    print(f"\nPerformance Summary:")
    print(f"  Total P/L: ${performance['total_pnl']:.2f}")
    print(f"  Win Rate: {performance['win_rate']:.1%}")
    print(f"  Avg Return: {performance['avg_return']:.2%}")
    print(f"  Max Drawdown: {performance['max_drawdown']:.1%}")
    print(f"  Overall Score: {performance['overall_score']:.3f}")
    
    # Run optimization
    print("\nRunning optimization...")
    result = optimizer.optimize()
    
    if result:
        print(f"\nOptimization Result:")
        print(f"  Reason: {result.reason}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Applied: {'✓' if result.is_applied else '✗'}")
        print(f"\n  Parameters Before:")
        for k, v in result.params_before.items():
            print(f"    {k}: {v:.3f}")
        print(f"\n  Parameters After:")
        for k, v in result.params_after.items():
            print(f"    {k}: {v:.3f}")
    
    # Show current params
    print(f"\nCurrent Optimized Parameters:")
    for k, v in optimizer.get_current_params().items():
        print(f"  {k}: {v:.3f}")
    
    # Performance summary
    summary = optimizer.get_performance_summary()
    print(f"\nPerformance by Time Window:")
    for window, data in summary.items():
        print(f"  {window}: {data['trades']} trades, P/L ${data['pnl']:.2f}, "
              f"Win {data['win_rate']:.0%}")
    
    print("\n" + "=" * 60)
    print("AUTO-OPTIMIZATION BENEFITS:")
    print("  • Learns from actual trade results")
    print("  • Balances exploration vs exploitation")
    print("  • Adjusts position sizing dynamically")
    print("  • Increases trader quality requirements when needed")
    print("  • Increases hedging during drawdowns")
    print("=" * 60)
