"""
Smart Capital Allocation System.

Allocates capital to traders based on performance.

Usage:
    from orchestrator.config_allocation import AllocationConfig, CapitalAllocator
    
    config = AllocationConfig(
        enabled=True,
        allocation_strategy="proportional",  # or "tiered", "kelly"
        base_allocation=10.0,
        max_allocation=100.0,
    )
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Capital allocation strategies."""
    EQUAL = "equal"           # Equal allocation to all
    PROPORTIONAL = "proportional"  # Proportional to performance
    KELLY = "kelly"           # Kelly criterion based
    TIERED = "tiered"         # Tiered by performance
    MOMENTUM = "momentum"     # Based on recent performance
    RISK_ADJUSTED = "risk_adjusted"  # Risk-adjusted returns


@dataclass
class AllocationConfig:
    """Configuration for capital allocation."""
    enabled: bool = True
    
    # Strategy
    strategy: AllocationStrategy = AllocationStrategy.PROPORTIONAL
    
    # Base settings
    base_allocation: float = 10.0  # Base allocation per trader
    max_allocation: float = 100.0  # Maximum allocation per trader
    min_allocation: float = 5.0    # Minimum allocation per trader
    
    # Proportional settings
    total_portfolio: float = 1000.0  # Total portfolio to allocate
    
    # Kelly settings
    kelly_fraction: float = 0.25  # Fraction of Kelly to use
    kelly_cap: float = 0.15       # Max Kelly allocation
    
    # Performance weighting
    performance_window: int = 30  # Days to look back
    return_weight: float = 0.5    # Weight for returns
    consistency_weight: float = 0.3  # Weight for consistency
    volume_weight: float = 0.2    # Weight for trade volume
    
    # Rebalancing
    rebalance_interval_hours: int = 24
    rebalance_threshold_pct: float = 0.20  # Rebalance if allocation off by 20%


@dataclass
class TraderAllocation:
    """Allocation for a single trader."""
    address: str
    allocated_amount: float
    performance_score: float
    recent_return: float
    consistency: float
    trades_count: int
    allocation_pct: float  # Percentage of total portfolio


class CapitalAllocator:
    """
    Manages capital allocation across traders.
    
    Key behavior:
    - Equal allocation by default
    - Proportional to performance (more to better traders)
    - Kelly criterion for optimal allocation
    - Risk-adjusted for stability
    """
    
    def __init__(self, config: Optional[AllocationConfig] = None):
        self.config = config or AllocationConfig()
        self.current_allocations: Dict[str, TraderAllocation] = {}
        self.last_rebalance: Optional[datetime] = None
        logger.info(f"CapitalAllocator initialized (enabled={self.config.enabled})")
    
    def calculate_allocations(
        self,
        traders: Dict[str, Dict[str, Any]],
    ) -> Dict[str, TraderAllocation]:
        """
        Calculate allocation for each trader.
        
        Args:
            traders: Dict mapping address -> trader data with:
                - score: Overall trader score
                - recent_return: Recent return percentage
                - win_rate: Win rate
                - total_pnl: Total P&L
                - trades_count: Number of trades
        
        Returns:
            Dict mapping address -> TraderAllocation
        """
        if not self.config.enabled:
            return {}
        
        # Calculate performance scores
        trader_scores = self._calculate_performance_scores(traders)
        
        # Apply allocation strategy
        if self.config.strategy == AllocationStrategy.EQUAL:
            allocations = self._allocate_equal(traders, len(traders))
        elif self.config.strategy == AllocationStrategy.PROPORTIONAL:
            allocations = self._allocate_proportional(traders, trader_scores)
        elif self.config.strategy == AllocationStrategy.KELLY:
            allocations = self._allocate_kelly(traders, trader_scores)
        elif self.config.strategy == AllocationStrategy.TIERED:
            allocations = self._allocate_tiered(traders, trader_scores)
        elif self.config.strategy == AllocationStrategy.MOMENTUM:
            allocations = self._allocate_momentum(traders, trader_scores)
        elif self.config.strategy == AllocationStrategy.RISK_ADJUSTED:
            allocations = self._allocate_risk_adjusted(traders, trader_scores)
        else:
            allocations = self._allocate_equal(traders, len(traders))
        
        self.current_allocations = allocations
        self.last_rebalance = datetime.utcnow()
        
        # Log summary
        total_allocated = sum(a.allocated_amount for a in allocations.values())
        logger.info(
            f"Allocated ${total_allocated:.2f} across {len(allocations)} traders "
            f"({self.config.strategy.value} strategy)"
        )
        
        return allocations
    
    def _calculate_performance_scores(
        self,
        traders: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate performance scores for traders."""
        scores = {}
        
        for address, data in traders.items():
            recent_return = data.get("recent_return", data.get("return_pct", 0))
            win_rate = data.get("win_rate", 0.5)
            total_pnl = data.get("total_pnl", 0)
            trades_count = data.get("trades_count", 1)
            
            # Calculate consistency (variance of returns, inverted)
            returns = data.get("returns", [])
            if len(returns) > 1:
                import statistics
                consistency = 1 / (1 + statistics.stdev(returns)) if statistics.stdev(returns) > 0 else 1
            else:
                consistency = 0.5
            
            # Volume score (normalized)
            volume_score = min(trades_count / 20, 1.0)
            
            # Combined score
            score = (
                recent_return * self.config.return_weight +
                consistency * self.config.consistency_weight +
                volume_score * self.config.volume_weight
            )
            
            scores[address] = max(0, min(1, score))
        
        return scores
    
    def _allocate_equal(
        self,
        traders: Dict[str, Dict[str, Any]],
        trader_count: int,
    ) -> Dict[str, TraderAllocation]:
        """Equal allocation to all traders."""
        if trader_count == 0:
            return {}
        
        allocation_per_trader = min(
            self.config.total_portfolio / trader_count,
            self.config.max_allocation
        )
        
        allocations = {}
        for address in traders.keys():
            allocations[address] = TraderAllocation(
                address=address,
                allocated_amount=allocation_per_trader,
                performance_score=0.5,
                recent_return=0,
                consistency=0.5,
                trades_count=traders[address].get("trades_count", 0),
                allocation_pct=allocation_per_trader / self.config.total_portfolio,
            )
        
        return allocations
    
    def _allocate_proportional(
        self,
        traders: Dict[str, Dict[str, Any]],
        scores: Dict[str, float],
    ) -> Dict[str, TraderAllocation]:
        """Proportional allocation based on performance."""
        if not scores:
            return {}
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1
        min_score = min(scores.values()) if scores else 0
        
        if max_score == min_score:
            normalized = {k: 0.5 for k in scores}
        else:
            normalized = {
                k: (v - min_score) / (max_score - min_score)
                for k, v in scores.items()
            }
        
        # Calculate total allocation pool
        # Use square of normalized score to amplify differences
        score_squares = {k: v ** 2 for k, v in normalized.items()}
        total_score = sum(score_squares.values())
        
        if total_score == 0:
            return self._allocate_equal(traders, len(traders))
        
        allocations = {}
        for address in traders.keys():
            score = normalized.get(address, 0)
            score_sq = score ** 2
            
            # Proportional allocation
            allocation = (score_sq / total_score) * self.config.total_portfolio
            
            # Apply min/max
            allocation = max(self.config.min_allocation, min(allocation, self.config.max_allocation))
            
            allocations[address] = TraderAllocation(
                address=address,
                allocated_amount=allocation,
                performance_score=score,
                recent_return=traders[address].get("recent_return", 0),
                consistency=traders[address].get("consistency", 0.5),
                trades_count=traders[address].get("trades_count", 0),
                allocation_pct=allocation / self.config.total_portfolio,
            )
        
        return allocations
    
    def _allocate_kelly(
        self,
        traders: Dict[str, Dict[str, Any]],
        scores: Dict[str, float],
    ) -> Dict[str, TraderAllocation]:
        """Kelly criterion based allocation."""
        allocations = self._allocate_proportional(traders, scores)
        
        kelly_multiplier = self.config.kelly_fraction
        
        for address, allocation in allocations.items():
            # Kelly formula: (bp - q) / b
            # Simplified: use performance score as edge
            win_rate = traders[address].get("win_rate", 0.5)
            avg_win = traders[address].get("avg_win", 0.1)
            avg_loss = traders[address].get("avg_loss", 0.05)
            
            if avg_loss > 0:
                kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly = max(0, kelly * kelly_multiplier)
                kelly = min(kelly, self.config.kelly_cap)
                
                allocation.allocated_amount *= (1 + kelly)
                allocation.allocated_amount = min(
                    allocation.allocated_amount,
                    self.config.max_allocation
                )
        
        return allocations
    
    def _allocate_tiered(
        self,
        traders: Dict[str, Dict[str, Any]],
        scores: Dict[str, float],
    ) -> Dict[str, TraderAllocation]:
        """Tiered allocation based on performance tiers."""
        # Sort traders by score
        sorted_traders = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Assign to tiers
        tier1_count = max(1, len(traders) // 3)
        tier2_count = max(1, len(traders) * 2 // 3) - tier1_count
        
        tier1_alloc = 0.50  # Top tier gets 50%
        tier2_alloc = 0.35  # Mid tier gets 35%
        tier3_alloc = 0.15  # Bottom tier gets 15%
        
        allocations = {}
        tier1_per_trader = (tier1_alloc * self.config.total_portfolio) / tier1_count if tier1_count > 0 else 0
        tier2_per_trader = (tier2_alloc * self.config.total_portfolio) / tier2_count if tier2_count > 0 else 0
        tier3_per_trader = (tier3_alloc * self.config.total_portfolio) / max(1, len(traders) - tier1_count - tier2_count)
        
        tier1_per_trader = min(tier1_per_trader, self.config.max_allocation)
        tier2_per_trader = min(tier2_per_trader, self.config.max_allocation)
        tier3_per_trader = max(self.config.min_allocation, min(tier3_per_trader, self.config.max_allocation))
        
        for i, (address, score) in enumerate(sorted_traders):
            if i < tier1_count:
                allocation = tier1_per_trader
            elif i < tier1_count + tier2_count:
                allocation = tier2_per_trader
            else:
                allocation = tier3_per_trader
            
            allocations[address] = TraderAllocation(
                address=address,
                allocated_amount=allocation,
                performance_score=score,
                recent_return=traders[address].get("recent_return", 0),
                consistency=traders[address].get("consistency", 0.5),
                trades_count=traders[address].get("trades_count", 0),
                allocation_pct=allocation / self.config.total_portfolio,
            )
        
        return allocations
    
    def _allocate_momentum(
        self,
        traders: Dict[str, Dict[str, Any]],
        scores: Dict[str, float],
    ) -> Dict[str, TraderAllocation]:
        """Momentum-based allocation (favor recent winners)."""
        # Boost scores based on recent performance
        boosted_scores = {}
        for address, score in scores.items():
            recent_return = traders[address].get("recent_return", 0)
            # Exponential boost for positive recent returns
            boost = 1 + max(0, recent_return) * 2
            boosted_scores[address] = score * boost
        
        return self._allocate_proportional(traders, boosted_scores)
    
    def _allocate_risk_adjusted(
        self,
        traders: Dict[str, Dict[str, Any]],
        scores: Dict[str, float],
    ) -> Dict[str, TraderAllocation]:
        """Risk-adjusted allocation."""
        # Calculate risk-adjusted scores
        risk_adjusted = {}
        for address, score in scores.items():
            volatility = traders[address].get("volatility", 0.1)
            drawdown = traders[address].get("max_drawdown", 0.1)
            
            # Lower risk = higher allocation
            risk_penalty = 1 / (1 + volatility + drawdown * 2)
            
            risk_adjusted[address] = score * risk_penalty
        
        return self._allocate_proportional(traders, risk_adjusted)
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get allocation summary."""
        if not self.current_allocations:
            return {"status": "no allocations"}
        
        total = sum(a.allocated_amount for a in self.current_allocations.values())
        
        return {
            "trader_count": len(self.current_allocations),
            "total_allocated": total,
            "avg_allocation": total / len(self.current_allocations),
            "strategy": self.config.strategy.value,
            "by_trader": {
                addr: {
                    "amount": alloc.allocated_amount,
                    "pct": alloc.allocation_pct,
                    "score": alloc.performance_score,
                }
                for addr, alloc in self.current_allocations.items()
            },
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get allocator status."""
        return {
            "enabled": self.config.enabled,
            "strategy": self.config.strategy.value,
            "traders_allocated": len(self.current_allocations),
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
        }


def create_allocation_config(
    strategy: str = "proportional",
    total_portfolio: float = 1000.0,
    base_allocation: float = 10.0,
) -> AllocationConfig:
    """Factory function to create allocation config."""
    strategy_map = {
        "equal": AllocationStrategy.EQUAL,
        "proportional": AllocationStrategy.PROPORTIONAL,
        "kelly": AllocationStrategy.KELLY,
        "tiered": AllocationStrategy.TIERED,
        "momentum": AllocationStrategy.MOMENTUM,
        "risk_adjusted": AllocationStrategy.RISK_ADJUSTED,
    }
    
    return AllocationConfig(
        enabled=True,
        strategy=strategy_map.get(strategy, AllocationStrategy.PROPORTIONAL),
        total_portfolio=total_portfolio,
        base_allocation=base_allocation,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("SMART CAPITAL ALLOCATION DEMO")
    print("=" * 60)
    
    # Create allocator with proportional strategy
    allocator = CapitalAllocator(
        create_allocation_config(
            strategy="proportional",
            total_portfolio=1000.0,
        )
    )
    
    # Mock trader data with different performance
    mock_traders = {
        "0xAAA111": {"recent_return": 0.15, "win_rate": 0.70, "trades_count": 30, "returns": [0.1, 0.2, 0.15]},
        "0xBBB222": {"recent_return": 0.08, "win_rate": 0.60, "trades_count": 25, "returns": [0.05, 0.1, 0.08]},
        "0xCCC333": {"recent_return": 0.02, "win_rate": 0.52, "trades_count": 20, "returns": [0.02, 0.03, 0.01]},
        "0xDDD444": {"recent_return": -0.05, "win_rate": 0.40, "trades_count": 15, "returns": [-0.02, -0.05, -0.08]},
        "0xEEE555": {"recent_return": 0.25, "win_rate": 0.80, "trades_count": 40, "returns": [0.2, 0.3, 0.25]},
    }
    
    # Calculate allocations
    print("\nCalculating proportional allocations...")
    allocations = allocator.calculate_allocations(mock_traders)
    
    # Show results
    print(f"\n{'='*60}")
    print("CAPITAL ALLOCATION RESULTS")
    print(f"{'='*60}")
    print(f"\nStrategy: {allocator.config.strategy.value}")
    print(f"Total Portfolio: ${allocator.config.total_portfolio:.2f}")
    print(f"\n{'Address':<15} {'Score':>8} {'Allocation':>12} {'Pct':>8}")
    print("-" * 45)
    
    sorted_allocations = sorted(
        allocations.values(),
        key=lambda x: x.allocated_amount,
        reverse=True
    )
    
    for alloc in sorted_allocations:
        print(f"{alloc.address[:12]}... {alloc.performance_score:>7.2f} "
              f"${alloc.allocated_amount:>10.2f} {alloc.allocation_pct:>7.1%}")
    
    print("-" * 45)
    total = sum(a.allocated_amount for a in allocations.values())
    print(f"{'TOTAL':<15} {'':<8} ${total:>10.2f}")
    
    # Compare strategies
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print(f"{'='*60}")
    
    for strategy_name in ["equal", "proportional", "tiered", "momentum"]:
        strategy_allocator = CapitalAllocator(
            create_allocation_config(strategy=strategy_name)
        )
        strategy_allocator.calculate_allocations(mock_traders)
        summary = strategy_allocator.get_allocation_summary()
        
        amounts = [a.allocated_amount for a in summary.get("by_trader", {}).values()]
        if amounts:
            print(f"\n{strategy_name}:")
            print(f"  Min: ${min(amounts):.2f}, Max: ${max(amounts):.2f}, "
                  f"Range: {max(amounts)/min(amounts):.1f}x")
    
    print("\n" + "=" * 60)
    print("ALLOCATION STRATEGY BENEFITS:")
    print("  • EQUAL: Simple, diversified")
    print("  • PROPORTIONAL: More to better performers")
    print("  • TIERED: Top performers get most capital")
    print("  • MOMENTUM: Favor recent winners")
    print("  • KELLY: Mathematically optimal sizing")
    print("=" * 60)
