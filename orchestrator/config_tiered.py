"""
Tiered Copying System.

Allocates more capital to top-performing traders, less to others.

Usage:
    from orchestrator.config_tiered import TieredCopyConfig, TieredCopyEngine
    
    config = TieredCopyConfig(
        enabled=True,
        tier1_traders=3,           # Top 3 get more
        tier1_multiplier=3.0,      # 3x normal position
        tier2_traders=7,           # Next 7 get normal
        tier2_multiplier=1.0,
        tier3_traders=10,          # Rest get less
        tier3_multiplier=0.5,
        rebalance_interval_hours=24,  # Recalculate tiers daily
        min_trades_for_tier1=10,   # Minimum trades for top tier
    )
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.config import CopyTradingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TierLevel(Enum):
    """Trader tier levels."""
    TIER_1 = "tier1"  # Top performers
    TIER_2 = "tier2"  # Mid performers
    TIER_3 = "tier3"  # Lower performers
    EXCLUDED = "excluded"


@dataclass
class TieredCopyConfig:
    """Configuration for tiered copying system."""
    enabled: bool = True
    
    # Tier 1 (Top performers)
    tier1_traders: int = 3
    tier1_multiplier: float = 3.0
    min_trades_for_tier1: int = 10
    min_pnl_for_tier1: float = 50.0
    
    # Tier 2 (Mid performers)
    tier2_traders: int = 7
    tier2_multiplier: float = 1.0
    min_trades_for_tier2: int = 5
    min_pnl_for_tier2: float = 10.0
    
    # Tier 3 (Lower performers)
    tier3_multiplier: float = 0.5
    tier3_max_traders: int = 10
    
    # Rebalancing
    rebalance_interval_hours: int = 24
    min_tier_change_pct: float = 0.10  # Only move tier if 10% improvement
    
    # Rankings
    ranking_metric: str = "overall_score"  # overall_score, growth_score, consistency
    ranking_direction: str = "desc"  # desc (higher is better) or asc


@dataclass
class TraderTier:
    """A trader's tier assignment."""
    address: str
    tier: TierLevel
    score: float
    total_pnl: float
    win_rate: float
    last_updated: datetime = field(default_factory=datetime.utcnow)


class TieredCopyEngine:
    """
    Manages tiered capital allocation across traders.
    
    Key behavior:
    - Top 3 traders get 3x normal position size
    - Next 7 get normal position
    - Remaining get 0.5x normal position
    - Rebalances daily based on performance
    """
    
    def __init__(self, config: Optional[TieredCopyConfig] = None):
        self.config = config or TieredCopyConfig()
        self.trader_tiers: Dict[str, TraderTier] = {}
        self.last_rebalance = datetime.utcnow()
        self.base_config = CopyTradingConfig()
        logger.info(f"TieredCopyEngine initialized (enabled={self.config.enabled})")
    
    def calculate_position_size(
        self,
        trader_address: str,
        trader_score: float,
        total_pnl: float,
        win_rate: float,
        base_position_size: float,
    ) -> float:
        """Calculate position size based on tier."""
        if not self.config.enabled:
            return base_position_size
        
        # Get or create tier
        tier = self._get_or_create_tier(
            trader_address, trader_score, total_pnl, win_rate
        )
        
        # Get multiplier
        multiplier = self._get_multiplier(tier.tier)
        
        # Calculate position
        position = base_position_size * multiplier
        
        logger.debug(
            f"Tiered position: {trader_address[:10]}... | "
            f"Tier: {tier.tier.value} | "
            f"Multiplier: {multiplier}x | "
            f"Position: ${position:.2f}"
        )
        
        return position
    
    def _get_or_create_tier(
        self,
        address: str,
        score: float,
        pnl: float,
        win_rate: float,
    ) -> TraderTier:
        """Get existing tier or create new one."""
        if address in self.trader_tiers:
            return self.trader_tiers[address]
        
        tier = TraderTier(
            address=address,
            tier=TierLevel.TIER_3,  # Default to lowest
            score=score,
            total_pnl=pnl,
            win_rate=win_rate,
        )
        self.trader_tiers[address] = tier
        return tier
    
    def _get_multiplier(self, tier: TierLevel) -> float:
        """Get position multiplier for tier."""
        if tier == TierLevel.TIER_1:
            return self.config.tier1_multiplier
        elif tier == TierLevel.TIER_2:
            return self.config.tier2_multiplier
        elif tier == TierLevel.TIER_3:
            return self.config.tier3_multiplier
        else:
            return 0.0
    
    def rebalance_tiers(
        self,
        traders: Dict[str, Dict[str, Any]],
    ) -> Dict[str, TraderTier]:
        """
        Recalculate all trader tiers based on performance.
        
        Args:
            traders: Dict mapping address -> trader data with:
                - score: Overall score
                - total_pnl: Total profit/loss
                - win_rate: Win rate percentage
                - trades_count: Number of trades
        
        Returns:
            Updated trader tiers
        """
        if not self.config.enabled:
            return self.trader_tiers
        
        now = datetime.utcnow()
        
        # Check if rebalance needed
        if (now - self.last_rebalance).total_seconds() < self.config.rebalance_interval_hours * 3600:
            return self.trader_tiers
        
        # Sort traders by ranking metric
        sorted_traders = self._sort_traders(traders)
        
        # Assign tiers
        tier1_count = 0
        tier2_count = 0
        
        for i, (address, data) in enumerate(sorted_traders):
            score = data.get("score", 0)
            pnl = data.get("total_pnl", 0)
            win_rate = data.get("win_rate", 0)
            trades_count = data.get("trades_count", 0)
            
            # Determine tier
            if self._qualifies_for_tier1(pnl, win_rate, trades_count):
                tier = TierLevel.TIER_1
                tier1_count += 1
            elif tier1_count < self.config.tier1_traders:
                tier = TierLevel.TIER_1
                tier1_count += 1
            elif tier2_count < self.config.tier2_traders:
                tier = TierLevel.TIER_2
                tier2_count += 1
            elif len([t for t in self.trader_tiers.values() if t.tier == TierLevel.TIER_3]) < self.config.tier3_max_traders:
                tier = TierLevel.TIER_3
            else:
                tier = TierLevel.EXCLUDED
            
            # Update tier
            if address in self.trader_tiers:
                old_tier = self.trader_tiers[address].tier
                if old_tier != tier:
                    logger.info(
                        f"Tier change: {address[:10]}... | "
                        f"{old_tier.value} -> {tier.value}"
                    )
            
            self.trader_tiers[address] = TraderTier(
                address=address,
                tier=tier,
                score=score,
                total_pnl=pnl,
                win_rate=win_rate,
                last_updated=now,
            )
        
        self.last_rebalance = now
        
        # Log summary
        tier_counts = {
            "tier1": sum(1 for t in self.trader_tiers.values() if t.tier == TierLevel.TIER_1),
            "tier2": sum(1 for t in self.trader_tiers.values() if t.tier == TierLevel.TIER_2),
            "tier3": sum(1 for t in self.trader_tiers.values() if t.tier == TierLevel.TIER_3),
            "excluded": sum(1 for t in self.trader_tiers.values() if t.tier == TierLevel.EXCLUDED),
        }
        logger.info(f"Rebalance complete: {tier_counts}")
        
        return self.trader_tiers
    
    def _sort_traders(
        self,
        traders: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Sort traders by ranking metric."""
        sorted_list = []
        
        for address, data in traders.items():
            metric = self.config.ranking_metric
            value = data.get(metric, 0)
            
            if self.config.ranking_direction == "desc":
                sorted_list.append((address, data))
                sorted_list.sort(key=lambda x: x[1].get(metric, 0), reverse=True)
            else:
                sorted_list.append((address, data))
                sorted_list.sort(key=lambda x: x[1].get(metric, 0))
        
        return sorted_list
    
    def _qualifies_for_tier1(
        self,
        pnl: float,
        win_rate: float,
        trades_count: int,
    ) -> bool:
        """Check if trader qualifies for tier 1."""
        if trades_count < self.config.min_trades_for_tier1:
            return False
        if pnl < self.config.min_pnl_for_tier1:
            return False
        return True
    
    def get_tier_summary(self) -> Dict[str, Any]:
        """Get summary of current tier distribution."""
        if not self.config.enabled:
            return {"enabled": False}
        
        tier_counts = {
            "tier1": [],
            "tier2": [],
            "tier3": [],
            "excluded": [],
        }
        
        for address, tier in self.trader_tiers.items():
            key = tier.tier.value
            tier_counts[key].append({
                "address": address[:10] + "...",
                "score": tier.score,
                "pnl": tier.total_pnl,
            })
        
        return {
            "enabled": True,
            "traders_per_tier": {
                "tier1": len(tier_counts["tier1"]),
                "tier2": len(tier_counts["tier2"]),
                "tier3": len(tier_counts["tier3"]),
            },
            "last_rebalance": self.last_rebalance.isoformat(),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "enabled": self.config.enabled,
            "trader_count": len(self.trader_tiers),
            "last_rebalance": self.last_rebalance.isoformat(),
            "summary": self.get_tier_summary(),
        }


def create_tiered_config(
    tier1_multiplier: float = 3.0,
    tier2_multiplier: float = 1.0,
    tier3_multiplier: float = 0.5,
    tier1_count: int = 3,
    tier2_count: int = 7,
    rebalance_hours: int = 24,
) -> TieredCopyConfig:
    """Factory function to create tiered copy config."""
    return TieredCopyConfig(
        enabled=True,
        tier1_traders=tier1_count,
        tier1_multiplier=tier1_multiplier,
        tier2_traders=tier2_count,
        tier2_multiplier=tier2_multiplier,
        tier3_multiplier=tier3_multiplier,
        rebalance_interval_hours=rebalance_hours,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TIERED COPYING SYSTEM DEMO")
    print("=" * 60)
    
    # Create engine
    engine = TieredCopyEngine(
        create_tiered_config(
            tier1_multiplier=3.0,
            tier1_count=2,
            tier2_count=5,
        )
    )
    
    # Mock trader data
    mock_traders = {
        "0xAAA": {"score": 0.9, "total_pnl": 500, "win_rate": 0.70, "trades_count": 50},
        "0xBBB": {"score": 0.8, "total_pnl": 300, "win_rate": 0.65, "trades_count": 40},
        "0xCCC": {"score": 0.7, "total_pnl": 150, "win_rate": 0.55, "trades_count": 30},
        "0xDDD": {"score": 0.6, "total_pnl": 80, "win_rate": 0.50, "trades_count": 20},
        "0xEEE": {"score": 0.5, "total_pnl": 30, "win_rate": 0.45, "trades_count": 15},
        "0xFFF": {"score": 0.4, "total_pnl": 10, "win_rate": 0.40, "trades_count": 10},
        "0xGGG": {"score": 0.3, "total_pnl": -20, "win_rate": 0.35, "trades_count": 8},
    }
    
    # Rebalance tiers
    print("\nRebalancing tiers...")
    engine.rebalance_tiers(mock_traders)
    
    # Show summary
    summary = engine.get_tier_summary()
    print(f"\nTier Distribution:")
    for tier, count in summary["traders_per_tier"].items():
        print(f"  {tier}: {count} traders")
    
    # Calculate positions
    print(f"\nPosition Allocation (base $10):")
    base_position = 10.0
    
    for address, tier in engine.trader_tiers.items():
        if tier.tier != TierLevel.EXCLUDED:
            position = engine.calculate_position_size(
                address, tier.score, tier.total_pnl, tier.win_rate, base_position
            )
            multiplier = engine._get_multiplier(tier.tier)
            print(f"  {address[:10]}... | Tier {tier.tier.value} | "
                  f"Multiplier: {multiplier}x | Position: ${position:.2f}")
    
    print("\n" + "=" * 60)
    print("TIERED COPYING BENEFITS:")
    print("  • Top traders get 3x capital allocation")
    print("  • Rebalances daily based on performance")
    print("  • Underperformers gradually reduced")
    print("  • Winners compound faster")
    print("=" * 60)
