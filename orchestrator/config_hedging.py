"""
Cross-Market Hedging System.

Automatically takes opposite positions to reduce risk.

Usage:
    from orchestrator.config_hedging import HedgingConfig, HedgingEngine
    
    config = HedgingConfig(
        enabled=True,
        hedge_pairs={
            "D": "R",           # Democrat vs Republican
            "Yes": "No",        # Binary outcome pairs
            "over": "under",    # Over/under pairs
        },
        hedge_pct=0.25,         # Hedge 25% of position
        max_hedge_cost=0.01,    # Max 1% of portfolio per hedge
    )
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HedgeType(Enum):
    """Types of hedging relationships."""
    BINARY = "binary"           # Yes/No pairs
    POLITICAL = "political"     # D/R pairs
    OVERUNDER = "overunder"     # Over/Under pairs
    CORRELATED = "correlated"   # Related markets
    NONE = "none"


@dataclass
class HedgeConfig:
    """Configuration for a specific hedge pair."""
    hedge_type: HedgeType = HedgeType.BINARY
    hedge_pct: float = 0.25      # Percentage of position to hedge
    max_hedge_pct: float = 0.02  # Max percentage of portfolio for hedge
    min_correlation: float = 0.8 # Minimum correlation for correlated hedges


@dataclass
class HedgingConfig:
    """Configuration for cross-market hedging."""
    enabled: bool = True
    
    # Hedge pairs (outcome -> opposite outcome)
    hedge_pairs: Dict[str, str] = field(default_factory=lambda: {
        "D": "R",
        "R": "D",
        "Yes": "No",
        "No": "Yes",
        "Over": "Under",
        "Under": "Over",
        "True": "False",
        "False": "True",
        "Biden": "Trump",
        "Trump": "Biden",
        "Harris": "DeSantis",
        "DeSantis": "Harris",
    })
    
    # Hedge relationships by market
    market_hedges: Dict[str, Dict[str, HedgeConfig]] = field(default_factory=lambda: {
        "politics": {
            "democrat_vs_republican": HedgeConfig(
                hedge_type=HedgeType.POLITICAL,
                hedge_pct=0.25,
                max_hedge_pct=0.02,
            ),
        },
        "sports": {
            "team1_vs_team2": HedgeConfig(
                hedge_type=HedgeType.BINARY,
                hedge_pct=0.20,
                max_hedge_pct=0.02,
            ),
        },
        "binary": {
            "yes_no": HedgeConfig(
                hedge_type=HedgeType.BINARY,
                hedge_pct=0.15,
                max_hedge_pct=0.01,
            ),
        },
    })
    
    # General settings
    auto_hedge_binary: bool = True  # Auto-hedge binary markets
    auto_hedge_correlated: bool = False  # Auto-hedge correlated markets
    max_total_hedge_pct: float = 0.10  # Max 10% of portfolio in hedges
    min_position_for_hedge: float = 5.0  # Min $5 position to consider hedging
    
    # Risk settings
    max_drawdown_before_hedging: float = 0.15  # Increase hedging at 15% drawdown
    hedge_aggression: float = 1.0  # Multiplier for hedge size (1.0 = normal, 1.5 = more aggressive)


@dataclass
class HedgeResult:
    """Result of hedge analysis for a trade."""
    original_trade: Dict[str, Any]
    hedge_market_id: Optional[str]
    hedge_outcome: Optional[str]
    hedge_amount: float
    hedge_type: HedgeType
    is_recommended: bool
    reason: str
    net_exposure_after_hedge: float


class HedgingEngine:
    """
    Manages cross-market hedging to reduce risk.
    
    Key behavior:
    - Identifies hedge opportunities
    - Calculates optimal hedge sizes
    - Manages overall hedge exposure
    - Increases hedging during drawdowns
    """
    
    def __init__(self, config: Optional[HedgingConfig] = None):
        self.config = config or HedgingConfig()
        self.active_hedges: List[Dict[str, Any]] = []
        self.hedge_exposure: float = 0.0
        self.last_hedge_time: Optional[datetime] = None
        logger.info(f"HedgingEngine initialized (enabled={self.config.enabled})")
    
    def analyze_hedge_opportunity(
        self,
        trade: Dict[str, Any],
        current_drawdown: float = 0.0,
    ) -> HedgeResult:
        """
        Analyze if a hedge is recommended for a trade.
        
        Args:
            trade: Trade dict with:
                - market_id: Market identifier
                - outcome: Trade outcome (Yes/No/D/R/etc)
                - amount: Position size
                - potential_return: Potential return percentage
            current_drawdown: Current portfolio drawdown
        
        Returns:
            HedgeResult with recommendation
        """
        if not self.config.enabled:
            return HedgeResult(
                original_trade=trade,
                hedge_market_id=None,
                hedge_outcome=None,
                hedge_amount=0,
                hedge_type=HedgeType.NONE,
                is_recommended=False,
                reason="Hedging disabled",
                net_exposure_after_hedge=trade.get("amount", 0),
            )
        
        outcome = trade.get("outcome", "")
        amount = trade.get("amount", 0)
        
        if amount < self.config.min_position_for_hedge:
            return HedgeResult(
                original_trade=trade,
                hedge_market_id=None,
                hedge_outcome=None,
                hedge_amount=0,
                hedge_type=HedgeType.NONE,
                is_recommended=False,
                reason=f"Position ${amount:.2f} below minimum ${self.config.min_position_for_hedge:.2f}",
                net_exposure_after_hedge=amount,
            )
        
        # Check if we already have too much hedge exposure
        if self.hedge_exposure >= self.config.max_total_hedge_pct * 1000:  # Assuming $1000 portfolio
            return HedgeResult(
                original_trade=trade,
                hedge_market_id=None,
                hedge_outcome=None,
                hedge_amount=0,
                hedge_type=HedgeType.NONE,
                is_recommended=False,
                reason="Max hedge exposure reached",
                net_exposure_after_hedge=amount,
            )
        
        # Find hedge outcome
        hedge_outcome = self._find_hedge_outcome(outcome)
        
        if not hedge_outcome:
            return HedgeResult(
                original_trade=trade,
                hedge_market_id=None,
                hedge_outcome=None,
                hedge_amount=0,
                hedge_type=HedgeType.NONE,
                is_recommended=False,
                reason=f"No hedge found for outcome: {outcome}",
                net_exposure_after_hedge=amount,
            )
        
        # Calculate hedge amount
        base_hedge_pct = self.config.hedge_aggression
        
        # Increase hedging during drawdown
        if current_drawdown > self.config.max_drawdown_before_hedging:
            base_hedge_pct *= 1.5  # 50% more hedging during drawdown
        
        hedge_amount = amount * base_hedge_pct
        
        # Cap hedge amount
        max_hedge_amount = 0.02 * 1000  # 2% of $1000 portfolio
        hedge_amount = min(hedge_amount, max_hedge_amount)
        
        # Calculate net exposure
        net_exposure = amount - hedge_amount
        
        # Determine hedge type
        hedge_type = self._get_hedge_type(outcome, hedge_outcome)
        
        return HedgeResult(
            original_trade=trade,
            hedge_market_id=f"hedge_{trade.get('market_id', '')}",
            hedge_outcome=hedge_outcome,
            hedge_amount=hedge_amount,
            hedge_type=hedge_type,
            is_recommended=True,
            reason=f"Hedge recommended for {outcome} -> {hedge_outcome}",
            net_exposure_after_hedge=net_exposure,
        )
    
    def _find_hedge_outcome(self, outcome: str) -> Optional[str]:
        """Find the opposite outcome for hedging."""
        return self.config.hedge_pairs.get(outcome)
    
    def _get_hedge_type(self, outcome: str, hedge_outcome: str) -> HedgeType:
        """Determine the type of hedge."""
        if outcome in ["D", "R", "Biden", "Trump", "Harris", "DeSantis"]:
            return HedgeType.POLITICAL
        elif outcome in ["Yes", "No", "True", "False"]:
            return HedgeType.BINARY
        elif outcome in ["Over", "Under"]:
            return HedgeType.OVERUNDER
        else:
            return HedgeType.BINARY
    
    def execute_hedge(self, hedge_result: HedgeResult) -> bool:
        """Execute a hedge trade."""
        if not hedge_result.is_recommended or hedge_result.hedge_amount <= 0:
            return False
        
        # Add to active hedges
        hedge = {
            "market_id": hedge_result.hedge_market_id,
            "outcome": hedge_result.hedge_outcome,
            "amount": hedge_result.hedge_amount,
            "type": hedge_result.hedge_type.value,
            "created_at": datetime.utcnow().isoformat(),
            "original_trade": hedge_result.original_trade.get("trade_id"),
        }
        
        self.active_hedges.append(hedge)
        self.hedge_exposure += hedge_result.hedge_amount
        self.last_hedge_time = datetime.utcnow()
        
        logger.info(
            f"Hedge executed: {hedge_result.hedge_outcome} "
            f"${hedge_result.hedge_amount:.2f} "
            f"(Type: {hedge_result.hedge_type.value})"
        )
        
        return True
    
    def close_hedge(self, hedge_id: int) -> bool:
        """Close a specific hedge."""
        if hedge_id < 0 or hedge_id >= len(self.active_hedges):
            return False
        
        hedge = self.active_hedges.pop(hedge_id)
        self.hedge_exposure -= hedge["amount"]
        
        logger.info(f"Hedge closed: {hedge['outcome']} ${hedge['amount']:.2f}")
        
        return True
    
    def close_expired_hedges(self, max_age_hours: int = 168) -> int:
        """Close hedges older than max_age_hours (default 1 week)."""
        now = datetime.utcnow()
        expired = []
        
        for i, hedge in enumerate(self.active_hedges):
            created = datetime.fromisoformat(hedge["created_at"])
            if (now - created).total_seconds() > max_age_hours * 3600:
                expired.append(i)
        
        # Close from end to preserve indices
        for i in reversed(expired):
            self.close_hedge(i)
        
        if expired:
            logger.info(f"Closed {len(expired)} expired hedges")
        
        return len(expired)
    
    def get_hedge_recommendations(
        self,
        trades: List[Dict[str, Any]],
        current_drawdown: float = 0.0,
    ) -> List[HedgeResult]:
        """Get hedge recommendations for multiple trades."""
        recommendations = []
        
        for trade in trades:
            result = self.analyze_hedge_opportunity(trade, current_drawdown)
            if result.is_recommended:
                recommendations.append(result)
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get hedging status."""
        return {
            "enabled": self.config.enabled,
            "active_hedges": len(self.active_hedges),
            "hedge_exposure": self.hedge_exposure,
            "max_exposure": self.config.max_total_hedge_pct * 1000,
            "last_hedge_time": self.last_hedge_time.isoformat() if self.last_hedge_time else None,
        }
    
    def get_hedge_summary(self) -> Dict[str, Any]:
        """Get summary of active hedges."""
        hedge_by_type = defaultdict(float)
        for hedge in self.active_hedges:
            hedge_by_type[hedge["type"]] += hedge["amount"]
        
        return {
            "total_hedges": len(self.active_hedges),
            "total_exposure": self.hedge_exposure,
            "by_type": dict(hedge_by_type),
            "recommendations_available": len(self.get_hedge_recommendations([])),
        }


def create_hedging_config(
    hedge_pct: float = 0.25,
    max_total_hedge: float = 0.10,
    auto_hedge: bool = True,
) -> HedgingConfig:
    """Factory function to create hedging config."""
    return HedgingConfig(
        enabled=True,
        auto_hedge_binary=auto_hedge,
        max_total_hedge_pct=max_total_hedge,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("CROSS-MARKET HEDGING SYSTEM DEMO")
    print("=" * 60)
    
    # Create engine
    engine = HedgingEngine(
        create_hedging_config(
            hedge_pct=0.25,
            max_total_hedge=0.10,
        )
    )
    
    # Mock trades
    mock_trades = [
        {"trade_id": "t1", "market_id": "m1", "outcome": "Yes", "amount": 50.0, "potential_return": 0.50},
        {"trade_id": "t2", "market_id": "m2", "outcome": "D", "amount": 100.0, "potential_return": 0.80},
        {"trade_id": "t3", "market_id": "m3", "outcome": "Biden", "amount": 25.0, "potential_return": 0.40},
        {"trade_id": "t4", "market_id": "m4", "outcome": "Over", "amount": 2.0, "potential_return": 0.50},  # Below min
    ]
    
    print("\nAnalyzing trades for hedging opportunities...")
    
    for trade in mock_trades:
        result = engine.analyze_hedge_opportunity(trade, current_drawdown=0.0)
        
        print(f"\n{'='*50}")
        print(f"Trade: {trade['outcome']} ${trade['amount']:.2f}")
        print(f"Hedge: {result.hedge_outcome} ${result.hedge_amount:.2f}")
        print(f"Net Exposure: ${result.net_exposure_after_hedge:.2f}")
        print(f"Type: {result.hedge_type.value}")
        print(f"Recommended: {'✓' if result.is_recommended else '✗'}")
        if result.reason:
            print(f"Reason: {result.reason}")
        
        # Execute hedge if recommended
        if result.is_recommended:
            engine.execute_hedge(result)
    
    print(f"\n{'='*50}")
    print(f"\nHEDGE SUMMARY:")
    print(f"  Active Hedges: {len(engine.active_hedges)}")
    print(f"  Total Exposure: ${engine.hedge_exposure:.2f}")
    
    summary = engine.get_hedge_summary()
    print(f"\nBy Type:")
    for htype, amount in summary["by_type"].items():
        print(f"  {htype}: ${amount:.2f}")
    
    print("\n" + "=" * 60)
    print("HEDGING BENEFITS:")
    print("  • Reduces portfolio volatility")
    print("  • Protects during drawdowns")
    print("  • Automatic binary market hedging")
    print("  • Political race hedging (D/R)")
    print("  • Increases hedging when losing")
    print("=" * 60)
