"""
Momentum Filtering System.

Filters traders based on RECENT performance, not historical.

Usage:
    from orchestrator.config_momentum import MomentumConfig, MomentumFilter
    
    config = MomentumConfig(
        enabled=True,
        lookback_days=30,           # Only look at last 30 days
        min_recent_return=0.02,     # Must be up 2%+ recently
        min_recent_trades=3,        # At least 3 recent trades
        decay_factor=0.9,           # Older trades matter less
    )
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumStatus(Enum):
    """Momentum filter result status."""
    HOT = "hot"           # Strong recent performance
    WARM = "warm"         # Good recent performance
    COOL = "cool"         # Average recent performance
    COLD = "cold"         # Poor recent performance
    INSUFFICIENT = "insufficient"  # Not enough data


@dataclass
class MomentumConfig:
    """Configuration for momentum filtering."""
    enabled: bool = True
    
    # Lookback window
    lookback_days: int = 30
    
    # Performance thresholds
    min_recent_return: float = 0.02      # 2% minimum recent return
    min_recent_pnl: float = 50.0         # $50 minimum recent profit
    min_recent_trades: int = 3           # Minimum trades in window
    
    # Hot trader thresholds
    hot_return_threshold: float = 0.10   # 10%+ = hot trader
    hot_trade_threshold: int = 10        # 10+ trades = active
    
    # Decay factor (exponential weighting)
    decay_factor: float = 0.9            # Each day back = 10% less weight
    decay_type: str = "exponential"      # exponential or linear
    
    # Scoring
    score_weight_return: float = 0.6     # Return weight
    score_weight_trades: float = 0.2     # Trade frequency weight
    score_weight_consistency: float = 0.2  # Consistency weight
    
    # Filtering
    max_cold_traders: int = 10           # Max cold traders to copy
    allow_warm_only: bool = False        # Only copy warm+ traders


@dataclass
class MomentumResult:
    """Result of momentum analysis for a trader."""
    address: str
    status: MomentumStatus
    score: float                    # 0-1 momentum score
    recent_return: float            # Return in lookback window
    recent_pnl: float               # P&L in lookback window
    recent_trades: int              # Number of trades in window
    weighted_score: float           # Decay-weighted score
    is_suitable: bool               # Passes filter
    rejection_reason: Optional[str] = None


class MomentumFilter:
    """
    Filters traders based on recent momentum.
    
    Key behavior:
    - Only copies traders who performed well recently
    - Uses exponential decay to weight recent trades more
    - Hot traders get priority
    - Cold traders are filtered out
    """
    
    def __init__(self, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig()
        self.trader_momentum: Dict[str, MomentumResult] = {}
        logger.info(f"MomentumFilter initialized (enabled={self.config.enabled})")
    
    def analyze_trader(
        self,
        address: str,
        trades: List[Dict[str, Any]],
    ) -> MomentumResult:
        """
        Analyze a trader's momentum.
        
        Args:
            address: Trader address
            trades: List of trade dicts with:
                - timestamp: Trade datetime
                - pnl: Profit/loss
                - return_pct: Return percentage
                - market_id: Market traded
        
        Returns:
            MomentumResult with analysis
        """
        if not self.config.enabled:
            return MomentumResult(
                address=address,
                status=MomentumStatus.WARM,
                score=0.5,
                recent_return=0,
                recent_pnl=0,
                recent_trades=len(trades),
                weighted_score=0.5,
                is_suitable=True,
            )
        
        # Filter trades to lookback window
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.lookback_days)
        recent_trades = [
            t for t in trades
            if datetime.fromisoformat(t.get("timestamp", "2024-01-01")) >= cutoff_date
        ]
        
        # Check minimum trades
        if len(recent_trades) < self.config.min_recent_trades:
            result = MomentumResult(
                address=address,
                status=MomentumStatus.INSUFFICIENT,
                score=0,
                recent_return=0,
                recent_pnl=0,
                recent_trades=len(recent_trades),
                weighted_score=0,
                is_suitable=False,
                rejection_reason=f"Insufficient recent trades: {len(recent_trades)} < {self.config.min_recent_trades}",
            )
            self.trader_momentum[address] = result
            return result
        
        # Calculate metrics
        total_pnl = sum(t.get("pnl", 0) for t in recent_trades)
        returns = [t.get("return_pct", 0) for t in recent_trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        
        # Calculate weighted score (recent trades weighted more)
        weighted_score = self._calculate_weighted_score(recent_trades)
        
        # Calculate overall momentum score
        score = self._calculate_momentum_score(
            avg_return, len(recent_trades), weighted_score
        )
        
        # Determine status
        status = self._get_status(avg_return, len(recent_trades))
        
        # Check thresholds
        is_suitable = True
        rejection_reason = None
        
        if avg_return < self.config.min_recent_return:
            is_suitable = False
            rejection_reason = f"Recent return {avg_return:.2%} below minimum {self.config.min_recent_return:.2%}"
        
        if total_pnl < self.config.min_recent_pnl:
            is_suitable = False
            rejection_reason = f"Recent P&L ${total_pnl:.2f} below minimum ${self.config.min_recent_pnl:.2f}"
        
        if self.config.allow_warm_only and status in [MomentumStatus.COOL, MomentumStatus.COLD]:
            is_suitable = False
            rejection_reason = f"Status {status.value} not warm enough for filtering"
        
        result = MomentumResult(
            address=address,
            status=status,
            score=score,
            recent_return=avg_return,
            recent_pnl=total_pnl,
            recent_trades=len(recent_trades),
            weighted_score=weighted_score,
            is_suitable=is_suitable,
            rejection_reason=rejection_reason,
        )
        
        self.trader_momentum[address] = result
        
        logger.debug(
            f"Momentum analysis: {address[:10]}... | "
            f"Status: {status.value} | "
            f"Score: {score:.2f} | "
            f"Return: {avg_return:.2%}"
        )
        
        return result
    
    def _calculate_weighted_score(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate weighted score with decay."""
        if not trades:
            return 0
        
        if self.config.decay_type == "exponential":
            return self._exponential_decay_score(trades)
        else:
            return self._linear_decay_score(trades)
    
    def _exponential_decay_score(self, trades: List[Dict[str, Any]]) -> float:
        """Exponential decay - newer trades weighted more."""
        # Sort by timestamp (newest first)
        sorted_trades = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        weighted_sum = 0
        weight_sum = 0
        
        for i, trade in enumerate(sorted_trades):
            weight = self.config.decay_factor ** i
            return_pct = trade.get("return_pct", 0)
            weighted_sum += return_pct * weight
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0
    
    def _linear_decay_score(self, trades: List[Dict[str, Any]]) -> float:
        """Linear decay - each trade back reduces weight equally."""
        sorted_trades = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        weighted_sum = 0
        max_weight = len(sorted_trades)
        
        for i, trade in enumerate(sorted_trades):
            weight = 1 - (i / max_weight) if max_weight > 0 else 0
            return_pct = trade.get("return_pct", 0)
            weighted_sum += return_pct * weight
        
        return weighted_sum / sum(range(1, max_weight + 1)) if max_weight > 0 else 0
    
    def _calculate_momentum_score(
        self,
        avg_return: float,
        trade_count: int,
        weighted_score: float,
    ) -> float:
        """Calculate overall momentum score (0-1)."""
        # Return component
        return_score = min(max(avg_return / 0.20, 0), 1.0)  # 20% return = max score
        
        # Trade frequency component
        freq_score = min(trade_count / 20, 1.0)  # 20 trades = max score
        
        # Consistency component
        consistency_score = min(max(weighted_score / 0.15, 0), 1.0)
        
        # Weighted combination
        score = (
            return_score * self.config.score_weight_return +
            freq_score * self.config.score_weight_trades +
            consistency_score * self.config.score_weight_consistency
        )
        
        return min(max(score, 0), 1.0)
    
    def _get_status(self, avg_return: float, trade_count: int) -> MomentumStatus:
        """Determine momentum status based on metrics."""
        if avg_return >= self.config.hot_return_threshold and trade_count >= self.config.hot_trade_threshold:
            return MomentumStatus.HOT
        elif avg_return >= self.config.min_recent_return * 2:
            return MomentumStatus.WARM
        elif avg_return >= self.config.min_recent_return:
            return MomentumStatus.COOL
        else:
            return MomentumStatus.COLD
    
    def filter_traders(
        self,
        traders: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, MomentumResult]:
        """
        Filter multiple traders based on momentum.
        
        Args:
            traders: Dict mapping address -> trades list
        
        Returns:
            Dict mapping address -> MomentumResult
        """
        results = {}
        
        for address, trades in traders.items():
            result = self.analyze_trader(address, trades)
            results[address] = result
        
        # Log summary
        status_counts = {
            "hot": sum(1 for r in results.values() if r.status == MomentumStatus.HOT),
            "warm": sum(1 for r in results.values() if r.status == MomentumStatus.WARM),
            "cool": sum(1 for r in results.values() if r.status == MomentumStatus.COOL),
            "cold": sum(1 for r in results.values() if r.status == MomentumStatus.COLD),
            "insufficient": sum(1 for r in results.values() if r.status == MomentumStatus.INSUFFICIENT),
            "suitable": sum(1 for r in results.values() if r.is_suitable),
        }
        
        logger.info(
            f"Momentum filter complete: "
            f"hot={status_counts['hot']}, "
            f"warm={status_counts['warm']}, "
            f"cool={status_counts['cool']}, "
            f"cold={status_counts['cold']}, "
            f"suitable={status_counts['suitable']}"
        )
        
        return results
    
    def get_suitable_traders(
        self,
        traders: Dict[str, List[Dict[str, Any]]],
        limit: int = 20,
    ) -> List[Tuple[str, MomentumResult]]:
        """
        Get list of suitable traders sorted by score.
        
        Args:
            traders: Dict mapping address -> trades list
            limit: Maximum traders to return
        
        Returns:
            List of (address, MomentumResult) tuples, sorted by score
        """
        results = self.filter_traders(traders)
        
        suitable = [
            (addr, result) for addr, result in results.items()
            if result.is_suitable
        ]
        
        suitable.sort(key=lambda x: x[1].score, reverse=True)
        
        return suitable[:limit]
    
    def get_hot_traders(
        self,
        traders: Dict[str, List[Dict[str, Any]]],
        limit: int = 10,
    ) -> List[Tuple[str, MomentumResult]]:
        """Get only hot momentum traders."""
        results = self.filter_traders(traders)
        
        hot = [
            (addr, result) for addr, result in results.items()
            if result.status == MomentumStatus.HOT
        ]
        
        hot.sort(key=lambda x: x[1].score, reverse=True)
        
        return hot[:limit]
    
    def get_status(self) -> Dict[str, Any]:
        """Get filter status."""
        return {
            "enabled": self.config.enabled,
            "lookback_days": self.config.lookback_days,
            "traders_analyzed": len(self.trader_momentum),
            "momentum_counts": {
                "hot": sum(1 for r in self.trader_momentum.values() if r.status == MomentumStatus.HOT),
                "warm": sum(1 for r in self.trader_momentum.values() if r.status == MomentumStatus.WARM),
                "cool": sum(1 for r in self.trader_momentum.values() if r.status == MomentumStatus.COOL),
                "cold": sum(1 for r in self.trader_momentum.values() if r.status == MomentumStatus.COLD),
            },
        }


def create_momentum_config(
    lookback_days: int = 30,
    min_return: float = 0.02,
    min_trades: int = 3,
    decay_factor: float = 0.9,
) -> MomentumConfig:
    """Factory function to create momentum config."""
    return MomentumConfig(
        enabled=True,
        lookback_days=lookback_days,
        min_recent_return=min_return,
        min_recent_trades=min_trades,
        decay_factor=decay_factor,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("MOMENTUM FILTERING SYSTEM DEMO")
    print("=" * 60)
    
    # Create filter
    filter = MomentumFilter(
        create_momentum_config(
            lookback_days=30,
            min_return=0.02,
            min_trades=3,
        )
    )
    
    # Mock trader data with different momentum profiles
    now = datetime.utcnow()
    
    mock_traders = {
        "0xHOT111": [
            {"timestamp": (now - timedelta(days=1)).isoformat(), "pnl": 50, "return_pct": 0.10},
            {"timestamp": (now - timedelta(days=3)).isoformat(), "pnl": 30, "return_pct": 0.08},
            {"timestamp": (now - timedelta(days=5)).isoformat(), "pnl": 40, "return_pct": 0.12},
            {"timestamp": (now - timedelta(days=7)).isoformat(), "pnl": 25, "return_pct": 0.09},
            {"timestamp": (now - timedelta(days=10)).isoformat(), "pnl": 35, "return_pct": 0.11},
        ],
        "0xWARM222": [
            {"timestamp": (now - timedelta(days=2)).isoformat(), "pnl": 20, "return_pct": 0.05},
            {"timestamp": (now - timedelta(days=8)).isoformat(), "pnl": 15, "return_pct": 0.04},
            {"timestamp": (now - timedelta(days=15)).isoformat(), "pnl": 18, "return_pct": 0.03},
            {"timestamp": (now - timedelta(days=20)).isoformat(), "pnl": 12, "return_pct": 0.03},
        ],
        "0xCOLD333": [
            {"timestamp": (now - timedelta(days=1)).isoformat(), "pnl": -10, "return_pct": -0.05},
            {"timestamp": (now - timedelta(days=5)).isoformat(), "pnl": -5, "return_pct": -0.02},
            {"timestamp": (now - timedelta(days=10)).isoformat(), "pnl": -8, "return_pct": -0.03},
        ],
        "0xOLD444": [  # Not enough recent trades
            {"timestamp": (now - timedelta(days=60)).isoformat(), "pnl": 100, "return_pct": 0.20},
            {"timestamp": (now - timedelta(days=90)).isoformat(), "pnl": 80, "return_pct": 0.15},
        ],
    }
    
    # Filter traders
    results = filter.filter_traders(mock_traders)
    
    print("\nMomentum Analysis Results:")
    print("-" * 60)
    
    for address, result in results.items():
        status_emoji = {"hot": "🔥", "warm": "☀️", "cool": "😐", "cold": "❄️", "insufficient": "❓"}
        emoji = status_emoji.get(result.status.value, "?")
        
        print(f"\n{emoji} {address[:10]}...")
        print(f"   Status: {result.status.value}")
        print(f"   Score: {result.score:.2f}")
        print(f"   Recent Return: {result.recent_return:.2%}")
        print(f"   Recent P&L: ${result.recent_pnl:.2f}")
        print(f"   Recent Trades: {result.recent_trades}")
        print(f"   Suitable: {'✓' if result.is_suitable else '✗'}")
        if result.rejection_reason:
            print(f"   Reason: {result.rejection_reason}")
    
    # Get hot traders only
    hot = filter.get_hot_traders(mock_traders)
    print(f"\n🔥 Hot Traders: {len(hot)}")
    
    # Get all suitable
    suitable = filter.get_suitable_traders(mock_traders)
    print(f"✓ Suitable Traders: {len(suitable)}")
    
    print("\n" + "=" * 60)
    print("MOMENTUM FILTERING BENEFITS:")
    print("  • Catches traders on hot streaks")
    print("  • Avoids cold/trending-down traders")
    print("  • Exponential decay weights recent trades")
    print("  • Filters out insufficient data")
    print("=" * 60)
