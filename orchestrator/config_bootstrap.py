"""
Quick-Start Bootstrap System.

Starts with proven patterns before switching to discovery.

Usage:
    from orchestrator.config_bootstrap import BootstrapConfig, BootstrapEngine
    
    config = BootstrapConfig(
        enabled=True,
        bootstrap_days=14,           # Use bootstrap for first 2 weeks
        use_historical_winners=True, # Copy last month's best
        auto_discover_after=True,    # Then switch to discovery
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


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap system."""
    enabled: bool = True
    
    # Bootstrap period
    bootstrap_days: int = 14  # Days to use bootstrap mode
    bootstrap_hours: int = field(default_factory=lambda: 14 * 24)
    
    # Historical winners
    use_historical_winners: bool = True
    historical_window_days: int = 30  # Look back for historical winners
    min_historical_trades: int = 10   # Minimum trades to consider
    top_winners_count: int = 10       # Number of historical winners to copy
    
    # Proven patterns
    use_proven_patterns: bool = True
    pattern_categories: List[str] = field(default_factory=lambda: [
        "presidential_election",
        "fed_decision",
        "sports_finals",
        "economic_report",
    ])
    pattern_min_accuracy: float = 0.60  # Minimum pattern accuracy
    
    # Auto-discovery transition
    auto_discover_after: bool = True
    discovery_threshold_pnl: float = 0.05  # Switch if 5% profit in bootstrap
    discovery_threshold_trades: int = 20   # Or 20 trades
    
    # Initial capital allocation
    bootstrap_capital_pct: float = 0.80  # 80% of capital in bootstrap trades
    discovery_capital_pct: float = 0.20  # 20% for discovery
    
    # Risk during bootstrap
    bootstrap_position_size_pct: float = 0.05  # Conservative sizing
    max_bootstrap_risk_pct: float = 0.10  # Max 10% at risk during bootstrap


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""
    mode: str  # "bootstrap" or "discovered"
    traders_copied: List[str]
    patterns_followed: List[str]
    capital_allocated: float
    should_switch_to_discovery: bool
    switch_reason: Optional[str]
    bootstrap_ends_at: datetime


class BootstrapEngine:
    """
    Manages bootstrap period for faster starts.
    
    Key behavior:
    - Starts by copying historical winners
    - Uses proven event patterns
    - Gradually transitions to discovery
    - Protects capital during bootstrap
    """
    
    def __init__(self, config: Optional[BootstrapConfig] = None):
        self.config = config or BootstrapConfig()
        self.start_time: Optional[datetime] = None
        self.bootstrap_end_time: Optional[datetime] = None
        self.current_mode: str = "bootstrap"  # or "discovered"
        self.copied_traders: List[str] = []
        self.followed_patterns: List[str] = []
        self.has_switched: bool = False
        logger.info(f"BootstrapEngine initialized (enabled={self.config.enabled})")
    
    def start_bootstrap(self, start_time: Optional[datetime] = None):
        """Initialize bootstrap period."""
        if not self.config.enabled:
            logger.info("Bootstrap disabled, skipping")
            return
        
        self.start_time = start_time or datetime.utcnow()
        self.bootstrap_end_time = self.start_time + timedelta(hours=self.config.bootstrap_hours)
        self.current_mode = "bootstrap"
        
        logger.info(
            f"Bootstrap started at {self.start_time.isoformat()}, "
            f"ends at {self.bootstrap_end_time.isoformat()} "
            f"({self.config.bootstrap_days} days)"
        )
    
    def get_current_mode(self) -> str:
        """Get current mode (bootstrap or discovered)."""
        if not self.config.enabled:
            return "disabled"
        
        if self.current_mode == "bootstrap":
            # Check if bootstrap period is over
            if datetime.utcnow() >= self.bootstrap_end_time:
                self._switch_to_discovery("Bootstrap period completed")
                return "discovered"
            
            # Check if we should switch early
            if self._should_switch_early():
                self._switch_to_discovery("Performance threshold met")
                return "discovered"
        
        return self.current_mode
    
    def _should_switch_early(self) -> bool:
        """Check if we should switch to discovery early."""
        # This would check actual performance metrics
        # For now, return False (only switch after bootstrap period)
        return False
    
    def _switch_to_discovery(self, reason: str):
        """Switch from bootstrap to discovery mode."""
        if self.has_switched:
            return
        
        self.current_mode = "discovered"
        self.has_switched = True
        
        logger.info(f"Switched to discovery mode: {reason}")
    
    def get_bootstrap_recommendations(
        self,
        historical_winners: Dict[str, Dict[str, Any]],
        proven_patterns: Dict[str, Any],
    ) -> BootstrapResult:
        """
        Get recommendations for bootstrap trading.
        
        Args:
            historical_winners: Dict of historical top traders
            proven_patterns: Dict of proven trading patterns
        
        Returns:
            BootstrapResult with recommendations
        """
        if not self.config.enabled or self.get_current_mode() != "bootstrap":
            return BootstrapResult(
                mode="discovered",
                traders_copied=[],
                patterns_followed=[],
                capital_allocated=0,
                should_switch_to_discovery=False,
                switch_reason=None,
                bootstrap_ends_at=self.bootstrap_end_time or datetime.utcnow(),
            )
        
        # Select historical winners to copy
        traders_to_copy = self._select_historical_winners(historical_winners)
        
        # Select patterns to follow
        patterns_to_follow = self._select_patterns(proven_patterns)
        
        # Calculate capital allocation
        capital = self.config.bootstrap_capital_pct
        
        # Check if should switch
        should_switch, switch_reason = self._check_switch_condition()
        
        result = BootstrapResult(
            mode="bootstrap",
            traders_copied=traders_to_copy,
            patterns_followed=patterns_to_follow,
            capital_allocated=capital,
            should_switch_to_discovery=should_switch,
            switch_reason=switch_reason,
            bootstrap_ends_at=self.bootstrap_end_time,
        )
        
        logger.info(
            f"Bootstrap recommendations: {len(traders_to_copy)} traders, "
            f"{len(patterns_to_follow)} patterns, "
            f"{capital:.0%} capital"
        )
        
        return result
    
    def _select_historical_winners(
        self,
        historical_winners: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Select historical winners to copy."""
        if not self.config.use_historical_winners:
            return []
        
        # Filter by minimum trades
        qualified = {
            addr: data for addr, data in historical_winners.items()
            if data.get("trades_count", 0) >= self.config.min_historical_trades
        }
        
        # Sort by performance
        sorted_winners = sorted(
            qualified.items(),
            key=lambda x: x[1].get("total_pnl", 0),
            reverse=True
        )
        
        # Take top N
        top_winners = [
            addr for addr, _ in sorted_winners[:self.config.top_winners_count]
        ]
        
        self.copied_traders = top_winners
        
        logger.info(f"Selected {len(top_winners)} historical winners to copy")
        
        return top_winners
    
    def _select_patterns(
        self,
        proven_patterns: Dict[str, Any],
    ) -> List[str]:
        """Select proven patterns to follow."""
        if not self.config.use_proven_patterns:
            return []
        
        selected_patterns = []
        
        for category in self.config.pattern_categories:
            if category in proven_patterns:
                accuracy = proven_patterns[category].get("accuracy", 0)
                if accuracy >= self.config.pattern_min_accuracy:
                    selected_patterns.append(category)
        
        self.followed_patterns = selected_patterns
        
        logger.info(f"Selected {len(selected_patterns)} proven patterns")
        
        return selected_patterns
    
    def _check_switch_condition(self) -> Tuple[bool, Optional[str]]:
        """Check if should switch to discovery mode."""
        # Check time-based
        if datetime.utcnow() >= self.bootstrap_end_time:
            return True, "Bootstrap period completed"
        
        # Check performance-based (would check actual metrics)
        # For now, only time-based switching
        return False, None
    
    def get_status(self) -> Dict[str, Any]:
        """Get bootstrap status."""
        return {
            "enabled": self.config.enabled,
            "mode": self.current_mode,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.bootstrap_end_time.isoformat() if self.bootstrap_end_time else None,
            "has_switched": self.has_switched,
            "traders_copied": len(self.copied_traders),
            "patterns_followed": len(self.followed_patterns),
            "time_remaining_hours": (
                (self.bootstrap_end_time - datetime.utcnow()).total_seconds() / 3600
                if self.bootstrap_end_time else 0
            ),
        }
    
    def get_bootstrap_summary(self) -> Dict[str, Any]:
        """Get bootstrap summary."""
        return {
            "mode": self.current_mode,
            "bootstrap_ends_at": self.bootstrap_end_time.isoformat() if self.bootstrap_end_time else None,
            "historical_winners_copied": self.copied_traders,
            "patterns_being_followed": self.followed_patterns,
            "capital_allocation": {
                "bootstrap": self.config.bootstrap_capital_pct,
                "discovery": self.config.discovery_capital_pct,
            },
            "position_size": self.config.bootstrap_position_size_pct,
            "max_risk": self.config.max_bootstrap_risk_pct,
        }


def create_bootstrap_config(
    bootstrap_days: int = 14,
    use_winners: bool = True,
    use_patterns: bool = True,
) -> BootstrapConfig:
    """Factory function to create bootstrap config."""
    return BootstrapConfig(
        enabled=True,
        bootstrap_days=bootstrap_days,
        use_historical_winners=use_winners,
        use_proven_patterns=use_patterns,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("QUICK-START BOOTSTRAP DEMO")
    print("=" * 60)
    
    # Create bootstrap engine
    engine = BootstrapEngine(
        create_bootstrap_config(
            bootstrap_days=14,
            use_winners=True,
            use_patterns=True,
        )
    )
    
    # Start bootstrap
    engine.start_bootstrap()
    
    # Mock historical winners
    historical_winners = {
        "0xWINNER1": {"total_pnl": 500, "win_rate": 0.75, "trades_count": 50},
        "0xWINNER2": {"total_pnl": 400, "win_rate": 0.70, "trades_count": 45},
        "0xWINNER3": {"total_pnl": 350, "win_rate": 0.68, "trades_count": 40},
        "0xWINNER4": {"total_pnl": 300, "win_rate": 0.65, "trades_count": 35},
        "0xLOSER1": {"total_pnl": -100, "win_rate": 0.30, "trades_count": 30},
    }
    
    # Mock proven patterns
    proven_patterns = {
        "presidential_election": {"accuracy": 0.75, "trades": 100},
        "fed_decision": {"accuracy": 0.70, "trades": 80},
        "sports_finals": {"accuracy": 0.65, "trades": 60},
        "random_opinion": {"accuracy": 0.45, "trades": 50},
    }
    
    # Get recommendations
    result = engine.get_bootstrap_recommendations(historical_winners, proven_patterns)
    
    print(f"\n{'='*60}")
    print("BOOTSTRAP RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print(f"\nMode: {result.mode}")
    print(f"Bootstrap Ends: {result.bootstrap_ends_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"Capital Allocated: {result.capital_allocated:.0%}")
    
    print(f"\nHistorical Winners to Copy ({len(result.traders_copied)}):")
    for trader in result.traders_copied:
        print(f"  ✓ {trader}")
    
    print(f"\nPatterns to Follow ({len(result.patterns_followed)}):")
    for pattern in result.patterns_followed:
        print(f"  ✓ {pattern} (accuracy: {proven_patterns[pattern]['accuracy']:.0%})")
    
    if result.should_switch_to_discovery:
        print(f"\n⚠️  Should Switch: {result.switch_reason}")
    
    # Show status
    status = engine.get_status()
    print(f"\n{'='*60}")
    print("BOOTSTRAP STATUS")
    print(f"{'='*60}")
    print(f"Mode: {status['mode']}")
    print(f"Time Remaining: {status['time_remaining_hours']:.1f} hours")
    print(f"Traders Copied: {status['traders_copied']}")
    print(f"Patterns Following: {status['patterns_followed']}")
    
    # Summary
    summary = engine.get_bootstrap_summary()
    print(f"\n{'='*60}")
    print("BOOTSTRAP SUMMARY")
    print(f"{'='*60}")
    print(f"  Bootstrap Capital: {summary['capital_allocation']['bootstrap']:.0%}")
    print(f"  Discovery Capital: {summary['capital_allocation']['discovery']:.0%}")
    print(f"  Position Size: {summary['position_size']:.0%}")
    print(f"  Max Risk: {summary['max_risk']:.0%}")
    
    print("\n" + "=" * 60)
    print("BOOTSTRAP BENEFITS:")
    print("  • Starts with proven winners immediately")
    print("  • Uses historical performance data")
    print("  • Follows reliable event patterns")
    print("  • Conservative sizing protects capital")
    print("  • Auto-switches to discovery when ready")
    print("=" * 60)
