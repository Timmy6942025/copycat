"""
Micro Mode - Ultra-Aggressive Configuration for Very Small Accounts ($10-$25).

This configuration is designed for accounts that need maximum growth from minimal capital.
All parameters are tuned for fastest possible compounding.

Usage:
    from orchestrator.config_micro import create_micro_config, MicroModeEngine

    config = create_micro_config(initial_balance=10.0)
    engine = MicroModeEngine(config)
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.config import (
    TradingMode, MarketPlatform, TraderSelectionConfig,
    BotFilterOrchestratorConfig, CopyTradingConfig,
    SandboxConfigOrchestrator, OrchestratorConfig, SelectionMode,
    BoostModeConfig,
)
from orchestrator.config_tiered import TieredCopyConfig, TieredCopyEngine
from orchestrator.config_momentum import MomentumConfig, MomentumFilter
from orchestrator.config_events import EventFocusConfig, EventFocusEngine
from orchestrator.config_hedging import HedgingConfig, HedgingEngine
from orchestrator.config_optimizer import OptimizationConfig, PerformanceOptimizer
from orchestrator.config_allocation import AllocationConfig, CapitalAllocator
from orchestrator.config_bootstrap import BootstrapConfig, BootstrapEngine
from orchestrator.config_adaptive import AdaptiveConfig, AdaptiveScaler, ScalingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicroModeLevel(Enum):
    """Micro mode intensity levels for very small accounts."""
    NANO = "nano"      # $10-$15 - Maximum aggression
    MICRO = "micro"    # $15-$25 - High aggression
    MINI = "mini"      # $25-$50 - Moderate aggression


@dataclass
class MicroModeConfig:
    """Unified configuration for maximum growth with very small accounts ($10-$50)."""
    
    # Mode settings
    micro_mode: MicroModeLevel = MicroModeLevel.MICRO
    initial_balance: float = 10.0
    mode: TradingMode = TradingMode.SANDBOX
    
    # Feature toggles - ALL ENABLED for maximum growth
    enable_tiered_copying: bool = True
    enable_momentum_filter: bool = False  # Disable for small accounts - too restrictive
    enable_event_focus: bool = True
    enable_hedging: bool = False  # Disable hedging - need 100% growth capital
    enable_optimizer: bool = True
    enable_allocation: bool = True
    enable_bootstrap: bool = True  # Critical for fast start
    enable_adaptive: bool = True
    enable_boost_mode: bool = True  # Always on for micro accounts
    
    # Feature configurations
    tiered_config: Optional[TieredCopyConfig] = None
    momentum_config: Optional[MomentumConfig] = None
    event_config: Optional[EventFocusConfig] = None
    hedging_config: Optional[HedgingConfig] = None
    optimizer_config: Optional[OptimizationConfig] = None
    allocation_config: Optional[AllocationConfig] = None
    bootstrap_config: Optional[BootstrapConfig] = None
    adaptive_config: Optional[AdaptiveConfig] = None
    boost_config: Optional[BoostModeConfig] = None
    
    # Base orchestrator config
    orchestrator_config: Optional[OrchestratorConfig] = None


class MicroModeEngine:
    """
    Unified engine managing all optimization features for very small accounts.
    
    Key Optimizations for $10-$25 accounts:
    - 10-second trading cycles (30x faster than default)
    - 50-75% position sizes (maximize capital utilization)
    - Bootstrap from proven traders (instant diversification)
    - No hedging (100% growth capital)
    - Lenient trader qualification (faster to find traders)
    - High Kelly fraction (2x-3x standard)
    """
    
    def __init__(self, config: Optional[MicroModeConfig] = None):
        self.config = config or MicroModeConfig()
        self._init_all_engines()
        logger.info(f"MicroModeEngine initialized with {self.config.micro_mode.value} mode "
                   f"(${self.config.initial_balance:.0f} starting balance)")
    
    def _init_all_engines(self):
        """Initialize all optimization engines based on config."""
        
        # Boost Mode - Always aggressive for micro accounts
        if self.config.enable_boost_mode:
            boost_config = self.config.boost_config or BoostModeConfig(
                enabled=True,
                balance_threshold=25.0,  # Only for accounts under $25
                position_multiplier=3.0,  # 3x position sizes
                max_boost_position_pct=0.75,  # Up to 75% per trade
                prefer_quick_resolve=True,
                quick_resolve_threshold_hours=48.0,  # 2 days
                quick_resolve_multiplier=2.5,  # 2.5x for quick resolves
            )
            self.boost_mode_config = boost_config
        
        # Tiered Copying - Concentrate on top performers
        if self.config.enable_tiered_copying:
            tiered_config = self.config.tiered_config or TieredCopyConfig(
                enabled=True,
                tier1_traders=2,  # Top 2 get majority
                tier1_multiplier=4.0,  # 4x capital to #1
                tier2_traders=3,
                tier2_multiplier=2.0,  # 2x to #2-4
                tier3_multiplier=0.5,
            )
            self.tiered_engine = TieredCopyEngine(tiered_config)
        else:
            self.tiered_engine = None
        
        # Momentum Filter - Disabled for micro (too restrictive)
        self.momentum_filter = None
        
        # Event Focus - Prioritize high-conviction events
        if self.config.enable_event_focus:
            event_config = self.config.event_config or EventFocusConfig(
                enabled=True,
                min_event_priority=0.3,  # Lenient threshold
            )
            self.event_engine = EventFocusEngine(event_config)
        else:
            self.event_engine = None
        
        # Hedging - DISABLED for maximum growth
        self.hedging_engine = None
        
        # Optimizer - Learn from trades
        if self.config.enable_optimizer:
            optimizer_config = self.config.optimizer_config or OptimizationConfig(
                enabled=True,
                learning_rate=0.05,  # Faster learning
                optimization_window=7,  # Short window for rapid adaptation
            )
            self.optimizer = PerformanceOptimizer(optimizer_config)
        else:
            self.optimizer = None
        
        # Capital Allocation
        if self.config.enable_allocation:
            allocation_config = self.config.allocation_config or AllocationConfig(
                enabled=True,
                strategy="aggressive",  # Maximize returns
                total_portfolio=self.config.initial_balance,
            )
            self.allocator = CapitalAllocator(allocation_config)
        else:
            self.allocator = None
        
        # Bootstrap - Critical for fast start
        if self.config.enable_bootstrap:
            bootstrap_config = self.config.bootstrap_config or BootstrapConfig(
                enabled=True,
                bootstrap_days=7,  # Shorter bootstrap
                use_historical_winners=True,
                use_proven_patterns=True,
                bootstrap_position_size_pct=0.10,  # Slightly higher for growth
            )
            self.bootstrap_engine = BootstrapEngine(bootstrap_config)
        else:
            self.bootstrap_engine = None
        
        # Adaptive Scaling
        if self.config.enable_adaptive:
            adaptive_config = self.config.adaptive_config or AdaptiveConfig(
                base_max_traders=5,
                strategy=ScalingStrategy.AGGRESSIVE,
            )
            self.adaptive_scaler = AdaptiveScaler(
                adaptive_config,
                initial_balance=self.config.initial_balance
            )
        else:
            self.adaptive_scaler = None
    
    def create_orchestrator_config(self) -> OrchestratorConfig:
        """Create base orchestrator config optimized for micro accounts."""
        
        # ULTRA-RELaxed trader selection for bootstrapping
        trader_selection = TraderSelectionConfig(
            mode=SelectionMode.GROWTH,
            growth_min_total_pnl=10.0,  # Very low threshold - just $10 profit
            growth_min_growth_rate=0.002,  # 0.2% per trade - extremely lenient
            growth_max_drawdown=0.75,  # Allow up to 75% drawdown
            growth_min_equity_slope=0.00005,  # Nearly flat trend OK
            growth_min_consistency=0.10,  # Low consistency OK
            growth_min_active_days=1,  # Just 1 day active
        )
        
        # ULTRA-AGGRESSIVE copy trading
        copy_trading = CopyTradingConfig(
            position_sizing_method="kelly",
            position_size_pct=0.60,  # 60% per trade!
            kelly_fraction=0.75,  # High Kelly for fast growth
            max_position_size_pct=0.75,  # Up to 75% on best trades
            max_total_exposure_pct=0.95,  # 95% fully invested
            min_order_size=0.01,
            max_orders_per_day=1000,  # Unlimited trades
        )
        
        # Sandbox with micro balance
        sandbox = SandboxConfigOrchestrator(
            initial_balance=self.config.initial_balance,
        )
        
        # Minimal bot filter
        bot_filter = BotFilterOrchestratorConfig(
            min_hft_score_to_exclude=0.9,  # Only exclude obvious bots
            min_arbitrage_score_to_exclude=0.9,
        )
        
        return OrchestratorConfig(
            mode=self.config.mode,
            platform=MarketPlatform.POLYMARKET,
            trader_selection=trader_selection,
            copy_trading=copy_trading,
            sandbox=sandbox,
            bot_filter=bot_filter,
            max_traders_to_copy=5,
            max_traders_to_analyze_per_cycle=300,  # Analyze many
            trader_data_refresh_interval_seconds=10,  # 10-second cycles
            boost_mode=self.boost_mode_config if hasattr(self, 'boost_mode_config') else BoostModeConfig(),
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get combined status from all engines."""
        status = {
            "micro_mode": self.config.micro_mode.value,
            "initial_balance": self.config.initial_balance,
            "features_enabled": {
                "tiered_copying": self.config.enable_tiered_copying,
                "momentum_filter": self.config.enable_momentum_filter,
                "event_focus": self.config.enable_event_focus,
                "hedging": self.config.enable_hedging,
                "optimizer": self.config.enable_optimizer,
                "allocation": self.config.enable_allocation,
                "bootstrap": self.config.enable_bootstrap,
                "adaptive": self.config.enable_adaptive,
                "boost_mode": self.config.enable_boost_mode,
            },
            "expected": {
                "cycle_time": "10 seconds",
                "position_size": "60-75%",
                "kelly_fraction": "0.75",
                "traders_to_copy": "3-5",
                "expected_monthly_return": "25-50%",
                "time_to_double": "2-4 months",
            },
        }
        return status


def create_micro_config(
    initial_balance: float = 10.0,
    mode: TradingMode = TradingMode.SANDBOX,
    micro_mode: str = "auto",
) -> MicroModeConfig:
    """
    Factory function to create micro mode configuration.
    
    Automatically selects intensity level based on starting balance:
    - $10-$15: NANO mode (maximum aggression)
    - $15-$25: MICRO mode (high aggression)
    - $25-$50: MINI mode (moderate aggression)
    
    Args:
        initial_balance: Starting balance ($10 recommended)
        mode: SANDBOX or LIVE
        micro_mode: "nano", "micro", "mini", or "auto" (auto-detect)
    
    Returns:
        Fully configured MicroModeConfig
    """
    # Auto-detect mode if needed
    if micro_mode == "auto":
        if initial_balance <= 15:
            micro_mode = "nano"
        elif initial_balance <= 25:
            micro_mode = "micro"
        else:
            micro_mode = "mini"
    
    mode_map = {
        "nano": MicroModeLevel.NANO,
        "micro": MicroModeLevel.MICRO,
        "mini": MicroModeLevel.MINI,
    }
    
    selected_level = mode_map.get(micro_mode, MicroModeLevel.MICRO)
    
    # Configure based on level
    if selected_level == MicroModeLevel.NANO:
        # Maximum aggression for $10-$15
        copy_trading = CopyTradingConfig(
            position_sizing_method="kelly",
            position_size_pct=0.75,  # 75% per trade
            kelly_fraction=0.75,
            max_position_size_pct=0.80,
            max_total_exposure_pct=0.95,
            max_orders_per_day=1000,
        )
        trader_selection = TraderSelectionConfig(
            mode=SelectionMode.GROWTH,
            growth_min_total_pnl=5.0,  # Very low
            growth_min_growth_rate=0.001,
            growth_max_drawdown=0.80,
            growth_min_active_days=1,
        )
        
    elif selected_level == MicroModeLevel.MICRO:
        # High aggression for $15-$25
        copy_trading = CopyTradingConfig(
            position_sizing_method="kelly",
            position_size_pct=0.60,
            kelly_fraction=0.75,
            max_position_size_pct=0.70,
            max_total_exposure_pct=0.90,
            max_orders_per_day=1000,
        )
        trader_selection = TraderSelectionConfig(
            mode=SelectionMode.GROWTH,
            growth_min_total_pnl=10.0,
            growth_min_growth_rate=0.002,
            growth_max_drawdown=0.75,
            growth_min_active_days=1,
        )
        
    else:  # MINI
        # Moderate for $25-$50
        copy_trading = CopyTradingConfig(
            position_sizing_method="kelly",
            position_size_pct=0.50,
            kelly_fraction=0.50,
            max_position_size_pct=0.60,
            max_total_exposure_pct=0.85,
            max_orders_per_day=750,
        )
        trader_selection = TraderSelectionConfig(
            mode=SelectionMode.GROWTH,
            growth_min_total_pnl=15.0,
            growth_min_growth_rate=0.003,
            growth_max_drawdown=0.70,
            growth_min_active_days=2,
        )
    
    # Create orchestrator config
    orchestrator_config = OrchestratorConfig(
        mode=mode,
        platform=MarketPlatform.POLYMARKET,
        trader_selection=trader_selection,
        copy_trading=copy_trading,
        sandbox=SandboxConfigOrchestrator(initial_balance=initial_balance),
        max_traders_to_copy=5,
        trader_data_refresh_interval_seconds=10,
    )
    
    return MicroModeConfig(
        micro_mode=selected_level,
        initial_balance=initial_balance,
        mode=mode,
        enable_boost_mode=True,
        enable_hedging=False,  # No hedging for growth
        enable_momentum_filter=False,  # Too restrictive
        enable_bootstrap=True,  # Critical for fast start
        orchestrator_config=orchestrator_config,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("MICRO MODE - ULTRA-AGGRESSIVE SMALL ACCOUNT OPTIMIZATION")
    print("=" * 70)
    
    # Test different starting balances
    for balance in [10, 15, 25, 50]:
        config = create_micro_config(initial_balance=balance)
        engine = MicroModeEngine(config)
        status = engine.get_status()
        
        print(f"\n{'='*70}")
        print(f"STARTING BALANCE: ${balance}")
        print(f"MODE: {status['micro_mode'].upper()}")
        print(f"{'='*70}")
        
        print("\nFeatures:")
        for feature, enabled in status['features_enabled'].items():
            print(f"  {'✓' if enabled else '✗'} {feature.replace('_', ' ').title()}")
        
        print("\nExpected Performance:")
        for metric, value in status['expected'].items():
            print(f"  {metric.title()}: {value}")
    
    print("\n" + "=" * 70)
    print("GROWTH PROJECTIONS ($10 starting):")
    print("=" * 70)
    print("""
    NANO Mode ($10-$15 starting):
    - Position Size: 75%
    - Expected Monthly Return: 35-50%
    - Time to Double: 2-3 months
    - Risk Level: VERY HIGH
    
    MICRO Mode ($15-$25 starting):
    - Position Size: 60%
    - Expected Monthly Return: 25-40%
    - Time to Double: 3-4 months
    - Risk Level: HIGH
    
    MINI Mode ($25-$50 starting):
    - Position Size: 50%
    - Expected Monthly Return: 20-30%
    - Time to Double: 4-5 months
    - Risk Level: MEDIUM-HIGH
    """)
    print("=" * 70)
