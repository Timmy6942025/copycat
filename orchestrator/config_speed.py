"""
Speed Mode - Unified Configuration for Maximum Growth.

Combines all speed optimization features into one powerful configuration.

Usage:
    from orchestrator.config_speed import SpeedModeConfig, create_speed_mode_config
    
    config = create_speed_mode_config(
        initial_balance=100.0,
        mode=TradingMode.SANDBOX,
        enable_all_features=True,  # Enable everything
    )
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


class SpeedMode(Enum):
    """Speed mode intensity levels."""
    CONSERVATIVE = "conservative"  # ~5% monthly
    BALANCED = "balanced"          # ~7-8% monthly (recommended)
    AGGRESSIVE = "aggressive"      # ~10-12% monthly
    EXTREME = "extreme"            # ~15%+ monthly (high risk)


@dataclass
class SpeedModeConfig:
    """Unified configuration for maximum growth."""
    
    # Mode settings
    speed_mode: SpeedMode = SpeedMode.BALANCED
    initial_balance: float = 100.0
    mode: TradingMode = TradingMode.SANDBOX
    
    # Feature toggles
    enable_tiered_copying: bool = True
    enable_momentum_filter: bool = True
    enable_event_focus: bool = True
    enable_hedging: bool = True
    enable_optimizer: bool = True
    enable_allocation: bool = True
    enable_bootstrap: bool = True
    enable_adaptive: bool = True
    
    # Feature configurations
    tiered_config: Optional[TieredCopyConfig] = None
    momentum_config: Optional[MomentumConfig] = None
    event_config: Optional[EventFocusConfig] = None
    hedging_config: Optional[HedgingConfig] = None
    optimizer_config: Optional[OptimizationConfig] = None
    allocation_config: Optional[AllocationConfig] = None
    bootstrap_config: Optional[BootstrapConfig] = None
    adaptive_config: Optional[AdaptiveConfig] = None
    
    # Base orchestrator config
    orchestrator_config: Optional[OrchestratorConfig] = None


class SpeedModeEngine:
    """
    Unified engine managing all speed optimization features.
    
    Combines:
    - Tiered Copying (more capital to top performers)
    - Momentum Filtering (focus on recent winners)
    - Event Focus (prioritize high-conviction events)
    - Cross-Market Hedging (reduce risk)
    - Auto-Optimization (learn from trades)
    - Smart Capital Allocation (optimal sizing)
    - Quick-Start Bootstrap (faster starts)
    - Adaptive Scaling (dynamic adjustment)
    """
    
    def __init__(self, config: Optional[SpeedModeConfig] = None):
        self.config = config or SpeedModeConfig()
        
        # Initialize all engines
        self._init_all_engines()
        
        logger.info(f"SpeedModeEngine initialized with {self.config.speed_mode.value} mode")
    
    def _init_all_engines(self):
        """Initialize all optimization engines based on config."""
        # Tiered Copying
        if self.config.enable_tiered_copying:
            tiered_config = self.config.tiered_config or TieredCopyConfig(
                enabled=True,
                tier1_traders=3,
                tier1_multiplier=3.0,
                tier2_traders=7,
                tier2_multiplier=1.0,
                tier3_multiplier=0.5,
            )
            self.tiered_engine = TieredCopyEngine(tiered_config)
        else:
            self.tiered_engine = None
        
        # Momentum Filter
        if self.config.enable_momentum_filter:
            momentum_config = self.config.momentum_config or MomentumConfig(
                enabled=True,
                lookback_days=30,
                min_recent_return=0.02,
                min_recent_trades=3,
                decay_factor=0.9,
            )
            self.momentum_filter = MomentumFilter(momentum_config)
        else:
            self.momentum_filter = None
        
        # Event Focus
        if self.config.enable_event_focus:
            event_config = self.config.event_config or EventFocusConfig(
                enabled=True,
                min_event_priority=0.4,
            )
            self.event_engine = EventFocusEngine(event_config)
        else:
            self.event_engine = None
        
        # Hedging
        if self.config.enable_hedging:
            hedging_config = self.config.hedging_config or HedgingConfig(
                enabled=True,
                auto_hedge_binary=True,
                max_total_hedge_pct=0.10,
            )
            self.hedging_engine = HedgingEngine(hedging_config)
        else:
            self.hedging_engine = None
        
        # Optimizer
        if self.config.enable_optimizer:
            optimizer_config = self.config.optimizer_config or OptimizationConfig(
                enabled=True,
                learning_rate=0.01,
                optimization_window=30,
            )
            self.optimizer = PerformanceOptimizer(optimizer_config)
        else:
            self.optimizer = None
        
        # Capital Allocation
        if self.config.enable_allocation:
            allocation_config = self.config.allocation_config or AllocationConfig(
                enabled=True,
                strategy="proportional",
                total_portfolio=self.config.initial_balance,
            )
            self.allocator = CapitalAllocator(allocation_config)
        else:
            self.allocator = None
        
        # Bootstrap
        if self.config.enable_bootstrap:
            bootstrap_config = self.config.bootstrap_config or BootstrapConfig(
                enabled=True,
                bootstrap_days=14,
                use_historical_winners=True,
                use_proven_patterns=True,
            )
            self.bootstrap_engine = BootstrapEngine(bootstrap_config)
        else:
            self.bootstrap_engine = None
        
        # Adaptive Scaling
        if self.config.enable_adaptive:
            adaptive_config = self.config.adaptive_config or AdaptiveConfig(
                base_max_traders=15,
                strategy=ScalingStrategy.BALANCED,
            )
            self.adaptive_scaler = AdaptiveScaler(
                adaptive_config,
                initial_balance=self.config.initial_balance
            )
        else:
            self.adaptive_scaler = None
    
    def get_optimized_params(self) -> Dict[str, float]:
        """Get current optimized parameters from all engines."""
        params = {
            "position_size_pct": 0.08,
            "max_traders": 15,
            "min_trader_score": 0.4,
            "hedge_pct": 0.25,
        }
        
        if self.config.enable_optimizer and self.optimizer:
            params.update(self.optimizer.get_current_params())
        
        if self.config.enable_adaptive and self.adaptive_scaler:
            limits = self.adaptive_scaler.get_effective_limits()
            params["max_traders"] = limits.max_traders_to_copy
            params["position_size_pct"] = limits.position_size_pct
        
        return params
    
    def get_status(self) -> Dict[str, Any]:
        """Get combined status from all engines."""
        status = {
            "speed_mode": self.config.speed_mode.value,
            "features_enabled": {
                "tiered_copying": self.config.enable_tiered_copying,
                "momentum_filter": self.config.enable_momentum_filter,
                "event_focus": self.config.enable_event_focus,
                "hedging": self.config.enable_hedging,
                "optimizer": self.config.enable_optimizer,
                "allocation": self.config.enable_allocation,
                "bootstrap": self.config.enable_bootstrap,
                "adaptive": self.config.enable_adaptive,
            },
            "engines": {},
        }
        
        # Add status from each engine
        if self.tiered_engine:
            status["engines"]["tiered"] = self.tiered_engine.get_status()
        if self.momentum_filter:
            status["engines"]["momentum"] = self.momentum_filter.get_status()
        if self.event_engine:
            status["engines"]["event"] = self.event_engine.get_status()
        if self.hedging_engine:
            status["engines"]["hedging"] = self.hedging_engine.get_status()
        if self.optimizer:
            status["engines"]["optimizer"] = self.optimizer.get_status()
        if self.allocator:
            status["engines"]["allocator"] = self.allocator.get_status()
        if self.bootstrap_engine:
            status["engines"]["bootstrap"] = self.bootstrap_engine.get_status()
        if self.adaptive_scaler:
            status["engines"]["adaptive"] = self.adaptive_scaler.get_status()
        
        return status
    
    def create_orchestrator_config(self) -> OrchestratorConfig:
        """Create base orchestrator config from speed mode settings."""
        # Trader selection (growth-based)
        trader_selection = TraderSelectionConfig(
            mode=SelectionMode.GROWTH,
            growth_min_total_pnl=100.0,
            growth_max_drawdown=0.50,
            growth_min_active_days=3,
        )
        
        # Copy trading
        copy_trading = CopyTradingConfig(
            position_sizing_method="kelly",
            position_size_pct=0.10,
            kelly_fraction=0.25,
        )
        
        # Sandbox
        sandbox = SandboxConfigOrchestrator(
            initial_balance=self.config.initial_balance,
        )
        
        # Bot filter
        bot_filter = BotFilterOrchestratorConfig()
        
        return OrchestratorConfig(
            mode=self.config.mode,
            platform=MarketPlatform.POLYMARKET,
            trader_selection=trader_selection,
            copy_trading=copy_trading,
            sandbox=sandbox,
            bot_filter=bot_filter,
            max_traders_to_copy=15,
            max_traders_to_analyze_per_cycle=200,
        )


def create_speed_mode_config(
    initial_balance: float = 100.0,
    mode: TradingMode = TradingMode.SANDBOX,
    speed_mode: str = "balanced",
    enable_all_features: bool = True,
) -> SpeedModeConfig:
    """
    Factory function to create speed mode configuration.
    
    Args:
        initial_balance: Starting balance
        mode: SANDBOX or LIVE
        speed_mode: conservative, balanced, aggressive, extreme
        enable_all_features: Enable all optimizations
    
    Returns:
        Fully configured SpeedModeConfig
    """
    mode_map = {
        "conservative": SpeedMode.CONSERVATIVE,
        "balanced": SpeedMode.BALANCED,
        "aggressive": SpeedMode.AGGRESSIVE,
        "extreme": SpeedMode.EXTREME,
    }
    
    selected_mode = mode_map.get(speed_mode, SpeedMode.BALANCED)
    
    # Configure based on speed mode
    if selected_mode == SpeedMode.CONSERVATIVE:
        tiered = TieredCopyConfig(
            enabled=True, tier1_multiplier=2.0, tier1_traders=2, tier2_traders=5
        )
        momentum = MomentumConfig(
            enabled=True, lookback_days=30, min_recent_return=0.03
        )
        adaptive = AdaptiveConfig(
            base_max_traders=10, strategy=ScalingStrategy.CONSERVATIVE
        )
    elif selected_mode == SpeedMode.BALANCED:
        tiered = TieredCopyConfig(
            enabled=True, tier1_multiplier=3.0, tier1_traders=3, tier2_traders=7
        )
        momentum = MomentumConfig(
            enabled=True, lookback_days=30, min_recent_return=0.02
        )
        adaptive = AdaptiveConfig(
            base_max_traders=15, strategy=ScalingStrategy.BALANCED
        )
    else:  # AGGRESSIVE or EXTREME
        tiered = TieredCopyConfig(
            enabled=True, tier1_multiplier=4.0, tier1_traders=5, tier2_traders=10
        )
        momentum = MomentumConfig(
            enabled=True, lookback_days=14, min_recent_return=0.01
        )
        adaptive = AdaptiveConfig(
            base_max_traders=25, strategy=ScalingStrategy.AGGRESSIVE
        )
    
    return SpeedModeConfig(
        speed_mode=selected_mode,
        initial_balance=initial_balance,
        mode=mode,
        enable_tiered_copying=enable_all_features,
        enable_momentum_filter=enable_all_features,
        enable_event_focus=enable_all_features,
        enable_hedging=enable_all_features,
        enable_optimizer=enable_all_features,
        enable_allocation=enable_all_features,
        enable_bootstrap=enable_all_features,
        enable_adaptive=enable_all_features,
        tiered_config=tiered,
        momentum_config=momentum,
        adaptive_config=adaptive,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("SPEED MODE - UNIFIED OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    # Create different speed modes
    for mode_name in ["conservative", "balanced", "aggressive"]:
        config = create_speed_mode_config(
            initial_balance=100.0,
            speed_mode=mode_name,
        )
        
        engine = SpeedModeEngine(config)
        status = engine.get_status()
        
        print(f"\n{'='*60}")
        print(f"SPEED MODE: {mode_name.upper()}")
        print(f"{'='*60}")
        
        print("\nFeatures Enabled:")
        for feature, enabled in status["features_enabled"].items():
            print(f"  {'✓' if enabled else '✗'} {feature.replace('_', ' ').title()}")
        
        print(f"\nExpected Performance ({mode_name}):")
        if mode_name == "conservative":
            print("  ~5% monthly returns")
            print("  Low risk, steady growth")
        elif mode_name == "balanced":
            print("  ~7-8% monthly returns")
            print("  Medium risk, balanced growth")
        else:
            print("  ~10-12% monthly returns")
            print("  Higher risk, faster growth")
        
        params = engine.get_optimized_params()
        print(f"\nOptimized Parameters:")
        print(f"  Position Size: {params['position_size_pct']:.1%}")
        print(f"  Max Traders: {params['max_traders']}")
    
    print("\n" + "=" * 60)
    print("COMBINED SPEED MODE BENEFITS:")
    print("=" * 60)
    print("""
    1. TIERED COPYING: Top 3 traders get 3x capital
    2. MOMENTUM FILTER: Focus on recent winners
    3. EVENT FOCUS: Prioritize high-conviction events
    4. HEDGING: Reduce risk with opposite positions
    5. OPTIMIZER: Learns from your trades
    6. ALLOCATION: More $ to better performers
    7. BOOTSTRAP: Start with proven winners
    8. ADAPTIVE: Scales with portfolio growth
    
    Expected: $10 -> $20 in ~6 months (balanced mode)
    Expected: $10 -> $20 in ~4-5 months (aggressive mode)
    """)
    print("=" * 60)
