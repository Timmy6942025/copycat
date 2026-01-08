"""
Aggressive Growth Configuration for CopyCat.

Optimized for FASTER profits while maintaining reasonable consistency.
Use SANDBOX mode first to validate, then switch to LIVE for real trading.

Created for: Faster compounding with $10 starting balance
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from orchestrator.config import (
    TradingMode,
    MarketPlatform,
    TraderSelectionConfig,
    BotFilterOrchestratorConfig,
    CopyTradingConfig,
    SandboxConfigOrchestrator,
)


class GrowthStrategy(Enum):
    """Aggressive growth strategy levels."""
    MODERATE = "moderate"  # 5-6% monthly
    AGGRESSIVE = "aggressive"  # 7-9% monthly
    VERY_AGGRESSIVE = "very_aggressive"  # 10-15% monthly


@dataclass
class AggressiveGrowthConfig:
    """
    Aggressive growth configuration for CopyCat.
    
    Optimized for faster compounding while maintaining risk controls.
    
    Strategy: Copy more traders with slightly lower quality thresholds,
    but use larger position sizing and Kelly criterion.
    """
    
    # =================================================================
    # Core Settings
    # =================================================================
    mode: TradingMode = TradingMode.SANDBOX
    platform: MarketPlatform = MarketPlatform.POLYMARKET
    
    # =================================================================
    # Growth Strategy
    # =================================================================
    strategy: GrowthStrategy = GrowthStrategy.AGGRESSIVE
    
    # =================================================================
    # Trader Selection (Slightly relaxed for more opportunities)
    # =================================================================
    # Lower thresholds = more traders to copy = more trades
    trader_selection: TraderSelectionConfig = field(default_factory=lambda: TraderSelectionConfig(
        min_win_rate=0.52,        # 52% (down from 55%) - more traders qualify
        min_trades=8,             # 8 trades minimum (down from 10)
        max_drawdown=0.30,        # Allow 30% drawdown (more tolerance)
        min_sharpe_ratio=0.4,     # Lower Sharpe requirement
        min_profit_factor=0.9,    # Accept slightly negative profit factor if win rate high
        min_total_pnl=-100.0,     # Accept small losses if recent performance good
        max_avg_hold_time_hours=120.0,  # 5 days max hold (faster turnover)
        min_reputation_score=0.4, # Lower reputation threshold
    ))
    
    # =================================================================
    # Bot Filtering (Standard - still filter out HFT bots)
    # =================================================================
    bot_filter: BotFilterOrchestratorConfig = field(default_factory=lambda: BotFilterOrchestratorConfig(
        hft_max_hold_time_seconds=1.0,  # Filter <1 second trades
        hft_min_trades_per_minute=5,    # Filter 5+ trades/min
        arbitrage_max_profit_pct=0.5,   # Filter 50%+ arbitrage
        min_hft_score_to_exclude=0.7,
        min_arbitrage_score_to_exclude=0.7,
        min_pattern_score_to_exclude=0.7,
    ))
    
    # =================================================================
    # Copy Trading (Larger positions = faster growth)
    # =================================================================
    copy_trading: CopyTradingConfig = field(default_factory=lambda: CopyTradingConfig(
        position_sizing_method="kelly",  # Kelly for mathematical optimality
        base_position_size=25.0,         # Larger base
        position_size_pct=0.10,          # 10% per trade (double default)
        kelly_fraction=0.25,             # 1/4 Kelly (conservative)
        max_position_size_pct=0.15,      # Cap at 15% per trade
        max_total_exposure_pct=0.60,     # 60% total exposure
        min_order_size=2.0,              # $2 minimum
        max_orders_per_day=100,          # More trades allowed
    ))
    
    # =================================================================
    # Sandbox Settings
    # =================================================================
    sandbox: SandboxConfigOrchestrator = field(default_factory=lambda: SandboxConfigOrchestrator(
        initial_balance=1000.0,  # Start with $1000 for faster testing
        simulate_slippage=True,
        simulate_fees=True,
        simulate_fill_probability=True,
    ))
    
    # =================================================================
    # Scanner Settings (Find more traders)
    # =================================================================
    max_traders_to_copy: int = 20  # Double the default
    max_traders_to_analyze_per_cycle: int = 200  # Scan more traders
    trader_data_refresh_interval_seconds: int = 180  # Refresh faster (3 min)
    
    # =================================================================
    # Risk Controls (Still have limits)
    # =================================================================
    max_daily_loss_pct: float = 0.05  # Stop if lose 5% in a day
    max_drawdown_stop: float = 0.25   # Stop if 25% drawdown
    min_traders_required: int = 3     # Always copy at least 3


def create_aggressive_config(
    strategy: GrowthStrategy = GrowthStrategy.AGGRESSIVE,
    initial_balance: float = 100.0,
    mode: TradingMode = TradingMode.SANDBOX,
) -> AggressiveGrowthConfig:
    """
    Factory function to create aggressive growth configuration.
    
    Args:
        strategy: Growth strategy level
        initial_balance: Starting balance
        mode: Trading mode (SANDBOX for testing, LIVE for real trading)
    
    Returns:
        Configured AggressiveGrowthConfig
    """
    
    # Adjust based on strategy
    if strategy == GrowthStrategy.MODERATE:
        return AggressiveGrowthConfig(
            strategy=strategy,
            mode=mode,
            sandbox=SandboxConfigOrchestrator(initial_balance=initial_balance),
            copy_trading=CopyTradingConfig(
                position_sizing_method="scaled",
                position_size_pct=0.06,
                kelly_fraction=0.25,
                max_position_size_pct=0.10,
            ),
            trader_selection=TraderSelectionConfig(
                min_win_rate=0.54,
                min_trades=10,
                max_drawdown=0.25,
                min_sharpe_ratio=0.45,
            ),
            max_traders_to_copy=15,
        )
    
    elif strategy == GrowthStrategy.AGGRESSIVE:
        return AggressiveGrowthConfig(
            strategy=strategy,
            mode=mode,
            sandbox=SandboxConfigOrchestrator(initial_balance=initial_balance),
            copy_trading=CopyTradingConfig(
                position_sizing_method="kelly",
                position_size_pct=0.10,
                kelly_fraction=0.25,
                max_position_size_pct=0.15,
            ),
            trader_selection=TraderSelectionConfig(
                min_win_rate=0.52,
                min_trades=8,
                max_drawdown=0.30,
                min_sharpe_ratio=0.4,
            ),
            max_traders_to_copy=20,
        )
    
    else:  # VERY_AGGRESSIVE
        return AggressiveGrowthConfig(
            strategy=strategy,
            mode=mode,
            sandbox=SandboxConfigOrchestrator(initial_balance=initial_balance),
            copy_trading=CopyTradingConfig(
                position_sizing_method="kelly",
                position_size_pct=0.15,
                kelly_fraction=0.20,  # Even more aggressive
                max_position_size_pct=0.20,
                max_total_exposure_pct=0.70,
            ),
            trader_selection=TraderSelectionConfig(
                min_win_rate=0.50,
                min_trades=5,
                max_drawdown=0.40,
                min_sharpe_ratio=0.3,
            ),
            max_traders_to_copy=30,
            max_daily_loss_pct=0.08,  # Allow bigger daily swings
            max_drawdown_stop=0.35,
        )


# =================================================================
# Quick Usage Examples
# =================================================================

def example_moderate_growth():
    """Moderate growth - ~5-6% monthly, lower risk."""
    from orchestrator.engine import CopyCatOrchestrator
    
    config = create_aggressive_config(
        strategy=GrowthStrategy.MODERATE,
        initial_balance=100.0,
        mode=TradingMode.SANDBOX,
    )
    
    orchestrator = CopyCatOrchestrator(config)
    return orchestrator


def example_aggressive_growth():
    """Aggressive growth - ~7-9% monthly, moderate risk."""
    from orchestrator.engine import CopyCatOrchestrator
    
    config = create_aggressive_config(
        strategy=GrowthStrategy.AGGRESSIVE,
        initial_balance=100.0,
        mode=TradingMode.SANDBOX,
    )
    
    orchestrator = CopyCatOrchestrator(config)
    return orchestrator


def example_very_aggressive():
    """Very aggressive - ~10-15% monthly, high risk."""
    from orchestrator.engine import CopyCatOrchestrator
    
    config = create_aggressive_config(
        strategy=GrowthStrategy.VERY_AGGRESSIVE,
        initial_balance=100.0,
        mode=TradingMode.SANDBOX,
    )
    
    orchestrator = CopyCatOrchestrator(config)
    return orchestrator


if __name__ == "__main__":
    # Print growth projections
    print("=" * 60)
    print("AGGRESSIVE GROWTH CONFIGURATIONS")
    print("=" * 60)
    
    for strategy in GrowthStrategy:
        print(f"\n{strategy.value.upper()}:")
        print("-" * 40)
        
        config = create_aggressive_config(strategy=strategy)
        
        print(f"  Position Size: {config.copy_trading.position_size_pct:.0%}")
        print(f"  Max Traders: {config.max_traders_to_copy}")
        print(f"  Min Win Rate: {config.trader_selection.min_win_rate:.0%}")
        print(f"  Max Drawdown: {config.trader_selection.max_drawdown:.0%}")
        print(f"  Kelly Method: {config.copy_trading.position_sizing_method}")
        
        # Project growth
        monthly_returns = {"moderate": 0.055, "aggressive": 0.08, "very_aggressive": 0.12}
        rate = monthly_returns[strategy.value]
        
        print(f"\n  Projected Growth (${100}):")
        print(f"    Month 3: ${100 * (1 + rate) ** 3:.0f}")
        print(f"    Month 6: ${100 * (1 + rate) ** 6:.0f}")
        print(f"    Month 12: ${100 * (1 + rate) ** 12:.0f}")
        print(f"    Month 24: ${100 * (1 + rate) ** 24:.0f}")
