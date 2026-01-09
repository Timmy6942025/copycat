"""
Balanced Configuration
Risk-balanced approach with moderate growth potential.
"""

from orchestrator.config import (
    OrchestratorConfig, TradingMode, MarketPlatform,
    TraderSelectionConfig, BotFilterOrchestratorConfig,
    CopyTradingConfig, SandboxConfigOrchestrator, SelectionMode,
)

config = OrchestratorConfig(
    mode=TradingMode.SANDBOX,
    platform=MarketPlatform.POLYMARKET,
    
    trader_selection=TraderSelectionConfig(
        mode=SelectionMode.GROWTH,
        growth_min_total_pnl=100.0,
        growth_max_drawdown=0.25,
        growth_min_active_days=3,
        min_win_rate=0.45,
        min_trades=15,
    ),
    
    copy_trading=CopyTradingConfig(
        position_sizing_method="percentage",
        position_size_pct=0.05,
        base_position_size=25.0,
        max_position_size_pct=0.08,
    ),
    
    bot_filter=BotFilterOrchestratorConfig(
        hft_max_hold_time_seconds=2.0,
        hft_min_trades_per_minute=3,
        min_hft_score_to_exclude=0.6,
    ),
    
    sandbox=SandboxConfigOrchestrator(
        initial_balance=100.0,
        simulate_slippage=True,
        simulate_fees=True,
    ),
    
    max_traders_to_copy=10,
    max_traders_to_analyze_per_cycle=100,
)
