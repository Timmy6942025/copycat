"""
Aggressive Configuration
Higher risk, higher potential returns.
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
        growth_min_total_pnl=200.0,
        growth_max_drawdown=0.40,
        growth_min_active_days=1,
        min_win_rate=0.35,
        min_trades=5,
    ),
    
    copy_trading=CopyTradingConfig(
        position_sizing_method="kelly",
        position_size_pct=0.10,
        kelly_fraction=0.25,
        base_position_size=50.0,
        max_position_size_pct=0.15,
    ),
    
    bot_filter=BotFilterOrchestratorConfig(
        hft_max_hold_time_seconds=1.0,
        hft_min_trades_per_minute=5,
        min_hft_score_to_exclude=0.8,
    ),
    
    sandbox=SandboxConfigOrchestrator(
        initial_balance=100.0,
        simulate_slippage=True,
        simulate_fees=True,
    ),
    
    max_traders_to_copy=20,
    max_traders_to_analyze_per_cycle=200,
)
