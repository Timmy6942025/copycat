# Conservative Configuration Example
# Low risk, steady growth - good for beginners
# Copy to config.py and modify as needed

from orchestrator.config import (
    OrchestratorConfig, TradingMode, MarketPlatform,
    TraderSelectionConfig, BotFilterOrchestratorConfig,
    CopyTradingConfig, SandboxConfigOrchestrator, SelectionMode,
)

config = OrchestratorConfig(
    # Basic settings
    mode=TradingMode.SANDBOX,  # Change to LIVE for real trading
    platform=MarketPlatform.POLYMARKET,
    
    # Trader selection - conservative criteria
    trader_selection=TraderSelectionConfig(
        mode=SelectionMode.GROWTH,
        growth_min_total_pnl=50.0,        # Lower profit threshold
        growth_max_drawdown=0.15,         # Low drawdown tolerance
        growth_min_active_days=5,         # More established traders
        min_win_rate=0.50,
        min_trades=20,
    ),
    
    # Copy trading - smaller positions
    copy_trading=CopyTradingConfig(
        position_sizing_method="percentage",
        position_size_pct=0.03,           # 3% per trade (conservative)
        base_position_size=10.0,
        max_position_size_pct=0.05,
    ),
    
    # Bot filtering
    bot_filter=BotFilterOrchestratorConfig(
        hft_max_hold_time_seconds=5.0,
        hft_min_trades_per_minute=2,
        min_hft_score_to_exclude=0.5,
    ),
    
    # Sandbox settings
    sandbox=SandboxConfigOrchestrator(
        initial_balance=100.0,
        simulate_slippage=True,
        simulate_fees=True,
    ),
    
    # Limits
    max_traders_to_copy=5,
    max_traders_to_analyze_per_cycle=50,
)

print("Conservative config loaded - edit this file to customize")
