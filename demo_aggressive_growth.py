"""
Quick Start: Aggressive Growth Mode

Run this to start CopyCat with aggressive growth settings.

Usage:
    python demo_aggressive_growth.py [--strategy moderate|aggressive|very-aggressive] [--balance 100]
    
Example:
    python demo_aggressive_growth.py --strategy aggressive --balance 50
"""

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.config import TradingMode
from orchestrator.config_aggressive import (
    create_aggressive_config,
    GrowthStrategy,
    AggressiveGrowthConfig,
)
from orchestrator.engine import CopyCatOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="CopyCat Aggressive Growth Mode")
    parser.add_argument(
        "--strategy",
        choices=["moderate", "aggressive", "very-aggressive"],
        default="aggressive",
        help="Growth strategy (default: aggressive)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=100.0,
        help="Initial balance (default: $100)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use LIVE mode (real money, not recommended for testing)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Convert strategy string to enum
    strategy_map = {
        "moderate": GrowthStrategy.MODERATE,
        "aggressive": GrowthStrategy.AGGRESSIVE,
        "very-aggressive": GrowthStrategy.VERY_AGGRESSIVE,
    }
    strategy = strategy_map[args.strategy]
    
    # Determine mode
    mode = TradingMode.LIVE if args.live else TradingMode.SANDBOX
    
    print("=" * 60)
    print(f"COPYCAT AGGRESSIVE GROWTH - {args.strategy.upper()}")
    print("=" * 60)
    print(f"Mode: {'LIVE (REAL MONEY)' if args.live else 'SANDBOX (PAPER TRADING)'}")
    print(f"Starting Balance: ${args.balance:,.2f}")
    print("=" * 60)
    
    # Create configuration
    config = create_aggressive_config(
        strategy=strategy,
        initial_balance=args.balance,
        mode=mode,
    )
    
    print("\nConfiguration:")
    print(f"  Position Size: {config.copy_trading.position_size_pct:.0%}")
    print(f"  Max Traders: {config.max_traders_to_copy}")
    print(f"  Min Win Rate: {config.trader_selection.min_win_rate:.0%}")
    print(f"  Sizing Method: {config.copy_trading.position_sizing_method}")
    print(f"  Quick Resolve: {config.boost_mode.prefer_quick_resolve}")
    print(f"  Quick Threshold: {config.boost_mode.quick_resolve_threshold_hours}h")
    print(f"  Quick Multiplier: x{config.boost_mode.quick_resolve_multiplier}")
    
    if not args.live:
        print(f"\nSandbox Balance: ${config.sandbox.initial_balance:,.2f}")
    
    # Create orchestrator
    orchestrator = CopyCatOrchestrator(config)
    
    print("\n" + "=" * 60)
    print("STARTING ORCHESTRATOR")
    print("=" * 60)
    
    # Start
    await orchestrator.start()
    
    # Wait a bit for initial scanning
    await asyncio.sleep(5)
    
    # Get status
    status = orchestrator.get_status()
    
    print(f"\nStatus:")
    print(f"  Running: {status['is_running']}")
    print(f"  Mode: {status['mode']}")
    print(f"  Copied Traders: {len(status.get('copied_traders', {}))}")
    print(f"  Trades Executed: {status.get('trades_executed', 0)}")
    print(f"  Total P&L: ${status.get('total_pnl', 0):,.2f}")
    
    print("\n" + "=" * 60)
    print("Bot is running! Press Ctrl+C to stop.")
    print("=" * 60)
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(30)
            status = orchestrator.get_status()
            print(f"[{status['mode']}] P&L: ${status.get('total_pnl', 0):,.2f} | "
                  f"Traders: {len(status.get('copied_traders', {}))} | "
                  f"Trades: {status.get('trades_executed', 0)}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await orchestrator.stop()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
