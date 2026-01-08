#!/usr/bin/env python3
"""
Standalone Sandbox Mode Example.

This example shows how to run the CopyCat trading bot in sandbox mode
WITHOUT any API keys, wallet connections, or external credentials.

All data is simulated - perfect for testing and learning!

Usage:
    python example_sandbox_standalone.py
"""

import asyncio
from datetime import datetime


async def main():
    print("=" * 70)
    print("CopyCat Sandbox Mode - No Credentials Required!")
    print("=" * 70)
    print()
    print("This runs the trading bot using simulated data:")
    print("  - 50+ simulated markets (politics, sports, crypto, etc.)")
    print("  - 100+ simulated traders with realistic performance")
    print("  - Virtual portfolio with $10,000 starting balance")
    print("  - Copy trading simulation")
    print()
    print("No API keys needed. No wallet connections. No external requests.")
    print("=" * 70)
    print()

    # =========================================================================
    # Method 1: Using the Mock API Client Directly
    # =========================================================================
    print("[1/2] Using Mock API Client Directly")
    print("-" * 50)

    from api_clients.mock import MockMarketAPIClient

    # Create mock client - no credentials!
    client = MockMarketAPIClient(
        platform="polymarket",
        initial_balance=10000.0,
        num_markets=20,
        num_traders=30,
    )

    # Get some markets
    markets = await client.get_markets(limit=5)
    print(f"Markets available: {len(markets)}")
    for m in markets[:3]:
        price = m.get("outcome_price", {}).get("yes", 0)
        print(f"  - {m['question']}: ${price:.2f}")

    # Get some trades from traders
    trades = await client.get_trades(limit=5)
    print(f"\nRecent trades: {len(trades)}")
    for t in trades[:3]:
        print(f"  - {t.side.value.upper()} {t.quantity:.0f} @ ${t.price:.2f} ({t.outcome})")

    # Get trader's balance (simulated)
    balance = await client.get_balance("0x123456")
    print(f"\nBalance for 0x123456: ${balance:,.2f}")

    print()
    print("-" * 50)
    print()

    # =========================================================================
    # Method 2: Using the Orchestrator in Sandbox Mode
    # =========================================================================
    print("[2/2] Using Orchestrator in Sandbox Mode")
    print("-" * 50)

    from orchestrator.config import OrchestratorConfig, TradingMode
    from orchestrator.engine import CopyCatOrchestrator

    # Create config - sandbox mode, no API keys!
    config = OrchestratorConfig(
        mode=TradingMode.SANDBOX,
        max_traders_to_copy=5,
        max_traders_to_analyze_per_cycle=10,
        trader_data_refresh_interval_seconds=1,  # Fast for demo
    )

    # Initialize orchestrator (uses mock client automatically)
    orchestrator = CopyCatOrchestrator(config)

    print(f"Orchestrator initialized in {orchestrator.config.mode.value} mode")
    print(f"API clients: {len(orchestrator.api_clients)}")
    print(f"No API keys required!")

    # Run one trading cycle
    print("\nRunning trading cycle...")
    result = await orchestrator._run_trading_cycle()

    if result.success:
        print(f"✓ Cycle completed!")
        print(f"  Traders analyzed: {result.details.get('traders_analyzed', 0)}")
        print(f"  Copied traders: {result.details.get('copied_traders', 0)}")

        # Check performance
        status = orchestrator.get_status()
        print(f"\n  Portfolio Performance:")
        print(f"    Total PnL: ${status['total_pnl']:,.2f}")
        print(f"    Win Rate: {status['win_rate']:.1%}")
        print(f"    Trades Executed: {status['trades_executed']}")
    else:
        print(f"✗ Cycle failed: {result.message}")

    print()
    print("=" * 70)
    print("Sandbox mode complete!")
    print()
    print("Key features:")
    print("  • Runs completely offline")
    print("  • No API keys or credentials needed")
    print("  • Simulated markets and traders")
    print("  • Realistic trading simulation")
    print()
    print("To run a full simulation, use:")
    print("  orchestrator.start()")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
