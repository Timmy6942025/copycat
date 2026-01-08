#!/usr/bin/env python3
"""
Demo script for the CopyCat Sandbox TUI Dashboard.

This script demonstrates the TUI dashboard functionality with sample data.
"""

import asyncio
from datetime import datetime
from sandbox import SandboxRunner, SandboxConfig, VirtualOrder, run_dashboard


async def demo_dashboard():
    """Run a quick demo of the dashboard."""
    print("=" * 60)
    print("CopyCat Sandbox TUI Dashboard Demo")
    print("=" * 60)
    print()
    print("This demo will show you how to use the TUI dashboard.")
    print()
    print("To run the full interactive dashboard:")
    print("  python -m sandbox.dashboard")
    print()
    print("Or via the CLI:")
    print("  python -m sandbox dashboard --dashboard-mode rich")
    print()
    print("For simple output mode:")
    print("  python -m sandbox dashboard --dashboard-mode simple")
    print()
    print("Press Ctrl+C to exit at any time.")
    print()
    
    # Initialize a sandbox runner
    config = SandboxConfig(initial_balance=10000)
    runner = SandboxRunner(config)
    
    # Set up fallback market data
    def get_market_data(market_id: str):
        return {
            "market_id": market_id,
            "current_price": 0.5 + hash(market_id) % 50 / 100,
            "previous_price": 0.5,
            "volatility": 0.02,
        }
    runner.set_market_data_callback(get_market_data)
    
    # Execute some demo trades
    markets = ["bitcoin", "AAPL", "ethereum", "BTC-YES", "TRUMP-2024"]
    
    print("Executing demo trades...")
    for i, market_id in enumerate(markets):
        order = VirtualOrder(
            order_id=f"demo_{i}_{datetime.utcnow().timestamp()}",
            market_id=market_id,
            side="buy",
            quantity=100 + i * 50,
            order_type="market",
            timestamp=datetime.utcnow(),
        )
        
        result = await runner.execute_order(order)
        print(f"  Order {order.order_id[:12]} - {result.status} @ ${result.average_price:.4f}")
        
        # Simulate price movement
        await runner.update_market_prices({market_id: 0.55 + i * 0.1})
    
    # Show summary
    summary = runner.get_portfolio_summary()
    metrics = runner.get_performance_metrics()
    
    print()
    print("Portfolio Summary:")
    print(f"  Balance: ${summary.balance:,.2f}")
    print(f"  Positions Value: ${summary.positions_value:,.2f}")
    print(f"  Total Value: ${summary.total_value:,.2f}")
    print(f"  Unrealized P&L: ${summary.unrealized_pnl:,.2f}")
    print()
    print("Performance Metrics:")
    print(f"  Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:.2%})")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print()
    print("=" * 60)
    print("Demo complete! Run the interactive dashboard for more features.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(demo_dashboard())
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
