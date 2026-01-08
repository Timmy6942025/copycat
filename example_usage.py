"""
Example usage script for the sandbox simulation module.
Demonstrates how to use the sandbox for paper trading simulation.
"""

import asyncio
from sandbox import (
    SandboxRunner, SandboxConfig, VirtualOrder, MarketData,
    OrderSide, PerformanceMetrics
)


async def main():
    """Run example sandbox simulation."""
    print("CopyCat Sandbox Simulation Demo")
    print("=" * 50)

    # Create configuration
    config = SandboxConfig(
        initial_balance=10000.0,
        max_position_size_pct=0.10,
        simulate_slippage=True,
        simulate_fees=True
    )

    # Initialize runner
    runner = SandboxRunner(config)

    # Define market data callback
    def get_market_data(market_id: str) -> MarketData:
        return MarketData(
            market_id=market_id,
            current_price=0.50 + hash(market_id) % 50 / 100,
            previous_price=0.50
        )

    runner.set_market_data_callback(get_market_data)

    # Execute demo trades
    demo_markets = [
        ("will_trump_win_2024", 500.0),
        ("btc_above_100k", 300.0),
        ("fed_cuts_rates_q1", 200.0),
    ]

    print("\nExecuting Demo Trades:")
    print("-" * 50)

    for i, (market_id, amount) in enumerate(demo_markets):
        order = VirtualOrder(
            market_id=market_id,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            quantity=amount,
            outcome="YES",
            source_trader_id=f"trader_{i}"
        )

        result = await runner.execute_order(order)

        status_symbol = "✓" if result.status.value == "filled" else "✗"
        print(f"{status_symbol} {market_id}: {result.status.value} "
              f"@ ${result.average_price:.3f} "
              f"(Qty: {result.filled_quantity:.1f})")

        # Simulate price movement
        new_price = 0.45 + hash(market_id) % 60 / 100
        await runner.update_market_prices({market_id: new_price})

    # Get results
    print("\n" + "=" * 50)
    print("Simulation Results:")
    print("-" * 50)

    summary = runner.get_portfolio_summary()
    metrics = runner.get_performance_metrics()

    print(f"Portfolio Value: ${summary.total_value:,.2f}")
    print(f"Cash Balance:    ${summary.balance:,.2f}")
    print(f"Positions Value: ${summary.positions_value:,.2f}")
    print(f"Positions:       {summary.position_count}")
    print()
    print(f"Total P&L:       ${metrics.total_pnl:+,.2f} ({metrics.total_pnl_pct:+.2%})")
    print(f"Win Rate:        {metrics.win_rate:.1%}")
    print(f"Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown:    {metrics.max_drawdown:.2%}")
    print(f"Total Trades:    {metrics.total_trades}")
    print(f"Profitable:      {metrics.winning_trades}")
    print(f"Losing:          {metrics.losing_trades}")

    # Generate and save report
    print("\n" + "=" * 50)
    report_path = runner.save_report("demo_report.md")
    print(f"Report saved to: {report_path}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
