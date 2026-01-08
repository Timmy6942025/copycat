#!/usr/bin/env python3
"""
Comprehensive Sandbox Module Demo - Paper Trading with Real Market Data.

This example demonstrates the complete sandbox functionality:
1. Configuration setup
2. Order execution simulation
3. Portfolio management
4. Performance tracking
5. Report generation
6. Backtesting capabilities

The sandbox allows you to test copy trading strategies with virtual money
while using real-time market data from Polymarket or simulated data.

Usage:
    python demo_sandbox.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import datetime, timedelta
import uuid


async def demo_basic_order_execution():
    """Demo 1: Basic order execution in sandbox mode."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Order Execution")
    print("=" * 70)
    
    from sandbox.config import SandboxConfig, VirtualOrder
    from sandbox.executor import VirtualOrderExecutor
    
    # Setup configuration
    config = SandboxConfig(
        initial_balance=10000.0,
        simulate_slippage=True,
        simulate_fees=True,
        fee_model="polymarket"
    )
    
    executor = VirtualOrderExecutor(config)
    
    # Create a virtual order
    order = VirtualOrder(
        order_id=f"demo_order_{uuid.uuid4().hex[:8]}",
        market_id="polymarket_us_election_2024",
        side="buy",
        quantity=100.0,  # $100 position
        order_type="market",
        timestamp=datetime.utcnow(),
        trader_total_volume_30d=50000  # Simulated volume for fee calculation
    )
    
    # Simulated market data (would come from real API in production)
    market_data = {
        "market_id": "polymarket_us_election_2024",
        "current_price": 0.52,  # YES outcome at $0.52
        "previous_price": 0.50,
        "volatility": 0.02,
        "orderbook": {
            "bids": [{"price": 0.51, "size": 5000}],
            "asks": [{"price": 0.53, "size": 5000}],
            "last_trade_price": 0.52
        }
    }
    
    print(f"Order: {order.side.upper()} ${order.quantity:.2f} of {order.market_id}")
    print(f"Market Price: ${market_data['current_price']:.2f}")
    
    # Execute the order
    result = await executor.execute_virtual_order(order, market_data)
    
    print(f"\nExecution Result:")
    print(f"  Status: {result.status}")
    print(f"  Filled Quantity: {result.filled_quantity:.2f}")
    print(f"  Average Price: ${result.average_price:.4f}")
    print(f"  Fees: ${result.total_fees:.4f}")
    print(f"  Slippage: {result.slippage:.4f} ({result.slippage*100:.2f}%)")
    
    return result


async def demo_portfolio_management():
    """Demo 2: Portfolio management and position tracking."""
    print("\n" + "=" * 70)
    print("DEMO 2: Portfolio Management")
    print("=" * 70)
    
    from sandbox.config import SandboxConfig
    from sandbox.portfolio import VirtualPortfolioManager
    
    # Initialize portfolio manager
    config = SandboxConfig(initial_balance=10000.0)
    portfolio = VirtualPortfolioManager(config)
    
    print(f"Initial Balance: ${portfolio.balance:,.2f}")
    
    # Simulate a copy trade (copying a successful trader)
    source_trade = {
        "trade_id": f"copy_trade_{uuid.uuid4().hex[:8]}",
        "market_id": "polymarket_crypto_btc_2024",
        "outcome": "YES",
        "trader_address": "0xSuccessTrader123",
        "price": 0.65,
        "quantity": 50.0
    }
    
    trader_config = {
        "position_sizing_method": "percentage",
        "position_size_pct": 0.02,  # 2% of portfolio
        "trader_score": 0.85  # High confidence trader
    }
    
    print(f"\nCopying trade from {source_trade['trader_address'][:8]}...")
    print(f"Trader Confidence: {trader_config['trader_score']:.0%}")
    print(f"Position Size: {trader_config['position_size_pct']:.1%} of portfolio")
    
    # Execute copy trade
    result = await portfolio.execute_copy_trade(source_trade, trader_config)
    
    print(f"\nCopy Trade Result:")
    print(f"  Status: {result.status}")
    print(f"  Position Size: ${result.position_size:.2f}")
    print(f"  Execution Price: ${result.execution_price:.4f}")
    
    # Get portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"  Balance: ${summary.balance:,.2f}")
    print(f"  Positions Value: ${summary.positions_value:,.2f}")
    print(f"  Total Value: ${summary.total_value:,.2f}")
    print(f"  Unrealized P&L: ${summary.unrealized_pnl:,.2f}")
    print(f"  Position Count: {summary.position_count}")
    print(f"  Exposure: {summary.exposure_pct:.1%}")
    
    # Simulate price movement
    print(f"\nSimulating price movement...")
    portfolio.update_market_prices({
        "polymarket_crypto_btc_2024": 0.72  # Price moved up
    })
    
    summary = portfolio.get_portfolio_summary()
    print(f"  New Unrealized P&L: ${summary.unrealized_pnl:,.2f}")
    
    return portfolio


async def demo_performance_tracking():
    """Demo 3: Performance metrics and analytics."""
    print("\n" + "=" * 70)
    print("DEMO 3: Performance Tracking")
    print("=" * 70)
    
    from sandbox.analytics import PerformanceTracker, PerformanceReporter
    from sandbox.config import VirtualTrade
    
    # Initialize performance tracker
    tracker = PerformanceTracker(initial_balance=10000.0)
    
    print(f"Initial Balance: ${tracker.initial_balance:,.2f}")
    
    # Record some sample trades (simulating copy trading results)
    sample_trades = [
        VirtualTrade(
            trade_id="trade_001",
            market_id="market_a",
            outcome="YES",
            quantity=100,
            entry_price=0.50,
            exit_price=0.60,
            profit=10.0,
            roi=0.20,
            timestamp=datetime.utcnow() - timedelta(days=5),
            source_trader="0xTraderA",
            fees=0.50,
            slippage=0.01
        ),
        VirtualTrade(
            trade_id="trade_002",
            market_id="market_b",
            outcome="YES",
            quantity=50,
            entry_price=0.30,
            exit_price=0.25,
            profit=-2.50,
            roi=-0.17,
            timestamp=datetime.utcnow() - timedelta(days=3),
            source_trader="0xTraderB",
            fees=0.25,
            slippage=0.02
        ),
        VirtualTrade(
            trade_id="trade_003",
            market_id="market_c",
            outcome="YES",
            quantity=75,
            entry_price=0.80,
            exit_price=0.85,
            profit=3.75,
            roi=0.06,
            timestamp=datetime.utcnow() - timedelta(days=1),
            source_trader="0xTraderA",
            fees=0.40,
            slippage=0.01
        ),
    ]
    
    print(f"\nRecording {len(sample_trades)} sample trades...")
    for trade in sample_trades:
        tracker.record_trade(trade)
        profit_str = f"+${trade.profit:.2f}" if trade.profit > 0 else f"-${abs(trade.profit):.2f}"
        print(f"  {trade.source_trader[:8]}: {profit_str} on {trade.market_id}")
    
    # Calculate performance metrics
    print(f"\nCalculating performance metrics...")
    metrics = tracker.calculate_metrics()
    
    print(f"\nPerformance Summary:")
    print(f"  Starting Balance: ${metrics.starting_balance:,.2f}")
    print(f"  Ending Balance: ${metrics.ending_balance:,.2f}")
    print(f"  Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:.2%})")
    print(f"  Annualized Return: {metrics.annualized_return:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.1%} ({metrics.winning_trades} wins / {metrics.total_trades} trades)")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    
    # Generate performance report
    print(f"\nGenerating performance report...")
    reporter = PerformanceReporter(output_path="./sandbox_results")
    report = reporter.generate_report(
        metrics=metrics,
        trades=tracker.trades,
        equity_curve=tracker.equity_curve
    )
    
    print(f"  Report generated ({len(report)} characters)")
    
    # Save the report
    report_path = reporter.save_report("demo_performance_report.md")
    print(f"  Report saved to: {report_path}")
    
    return metrics


async def demo_backtesting():
    """Demo 4: Backtesting with historical data."""
    print("\n" + "=" * 70)
    print("DEMO 4: Backtesting")
    print("=" * 70)
    
    from sandbox.backtest import BacktestEngine, BacktestConfig
    from datetime import datetime, timedelta
    
    # Configure backtest
    backtest_config = BacktestConfig(
        starting_balance=10000.0,
        output_path="./backtest_results"
    )
    
    engine = BacktestEngine(backtest_config)
    
    print(f"Backtest Configuration:")
    print(f"  Starting Balance: ${backtest_config.starting_balance:,.2f}")
    print(f"  Output Path: {backtest_config.output_path}")
    
    # Define backtest period (last 30 days)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nBacktest Period:")
    print(f"  Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"  End: {end_date.strftime('%Y-%m-%d')}")
    
    # Trader selection configuration
    trader_selection_config = {
        "min_win_rate": 0.50,
        "min_trades": 10,
        "max_bot_score": 0.3,
        "max_insider_score": 0.5,
        "min_reputation_score": 0.6
    }
    
    # Copy trading configuration
    copy_trading_config = {
        "position_sizing_method": "scaled",
        "position_size_pct": 0.02,
        "kelly_fraction": 0.25
    }
    
    print(f"\nTrader Selection Criteria:")
    for key, value in trader_selection_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nCopy Trading Configuration:")
    for key, value in copy_trading_config.items():
        print(f"  {key}: {value}")
    
    # Run backtest (note: will use simulated data since no historical API)
    print(f"\nRunning backtest...")
    print(f"  (Using simulated historical data for demonstration)")
    
    result = await engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        trader_selection_config=trader_selection_config,
        copy_trading_config=copy_trading_config
    )
    
    print(f"\nBacktest Results:")
    print(f"  Starting Balance: ${result.starting_balance:,.2f}")
    print(f"  Ending Balance: ${result.ending_balance:,.2f}")
    print(f"  Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:.2%})")
    print(f"  Total Trades: {len(result.trades)}")
    
    if result.metrics:
        print(f"\nPerformance Metrics:")
        print(f"  Win Rate: {result.metrics.win_rate:.1%}")
        print(f"  Profit Factor: {result.metrics.profit_factor:.2f}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
    
    return result


async def demo_sandbox_runner():
    """Demo 5: Using the SandboxRunner orchestrator."""
    print("\n" + "=" * 70)
    print("DEMO 5: SandboxRunner Orchestrator")
    print("=" * 70)
    
    from sandbox.runner import SandboxRunner
    from sandbox.config import SandboxConfig, VirtualOrder
    from datetime import datetime
    
    # Initialize sandbox runner
    config = SandboxConfig(
        initial_balance=10000.0,
        simulate_slippage=True,
        simulate_fees=True,
        results_storage_path="./sandbox_results"
    )
    
    runner = SandboxRunner(config)
    
    print(f"SandboxRunner initialized:")
    print(f"  Initial Balance: ${config.initial_balance:,.2f}")
    print(f"  Slippage Simulation: {config.simulate_slippage}")
    print(f"  Fee Simulation: {config.simulate_fees}")
    
    # Set up market data callback (simulated)
    def get_simulated_market_data(market_id: str):
        # Simulated market data based on market ID
        base_prices = {
            "btc_usd": 65000.0,
            "eth_usd": 3500.0,
            "aapl_usd": 175.0,
        }
        base_price = base_prices.get(market_id, 0.5)
        
        return {
            "market_id": market_id,
            "current_price": base_price,
            "previous_price": base_price * 0.99,
            "volatility": 0.02,
        }
    
    runner.set_market_data_callback(get_simulated_market_data)
    print(f"\nMarket data callback configured (simulated data)")
    
    # Execute some demo orders
    print(f"\nExecuting demo orders...")
    demo_markets = [
        ("btc_usd", "buy", 100.0),
        ("eth_usd", "buy", 200.0),
        ("aapl_usd", "sell", 50.0),
    ]
    
    for i, (market_id, side, quantity) in enumerate(demo_markets):
        order = VirtualOrder(
            order_id=f"runner_demo_{i}_{uuid.uuid4().hex[:8]}",
            market_id=market_id,
            side=side,
            quantity=quantity,
            order_type="market",
            timestamp=datetime.utcnow(),
        )
        
        result = await runner.execute_order(order)
        print(f"  Order {i+1}: {order.market_id} {order.side.upper()} ${order.quantity:.0f} -> {result.status}")
    
    # Get final state
    summary = runner.get_portfolio_summary()
    metrics = runner.get_performance_metrics()
    
    print(f"\nFinal Portfolio State:")
    print(f"  Balance: ${summary.balance:,.2f}")
    print(f"  Positions Value: ${summary.positions_value:,.2f}")
    print(f"  Total Value: ${summary.total_value:,.2f}")
    print(f"  Unrealized P&L: ${summary.unrealized_pnl:,.2f}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Total P&L: ${metrics.total_pnl:,.2f}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    
    # Generate and save report
    report_path = runner.save_report("runner_demo_report.md")
    print(f"\nReport saved to: {report_path}")
    
    return runner


async def main():
    """Run all sandbox demos."""
    print("=" * 70)
    print("SANDBOX MODULE COMPREHENSIVE DEMO")
    print("Paper Trading with Real-Time Market Data")
    print("=" * 70)
    print()
    print("This demo showcases the complete sandbox functionality:")
    print("  1. Order Execution Simulation")
    print("  2. Portfolio Management")
    print("  3. Performance Tracking & Analytics")
    print("  4. Backtesting Capabilities")
    print("  5. SandboxRunner Orchestrator")
    print()
    print("All simulations use virtual money - no real funds required!")
    print("=" * 70)
    
    # Run all demos
    await demo_basic_order_execution()
    await demo_portfolio_management()
    await demo_performance_tracking()
    await demo_backtesting()
    await demo_sandbox_runner()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  ✓ Sandbox module provides risk-free trading simulation")
    print("  ✓ Can use real market data from Polymarket (or simulated)")
    print("  ✓ Comprehensive performance tracking and analytics")
    print("  ✓ Backtesting capabilities for strategy validation")
    print("  ✓ Full orchestrator for automated paper trading")
    print()
    print("Next Steps:")
    print("  • Integrate with real Polymarket API for live data")
    print("  • Add trader identification and copy trading logic")
    print("  • Implement risk management rules")
    print("  • Deploy for continuous paper trading simulation")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
