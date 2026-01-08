"""
Main Sandbox Runner - Orchestrates the entire sandbox simulation.
Coordinates data feeds, order execution, portfolio management, and performance tracking.

Supports real-time market data from:
- CoinGecko (cryptocurrency)
- Yahoo Finance (stocks)
- Polymarket (prediction markets)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from sandbox.config import (
    SandboxConfig, SimulationState, VirtualOrder, VirtualOrderResult,
    VirtualOrder, OrderStatus
)
from sandbox.executor import VirtualOrderExecutor
from sandbox.portfolio import VirtualPortfolioManager
from sandbox.analytics import PerformanceTracker, PerformanceReporter
from sandbox.data_providers import (
    UnifiedDataProvider,
    DataProviderConfig,
    CoinGeckoProvider,
    YahooFinanceProvider,
    PolymarketDataProvider,
)
from sandbox.data_providers.feed import RealtimeDataFeed, FeedConfig, FeedStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SandboxRunner:
    """Main orchestrator for sandbox simulation with real-time data support."""

    def __init__(
        self,
        config: SandboxConfig = None,
        data_provider_config: DataProviderConfig = None,
        feed_config: FeedConfig = None,
    ):
        """Initialize sandbox runner with optional data providers."""
        self.config = config or SandboxConfig()
        self.data_provider_config = data_provider_config or DataProviderConfig()
        self.feed_config = feed_config or FeedConfig()

        # Core components
        self.executor = VirtualOrderExecutor(self.config)
        self.portfolio_manager = VirtualPortfolioManager(self.config)
        self.tracker = PerformanceTracker(self.config.initial_balance)
        self.reporter = PerformanceReporter(self.config.results_storage_path)

        # State
        self.state = SimulationState(balance=self.config.initial_balance)
        self.state.total_value = self.config.initial_balance

        # Data providers
        self.data_provider: Optional[UnifiedDataProvider] = None
        self.data_feed: Optional[RealtimeDataFeed] = None

        # Callbacks
        self.is_running = False
        self.market_data_callback: Optional[Callable[[str], Dict[str, Any]]] = None
        self.order_callback: Optional[Callable[[VirtualOrder], None]] = None

    def enable_realtime_data(
        self,
        enable_crypto: bool = True,
        enable_stocks: bool = True,
        enable_prediction: bool = True,
    ):
        """Enable real-time data from configured providers."""
        self.data_provider_config.enable_crypto = enable_crypto
        self.data_provider_config.enable_stocks = enable_stocks
        self.data_provider_config.enable_prediction = enable_prediction

        self.data_provider = UnifiedDataProvider(self.data_provider_config)
        logger.info(f"Real-time data enabled: crypto={enable_crypto}, stocks={enable_stocks}, prediction={enable_prediction}")

    async def start_data_feed(self, market_ids: List[str]):
        """Start real-time data feed for specified markets."""
        if not self.data_provider:
            self.enable_realtime_data()

        if not self.data_provider:
            logger.warning("Data provider not available")
            return

        self.data_feed = RealtimeDataFeed(self.feed_config)

        # Set up callback to get market data from unified provider
        async def get_market_data(market_id: str) -> Dict[str, Any]:
            data = await self.data_provider.get_market_data(market_id)
            if data:
                return data.to_sandbox_format()
            return self.data_provider.get_sandbox_market_data(market_id)

        # Start the feed
        await self.data_feed.start(market_ids, get_market_data)

        # Set up price update callback to update portfolio
        def on_price_update(market_id: str, price: float):
            asyncio.create_task(self._handle_price_update(market_id, price))

        self.data_feed.on_price_update = on_price_update
        logger.info(f"Data feed started for {len(market_ids)} markets")

    async def stop_data_feed(self):
        """Stop the real-time data feed."""
        if self.data_feed:
            await self.data_feed.stop()
            self.data_feed = None
            logger.info("Data feed stopped")

    async def _handle_price_update(self, market_id: str, price: float):
        """Handle price update from data feed."""
        await self.update_market_prices({market_id: price})

    def set_market_data_callback(self, callback: Callable[[str], Dict[str, Any]]):
        """Set callback to provide market data for a market ID."""
        self.market_data_callback = callback

    def set_order_callback(self, callback: Callable[[VirtualOrder], None]):
        """Set callback for when orders are executed."""
        self.order_callback = callback

    async def execute_order(self, order: VirtualOrder) -> VirtualOrderResult:
        """Execute a virtual order through the simulation."""
        market_data = self._get_market_data(order.market_id)
        result = await self.executor.execute_virtual_order(order, market_data)

        if result.status == OrderStatus.FILLED.value:
            await self.portfolio_manager.execute_copy_trade(
                source_trade={
                    "trade_id": order.order_id,
                    "market_id": order.market_id,
                    "outcome": order.outcome if hasattr(order, 'outcome') else "YES",
                    "trader_address": order.source_trader,
                },
                trader_config={}
            )

        if self.order_callback:
            self.order_callback(order)

        return result

    def _get_market_data(self, market_id: str) -> Dict[str, Any]:
        """Get market data for a market."""
        if self.market_data_callback:
            return self.market_data_callback(market_id)

        if self.data_feed:
            latest_price = self.data_feed.get_latest_price(market_id)
            if latest_price is not None:
                return {
                    "market_id": market_id,
                    "current_price": latest_price,
                    "previous_price": latest_price,
                    "volatility": 0.02,
                }

        if self.data_provider:
            return self.data_provider.get_sandbox_market_data(market_id)

        return {
            "market_id": market_id,
            "current_price": 0.5,
            "previous_price": 0.5,
            "volatility": 0.02,
        }

    async def update_market_prices(self, updates: Dict[str, float]):
        """Update market prices for all positions."""
        self.portfolio_manager.update_market_prices(updates)
        self.tracker._update_equity_curve()

    def get_state(self) -> SimulationState:
        """Get current simulation state."""
        summary = self.portfolio_manager.get_portfolio_summary()
        self.state.balance = summary.balance
        self.state.total_value = summary.total_value
        return self.state

    def get_portfolio_summary(self):
        """Get portfolio summary."""
        return self.portfolio_manager.get_portfolio_summary()

    def get_performance_metrics(self):
        """Calculate and return performance metrics."""
        return self.tracker.calculate_metrics()

    def generate_report(self) -> str:
        """Generate performance report."""
        metrics = self.get_performance_metrics()
        return self.reporter.generate_report(
            metrics=metrics,
            trades=self.portfolio_manager.completed_trades,
            equity_curve=self.tracker.equity_curve
        )

    def save_report(self, filename: str = None) -> str:
        """Generate and save performance report."""
        report = self.generate_report()
        return self.reporter.save_report(report, filename)

    def reset(self):
        """Reset simulation to initial state."""
        self.state = SimulationState(balance=self.config.initial_balance)
        self.state.total_value = self.config.initial_balance

        self.portfolio_manager = VirtualPortfolioManager(self.config)
        self.tracker = PerformanceTracker(self.config.initial_balance)

    async def close(self):
        """Close all connections and clean up."""
        await self.stop_data_feed()

        if self.data_provider:
            await self.data_provider.close()
            self.data_provider = None

        if self.executor and hasattr(self.executor, 'close'):
            await self.executor.close()


async def run_demo_with_realtime_data():
    """Run a demo simulation with real-time data."""
    from sandbox.config import SandboxConfig, VirtualOrder

    print("=" * 60)
    print("Sandbox Simulation with Real-Time Data")
    print("=" * 60)

    config = SandboxConfig(initial_balance=10000)
    runner = SandboxRunner(config)

    # Enable real-time data
    runner.enable_realtime_data(
        enable_crypto=True,
        enable_stocks=True,
        enable_prediction=True,
    )

    # Markets to track
    market_ids = ["bitcoin", "AAPL", "ethereum"]

    # Start data feed
    await runner.start_data_feed(market_ids)

    # Wait for initial data
    print("\nWaiting for market data...")
    await asyncio.sleep(2)

    # Get latest prices
    if runner.data_feed:
        prices = runner.data_feed.get_all_latest_prices()
        print("\nLatest Prices:")
        for market_id, price in prices.items():
            print(f"  {market_id}: ${price:,.2f}")

    # Execute some demo orders
    print("\nExecuting demo orders...")
    markets = ["bitcoin", "AAPL", "ethereum"]

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

        await asyncio.sleep(0.5)

    # Get results
    summary = runner.get_portfolio_summary()
    metrics = runner.get_performance_metrics()

    print(f"\nPortfolio Summary:")
    print(f"  Balance: ${summary.balance:,.2f}")
    print(f"  Positions Value: ${summary.positions_value:,.2f}")
    print(f"  Total Value: ${summary.total_value:,.2f}")
    print(f"  Unrealized P&L: ${summary.unrealized_pnl:,.2f}")

    # Generate report
    report = runner.save_report("realtime_demo_report.md")
    print(f"\nReport saved to: {report}")

    # Clean up
    await runner.close()

    print("\nDemo completed!")


async def run_demo_without_realtime():
    """Run a demo simulation without real-time data (fallback mode)."""
    from sandbox.config import SandboxConfig, VirtualOrder

    print("=" * 60)
    print("Sandbox Simulation (Fallback Mode)")
    print("=" * 60)

    config = SandboxConfig(initial_balance=10000)
    runner = SandboxRunner(config)

    # Set up fallback market data callback
    def get_market_data(market_id: str) -> Dict[str, Any]:
        return {
            "market_id": market_id,
            "current_price": 0.5 + hash(market_id) % 50 / 100,
            "previous_price": 0.5,
            "volatility": 0.02,
        }

    runner.set_market_data_callback(get_market_data)

    # Execute demo orders
    print("\nExecuting demo orders...")
    markets = ["market_a", "market_b", "market_c"]

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

    # Get results
    summary = runner.get_portfolio_summary()
    metrics = runner.get_performance_metrics()

    print(f"\nPortfolio Summary:")
    print(f"  Balance: ${summary.balance:,.2f}")
    print(f"  Positions Value: ${summary.positions_value:,.2f}")
    print(f"  Total Value: ${summary.total_value:,.2f}")
    print(f"  Unrealized P&L: ${summary.unrealized_pnl:,.2f}")

    # Generate report
    report = runner.save_report("fallback_demo_report.md")
    print(f"\nReport saved to: {report}")

    print("\nDemo completed!")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='CopyCat Sandbox Simulation')
    parser.add_argument(
        '--mode',
        choices=['realtime', 'fallback'],
        default='fallback',
        help='Simulation mode (realtime uses live APIs, fallback uses simulated data)'
    )
    parser.add_argument(
        '--balance',
        type=float,
        default=10000,
        help='Initial balance'
    )

    args = parser.parse_args()

    if args.mode == 'realtime':
        await run_demo_with_realtime_data()
    else:
        await run_demo_without_realtime()


if __name__ == "__main__":
    asyncio.run(main())
