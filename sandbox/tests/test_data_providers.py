"""
Integration test for sandbox data providers.
Tests the real-time data integration with free APIs.
"""

import asyncio
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/home/timmy/copycat')


def test_data_provider_imports():
    """Test that all data provider modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Data Provider Imports")
    print("=" * 60)

    try:
        from sandbox.data_providers import (
            CoinGeckoProvider,
            YahooFinanceProvider,
            PolymarketDataProvider,
            UnifiedDataProvider,
            DataProviderConfig,
        )
        print("\n✓ All data provider modules imported successfully!")
        return True

    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        print("  Some dependencies may be missing. Run: pip install yfinance")
        return False


def test_sandbox_runner_import():
    """Test that sandbox runner can be imported."""
    print("\n" + "=" * 60)
    print("Testing Sandbox Runner Import")
    print("=" * 60)

    try:
        from sandbox.runner import SandboxRunner
        from sandbox.config import SandboxConfig

        print("\n✓ Sandbox runner imported successfully!")
        return True

    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False


async def test_sandbox_runner_basic():
    """Test basic sandbox runner functionality without external APIs."""
    print("\n" + "=" * 60)
    print("Testing Sandbox Runner (Fallback Mode)")
    print("=" * 60)

    from sandbox.runner import SandboxRunner
    from sandbox.config import SandboxConfig

    config = SandboxConfig(initial_balance=10000)
    runner = SandboxRunner(config)

    # Set up fallback market data callback
    def get_market_data(market_id: str):
        return {
            "market_id": market_id,
            "current_price": 0.5 + hash(market_id) % 50 / 100,
            "previous_price": 0.5,
            "volatility": 0.02,
        }

    runner.set_market_data_callback(get_market_data)

    try:
        # Get state
        state = runner.get_state()
        print(f"\nInitial balance: ${state.balance:,.2f}")

        # Execute demo orders
        from sandbox.config import VirtualOrder

        markets = ["market_a", "market_b", "market_c"]
        print("\nExecuting demo orders:")

        for i, market_id in enumerate(markets):
            order = VirtualOrder(
                order_id=f"test_{i}_{datetime.utcnow().timestamp()}",
                market_id=market_id,
                side="buy",
                quantity=100 + i * 50,
                order_type="market",
                timestamp=datetime.utcnow(),
            )

            result = await runner.execute_order(order)
            print(f"  Order {order.order_id[:12]} - {result.status} @ ${result.average_price:.4f}")

        # Get portfolio summary
        summary = runner.get_portfolio_summary()
        print(f"\nPortfolio Summary:")
        print(f"  Balance: ${summary.balance:,.2f}")
        print(f"  Positions Value: ${summary.positions_value:,.2f}")
        print(f"  Total Value: ${summary.total_value:,.2f}")

        # Generate report
        report = runner.save_report("integration_test_report.md")
        print(f"\nReport saved to: {report}")

        print("\n✓ Sandbox runner test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Sandbox runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_realtime_data_feed():
    """Test real-time data feed with simulated data."""
    print("\n" + "=" * 60)
    print("Testing Real-Time Data Feed (Simulated)")
    print("=" * 60)

    from sandbox.data_providers.feed import RealtimeDataFeed, FeedConfig
    from sandbox.runner import SandboxRunner
    from sandbox.config import SandboxConfig

    # Create feed
    feed_config = FeedConfig(update_interval=0.5)
    feed = RealtimeDataFeed(feed_config)

    # Mock provider callback
    async def mock_provider(market_id: str):
        return {
            "market_id": market_id,
            "current_price": 100 + hash(market_id) % 10,
            "volume_24h": 1000000,
            "volatility": 0.02,
        }

    try:
        # Start feed
        markets = ["market_a", "market_b"]
        await feed.start(markets, mock_provider)

        print(f"\nStarted feed for {len(markets)} markets")

        # Wait for updates
        await asyncio.sleep(2)

        # Get state
        state = feed.get_state()
        print(f"  Status: {state.status.value}")
        print(f"  Markets tracked: {state.markets_tracked}")
        print(f"  Updates received: {state.updates_received}")

        # Get latest prices
        prices = feed.get_all_latest_prices()
        print(f"\nLatest prices:")
        for market_id, price in prices.items():
            print(f"  {market_id}: ${price:,.2f}")

        # Stop feed
        await feed.stop()

        print("\n✓ Real-time data feed test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Real-time data feed test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if feed.get_state().status.value != "stopped":
            await feed.stop()


async def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("SANDBOX DATA PROVIDER INTEGRATION TESTS")
    print("=" * 60)

    results = []

    # Test imports
    results.append(("Data Provider Imports", test_data_provider_imports()))

    # Test runner import
    results.append(("Sandbox Runner Import", test_sandbox_runner_import()))

    # Run async tests
    try:
        results.append(("Sandbox Runner (Fallback)", await test_sandbox_runner_basic()))
    except Exception as e:
        print(f"\n! Sandbox runner test failed: {e}")
        results.append(("Sandbox Runner (Fallback)", False))

    try:
        results.append(("Real-Time Data Feed", await test_realtime_data_feed()))
    except Exception as e:
        print(f"\n! Real-time data feed test failed: {e}")
        results.append(("Real-Time Data Feed", False))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed > 0:
        print("\nNote: Some tests may require additional dependencies:")
        print("  pip install yfinance aiohttp")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
