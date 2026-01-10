#!/usr/bin/env python3
"""
Demo: Topic-Based Basket Trading
Demonstrates basket-based consensus trading system.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from api_clients.polymarket.data_api import DataAPIClient
from basket_trading import (
    BasketConfig,
    Topic,
)
from basket_trading.orchestrator import BasketTradingOrchestrator
from basket_trading.sandbox_runner import BasketSandboxRunner


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_geopolitics_basket():
    """Demo geopolitics basket trading."""
    print("\n" + "=" * 70)
    print("DEMO: Topic-Based Basket Trading")
    print("=" * 70 + "\n")

    data_client = DataAPIClient()

    config = BasketConfig(
        min_wallet_age_months=6,
        min_consensus_pct=0.80,
        max_price_band_pct=0.05,
        max_spread_pct=0.10,
        min_basket_size=10,
        max_basket_size=50,
    )

    runner = BasketSandboxRunner(
        data_client=data_client,
        config=config,
        initial_balance=10000.0,
    )

    geopolitics_wallets = [
        "0x1234567890abcdef1234567890abcdef1234567890ab",
        "0x2345678901abcdef1234567890abcdef1234567890cd",
        "0x345678901234567890abcdef1234567890def",
        "0x45678901234567890abcdef1234567890eff",
        "0x56789012345678901234567890abcdef12345678901",
        "0x6789012345678901234567890abcdef123456789012",
        "0x789012345678901234567890abcdef123456789023",
        "0x89012345678901234567890abcdef123456789034",
        "0x9012345678901234567890abcdef123456789045",
    ]

    topic_wallets = {
        Topic.GEOPOLITICS: geopolitics_wallets,
        Topic.POLITICS: geopolitics_wallets[:5],
        Topic.ELECTIONS: geopolitics_wallets[:7],
        Topic.CRYPTO: geopolitics_wallets[:4],
    }

    print(f"Configuring {len(topic_wallets)} topic baskets:")
    for topic, addresses in topic_wallets.items():
        print(f"  - {topic.value}: {len(addresses)} wallets")

    print("\nInitializing basket trading system...")
    await runner.orchestrator.initialize_topic_baskets(topic_wallets)
    await runner.orchestrator.detect_and_assign_clusters()
    await runner.orchestrator.rank_all_baskets()

    print("\nBasket Statistics:")
    stats = runner.orchestrator.get_basket_stats()
    for topic, stat in stats.items():
        status = "✓ Valid" if stat["is_valid"] else "✗ Invalid"
        print(f"  {topic.value}:")
        print(f"    Total wallets: {stat['total_wallets']}")
        print(f"    Effective wallets: {stat['effective_wallets']}")
        print(f"    Active wallets: {stat['active_wallets']}")
        print(f"    Status: {status}")

    print("\nScanning for consensus signals...")
    signals = await runner.orchestrator.scan_for_signals()

    print(f"\nFound {len(signals)} new signals:")
    for i, signal in enumerate(signals[:5], 1):
        print(f"\n  Signal #{i}:")
        print(f"    Market: {signal.market_title}")
        print(f"    Outcome: {signal.outcome}")
        print(f"    Side: {signal.side.value}")
        print(f"    Consensus: {signal.consensus_pct:.1%}")
        print(f"    Participants: {signal.participating_wallets}/{signal.total_wallets}")
        print(f"    Avg entry: ${signal.avg_entry_price:.4f}")
        print(f"    Price band: ${signal.price_band_low:.4f} - ${signal.price_band_high:.4f}")
        print(f"    Spread: {signal.spread_pct:.1%}")
        print(f"    Strength: {signal.signal_strength:.2f}")

    if signals:
        print("\nExecuting top 3 strongest signals...")
        results = await runner.orchestrator.execute_top_signals(limit=3)

        print("\nExecution Results:")
        for i, result in enumerate(results, 1):
            status = "✓ Executed" if result.executed else "✗ Rejected"
            print(f"  {status}: {signal.market_title}")
            if result.rejection_reason:
                print(f"    Reason: {result.rejection_reason}")
            else:
                print(f"    Price: ${result.execution_price:.4f}")
                print(f"    Qty: ${result.quantity:.2f}")

    print("\nPerformance Summary:")
    summary = runner.get_performance_summary()
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Successful: {summary['successful_trades']}")
    print(f"  P&L: ${summary['pnl']:.2f}")
    print(f"  Win rate: {summary['win_rate']:.1%}")
    print(f"  Portfolio value: ${summary['portfolio_value']:.2f}")
    print(f"  Positions: {summary['positions_count']}")
    print(f"  Exposure: {summary['exposure_pct']:.1%}")

    await data_client.close()
    await runner.stop()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70 + "\n")


async def demo_signal_filtering():
    """Demo signal filtering logic."""
    print("\n" + "=" * 70)
    print("DEMO: Signal Filtering and Validation")
    print("=" * 70 + "\n")

    config = BasketConfig()

    test_cases = [
        {
            "name": "Strong Signal",
            "consensus": 0.90,
            "price_band": 0.03,
            "spread": 0.04,
        },
        {
            "name": "Weak Consensus",
            "consensus": 0.70,
            "price_band": 0.04,
            "spread": 0.08,
        },
        {
            "name": "Wide Price Band",
            "consensus": 0.90,
            "price_band": 0.08,
            "spread": 0.05,
        },
        {
            "name": "Cooked Spread",
            "consensus": 0.90,
            "price_band": 0.03,
            "spread": 0.12,
        },
    ]

    print("Testing signal validation logic:\n")
    for test in test_cases:
        print(f"  {test['name']}:")
        print(f"    Consensus: {test['consensus']:.0%} (min: {config.min_consensus_pct:.0%})")
        print(f"    Price band: {test['price_band']:.0%} (max: {config.max_price_band_pct:.0%})")
        print(f"    Spread: {test['spread']:.0%} (max: {config.max_spread_pct:.0%})")

        would_pass = (
            test['consensus'] >= config.min_consensus_pct and
            test['price_band'] <= config.max_price_band_pct and
            test['spread'] <= config.max_spread_pct
        )

        status = "✓ PASS" if would_pass else "✗ FAIL"
        print(f"    Result: {status}\n")

    print("=" * 70 + "\n")


async def main():
    """Run all demos."""
    try:
        await demo_signal_filtering()
        await demo_geopolitics_basket()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
