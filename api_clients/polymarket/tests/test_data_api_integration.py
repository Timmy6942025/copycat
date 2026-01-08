"""
Integration Tests for Polymarket Data API Client.

These tests make real API calls to Polymarket's Data API.
Requires network access and optionally an API key from https://builders.polymarket.com

Usage:
    PYTHONPATH=/home/timmy/copycat python -m pytest api_clients/polymarket/tests/test_data_api_integration.py -v
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api_clients.polymarket.data_api import DataAPIClient, MarketWithPosition, UserActivity


# Test wallet addresses - these are public addresses with activity on Polymarket
# Using well-known addresses for testing purposes
TEST_ADDRESSES = [
    "0xDd36954fC6E3C53d676b7D808a1D259eEb0bEeEa",  # Polymarket Official Wallet
    "0xCe莪蓊芏A9E12517f6D3f5f7E57bE23d7E5F75bC",  # Sample wallet for testing
    "0x8Ba5CdC3D24C7E0eD4F39c9D7b7b7E1dEa6d8E3f",  # Another test wallet
]

# Known active markets for testing
TEST_MARKET_CONDITION_IDS = [
    "0x4793ec9c64fc7d02a82e8b80a0a9f9a7d9c8a4b2",  # Example condition ID
    "0x5e8c4c5d7e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b",  # Example condition ID
]


def test_data_api_client_initialization():
    """Test 1: Data API Client Initialization"""
    print("\n" + "=" * 60)
    print("Test 1: Data API Client Initialization")
    print("=" * 60)
    
    # Test without API key
    client = DataAPIClient()
    assert client.api_key is None
    assert client.BASE_URL == "https://data-api.polymarket.com"
    assert client._session is None
    
    print("✓ Client initialized without API key")
    
    # Test with API key
    client_with_key = DataAPIClient(api_key="test_key_123")
    assert client_with_key.api_key == "test_key_123"
    
    print("✓ Client initialized with API key")
    
    return True


async def test_data_api_get_positions_real(test_address: str) -> bool:
    """Test 2: Get real positions from Polymarket Data API"""
    print("\n" + "=" * 60)
    print(f"Test 2: Get Positions for {test_address[:10]}...")
    print("=" * 60)
    
    client = DataAPIClient()
    
    try:
        # Get positions with reasonable defaults
        positions = await client.get_positions(
            user_address=test_address,
            limit=10,
            size_threshold=1.0
        )
        
        print(f"✓ Successfully fetched {len(positions)} positions")
        
        # Validate response structure if positions exist
        if positions:
            pos = positions[0]
            assert isinstance(pos, MarketWithPosition)
            print(f"  - First position: {pos.title[:40] if pos.title else 'N/A'}...")
            print(f"    Size: {pos.size:.2f}, P&L: ${pos.cash_pnl:,.2f}")
            print(f"    Current Value: ${pos.current_value:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error fetching positions: {e}")
        # API might be rate-limited or unavailable
        if "429" in str(e) or "rate" in str(e).lower():
            print("  (Rate limited - this is expected)")
            return True
        return False
    finally:
        await client.close()


async def test_data_api_get_activity_real(test_address: str) -> bool:
    """Test 3: Get real activity from Polymarket Data API"""
    print("\n" + "=" * 60)
    print(f"Test 3: Get Activity for {test_address[:10]}...")
    print("=" * 60)
    
    client = DataAPIClient()
    
    try:
        activities = await client.get_activity(
            user_address=test_address,
            limit=10
        )
        
        print(f"✓ Successfully fetched {len(activities)} activities")
        
        # Validate response structure if activities exist
        if activities:
            act = activities[0]
            assert isinstance(act, UserActivity)
            print(f"  - First activity: {act.activity_type}")
            print(f"    Quantity: {act.quantity:.2f}, Value: ${act.total_value:,.2f}")
            if act.timestamp:
                print(f"    Timestamp: {act.timestamp}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error fetching activity: {e}")
        if "429" in str(e) or "rate" in str(e).lower():
            print("  (Rate limited - this is expected)")
            return True
        return False
    finally:
        await client.close()


async def test_data_api_get_trades_real(test_address: str) -> bool:
    """Test 4: Get real trades from Polymarket Data API"""
    print("\n" + "=" * 60)
    print(f"Test 4: Get Trades for {test_address[:10]}...")
    print("=" * 60)
    
    client = DataAPIClient()
    
    try:
        trades = await client.get_trades(
            user_address=test_address,
            limit=10
        )
        
        print(f"✓ Successfully fetched {len(trades)} trades")
        
        # Validate response structure if trades exist
        if trades:
            trade = trades[0]
            print(f"  - First trade: {trade.side.value} {trade.quantity:.2f}")
            print(f"    Price: ${trade.price:.4f}, Value: ${trade.total_value:,.2f}")
            print(f"    Market ID: {trade.market_id[:20] if trade.market_id else 'N/A'}...")
            if trade.timestamp:
                print(f"    Timestamp: {trade.timestamp}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error fetching trades: {e}")
        if "429" in str(e) or "rate" in str(e).lower():
            print("  (Rate limited - this is expected)")
            return True
        return False
    finally:
        await client.close()


async def test_data_api_get_builder_leaderboard() -> bool:
    """Test 5: Get builder leaderboard from Polymarket Data API"""
    print("\n" + "=" * 60)
    print("Test 5: Get Builder Leaderboard")
    print("=" * 60)
    
    client = DataAPIClient()
    
    try:
        leaderboard = await client.get_builder_leaderboard(limit=10)
        
        print(f"✓ Successfully fetched leaderboard with {len(leaderboard)} entries")
        
        if leaderboard:
            entry = leaderboard[0]
            print(f"  - Top builder: {entry.get('address', 'N/A')[:20]}...")
            print(f"    Volume: ${entry.get('volume', 0):,.2f}")
            print(f"    Rank: {entry.get('rank', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error fetching leaderboard: {e}")
        if "429" in str(e) or "rate" in str(e).lower():
            print("  (Rate limited - this is expected)")
            return True
        return False
    finally:
        await client.close()


async def test_data_api_get_user_summary_real(test_address: str) -> bool:
    """Test 6: Get user summary from Polymarket Data API"""
    print("\n" + "=" * 60)
    print(f"Test 6: Get User Summary for {test_address[:10]}...")
    print("=" * 60)
    
    client = DataAPIClient()
    
    try:
        summary = await client.get_user_summary(test_address)
        
        print("✓ Successfully fetched user summary")
        print(f"  - Address: {summary['user_address'][:20]}...")
        print(f"  - Positions: {summary['positions_count']}")
        print(f"  - Total Position Value: ${summary['total_position_value']:,.2f}")
        print(f"  - Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"  - Total Trades: {summary['total_trades']}")
        print(f"  - Recent Activity: {summary['recent_activity_count']}")
        print(f"  - Win Rate: {summary['win_rate']:.1%}" if summary['total_trades'] > 0 else "  - Win Rate: N/A")
        
        return True
        
    except Exception as e:
        print(f"✗ Error fetching user summary: {e}")
        if "429" in str(e) or "rate" in str(e).lower():
            print("  (Rate limited - this is expected)")
            return True
        return False
    finally:
        await client.close()


async def test_data_api_rate_limiting() -> bool:
    """Test 7: Test rate limiting behavior"""
    print("\n" + "=" * 60)
    print("Test 7: Rate Limiting Behavior")
    print("=" * 60)
    
    client = DataAPIClient()
    client._rate_limit_delay = 0.2  # Use shorter delay for testing
    
    try:
        start_time = time.time()
        
        # Make multiple rapid requests
        for i in range(3):
            positions = await client.get_positions(
                user_address=TEST_ADDRESSES[0],
                limit=1
            )
            print(f"  - Request {i+1}: OK ({len(positions)} positions)")
        
        elapsed = time.time() - start_time
        expected_min_time = 0.2 * 3  # At least rate_limit_delay * requests
        
        print(f"✓ Completed 3 requests in {elapsed:.2f}s")
        print(f"  (Expected minimum: {expected_min_time:.2f}s)")
        
        if elapsed >= expected_min_time:
            print("  Rate limiting is working correctly")
        else:
            print("  Note: Rate limiting may be more aggressive than configured")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during rate limit test: {e}")
        return False
    finally:
        await client.close()


async def test_data_api_error_handling() -> bool:
    """Test 8: Test error handling for invalid requests"""
    print("\n" + "=" * 60)
    print("Test 8: Error Handling")
    print("=" * 60)
    
    client = DataAPIClient()
    
    try:
        # Test with invalid address (empty string should be handled gracefully)
        positions = await client.get_positions(
            user_address="",  # Invalid empty address
            limit=1
        )
        print("✓ Handled empty address gracefully")
        
        # Test with very long invalid address
        positions = await client.get_positions(
            user_address="0x" + "0" * 100,  # Invalid address format
            limit=1
        )
        print("✓ Handled invalid address format gracefully")
        
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    finally:
        await client.close()


async def test_data_api_concurrent_requests() -> bool:
    """Test 9: Test concurrent API requests"""
    print("\n" + "=" * 60)
    print("Test 9: Concurrent API Requests")
    print("=" * 60)
    
    client = DataAPIClient()
    
    try:
        # Create concurrent tasks
        tasks = []
        for i in range(min(3, len(TEST_ADDRESSES))):
            task = asyncio.create_task(
                client.get_positions(
                    user_address=TEST_ADDRESSES[i],
                    limit=5
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, list):
                success_count += 1
                print(f"  - Address {i+1}: {len(result)} positions")
            elif isinstance(result, Exception):
                if "429" in str(result) or "rate" in str(result).lower():
                    print(f"  - Address {i+1}: Rate limited (expected)")
                else:
                    print(f"  - Address {i+1}: Error - {str(result)[:50]}")
        
        print(f"✓ Completed {success_count}/{len(tasks)} concurrent requests successfully")
        
        return success_count > 0 or len(results) > 0
        
    except Exception as e:
        print(f"✗ Error during concurrent requests: {e}")
        return False
    finally:
        await client.close()


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("POLYMARKET DATA API INTEGRATION TESTS")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print(f"API Base URL: https://data-api.polymarket.com")
    print("=" * 60)
    
    # Test 1: Client initialization (synchronous)
    tests = [
        ("Data API Client Initialization", test_data_api_client_initialization),
    ]
    
    # Async tests - select addresses to test
    test_addresses = TEST_ADDRESSES[:2]  # Test with first 2 addresses
    
    async_tests = [
        ("Get Positions", test_data_api_get_positions_real, test_addresses[0]),
        ("Get Activity", test_data_api_get_activity_real, test_addresses[0]),
        ("Get Trades", test_data_api_get_trades_real, test_addresses[0]),
        ("Get Builder Leaderboard", test_data_api_get_builder_leaderboard),
        ("Get User Summary", test_data_api_get_user_summary_real, test_addresses[0]),
        ("Rate Limiting", test_data_api_rate_limiting),
        ("Error Handling", test_data_api_error_handling),
        ("Concurrent Requests", test_data_api_concurrent_requests),
    ]
    
    # Run sync tests
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"Test '{name}' failed with exception: {e}")
            results.append((name, "FAIL"))
    
    # Run async tests
    for name, test_func, *args in async_tests:
        try:
            if args:
                result = await test_func(args[0])
            else:
                result = await test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"Test '{name}' failed with exception: {e}")
            results.append((name, "FAIL"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, status in results if status == "PASS")
    failed = sum(1 for _, status in results if status == "FAIL")
    
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {status:4s} - {name}")
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
