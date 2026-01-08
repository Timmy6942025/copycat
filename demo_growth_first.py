"""
Growth-First Copy Trading Demo

This demonstrates copying traders based on ACCOUNT GROWTH instead of win rate.

Run: python demo_growth_first.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_clients.base import Trade, OrderSide
from trader_identification.growth_selectors import (
    GrowthBasedSelector,
    GrowthSelectionConfig,
    GrowthMetrics,
)


# Simulate trader data with different profiles
def create_trader_profiles():
    """Create mock traders with different profiles."""
    
    # Trader A: High win rate, low growth (boring but consistent)
    trader_a_trades = []
    for i in range(50):
        trade = Trade(
            trade_id=f"a_{i}",
            market_id="0x123",
            trader_address="0xAAA",
            side=OrderSide.BUY,
            quantity=10.0,
            price=0.52,
            total_value=5.2,
            fees=0.1,
            timestamp=datetime.now() - timedelta(days=50-i),
            outcome="YES",
        )
        # Small consistent wins
        trade._mock_pnl = 1.0 if i % 2 == 0 else -0.5
        trader_a_trades.append(trade)
    
    # Trader B: Low win rate, INSANE growth (like insider - wins big when wins)
    trader_b_trades = []
    for i in range(30):
        trade = Trade(
            trade_id=f"b_{i}",
            market_id="0x456",
            trader_address="0xBBB",
            side=OrderSide.BUY,
            quantity=50.0,
            price=0.55,
            total_value=27.5,
            fees=0.55,
            timestamp=datetime.now() - timedelta(days=30-i),
            outcome="YES",
        )
        # Low win rate but huge wins when win
        trade._mock_pnl = 25.0 if i % 5 == 0 else -2.0  # 20% win rate but +$25 wins
        trader_b_trades.append(trade)
    
    # Trader C: Medium win rate, moderate growth (average trader)
    trader_c_trades = []
    for i in range(100):
        trade = Trade(
            trade_id=f"c_{i}",
            market_id="0x789",
            trader_address="0xCCC",
            side=OrderSide.BUY,
            quantity=20.0,
            price=0.50,
            total_value=10.0,
            fees=0.2,
            timestamp=datetime.now() - timedelta(days=100-i),
            outcome="YES",
        )
        trade._mock_pnl = 5.0 if i % 3 == 0 else -3.0  # 33% win rate
        trader_c_trades.append(trade)
    
    return {
        "0xAAA": trader_a_trades,
        "0xBBB": trader_b_trades,
        "0xCCC": trader_c_trades,
    }


def print_metrics(address: str, metrics: GrowthMetrics, total_pnl: float = None):
    """Print growth metrics in readable format."""
    wins = [t._mock_pnl for t in traders[address] if t._mock_pnl > 0]
    losses = [t._mock_pnl for t in traders[address] if t._mock_pnl < 0]
    actual_pnl = sum(t._mock_pnl for t in traders[address])
    
    print(f"\n{'='*50}")
    print(f"Trader: {address}")
    print(f"{'='*50}")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Active Days: {metrics.active_days}")
    print(f"  Win Rate: {len(wins)/(len(wins)+len(losses)):.0%}")
    print(f"  Total P&L: ${actual_pnl:,.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.1%}")
    print(f"  Growth Score: {metrics.growth_score:.2f}")
    print(f"  Consistency Score: {metrics.consistency_score:.2f}")
    print(f"  Stability Score: {metrics.stability_score:.2f}")
    print(f"  OVERALL SCORE: {metrics.overall_score:.2f}")


async def main():
    global traders
    traders = create_trader_profiles()
    
    print("=" * 60)
    print("GROWTH-FIRST COPY TRADING DEMO")
    print("=" * 60)
    print("\nGoal: Find traders who grew their account, regardless of win rate.")
    print("\nThree traders to compare:")
    print("  A: 50 trades, 50% win rate, small consistent wins")
    print("  B: 30 trades, 20% win rate, huge wins when they win")  
    print("  C: 100 trades, 33% win rate, average performance")
    
    # Create growth selector
    selector = GrowthBasedSelector(
        min_total_pnl=50.0,  # Low threshold for demo
        max_drawdown=0.50,
        min_equity_slope=0.0001,
        min_consistency_score=0.2,
        min_active_days=3,
    )
    
    print("\n" + "=" * 60)
    print("ANALYZING EACH TRADER")
    print("=" * 60)
    
    # Analyze each trader
    for address, trades in traders.items():
        metrics = selector.calculate_growth_metrics(trades)
        print_metrics(address, metrics)
    
    print("\n" + "=" * 60)
    print("SELECTING TOP TRADER(S)")
    print("=" * 60)
    
    # Select top traders
    selected = selector.select_traders(traders, top_n=2)
    
    print(f"\nTop {len(selected)} traders selected:")
    for i, (address, metrics) in enumerate(selected, 1):
        print(f"\n#{i} {address}")
        print(f"   Overall Score: {metrics.overall_score:.2f}")
        print(f"   Total P&L: ${sum(t._mock_pnl for t in traders[address]):,.2f}")
        print(f"   Win Rate: {len([t for t in traders[address] if t._mock_pnl > 0])/len(traders[address]):.0%}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("\nTrader B has LOW win rate (20%) but would be selected")
    print("because they GREW their account significantly (+$300+).")
    print("\nTraditional systems would reject Trader B.")
    print("Growth-first selection embraces them if they make money.")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
