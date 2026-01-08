"""
Mock/Simulated Market API Client for Sandbox Mode.

Implements the full MarketAPIClient interface using simulated data.
This allows sandbox mode to run identically to live mode without any:
- API keys
- External HTTP requests
- Wallet connections
- Exchange credentials

All data is procedurally generated to be realistic but completely fake.
"""

import asyncio
import hashlib
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .base import (
    MarketAPIClient,
    MarketData,
    OrderBook,
    Trade,
    Trader,
    TraderPerformance,
    Position,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
)

logger = logging.getLogger(__name__)


# Seeded random for reproducible results
SEED = 42


class MockMarketAPIClient(MarketAPIClient):
    """
    Mock API client that simulates Polymarket/Kalshi behavior.

    Implements the same interface as real API clients but:
    - Requires no API keys
    - Makes no external HTTP requests
    - Uses procedurally generated realistic data
    """

    def __init__(
        self,
        platform: str = "polymarket",
        seed: int = SEED,
        initial_balance: float = 10000.0,
        num_markets: int = 50,
        num_traders: int = 100,
    ):
        """Initialize mock API client with simulated data."""
        super().__init__(api_key=None)
        self.platform_name_val = platform
        self.seed = seed
        self.initial_balance = initial_balance

        # Initialize random with seed for reproducibility
        random.seed(seed)

        # Generate simulated data
        self._markets = self._generate_markets(num_markets)
        self._traders = self._generate_traders(num_traders)
        self._trades = self._generate_trades()
        self._orders: Dict[str, List[Order]] = {}
        self._positions: Dict[str, List[Position]] = {}
        self._balances: Dict[str, float] = {}

        # Track trade history for each trader
        self._trader_trade_history: Dict[str, List[Trade]] = {
            t["address"]: [] for t in self._traders
        }

        logger.info(f"Mock {platform} API initialized with {len(self._markets)} markets, {len(self._traders)} traders")

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return self.platform_name_val

    # =========================================================================
    # Market Data Methods
    # =========================================================================

    async def get_markets(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get list of markets."""
        markets = self._markets

        # Filter by category if specified
        if category:
            markets = [m for m in markets if m.get("category") == category]

        # Apply pagination
        start = min(offset, len(markets))
        end = min(offset + limit, len(markets))

        return markets[start:end]

    async def get_market_data(self, market_id: str) -> Optional[MarketData]:
        """Get market data for a specific market."""
        market = self._find_market(market_id)
        if not market:
            return None

        # Generate dynamic price data with some movement
        base_price = market.get("outcome_price", {}).get("yes", 0.5)
        volatility = 0.02

        # Add some simulated price movement
        price_change = random.uniform(-volatility, volatility)
        current_price = max(0.01, min(0.99, base_price + price_change))

        return MarketData(
            market_id=market_id,
            current_price=current_price,
            previous_price=base_price,
            mid_price=current_price,
            best_bid=current_price - 0.01,
            best_ask=current_price + 0.01,
            spread=0.02,
            volume_24h=market.get("volume_24h", 10000),
            liquidity=market.get("liquidity", 50000),
            volatility=volatility,
            last_updated=datetime.utcnow(),
        )

    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get order book for a specific market."""
        market = self._find_market(market_id)
        if not market:
            return None

        current_price = market.get("outcome_price", {}).get("yes", 0.5)

        # Generate bid/ask levels
        bids = []
        asks = []

        for i in range(10):
            bid_price = current_price - 0.01 * (i + 1) - random.uniform(0, 0.005)
            ask_price = current_price + 0.01 * (i + 1) + random.uniform(0, 0.005)

            if bid_price > 0:
                bids.append({
                    "price": round(bid_price, 4),
                    "size": random.randint(100, 10000),
                })

            if ask_price < 1:
                asks.append({
                    "price": round(ask_price, 4),
                    "size": random.randint(100, 10000),
                })

        return OrderBook(
            market_id=market_id,
            bids=bids[:10],
            asks=asks[:10],
            last_updated=datetime.utcnow(),
        )

    # =========================================================================
    # Trader Methods
    # =========================================================================

    async def get_trades(
        self,
        market_id: Optional[str] = None,
        trader_address: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """Get trades with optional filters."""
        trades = self._trades

        # Filter by market
        if market_id:
            trades = [t for t in trades if t.market_id == market_id]

        # Filter by trader
        if trader_address:
            trades = [t for t in trades if t.trader_address == trader_address]

        # Filter by time range
        if start_time:
            trades = [t for t in trades if t.timestamp >= start_time]
        if end_time:
            trades = [t for t in trades if t.timestamp <= end_time]

        # Sort by timestamp (newest first) and apply limit
        trades = sorted(trades, key=lambda t: t.timestamp, reverse=True)
        return trades[:limit]

    async def get_trader_info(self, trader_address: str) -> Optional[Trader]:
        """Get trader profile information."""
        trader = self._find_trader(trader_address)
        if not trader:
            return None

        return Trader(
            address=trader_address,
            username=trader.get("username"),
            avatar_url=trader.get("avatar_url"),
            bio=trader.get("bio"),
            is_verified=trader.get("is_verified", False),
            is_market_maker=trader.get("is_market_maker", False),
            total_trades=trader.get("total_trades", 0),
            total_volume=trader.get("total_volume", 0),
            first_trade_timestamp=trader.get("first_trade_timestamp"),
            last_trade_timestamp=trader.get("last_trade_timestamp"),
        )

    async def get_trader_positions(
        self, trader_address: str, market_id: Optional[str] = None
    ) -> List[Position]:
        """Get trader's current positions."""
        trader = self._find_trader(trader_address)
        if not trader:
            return []

        positions = self._positions.get(trader_address, [])

        if market_id:
            positions = [p for p in positions if p.market_id == market_id]

        return positions

    async def get_trader_orders(
        self,
        trader_address: str,
        market_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> List[Order]:
        """Get trader's orders."""
        orders = self._orders.get(trader_address, [])

        if market_id:
            orders = [o for o in orders if o.market_id == market_id]

        if status:
            orders = [o for o in orders if o.status == status]

        return orders

    # =========================================================================
    # Trading Methods
    # =========================================================================

    async def create_order(
        self,
        market_id: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        limit_price: Optional[float] = None,
        outcome: str = "YES",
    ) -> Order:
        """Create a new order (simulated - doesn't actually trade)."""
        order_id = str(uuid.uuid4())

        # Determine if order fills immediately (market orders always fill)
        if order_type == OrderType.MARKET:
            status = OrderStatus.FILLED
            filled_qty = quantity
            avg_price = limit_price or 0.5
        else:
            status = OrderStatus.OPEN
            filled_qty = 0
            avg_price = 0

        order = Order(
            order_id=order_id,
            market_id=market_id,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            filled_quantity=filled_qty,
            average_price=avg_price,
            status=status,
            fees=quantity * 0.02 if status == OrderStatus.FILLED else 0,  # 2% fee simulation
            timestamp=datetime.utcnow(),
            outcome=outcome,
        )

        # Store order
        if order.trader_address not in self._orders:
            self._orders[order.trader_address] = []
        self._orders[order.trader_address].append(order)

        logger.info(f"Mock order created: {order_id} {side.value} {quantity} {market_id}")

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        # Find the order across all traders
        for orders in self._orders.values():
            for order in orders:
                if order.order_id == order_id and order.status == OrderStatus.OPEN:
                    order.status = OrderStatus.CANCELLED
                    logger.info(f"Mock order cancelled: {order_id}")
                    return True

        return False

    async def get_balance(self, trader_address: str) -> float:
        """Get trader's balance."""
        if trader_address not in self._balances:
            self._balances[trader_address] = self.initial_balance
        return self._balances[trader_address]

    async def get_trader_performance(
        self,
        trader_address: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[TraderPerformance]:
        """Calculate trader performance from trades."""
        trades = await self.get_trades(trader_address=trader_address)

        if not trades:
            return None

        # Calculate performance metrics
        winning_trades = []
        losing_trades = []

        for trade in trades:
            # For prediction markets, a "winning" trade depends on outcome
            if trade.outcome == "YES" and trade.side == OrderSide.BUY and trade.price > 0.5:
                winning_trades.append(trade)
            elif trade.outcome == "NO" and trade.side == OrderSide.SELL and trade.price < 0.5:
                winning_trades.append(trade)
            else:
                losing_trades.append(trade)

        total_pnl = sum(t.total_value - t.fees for t in winning_trades) - \
                    sum(t.total_value + t.fees for t in losing_trades)

        total_volume = sum(t.total_value for t in trades)
        win_rate = len(winning_trades) / len(trades) if trades else 0

        return TraderPerformance(
            trader_address=trader_address,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / total_volume if total_volume > 0 else 0,
            win_rate=win_rate,
            profit_factor=(
                sum(t.total_value for t in winning_trades) /
                sum(t.total_value for t in losing_trades)
                if losing_trades and winning_trades else 0
            ),
            sharpe_ratio=random.uniform(0.5, 2.0),  # Simulated
            max_drawdown=random.uniform(0.05, 0.3),  # Simulated
            avg_hold_time_hours=random.uniform(1, 168),  # Simulated
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=sum(t.total_value for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            avg_loss=sum(t.total_value for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            total_volume=total_volume,
        )

    # =========================================================================
    # Data Generation Helpers
    # =========================================================================

    def _find_market(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Find a market by ID."""
        for market in self._markets:
            if market.get("id") == market_id or market_id in market.get("clobTokenIds", []):
                return market
        return None

    def _find_trader(self, trader_address: str) -> Optional[Dict[str, Any]]:
        """Find a trader by address."""
        for trader in self._traders:
            if trader.get("address") == trader_address:
                return trader
        return None

    def _generate_markets(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic mock markets."""
        categories = [
            "Politics", "Sports", "Crypto", "Science", "Economics",
            "Entertainment", "Technology", "Health", "Environment"
        ]

        topics = {
            "Politics": [
                "US Election 2024", "UK Election", "France Election",
                "Germany Coalition", "China Taiwan", "Israel Gaza"
            ],
            "Sports": [
                "Super Bowl", "World Cup", "NBA Finals", "Olympics",
                "Wimbledon", "US Open", "Tour de France"
            ],
            "Crypto": [
                "Bitcoin $100k", "Ethereum Merge", "Solana Adoption",
                "DeFi Regulation", "NFT Market", "Crypto ETFs"
            ],
            "Science": [
                "AI Alignment", "Climate Change", "Space Exploration",
                "CRISPR Cure", "Fusion Energy", "Quantum Computing"
            ],
            "Economics": [
                "US Recession", "Inflation Target", "Interest Rates",
                "GDP Growth", "Unemployment", "Housing Market"
            ],
            "Entertainment": [
                "Oscar Winners", "Grammy Winners", "Box Office",
                "Streaming Wars", "TV Show Renewals", "Celebrity News"
            ],
            "Technology": [
                "iPhone Release", "GPT-5 Release", "Metaverse",
                "Autonomous Vehicles", "5G Adoption", "Cloud Market"
            ],
            "Health": [
                "Pandemic End", "New Treatment", "Life Expectancy",
                "Mental Health", "Obesity Rates", "Vaccine Efficacy"
            ],
            "Environment": [
                "Net Zero 2050", "Electric Vehicles", "Carbon Tax",
                "Renewable Energy", "Biodiversity", "Ocean Cleanup"
            ],
        }

        markets = []
        used_titles = set()

        for i in range(count):
            # Pick a random category and topic
            category = random.choice(categories)
            topic_list = topics.get(category, ["General"])
            topic = random.choice(topic_list)

            # Create unique title
            base_title = f"{topic} - {random.randint(1, 100)}"
            if base_title in used_titles:
                base_title = f"{topic} - {random.randint(100, 1000)}"
            used_titles.add(base_title)

            # Generate price (biased towards center for realism)
            yes_price = max(0.05, min(0.95, random.triangular(0.2, 0.8, 0.5)))
            no_price = 1 - yes_price

            # Generate volume
            volume = random.uniform(1000, 100000)
            if random.random() < 0.1:  # Some markets are highly active
                volume *= random.uniform(10, 100)

            market = {
                "id": f"market_{i}",
                "slug": base_title.lower().replace(" ", "-").replace("$", ""),
                "question": f"Will {topic} happen?",
                "category": category,
                "outcome_price": {"yes": round(yes_price, 4), "no": round(no_price, 4)},
                "volume_24h": round(volume, 2),
                "volume": round(volume * random.uniform(1, 10), 2),
                "liquidity": round(volume * random.uniform(0.5, 2), 2),
                "clobTokenIds": [f"token_{i}_yes", f"token_{i}_no"],
                "active": random.random() < 0.9,
                "created": (datetime.utcnow() - timedelta(days=random.randint(1, 90))).isoformat(),
                "end_date": (datetime.utcnow() + timedelta(days=random.randint(1, 365))).isoformat(),
            }

            markets.append(market)

        return markets

    def _generate_traders(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic mock traders."""
        adjectives = [
            "Crypto", "Bullish", "Bearish", "Lucky", "Smart", "Quick",
            "Sharp", "Bold", "Calm", "Wise", "Swift", "Keen", "Sharp"
        ]
        nouns = [
            "Trader", "Hunter", "Seeker", "Trader", "Whale", "Dolphin",
            "Trader", "Pro", "Master", "Guru", "Expert", "Analyst", "Player"
        ]

        traders = []

        for i in range(count):
            username = f"{random.choice(adjectives)}{random.choice(nouns)}{random.randint(1, 999)}"

            # Generate performance stats
            total_trades = random.randint(10, 500)
            win_rate = max(0.3, min(0.8, random.triangular(0.35, 0.75, 0.55)))
            total_volume = random.uniform(1000, 100000)

            first_trade = datetime.utcnow() - timedelta(days=random.randint(30, 365))
            last_trade = datetime.utcnow() - timedelta(days=random.randint(0, 7))

            trader = {
                "address": f"0x{trader_address(i)}",
                "username": username,
                "avatar_url": f"https://api.dicebear.com/7.x/avataaars/svg?seed={username}",
                "bio": f"Experienced {random.choice(adjectives).lower()} trader in prediction markets.",
                "is_verified": random.random() < 0.1,  # 10% are verified
                "is_market_maker": random.random() < 0.05,  # 5% are market makers
                "total_trades": total_trades,
                "total_volume": round(total_volume, 2),
                "first_trade_timestamp": first_trade,
                "last_trade_timestamp": last_trade,
                "win_rate": win_rate,
            }

            traders.append(trader)

        return traders

    def _generate_trades(self) -> List[Trade]:
        """Generate mock trades from all traders across all markets."""
        trades = []

        for trader in self._traders:
            address = trader["address"]
            num_trades = min(trader["total_trades"], 50)  # Limit per trader for performance

            for _ in range(num_trades):
                # Pick a random market
                market = random.choice(self._markets)

                # Generate trade
                side = random.choice(list(OrderSide))
                outcome = "YES" if random.random() < market["outcome_price"]["yes"] else "NO"

                # Price depends on outcome
                base_price = market["outcome_price"]["yes"] if outcome == "YES" else market["outcome_price"]["no"]
                price = max(0.01, min(0.99, base_price + random.uniform(-0.05, 0.05)))
                quantity = random.uniform(10, 1000)
                total_value = quantity * price

                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    market_id=market["id"],
                    trader_address=address,
                    side=side,
                    quantity=round(quantity, 4),
                    price=round(price, 4),
                    total_value=round(total_value, 4),
                    fees=round(total_value * 0.02, 4),  # 2% fee
                    timestamp=trader["last_trade_timestamp"] - timedelta(
                        minutes=random.randint(0, 10000)
                    ),
                    outcome=outcome,
                    transaction_hash=f"0x{uuid.uuid4().hex[:64]}",
                )

                trades.append(trade)

        # Sort by timestamp
        trades.sort(key=lambda t: t.timestamp, reverse=True)

        return trades

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def _generate_address(self) -> str:
        """Generate a fake Ethereum address."""
        return f"0x{uuid.uuid4().hex[:40]}"

    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.5

        wins = 0
        for trade in trades:
            if trade.outcome == "YES" and trade.side == OrderSide.BUY:
                wins += 1
            elif trade.outcome == "NO" and trade.side == OrderSide.SELL:
                wins += 1

        return wins / len(trades)


def trader_address(index: int) -> str:
    """Generate a deterministic fake address for a trader index."""
    hash_input = f"trader_{index}_{SEED}"
    hash_bytes = hashlib.md5(hash_input.encode()).hexdigest()
    return hash_bytes[:40]


def create_mock_client(
    platform: str = "polymarket",
    initial_balance: float = 10000.0,
    num_markets: int = 50,
    num_traders: int = 100,
) -> MockMarketAPIClient:
    """Factory function to create a mock API client."""
    return MockMarketAPIClient(
        platform=platform,
        initial_balance=initial_balance,
        num_markets=num_markets,
        num_traders=num_traders,
    )
