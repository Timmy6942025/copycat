"""
Kalshi API client implementation.
Provides access to market data and trading for Kalshi exchange.
"""

import os
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..base import (
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


class KalshiAPIClient(MarketAPIClient):
    """
    Kalshi API client.
    Provides access to market data and trading for Kalshi exchange.
    """

    BASE_URL = "https://api.kalshi.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        demo: bool = False,
    ):
        """Initialize Kalshi API client."""
        super().__init__(api_key)
        self.api_key = api_key or os.getenv("KALSHI_API_KEY")
        self.demo = demo
        self._session: Optional[aiohttp.ClientSession] = None

        # Set rate limit based on tier (Basic tier: 20 read/s, 10 write/s)
        self.set_rate_limit(20)

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "kalshi"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to Kalshi API."""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"

        await asyncio.sleep(self._rate_limit_delay)

        async with session.request(method, url, params=params, json=data) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"Kalshi API error: {response.status} - {error_text}")
            return await response.json()

    # Market Data Endpoints

    async def get_markets(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get list of markets."""
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category

        data = await self._request("GET", "/exchange/v1/markets", params=params)
        return data.get("markets", data.get("data", []))

    async def get_market_data(self, market_id: str) -> Optional[MarketData]:
        """Get market data for a specific market."""
        try:
            data = await self._request("GET", f"/exchange/v1/markets/{market_id}")
            market = data.get("market", data)

            return MarketData(
                market_id=market_id,
                current_price=market.get("current_price", 0.5),
                previous_price=market.get("previous_price", 0.5),
                mid_price=market.get("mid_price", 0.5),
                best_bid=market.get("best_bid", 0.5),
                best_ask=market.get("best_ask", 0.5),
                spread=market.get("spread", 0.0),
                volume_24h=market.get("volume_24h", 0.0),
                liquidity=market.get("liquidity", 0.0),
                volatility=market.get("volatility", 0.0),
                last_updated=datetime.utcnow(),
            )
        except Exception:
            return None

    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get order book for a specific market."""
        try:
            data = await self._request("GET", f"/exchange/v1/markets/{market_id}/orderbook")
            orderbook = data.get("orderbook", data)

            return OrderBook(
                market_id=market_id,
                bids=orderbook.get("bids", []),
                asks=orderbook.get("asks", []),
                last_updated=datetime.utcnow(),
            )
        except Exception:
            return None

    async def get_trades(
        self,
        market_id: Optional[str] = None,
        trader_address: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """Get trades with optional filters."""
        params = {"limit": limit}
        if market_id:
            params["market_id"] = market_id
        if trader_address:
            params["trader_id"] = trader_address
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        try:
            data = await self._request("GET", "/exchange/v1/trades", params=params)
            trades = data.get("trades", data.get("data", []))

            return [
                Trade(
                    trade_id=t.get("trade_id", ""),
                    market_id=t.get("market_id", ""),
                    trader_address=t.get("trader_id", ""),
                    side=OrderSide(t.get("side", "buy")),
                    quantity=t.get("quantity", 0.0),
                    price=t.get("price", 0.0),
                    total_value=t.get("total_value", 0.0),
                    fees=t.get("fees", 0.0),
                    timestamp=_parse_datetime(t.get("timestamp")),
                    outcome=t.get("outcome", "YES"),
                    transaction_hash=t.get("transaction_hash"),
                )
                for t in trades
            ]
        except Exception:
            return []

    # Trader Endpoints

    async def get_trader_info(self, trader_address: str) -> Optional[Trader]:
        """Get trader profile information."""
        try:
            data = await self._request("GET", f"/exchange/v1/traders/{trader_address}")
            trader = data.get("trader", data)

            return Trader(
                address=trader_address,
                username=trader.get("username"),
                avatar_url=trader.get("avatar_url"),
                bio=trader.get("bio"),
                is_verified=trader.get("is_verified", False),
                total_trades=trader.get("total_trades", 0),
                total_volume=trader.get("total_volume", 0.0),
                first_trade_timestamp=_parse_datetime(trader.get("first_trade_at")),
                last_trade_timestamp=_parse_datetime(trader.get("last_trade_at")),
            )
        except Exception:
            return None

    async def get_trader_positions(
        self,
        trader_address: str,
        market_id: Optional[str] = None,
    ) -> List[Position]:
        """Get trader's current positions."""
        params = {"trader_id": trader_address}
        if market_id:
            params["market_id"] = market_id

        try:
            data = await self._request("GET", "/exchange/v1/positions", params=params)
            positions = data.get("positions", data.get("data", []))

            return [
                Position(
                    market_id=p.get("market_id", ""),
                    outcome=p.get("outcome", "YES"),
                    quantity=p.get("quantity", 0.0),
                    avg_price=p.get("avg_price", 0.0),
                    current_price=p.get("current_price", 0.0),
                    unrealized_pnl=p.get("unrealized_pnl", 0.0),
                    realized_pnl=p.get("realized_pnl", 0.0),
                    timestamp=datetime.utcnow(),
                )
                for p in positions
            ]
        except Exception:
            return []

    async def get_trader_orders(
        self,
        trader_address: str,
        market_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> List[Order]:
        """Get trader's orders."""
        params = {"trader_id": trader_address}
        if market_id:
            params["market_id"] = market_id

        try:
            data = await self._request("GET", "/exchange/v1/orders", params=params)
            orders = data.get("orders", data.get("data", []))

            # Filter by status if provided
            if status:
                orders = [o for o in orders if o.get("status") == status.value]

            return [
                Order(
                    order_id=o.get("order_id", ""),
                    market_id=o.get("market_id", ""),
                    side=OrderSide(o.get("side", "buy")),
                    order_type=OrderType(o.get("order_type", "limit")),
                    quantity=o.get("quantity", 0.0),
                    limit_price=o.get("limit_price"),
                    filled_quantity=o.get("filled_quantity", 0.0),
                    average_price=o.get("average_price", 0.0),
                    status=OrderStatus(o.get("status", "pending")),
                    fees=o.get("fees", 0.0),
                    timestamp=_parse_datetime(o.get("timestamp")),
                    outcome=o.get("outcome", "YES"),
                    trader_address=o.get("trader_id"),
                )
                for o in orders
            ]
        except Exception:
            return []

    # Trading Endpoints

    async def create_order(
        self,
        market_id: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        limit_price: Optional[float] = None,
        outcome: str = "YES",
    ) -> Order:
        """Create a new order."""
        order_data = {
            "market_id": market_id,
            "side": side.value,
            "order_type": order_type.value,
            "quantity": quantity,
            "outcome": outcome,
        }

        if order_type == OrderType.LIMIT and limit_price is not None:
            order_data["limit_price"] = limit_price

        try:
            data = await self._request("POST", "/exchange/v1/orders", data=order_data)
            order = data.get("order", data)

            return Order(
                order_id=order.get("order_id", ""),
                market_id=market_id,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.OPEN,
                timestamp=datetime.utcnow(),
                outcome=outcome,
            )
        except Exception:
            # Return a rejected order on failure
            return Order(
                order_id="",
                market_id=market_id,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.REJECTED,
                timestamp=datetime.utcnow(),
                outcome=outcome,
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await self._request("DELETE", f"/exchange/v1/orders/{order_id}")
            return True
        except Exception:
            return False

    async def cancel_all_orders(
        self,
        market_id: Optional[str] = None,
    ) -> bool:
        """Cancel all orders."""
        try:
            data = {"market_id": market_id} if market_id else {}
            await self._request("DELETE", "/exchange/v1/orders", data=data)
            return True
        except Exception:
            return False

    async def get_balance(self, trader_address: str) -> float:
        """Get trader's balance."""
        try:
            data = await self._request("GET", f"/exchange/v1/traders/{trader_address}/balance")
            return data.get("balance", 0.0)
        except Exception:
            return 0.0

    # Portfolio Endpoints

    async def get_portfolio(
        self,
        trader_address: str,
    ) -> Dict[str, Any]:
        """Get trader's portfolio summary."""
        try:
            return await self._request("GET", f"/exchange/v1/traders/{trader_address}/portfolio")
        except Exception:
            return {}

    async def get_trader_performance(
        self,
        trader_address: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[TraderPerformance]:
        """Calculate trader performance from trades."""
        trades = await self.get_trades(trader_address=trader_address, start_time=start_time, end_time=end_time)
        if not trades:
            return None

        return self._calculate_trader_performance(trader_address, trades)

    def _calculate_trader_performance(
        self,
        trader_address: str,
        trades: List[Trade],
    ) -> TraderPerformance:
        """Calculate performance metrics from trades."""
        if not trades:
            return TraderPerformance(trader_address=trader_address)

        # Calculate basic metrics
        winning_trades = [t for t in trades if t.price > 0.5 and t.side == OrderSide.BUY]  # Simplified
        losing_trades = [t for t in trades if t.price <= 0.5 and t.side == OrderSide.SELL]

        total_pnl = sum(t.total_value - t.fees for t in winning_trades) - sum(t.total_value + t.fees for t in losing_trades)
        total_volume = sum(t.total_value for t in trades)

        return TraderPerformance(
            trader_address=trader_address,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / total_volume if total_volume > 0 else 0,
            win_rate=len(winning_trades) / len(trades) if trades else 0,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_volume=total_volume,
        )


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
