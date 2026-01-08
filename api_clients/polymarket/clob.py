"""
CLOB API client for Polymarket.
Handles trading, order management, and ledger operations.
"""

import os
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from api_clients.base import (
    MarketData,
    OrderBook,
    Trade,
    Position,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
)


class CLOBAPIClient:
    """
    CLOB API client for Polymarket.
    Provides access to trading, order management, and ledger operations.
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize CLOB API client."""
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay = 0.011  # ~9000 requests per 10s

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
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
        """Make an HTTP request to CLOB API."""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"

        await asyncio.sleep(self._rate_limit_delay)

        async with session.request(method, url, params=params, json=data) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"CLOB API error: {response.status} - {error_text}")
            return await response.json()

    # Market Data Endpoints

    async def get_market_data(self, token_id: str) -> Optional[MarketData]:
        """
        Get market data for a specific market by token ID.

        NOTE: For Polymarket, it's more reliable to get prices from the Gamma API
        response which includes outcomePrices field. This method tries the CLOB API
        first, but falls back to None if it fails, allowing the caller to use
        Gamma API data instead.
        """
        try:
            # Try CLOB API first
            data = await self._request("GET", f"/price", params={"token_id": token_id})

            # If CLOB API returns empty/invalid data, return None so caller can use Gamma API data
            if not data or data.get("price") is None:
                return None

            return MarketData(
                market_id=token_id,
                current_price=data.get("price", 0.5),
                previous_price=data.get("previous_price", 0.5),
                mid_price=data.get("mid_price", 0.5),
                best_bid=data.get("best_bid", 0.5),
                best_ask=data.get("best_ask", 0.5),
                spread=data.get("spread", 0.0),
                volume_24h=data.get("volume_24h", 0.0),
                liquidity=data.get("liquidity", 0.0),
                volatility=data.get("volatility", 0.0),
                last_updated=datetime.utcnow(),
            )
        except Exception:
            # Return None on any error so caller can use Gamma API data
            return None

    @staticmethod
    def parse_market_data_from_gamma_response(market: Dict[str, Any]) -> Optional[MarketData]:
        """
        Parse market data from Gamma API response.

        The Gamma API returns clobTokenIds, outcomes, and outcomePrices as JSON stringified arrays.
        This method parses them and returns MarketData for the first token.

        Args:
            market: Raw market dict from Gamma API

        Returns:
            MarketData for the first outcome token, or None if parsing fails
        """
        try:
            # Parse JSON stringified fields
            token_ids = json.loads(market.get("clobTokenIds", "[]"))
            outcomes = json.loads(market.get("outcomes", "[]"))
            outcome_prices = json.loads(market.get("outcomePrices", "[]"))

            if not token_ids or not outcome_prices:
                return None

            # Get the first token (typically the "Yes" outcome)
            token_id = token_ids[0]
            current_price = float(outcome_prices[0]) if outcome_prices else 0.5

            # Calculate additional fields from other market data
            volume_24h = float(market.get("volume24hr", 0))
            liquidity = float(market.get("liquidity", 0))

            return MarketData(
                market_id=token_id,
                current_price=current_price,
                previous_price=current_price,  # Gamma API doesn't provide previous price
                mid_price=current_price,
                best_bid=current_price,  # Approximation
                best_ask=current_price,  # Approximation
                spread=0.0,  # Unknown from Gamma API
                volume_24h=volume_24h,
                liquidity=liquidity,
                volatility=0.0,  # Unknown from Gamma API
                last_updated=datetime.utcnow(),
            )
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"Error parsing market data: {e}")
            return None

    async def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
        """Get order book for a specific market by token ID."""
        try:
            data = await self._request("GET", f"/book", params={"token_id": token_id})
            if not data:
                return None

            return OrderBook(
                market_id=market_id,
                bids=data.get("bids", []),
                asks=data.get("asks", []),
                last_updated=datetime.utcnow(),
            )
        except Exception:
            return None

    async def get_multiple_orderbooks(
        self,
        market_ids: List[str],
    ) -> Dict[str, OrderBook]:
        """Get order books for multiple markets."""
        try:
            data = await self._request("GET", f"/books/{','.join(market_ids)}")
            orderbooks = {}
            for market_id in market_ids:
                if market_id in data:
                    orderbooks[market_id] = OrderBook(
                        market_id=market_id,
                        bids=data[market_id].get("bids", []),
                        asks=data[market_id].get("asks", []),
                        last_updated=datetime.utcnow(),
                    )
            return orderbooks
        except Exception:
            return {}

    async def get_midprice(self, market_id: str) -> Optional[float]:
        """Get mid price for a market."""
        try:
            data = await self._request("GET", f"/midprice/{market_id}")
            return data.get("price")
        except Exception:
            return None

    async def get_multiple_midprices(
        self,
        market_ids: List[str],
    ) -> Dict[str, float]:
        """Get mid prices for multiple markets."""
        try:
            data = await self._request("GET", f"/midprices/{','.join(market_ids)}")
            return data
        except Exception:
            return {}

    async def get_price_history(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        granularity: str = "1m",
    ) -> List[Dict[str, Any]]:
        """Get price history for a market."""
        try:
            params = {"market_id": market_id, "granularity": granularity}
            if start_time:
                params["start_time"] = start_time.isoformat()
            if end_time:
                params["end_time"] = end_time.isoformat()

            data = await self._request("GET", "/price-history", params=params)
            return data.get("history", [])
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
            order_data["price"] = limit_price

        try:
            data = await self._request("POST", "/order", data=order_data)
            return Order(
                order_id=data.get("order_id", ""),
                market_id=market_id,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.OPEN,
                timestamp=datetime.utcnow(),
                outcome=outcome,
            )
        except Exception as e:
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
            await self._request("DELETE", f"/order/{order_id}")
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
            await self._request("DELETE", "/cancel-all", data=data)
            return True
        except Exception:
            return False

    # Ledger Endpoints

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
            params["user"] = trader_address
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        try:
            data = await self._request("GET", "/trades", params=params)
            trades = data.get("trades", [])

            return [
                Trade(
                    trade_id=t.get("trade_id", ""),
                    market_id=t.get("market_id", ""),
                    trader_address=t.get("user_address", ""),
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

    async def get_orders(
        self,
        trader_address: Optional[str] = None,
        market_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> List[Order]:
        """Get orders with optional filters."""
        params = {}
        if trader_address:
            params["user"] = trader_address
        if market_id:
            params["market_id"] = market_id

        try:
            data = await self._request("GET", "/orders", params=params)
            orders = data.get("orders", [])

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
                    limit_price=o.get("price"),
                    filled_quantity=o.get("filled_quantity", 0.0),
                    average_price=o.get("average_price", 0.0),
                    status=OrderStatus(o.get("status", "pending")),
                    fees=o.get("fees", 0.0),
                    timestamp=_parse_datetime(o.get("timestamp")),
                    outcome=o.get("outcome", "YES"),
                    trader_address=o.get("user_address"),
                )
                for o in orders
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
        return await self.get_orders(trader_address=trader_address, market_id=market_id, status=status)

    async def get_trader_positions(
        self,
        trader_address: str,
        market_id: Optional[str] = None,
    ) -> List[Position]:
        """Get trader's current positions."""
        params = {"user": trader_address}
        if market_id:
            params["market_id"] = market_id

        try:
            data = await self._request("GET", "/positions", params=params)
            positions = data.get("positions", [])

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

    async def get_balance(self, trader_address: str) -> float:
        """Get trader's balance."""
        try:
            data = await self._request("GET", f"/balance/{trader_address}")
            return data.get("balance", 0.0)
        except Exception:
            return 0.0

    async def get_balance_allowance(
        self,
        trader_address: str,
    ) -> Dict[str, float]:
        """Get trader's balance allowance."""
        try:
            data = await self._request("GET", f"/balance-allowance/{trader_address}")
            return {
                "balance": data.get("balance", 0.0),
                "allowance": data.get("allowance", 0.0),
            }
        except Exception:
            return {"balance": 0.0, "allowance": 0.0}


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
