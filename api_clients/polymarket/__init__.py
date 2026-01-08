"""
Polymarket API client implementation.
Provides access to Gamma and CLOB APIs for market data and trading.
"""

import os
import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .gamma import GammaAPIClient
from .clob import CLOBAPIClient
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

logger = logging.getLogger(__name__)


@dataclass
class PolymarketAPIClient(MarketAPIClient):
    """
    Polymarket API client.
    Combines Gamma API (market data) and CLOB API (trading).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_gamma: bool = True,
        use_clob: bool = True,
    ):
        """Initialize Polymarket API client."""
        super().__init__(api_key)
        self.gamma = GammaAPIClient(api_key) if use_gamma else None
        self.clob = CLOBAPIClient(api_key) if use_clob else None
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "polymarket"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def get_markets(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get list of markets from Gamma API."""
        if self.gamma:
            return await self.gamma.get_markets(category, limit, offset)
        return []

    async def get_market_data(self, market_id: str) -> Optional[MarketData]:
        """
        Get market data from CLOB API first, fall back to Gamma API.

        This method tries to get market data from the CLOB API first.
        If that fails, it falls back to the Gamma API by fetching markets
        and parsing the outcomePrices field.

        Args:
            market_id: The token_id to fetch data for

        Returns:
            MarketData if successful, None otherwise
        """
        # Try CLOB API first
        if self.clob:
            clob_data = await self.clob.get_market_data(market_id)
            if clob_data:
                return clob_data

        # Fallback to Gamma API
        if self.gamma:
            try:
                # Get markets from Gamma API and find the one with matching token_id
                markets = await self.gamma.get_markets(limit=100)
                for market in markets:
                    # Parse clobTokenIds from JSON string
                    token_ids = json.loads(market.get("clobTokenIds", "[]"))
                    if market_id in token_ids:
                        # Use the static method to parse market data
                        return self.clob.parse_market_data_from_gamma_response(market) if self.clob else None
            except Exception as e:
                logger.warning(f"Error fetching market data from Gamma API: {e}")

        return None

    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get order book from CLOB API."""
        if self.clob:
            return await self.clob.get_orderbook(market_id)
        return None

    async def get_trades(
        self,
        market_id: Optional[str] = None,
        trader_address: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """Get trades from CLOB API."""
        if self.clob:
            return await self.clob.get_trades(market_id, trader_address, start_time, end_time, limit)
        return []

    async def get_trader_info(self, trader_address: str) -> Optional[Trader]:
        """Get trader profile from Gamma API."""
        if self.gamma:
            return await self.gamma.get_trader_info(trader_address)
        return None

    async def get_trader_positions(
        self, trader_address: str, market_id: Optional[str] = None
    ) -> List[Position]:
        """Get trader's positions from CLOB API."""
        if self.clob:
            return await self.clob.get_trader_positions(trader_address, market_id)
        return []

    async def get_trader_orders(
        self,
        trader_address: str,
        market_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> List[Order]:
        """Get trader's orders from CLOB API."""
        if self.clob:
            return await self.clob.get_trader_orders(trader_address, market_id, status)
        return []

    async def create_order(
        self,
        market_id: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        limit_price: Optional[float] = None,
        outcome: str = "YES",
    ) -> Order:
        """Create a new order via CLOB API."""
        if self.clob:
            return await self.clob.create_order(market_id, side, order_type, quantity, limit_price, outcome)
        raise ValueError("CLOB API not initialized")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order via CLOB API."""
        if self.clob:
            return await self.clob.cancel_order(order_id)
        return False

    async def get_balance(self, trader_address: str) -> float:
        """Get trader's balance from CLOB API."""
        if self.clob:
            return await self.clob.get_balance(trader_address)
        return 0.0

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
