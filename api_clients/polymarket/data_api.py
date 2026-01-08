"""
Data API client for Polymarket.
Handles user positions, activity, and trade history.
"""

import os
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from api_clients.base import (
    Trade,
    Position,
    Order,
    OrderSide,
    OrderStatus,
)


@dataclass
class UserActivity:
    """User activity record from Data API."""
    activity_id: str
    activity_type: str  # 'trade', 'deposit', 'withdrawal', etc.
    market_id: Optional[str]
    outcome: Optional[str]
    quantity: float
    price: float
    total_value: float
    fees: float
    status: str
    timestamp: datetime
    transaction_hash: Optional[str] = None


@dataclass
class MarketWithPosition:
    """Market information with user position."""
    market_id: str
    title: str
    slug: str
    outcome: str
    outcome_index: int
    size: float
    avg_price: float
    initial_value: float
    current_value: float
    cash_pnl: float
    percent_pnl: float
    total_bought: float
    realized_pnl: float
    percent_realized_pnl: float
    current_price: float
    redeemable: bool
    mergeable: bool
    end_date: Optional[str]
    icon: Optional[str]
    event_slug: Optional[str]


class DataAPIClient:
    """
    Data API client for Polymarket.
    
    Provides access to:
    - User positions
    - User activity
    - Trade history
    - Leaderboard data
    
    Docs: https://docs.polymarket.com/developers/misc-endpoints/data-api-get-positions
    """

    BASE_URL = "https://data-api.polymarket.com"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Data API client."""
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay = 0.1  # 10 requests per second

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
    ) -> Dict[str, Any]:
        """Make an HTTP request to Data API."""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"

        await asyncio.sleep(self._rate_limit_delay)

        async with session.request(method, url, params=params) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"Data API error: {response.status} - {error_text}")
            return await response.json()

    # =========================================================================
    # Positions Endpoints
    # =========================================================================

    async def get_positions(
        self,
        user_address: str,
        market_ids: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        size_threshold: float = 1.0,
        redeemable: bool = False,
        mergeable: bool = False,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "TOKENS",
        sort_direction: str = "DESC",
        title: Optional[str] = None,
    ) -> List[MarketWithPosition]:
        """
        Get current positions for a user.
        
        Args:
            user_address: User's wallet address (required)
            market_ids: Filter by specific markets (condition IDs)
            event_ids: Filter by specific events
            size_threshold: Minimum position size
            redeemable: Only show redeemable positions
            mergeable: Only show mergeable positions
            limit: Max results (0-500)
            offset: Pagination offset
            sort_by: Sort field (CURRENT, INITIAL, TOKENS, CASHPNL, etc.)
            sort_direction: ASC or DESC
            title: Filter by market title
        
        Returns:
            List of MarketWithPosition objects
        """
        params = {
            "user": user_address,
            "sizeThreshold": size_threshold,
            "limit": limit,
            "offset": offset,
            "sortBy": sort_by,
            "sortDirection": sort_direction,
        }

        # Convert boolean params to API-expected format (string "true"/"false" or omit)
        if redeemable:
            params["redeemable"] = "true"
        if mergeable:
            params["mergeable"] = "true"

        if market_ids:
            params["market"] = ",".join(market_ids)
        if event_ids:
            params["eventId"] = ",".join(event_ids)
        if title:
            params["title"] = title

        try:
            data = await self._request("GET", "/positions", params=params)
            positions = data if isinstance(data, list) else data.get("positions", [])

            return [
                MarketWithPosition(
                    market_id=p.get("conditionId", ""),
                    title=p.get("title", ""),
                    slug=p.get("slug", ""),
                    outcome=p.get("outcome", ""),
                    outcome_index=p.get("outcomeIndex", 0),
                    size=p.get("size", 0.0),
                    avg_price=p.get("avgPrice", 0.0),
                    initial_value=p.get("initialValue", 0.0),
                    current_value=p.get("currentValue", 0.0),
                    cash_pnl=p.get("cashPnl", 0.0),
                    percent_pnl=p.get("percentPnl", 0.0),
                    total_bought=p.get("totalBought", 0.0),
                    realized_pnl=p.get("realizedPnl", 0.0),
                    percent_realized_pnl=p.get("percentRealizedPnl", 0.0),
                    current_price=p.get("curPrice", 0.0),
                    redeemable=p.get("redeemable", False),
                    mergeable=p.get("mergeable", False),
                    end_date=p.get("endDate"),
                    icon=p.get("icon"),
                    event_slug=p.get("eventSlug"),
                )
                for p in positions
            ]
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    # =========================================================================
    # Activity Endpoints
    # =========================================================================

    async def get_activity(
        self,
        user_address: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[UserActivity]:
        """
        Get user activity.
        
        Args:
            user_address: User's wallet address (required)
            limit: Max results
            offset: Pagination offset
        
        Returns:
            List of UserActivity objects
        """
        params = {
            "user": user_address,
            "limit": limit,
            "offset": offset,
        }

        try:
            data = await self._request("GET", "/activity", params=params)
            activities = data if isinstance(data, list) else data.get("activity", [])

            return [
                UserActivity(
                    activity_id=a.get("id", ""),
                    activity_type=a.get("type", ""),
                    market_id=a.get("conditionId"),
                    outcome=a.get("outcome"),
                    quantity=a.get("size", 0.0),
                    price=a.get("price", 0.0),
                    total_value=a.get("totalValue", 0.0),
                    fees=a.get("fees", 0.0),
                    status=a.get("status", ""),
                    timestamp=_parse_datetime(a.get("timestamp")),
                    transaction_hash=a.get("transactionHash"),
                )
                for a in activities
            ]
        except Exception as e:
            print(f"Error fetching activity: {e}")
            return []

    # =========================================================================
    # Trades Endpoints
    # =========================================================================

    async def get_trades(
        self,
        user_address: str,
        market_ids: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """
        Get user trade history.
        
        Args:
            user_address: User's wallet address (required)
            market_ids: Filter by specific markets
            event_ids: Filter by specific events
            limit: Max results
            offset: Pagination offset
        
        Returns:
            List of Trade objects
        """
        params = {
            "user": user_address,
            "limit": limit,
            "offset": offset,
        }

        if market_ids:
            params["market"] = ",".join(market_ids)
        if event_ids:
            params["eventId"] = ",".join(event_ids)

        try:
            data = await self._request("GET", "/trades", params=params)
            trades = data if isinstance(data, list) else data.get("trades", [])

            return [
                Trade(
                    trade_id=t.get("id", ""),
                    market_id=t.get("conditionId", ""),
                    trader_address=user_address,
                    side=OrderSide.BUY if t.get("side", "").lower() == "buy" else OrderSide.SELL,
                    quantity=t.get("size", 0.0),
                    price=t.get("price", 0.0),
                    total_value=t.get("totalValue", 0.0),
                    fees=t.get("fees", 0.0),
                    timestamp=_parse_datetime(t.get("timestamp")),
                    outcome=t.get("outcome", ""),
                    transaction_hash=t.get("transactionHash"),
                )
                for t in trades
            ]
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return []

    # =========================================================================
    # Builder Leaderboard Endpoints
    # =========================================================================

    async def get_builder_leaderboard(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get builder leaderboard data.
        
        Returns:
            List of builder rankings
        """
        try:
            data = await self._request("GET", "/builders/leaderboard", params={"limit": limit})
            return data.get("leaderboard", [])
        except Exception as e:
            print(f"Error fetching builder leaderboard: {e}")
            return []

    async def get_builder_volume_timeseries(
        self,
        builder_address: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get daily volume time series for a builder.
        
        Args:
            builder_address: Builder's wallet address
            days: Number of days of data
        
        Returns:
            List of daily volume records
        """
        try:
            data = await self._request(
                "GET",
                f"/builders/volume/{builder_address}",
                params={"days": days}
            )
            return data.get("volume", [])
        except Exception as e:
            print(f"Error fetching builder volume: {e}")
            return []

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def get_user_summary(
        self,
        user_address: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of user data across all endpoints.
        
        Returns:
            Dict with positions, activity, and trades summary
        """
        positions = await self.get_positions(user_address)
        activity = await self.get_activity(user_address, limit=50)
        trades = await self.get_trades(user_address, limit=50)

        # Calculate summary stats
        total_position_value = sum(p.current_value for p in positions)
        total_pnl = sum(p.cash_pnl for p in positions)
        total_trades = len(trades)

        winning_trades = [t for t in trades if t.total_value > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        return {
            "user_address": user_address,
            "positions_count": len(positions),
            "total_position_value": total_position_value,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "recent_activity_count": len(activity),
            "win_rate": win_rate,
        }


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


# Convenience function to create Data API client
def create_data_client(api_key: Optional[str] = None) -> DataAPIClient:
    """Create a Data API client."""
    return DataAPIClient(api_key)
