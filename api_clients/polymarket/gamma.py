"""
Gamma API client for Polymarket.
Handles market data, profiles, and comments.
"""

import os
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from api_clients.base import (
    Trader,
    MarketData,
)


class GammaAPIClient:
    """
    Gamma API client for Polymarket.
    Provides access to market data, profiles, and comments.
    """

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gamma API client."""
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay = 0.25  # 4000 requests per 10s = 0.25s delay

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
        """Make an HTTP request to Gamma API."""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"

        await asyncio.sleep(self._rate_limit_delay)

        async with session.request(method, url, params=params, json=data) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"Gamma API error: {response.status} - {error_text}")
            return await response.json()

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

        data = await self._request("GET", "/markets", params=params)
        # API returns list directly, not wrapped in dict
        if isinstance(data, list):
            return data
        return data.get("markets", []) if isinstance(data, dict) else []

    async def get_market(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific market."""
        try:
            return await self._request("GET", f"/markets/{market_id}")
        except Exception:
            return None

    async def get_market_events(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get events for a market."""
        try:
            return await self._request("GET", f"/markets/{market_id}/events")
        except Exception:
            return None

    async def get_events(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get list of events."""
        params = {"limit": limit, "offset": offset}
        data = await self._request("GET", "/events", params=params)
        # API returns list directly, not wrapped in dict
        if isinstance(data, list):
            return data
        return data.get("events", []) if isinstance(data, dict) else []

    async def get_trader_info(self, trader_address: str) -> Optional[Trader]:
        """Get trader profile information."""
        try:
            data = await self._request("GET", f"/profiles/{trader_address}")
            if not data:
                return None

            return Trader(
                address=trader_address,
                username=data.get("username"),
                avatar_url=data.get("avatar_url"),
                bio=data.get("bio"),
                is_verified=data.get("is_verified", False),
                total_trades=data.get("total_trades", 0),
                total_volume=data.get("total_volume", 0.0),
                first_trade_timestamp=_parse_datetime(data.get("first_trade_at")),
                last_trade_timestamp=_parse_datetime(data.get("last_trade_at")),
            )
        except Exception:
            return None

    async def get_trader_positions(
        self,
        trader_address: str,
        market_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get trader's positions."""
        try:
            params = {"user": trader_address}
            if market_id:
                params["market_id"] = market_id

            data = await self._request("GET", "/positions", params=params)
            return data.get("positions", [])
        except Exception:
            return []

    async def get_trader_trades(
        self,
        trader_address: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get trader's trade history."""
        try:
            params = {"user": trader_address, "limit": limit, "offset": offset}
            data = await self._request("GET", "/trades", params=params)
            return data.get("trades", [])
        except Exception:
            return []

    async def get_market_comments(
        self,
        market_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get comments for a market."""
        try:
            params = {"limit": limit, "offset": offset}
            data = await self._request("GET", f"/markets/{market_id}/comments", params=params)
            return data.get("comments", [])
        except Exception:
            return []

    async def get_series(
        self,
        series_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a series by ID."""
        try:
            return await self._request("GET", f"/series/{series_id}")
        except Exception:
            return None

    async def search_markets(
        self,
        query: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search markets."""
        try:
            params = {"q": query, "limit": limit}
            data = await self._request("GET", "/search/markets", params=params)
            return data.get("results", [])
        except Exception:
            return []


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
