"""
Tests for Polymarket Data API client.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_clients.polymarket.data_api import (
    DataAPIClient,
    UserActivity,
    MarketWithPosition,
    create_data_client,
)


class TestDataAPIClient:
    """Tests for DataAPIClient."""

    @pytest.fixture
    def data_client(self):
        """Create Data API client."""
        client = DataAPIClient(api_key="test_api_key")
        return client

    def test_initialization(self, data_client):
        """Test client initialization."""
        assert data_client.api_key == "test_api_key"
        assert data_client._rate_limit_delay == 0.1

    def test_initialization_no_key(self):
        """Test initialization without API key."""
        client = DataAPIClient()
        assert client.api_key is None

    @pytest.mark.asyncio
    async def test_get_positions_success(self, data_client):
        """Test successful positions fetch."""
        mock_response = [
            {
                "conditionId": "0x123",
                "title": "Will BTC hit 100k?",
                "slug": "will-btc-hit-100k",
                "outcome": "YES",
                "outcomeIndex": 0,
                "size": 100.0,
                "avgPrice": 0.55,
                "initialValue": 55.0,
                "currentValue": 60.0,
                "cashPnl": 5.0,
                "percentPnl": 0.09,
                "totalBought": 100.0,
                "realizedPnl": 2.0,
                "percentRealizedPnl": 0.036,
                "curPrice": 0.60,
                "redeemable": True,
                "mergeable": False,
                "endDate": "2025-12-31",
                "icon": "https://example.com/icon.png",
                "eventSlug": "btc-100k",
            }
        ]
        
        # Mock the _request method
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await data_client.get_positions("0x1234567890abcdef")
            
            assert len(result) == 1
            assert result[0].market_id == "0x123"
            assert result[0].title == "Will BTC hit 100k?"
            assert result[0].size == 100.0
            assert result[0].cash_pnl == 5.0

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, data_client):
        """Test positions fetch with no positions."""
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = []
            
            result = await data_client.get_positions("0x1234567890abcdef")
            
            assert result == []

    @pytest.mark.asyncio
    async def test_get_positions_error(self, data_client):
        """Test positions fetch with error."""
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("API error")
            
            result = await data_client.get_positions("0x1234567890abcdef")
            
            assert result == []

    @pytest.mark.asyncio
    async def test_get_trades_success(self, data_client):
        """Test successful trades fetch."""
        mock_response = [
            {
                "id": "trade_123",
                "conditionId": "0x456",
                "size": 50.0,
                "price": 0.60,
                "totalValue": 30.0,
                "fees": 0.6,
                "side": "buy",
                "outcome": "YES",
                "timestamp": "2025-01-08T20:00:00Z",
                "transactionHash": "0xabc",
            }
        ]
        
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await data_client.get_trades("0x1234567890abcdef", limit=50)
            
            assert len(result) == 1
            assert result[0].trade_id == "trade_123"
            assert result[0].market_id == "0x456"
            assert result[0].quantity == 50.0

    @pytest.mark.asyncio
    async def test_get_trades_error(self, data_client):
        """Test trades fetch with error."""
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("API error")
            
            result = await data_client.get_trades("0x1234567890abcdef")
            
            assert result == []

    @pytest.mark.asyncio
    async def test_get_activity_success(self, data_client):
        """Test successful activity fetch."""
        mock_response = [
            {
                "id": "activity_123",
                "type": "trade",
                "conditionId": "0x789",
                "outcome": "NO",
                "size": 25.0,
                "price": 0.40,
                "totalValue": 10.0,
                "fees": 0.2,
                "status": "filled",
                "timestamp": "2025-01-08T19:00:00Z",
                "transactionHash": "0xdef",
            }
        ]
        
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await data_client.get_activity("0x1234567890abcdef", limit=10)
            
            assert len(result) == 1
            assert result[0].activity_id == "activity_123"
            assert result[0].activity_type == "trade"

    @pytest.mark.asyncio
    async def test_get_activity_error(self, data_client):
        """Test activity fetch with error."""
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("API error")
            
            result = await data_client.get_activity("0x1234567890abcdef")
            
            assert result == []

    @pytest.mark.asyncio
    async def test_get_user_summary(self, data_client):
        """Test user summary generation."""
        # Mock positions response
        positions_response = [
            {
                "conditionId": "0x123",
                "title": "Test Market",
                "currentValue": 60.0,
                "cashPnl": 5.0,
                "size": 100.0,
                "avgPrice": 0.55,
                "initialValue": 55.0,
                "percentPnl": 0.09,
                "totalBought": 100.0,
                "realizedPnl": 2.0,
                "percentRealizedPnl": 0.036,
                "curPrice": 0.60,
                "redeemable": True,
                "mergeable": False,
                "endDate": "2025-12-31",
                "icon": "https://example.com/icon.png",
                "eventSlug": "test-event",
                "outcome": "YES",
                "outcomeIndex": 0,
            }
        ]
        
        # Mock trades response  
        trades_response = [
            {
                "id": "trade_1",
                "conditionId": "0x123",
                "size": 50.0,
                "price": 0.60,
                "totalValue": 30.0,
                "fees": 0.6,
                "side": "buy",
                "outcome": "YES",
                "timestamp": "2025-01-08T20:00:00Z",
            }
        ]
        
        # Mock _request to return different responses
        async def mock_request_side_effect(method, endpoint, params=None):
            if endpoint == "/positions":
                return positions_response
            elif endpoint == "/activity":
                return []
            elif endpoint == "/trades":
                return trades_response
            return []
        
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = mock_request_side_effect
            
            result = await data_client.get_user_summary("0x1234567890abcdef")
            
            assert result["user_address"] == "0x1234567890abcdef"
            assert result["positions_count"] == 1
            assert result["total_pnl"] == 5.0

    @pytest.mark.asyncio
    async def test_get_builder_leaderboard(self, data_client):
        """Test builder leaderboard fetch."""
        mock_response = {
            "leaderboard": [
                {"address": "0xabc", "volume": 10000.0, "rank": 1},
                {"address": "0xdef", "volume": 5000.0, "rank": 2},
            ]
        }
        
        with patch.object(data_client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await data_client.get_builder_leaderboard(limit=10)
            
            assert len(result) == 2
            assert result[0]["rank"] == 1

    @pytest.mark.asyncio
    async def test_close_session(self, data_client):
        """Test session closure."""
        mock_session = AsyncMock()
        mock_session.closed = False
        data_client._session = mock_session
        
        await data_client.close()
        
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_already_closed(self, data_client):
        """Test closing already closed session."""
        mock_session = AsyncMock()
        mock_session.closed = True
        data_client._session = mock_session
        
        await data_client.close()
        
        mock_session.close.assert_not_called()


class TestUserActivity:
    """Tests for UserActivity dataclass."""

    def test_user_activity_creation(self):
        """Test UserActivity creation."""
        activity = UserActivity(
            activity_id="act_123",
            activity_type="trade",
            market_id="0x456",
            outcome="YES",
            quantity=100.0,
            price=0.55,
            total_value=55.0,
            fees=1.1,
            status="filled",
            timestamp=datetime.utcnow(),
        )
        
        assert activity.activity_id == "act_123"
        assert activity.activity_type == "trade"
        assert activity.quantity == 100.0


class TestMarketWithPosition:
    """Tests for MarketWithPosition dataclass."""

    def test_market_with_position_creation(self):
        """Test MarketWithPosition creation."""
        market = MarketWithPosition(
            market_id="0x123",
            title="Will ETH hit 5k?",
            slug="will-eth-hit-5k",
            outcome="YES",
            outcome_index=0,
            size=50.0,
            avg_price=0.45,
            initial_value=22.5,
            current_value=30.0,
            cash_pnl=7.5,
            percent_pnl=0.333,
            total_bought=50.0,
            realized_pnl=3.0,
            percent_realized_pnl=0.133,
            current_price=0.60,
            redeemable=True,
            mergeable=False,
            end_date="2025-06-30",
            icon="https://example.com/eth.png",
            event_slug="eth-5k",
        )
        
        assert market.market_id == "0x123"
        assert market.title == "Will ETH hit 5k?"
        assert market.size == 50.0
        assert market.cash_pnl == 7.5


class TestCreateDataClient:
    """Tests for create_data_client factory function."""

    def test_create_with_key(self):
        """Test creating client with API key."""
        client = create_data_client("my_api_key")
        assert client.api_key == "my_api_key"

    def test_create_without_key(self):
        """Test creating client without API key."""
        client = create_data_client()
        assert client.api_key is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
