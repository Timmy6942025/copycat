"""
Real-Time Data Feed Handler for Sandbox Simulation.

Provides continuous market data updates for sandbox simulations.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FeedStatus(Enum):
    """Status of the data feed."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class FeedConfig:
    """Configuration for data feed."""
    update_interval: float = 1.0  # seconds between updates
    max_history: int = 1000  # maximum price history points per market
    enable_price_history: bool = True
    enable_orderbook: bool = True
    enable_trades: bool = True
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0  # seconds to wait before reconnecting


@dataclass
class PricePoint:
    """Single price data point."""
    timestamp: datetime
    price: float
    volume: float


@dataclass
class FeedState:
    """Current state of the data feed."""
    status: FeedStatus = FeedStatus.STOPPED
    last_update: Optional[datetime] = None
    markets_tracked: int = 0
    updates_received: int = 0
    errors_count: int = 0


class RealtimeDataFeed:
    """
    Real-time data feed handler for sandbox simulations.
    
    Provides continuous market data updates to the sandbox simulation.
    Features:
    - Polling-based updates from configured providers
    - Price history tracking
    - Callback-based notifications
    - Automatic reconnection on errors
    """
    
    def __init__(
        self,
        config: Optional[FeedConfig] = None,
    ):
        """Initialize real-time data feed."""
        self.config = config or FeedConfig()
        self.state = FeedState()
        self._running = False
        self._paused = False
        self._update_tasks: Dict[str, asyncio.Task] = {}
        self._price_history: Dict[str, List[PricePoint]] = {}
        self._latest_prices: Dict[str, float] = {}
        
        # Callbacks
        self.on_price_update: Optional[Callable[[str, float], None]] = None
        self.on_market_update: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.on_status_change: Optional[Callable[[FeedStatus], None]] = None
        self.on_error: Optional[Callable[[str, Exception], None]] = None
    
    async def start(
        self,
        market_ids: List[str],
        provider_callback: Callable[[str], Dict[str, Any]],
    ):
        """
        Start the data feed for specified markets.
        
        Args:
            market_ids: List of market IDs to track
            provider_callback: Async callback to get market data (market_id -> data)
        """
        if self._running:
            logger.warning("Data feed already running")
            return
        
        self._running = True
        self._paused = False
        self._set_status(FeedStatus.RUNNING)
        
        # Initialize price history
        if self.config.enable_price_history:
            for market_id in market_ids:
                self._price_history[market_id] = []
        
        # Start update tasks for each market
        for market_id in market_ids:
            task = asyncio.create_task(
                self._update_loop(market_id, provider_callback)
            )
            self._update_tasks[market_id] = task
        
        self.state.markets_tracked = len(market_ids)
        logger.info(f"Started data feed for {len(market_ids)} markets")
    
    async def stop(self):
        """Stop the data feed."""
        self._running = False
        self._paused = False
        
        # Cancel all update tasks
        for task in self._update_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._update_tasks:
            await asyncio.gather(*self._update_tasks.values(), return_exceptions=True)
        
        self._update_tasks.clear()
        self._set_status(FeedStatus.STOPPED)
        logger.info("Data feed stopped")
    
    async def pause(self):
        """Pause the data feed."""
        if not self._running:
            return
        
        self._paused = True
        self._set_status(FeedStatus.PAUSED)
        logger.info("Data feed paused")
    
    async def resume(self):
        """Resume the data feed."""
        if not self._running:
            return
        
        self._paused = False
        self._set_status(FeedStatus.RUNNING)
        logger.info("Data feed resumed")
    
    async def add_market(
        self,
        market_id: str,
        provider_callback: Callable[[str], Dict[str, Any]],
    ):
        """Add a new market to track."""
        if market_id in self._update_tasks:
            logger.warning(f"Market {market_id} already tracked")
            return
        
        if self.config.enable_price_history:
            self._price_history[market_id] = []
        
        task = asyncio.create_task(
            self._update_loop(market_id, provider_callback)
        )
        self._update_tasks[market_id] = task
        self.state.markets_tracked += 1
        logger.info(f"Added market {market_id} to data feed")
    
    async def remove_market(self, market_id: str):
        """Remove a market from tracking."""
        if market_id not in self._update_tasks:
            return
        
        task = self._update_tasks[market_id]
        task.cancel()
        del self._update_tasks[market_id]
        
        if market_id in self._price_history:
            del self._price_history[market_id]
        if market_id in self._latest_prices:
            del self._latest_prices[market_id]
        
        self.state.markets_tracked -= 1
        logger.info(f"Removed market {market_id} from data feed")
    
    def get_price_history(
        self,
        market_id: str,
        limit: int = 100,
    ) -> List[PricePoint]:
        """Get price history for a market."""
        history = self._price_history.get(market_id, [])
        return history[-limit:] if limit > 0 else history
    
    def get_latest_price(self, market_id: str) -> Optional[float]:
        """Get the latest price for a market."""
        return self._latest_prices.get(market_id)
    
    def get_all_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all tracked markets."""
        return self._latest_prices.copy()
    
    async def _update_loop(
        self,
        market_id: str,
        provider_callback: Callable[[str], Dict[str, Any]],
    ):
        """Main update loop for a single market."""
        while self._running:
            if self._paused:
                await asyncio.sleep(0.1)
                continue
            
            try:
                # Get market data from provider
                data = await provider_callback(market_id)
                
                if data and "current_price" in data:
                    price = data["current_price"]
                    volume = data.get("volume_24h", 0)
                    timestamp = datetime.utcnow()
                    
                    # Update latest price
                    self._latest_prices[market_id] = price
                    
                    # Update price history
                    if self.config.enable_price_history:
                        point = PricePoint(
                            timestamp=timestamp,
                            price=price,
                            volume=volume,
                        )
                        history = self._price_history.get(market_id, [])
                        history.append(point)
                        
                        # Trim history if needed
                        if len(history) > self.config.max_history:
                            history.pop(0)
                        
                        self._price_history[market_id] = history
                    
                    # Notify callbacks
                    if self.on_price_update:
                        self.on_price_update(market_id, price)
                    
                    if self.on_market_update:
                        self.on_market_update(market_id, data)
                    
                    self.state.updates_received += 1
                    self.state.last_update = timestamp
                
            except Exception as e:
                self.state.errors_count += 1
                logger.error(f"Error updating market {market_id}: {e}")
                
                if self.on_error:
                    self.on_error(market_id, e)
                
                if self.config.auto_reconnect:
                    await asyncio.sleep(self.config.reconnect_delay)
                else:
                    break
            
            # Wait for next update
            await asyncio.sleep(self.config.update_interval)
    
    def _set_status(self, status: FeedStatus):
        """Update feed status."""
        self.state.status = status
        if self.on_status_change:
            self.on_status_change(status)
    
    def get_state(self) -> FeedState:
        """Get current feed state."""
        return self.state
    
    def reset_state(self):
        """Reset feed state counters."""
        self.state.updates_received = 0
        self.state.errors_count = 0
        self.state.last_update = None


async def test_realtime_feed():
    """Test the real-time data feed."""
    
    # Mock provider callback
    async def mock_provider(market_id: str) -> Dict[str, Any]:
        return {
            "market_id": market_id,
            "current_price": 100 + hash(market_id) % 10,
            "volume_24h": 1000000,
            "volatility": 0.02,
        }
    
    feed = RealtimeDataFeed(FeedConfig(update_interval=0.5))
    
    # Set up callbacks
    def on_price_update(market_id: str, price: float):
        print(f"Price update: {market_id} = ${price:.2f}")
    
    def on_status_change(status: FeedStatus):
        print(f"Feed status: {status.value}")
    
    feed.on_price_update = on_price_update
    feed.on_status_change = on_status_change
    
    # Start feed
    markets = ["market_a", "market_b", "market_c"]
    await feed.start(markets, mock_provider)
    
    # Let it run for a few seconds
    await asyncio.sleep(3)
    
    # Get state
    state = feed.get_state()
    print(f"\nFeed state:")
    print(f"  Status: {state.status.value}")
    print(f"  Markets tracked: {state.markets_tracked}")
    print(f"  Updates received: {state.updates_received}")
    print(f"  Errors: {state.errors_count}")
    
    # Get latest prices
    prices = feed.get_all_latest_prices()
    print(f"\nLatest prices:")
    for market_id, price in prices.items():
        print(f"  {market_id}: ${price:.2f}")
    
    # Get price history
    history = feed.get_price_history("market_a", limit=10)
    print(f"\nPrice history (market_a): {len(history)} points")
    
    # Stop feed
    await feed.stop()
    
    print("\nTest completed!")


if __name__ == "__main__":
    import sys
    asyncio.run(test_realtime_feed())
