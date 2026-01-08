"""
Market Data Cache for Sandbox Simulation.

Provides in-memory and persistent caching for market data with TTL support,
LRU eviction, and automatic refresh capabilities.
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Callable
from collections import OrderedDict
from enum import Enum

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Caching policies for different data types."""
    TIME_TO_LIVE = "ttl"  # Expire after TTL
    LRU = "lru"  # Least recently used eviction
    FIFO = "fifo"  # First in, first out
    WRITE_THROUGH = "write_through"  # Sync with disk
    REFRESH_AHEAD = "refresh_ahead"  # Auto-refresh before expiry


@dataclass
class CacheEntry(Generic[TypeVar("T")]):
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    is_dirty: bool = False  # Needs write to disk
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self):
        """Update access time and count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def get_age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()


@dataclass
class CacheConfig:
    """Configuration for market data cache."""
    # Size limits
    max_entries: int = 10000
    max_memory_mb: float = 100.0

    # TTL settings (seconds)
    default_ttl: float = 300.0  # 5 minutes
    price_ttl: float = 60.0  # 1 minute for prices
    metadata_ttl: float = 3600.0  # 1 hour for metadata

    # Policy
    eviction_policy: CachePolicy = CachePolicy.LRU
    refresh_ahead_fraction: float = 0.2  # Refresh when 20% of TTL remaining

    # Persistence
    persistence_enabled: bool = True
    persistence_dir: str = "./sandbox_cache"
    persistence_interval: float = 60.0  # Save every 60 seconds

    # Threading
    background_refresh: bool = True
    max_background_tasks: int = 10


class MarketDataCache:
    """
    High-performance cache for market data.

    Features:
    - Multiple caching policies (LRU, FIFO, TTL)
    - Automatic persistence to disk
    - Thread-safe operations
    - Memory-aware eviction
    - Refresh-ahead for frequently accessed data
    - Statistics and monitoring

    Usage:
        cache = MarketDataCache()
        cache.set("bitcoin_price", 50000.0, ttl=60.0)
        price = cache.get("bitcoin_price")
        cache.clear()
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._persist_lock = threading.Lock()
        self._refresh_tasks: Dict[str, asyncio.Task] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "expirations": 0,
            "persist_writes": 0,
            "persist_reads": 0,
        }
        self._last_persist_time = time.time()

        # Ensure persistence directory exists
        if self.config.persistence_enabled:
            Path(self.config.persistence_dir).mkdir(parents=True, exist_ok=True)

        # Start background persistence thread
        if self.config.persistence_enabled and self.config.persistence_interval > 0:
            self._start_persistence_thread()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return default

            if entry.is_expired():
                self._remove(key)
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return default

            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    async def get_async(self, key: str, default: Any = None) -> Any:
        """Async version of get."""
        return self.get(key, default)

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_dirty: bool = False,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            TTL: Time to live in seconds (None for default)
            metadata: Optional metadata
            is_dirty: Needs write to disk

        Returns:
            True if set successfully
        """
        ttl = ttl or self.config.default_ttl

        with self._lock:
            # Create or update entry
            if key in self._cache:
                entry = self._cache[key]
                entry.value = value
                entry.ttl_seconds = ttl
                entry.is_dirty = is_dirty
                if metadata:
                    entry.metadata.update(metadata)
            else:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl_seconds=ttl,
                    is_dirty=is_dirty,
                    metadata=metadata or {},
                )
                self._cache[key] = entry

            self._stats["sets"] += 1

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Check if we need to evict
            self._maybe_evict()

            return True

    async def set_async(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Async version of set."""
        return self.set(key, value, ttl, metadata)

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
        return False

    def clear(self, pattern: Optional[str] = None):
        """
        Clear cache, optionally matching a pattern.

        Args:
            pattern: Glob pattern to match keys (e.g., "bitcoin*")
        """
        with self._lock:
            if pattern is None:
                self._cache.clear()
                self._stats["evictions"] += len(self._cache)
            else:
                import fnmatch
                keys_to_remove = [
                    k for k in self._cache
                    if fnmatch.fnmatch(k, pattern)
                ]
                for key in keys_to_remove:
                    self._remove(key, lock=False)
                self._stats["evictions"] += len(keys_to_remove)

    def clear_expired(self) -> int:
        """Remove all expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove(key, lock=False)
            self._stats["expirations"] += len(expired_keys)
            return len(expired_keys)

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[float] = None,
    ) -> Any:
        """
        Get value or set using factory function.

        Args:
            key: Cache key
            factory: Function to create value if not cached
            ttl: Time to live

        Returns:
            Cached or newly created value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[float] = None,
    ) -> Any:
        """Async version of get_or_set."""
        return self.get_or_set(key, factory, ttl)

    def refresh_ahead(
        self,
        key: str,
        fetcher: Callable[[], Any],
        ttl: Optional[float] = None,
    ):
        """
        Set up refresh-ahead for a key.

        Entry will be automatically refreshed when TTL fraction remains.

        Args:
            key: Cache key
            fetcher: Function to fetch new value
            ttl: Time to live
        """
        ttl = ttl or self.config.default_ttl

        def do_refresh():
            try:
                new_value = fetcher()
                self.set(key, new_value, ttl)
            except Exception as e:
                logger.warning(f"Refresh-ahead failed for {key}: {e}")

        self.set(key, fetcher(), ttl)

        # Schedule background refresh
        refresh_delay = ttl * (1 - self.config.refresh_ahead_fraction)
        task = asyncio.create_task(self._schedule_refresh(key, fetcher, ttl, refresh_delay))
        self._refresh_tasks[key] = task

    async def _schedule_refresh(
        self,
        key: str,
        fetcher: Callable[[], Any],
        ttl: float,
        delay: float,
    ):
        """Schedule a background refresh."""
        await asyncio.sleep(delay)

        if key in self._cache and not self._cache[key].is_expired():
            try:
                new_value = fetcher()
                self.set(key, new_value, ttl)

                # Schedule next refresh
                next_delay = ttl * (1 - self.config.refresh_ahead_fraction)
                task = asyncio.create_task(
                    self._schedule_refresh(key, fetcher, ttl, next_delay)
                )
                self._refresh_tasks[key] = task
            except Exception as e:
                logger.warning(f"Background refresh failed for {key}: {e}")

    def cancel_refresh(self, key: str):
        """Cancel background refresh for a key."""
        if key in self._refresh_tasks:
            self._refresh_tasks[key].cancel()
            del self._refresh_tasks[key]

    def bulk_set(self, items: Dict[str, Any], ttl: Optional[float] = None):
        """
        Set multiple values at once.

        Args:
            items: Dict of key-value pairs
            ttl: Time to live for all items
        """
        for key, value in items.items():
            self.set(key, value, ttl)

    def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values at once.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dict of found key-value pairs
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally matching a pattern."""
        with self._lock:
            if pattern is None:
                return list(self._cache.keys())
            else:
                import fnmatch
                return [
                    k for k in self._cache
                    if fnmatch.fnmatch(k, pattern)
                ]

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired()

    def get_ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL for a key."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.ttl_seconds is None:
                return None
            remaining = entry.ttl_seconds - entry.get_age_seconds()
            return max(0, remaining)

    def set_ttl(self, key: str, ttl: float):
        """Update TTL for a key."""
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                entry.ttl_seconds = ttl

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total * 100
                if total > 0 else 0
            )

            return {
                "size": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": f"{hit_rate:.2f}%",
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
                "memory_usage_mb": self._estimate_memory_mb(),
                "persist_writes": self._stats["persist_writes"],
                "persist_reads": self._stats["persist_reads"],
            }

    def reset_stats(self):
        """Reset statistics counters."""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0

    async def persist_to_disk(self, filepath: Optional[str] = None):
        """Save cache to disk."""
        if not self.config.persistence_enabled:
            return

        filepath = filepath or os.path.join(
            self.config.persistence_dir,
            "market_cache.json"
        )

        with self._persist_lock:
            data = {}
            with self._lock:
                for key, entry in self._cache.items():
                    if not entry.is_expired():
                        data[key] = {
                            "value": entry.value,
                            "created_at": entry.created_at.isoformat(),
                            "ttl_seconds": entry.ttl_seconds,
                            "metadata": entry.metadata,
                        }

            try:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
                self._stats["persist_writes"] += 1
                logger.debug(f"Persisted {len(data)} entries to {filepath}")
            except Exception as e:
                logger.error(f"Failed to persist cache: {e}")

    async def load_from_disk(self, filepath: Optional[str] = None) -> int:
        """Load cache from disk."""
        if not self.config.persistence_enabled:
            return 0

        filepath = filepath or os.path.join(
            self.config.persistence_dir,
            "market_cache.json"
        )

        if not os.path.exists(filepath):
            return 0

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            loaded = 0
            with self._lock:
                for key, entry_data in data.items():
                    created_at = datetime.fromisoformat(entry_data["created_at"])
                    age = (datetime.utcnow() - created_at).total_seconds()
                    ttl = entry_data.get("ttl_seconds")

                    # Skip if expired
                    if ttl is not None and age > ttl:
                        continue

                    self._cache[key] = CacheEntry(
                        key=key,
                        value=entry_data["value"],
                        created_at=created_at,
                        ttl_seconds=ttl,
                        metadata=entry_data.get("metadata", {}),
                    )
                    loaded += 1

            self._stats["persist_reads"] += 1
            logger.info(f"Loaded {loaded} entries from {filepath}")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return 0

    def _remove(self, key: str, lock: bool = True):
        """Remove entry from cache."""
        if lock:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
        else:
            if key in self._cache:
                del self._cache[key]

        self.cancel_refresh(key)

    def _maybe_evict(self):
        """Evict entries if cache is full."""
        if len(self._cache) > self.config.max_entries:
            if self.config.eviction_policy == CachePolicy.LRU:
                # Remove oldest entries
                num_to_remove = len(self._cache) - self.config.max_entries
                for _ in range(num_to_remove):
                    if self._cache:
                        self._cache.popitem(last=False)
                        self._stats["evictions"] += 1
            elif self.config.eviction_policy == CachePolicy.FIFO:
                # Remove oldest entries by creation time
                num_to_remove = len(self._cache) - self.config.max_entries
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].created_at
                )
                for i in range(num_to_remove):
                    key = sorted_entries[i][0]
                    del self._cache[key]
                    self._stats["evictions"] += 1

        # Check memory limit
        if self._estimate_memory_mb() > self.config.max_memory_mb:
            self._evict_by_memory()

    def _evict_by_memory(self, target_mb: Optional[float] = None):
        """Evict entries to reduce memory usage."""
        target_mb = target_mb or self.config.max_memory_mb * 0.8
        current_mb = self._estimate_memory_mb()

        while current_mb > target_mb and len(self._cache) > 0:
            # Remove least recently used
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            current_mb = self._estimate_memory_mb()

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        import sys
        total = 0
        with self._lock:
            for entry in self._cache.values():
                total += sys.getsizeof(entry.value)
                total += sys.getsizeof(str(entry.key))
        return total / (1024 * 1024)

    def _start_persistence_thread(self):
        """Start background thread for periodic persistence."""
        def persist_loop():
            while True:
                time.sleep(self.config.persistence_interval)
                try:
                    asyncio.run(self.persist_to_disk())
                except Exception as e:
                    logger.error(f"Persistence thread error: {e}")

        thread = threading.Thread(target=persist_loop, daemon=True)
        thread.start()


# Specialized cache for market prices
class PriceCache(MarketDataCache):
    """
    Specialized cache for market prices with shorter TTL.
    """

    def __init__(self):
        config = CacheConfig(
            default_ttl=60.0,  # 1 minute for prices
            max_entries=1000,
            persistence_enabled=True,
            persistence_dir="./sandbox_cache/prices",
        )
        super().__init__(config)

    def set_price(
        self,
        market_id: str,
        price: float,
        timestamp: Optional[datetime] = None,
    ):
        """Set price for a market."""
        key = f"price:{market_id}"
        metadata = {"timestamp": (timestamp or datetime.utcnow()).isoformat()}
        self.set(key, price, ttl=self.config.price_ttl, metadata=metadata)

    def get_price(self, market_id: str) -> Optional[float]:
        """Get price for a market."""
        key = f"price:{market_id}"
        return self.get(key)

    def get_price_with_metadata(self, market_id: str) -> Dict[str, Any]:
        """Get price and metadata for a market."""
        key = f"price:{market_id}"
        entry = self._cache.get(key)
        if entry is None:
            return {"price": None, "timestamp": None, "age": None}
        return {
            "price": entry.value,
            "timestamp": entry.metadata.get("timestamp"),
            "age": entry.get_age_seconds(),
        }

    def set_bulk_prices(self, prices: Dict[str, float]):
        """Set multiple prices at once."""
        for market_id, price in prices.items():
            self.set_price(market_id, price)


# Specialized cache for order book data
class OrderBookCache(MarketDataCache):
    """
    Specialized cache for order book data.
    """

    def __init__(self):
        config = CacheConfig(
            default_ttl=30.0,  # 30 seconds for order books
            max_entries=500,
            persistence_enabled=False,  # Don't persist order books
        )
        super().__init__(config)

    def set_orderbook(
        self,
        market_id: str,
        orderbook: Dict[str, Any],
    ):
        """Set order book for a market."""
        key = f"orderbook:{market_id}"
        self.set(key, orderbook, ttl=self.config.default_ttl)

    def get_orderbook(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get order book for a market."""
        key = f"orderbook:{market_id}"
        return self.get(key)


# Factory function for creating appropriate cache
def create_cache(cache_type: str = "market") -> MarketDataCache:
    """
    Create appropriate cache based on type.

    Args:
        cache_type: Type of cache ("market", "price", "orderbook")

    Returns:
        Configured cache instance
    """
    if cache_type == "price":
        return PriceCache()
    elif cache_type == "orderbook":
        return OrderBookCache()
    else:
        return MarketDataCache()


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)

    # Create cache
    cache = MarketDataCache()

    # Set some values
    cache.set("bitcoin", 50000.0, ttl=60.0)
    cache.set("ethereum", 3000.0, ttl=60.0)
    cache.set("metadata:bitcoin", {"market_cap": "1T"}, ttl=3600.0)

    # Get values
    print(f"Bitcoin: ${cache.get('bitcoin')}")
    print(f"Ethereum: ${cache.get('ethereum')}")
    print(f"Bitcoin metadata: {cache.get('metadata:bitcoin')}")

    # Stats
    print(f"\nCache stats: {cache.get_stats()}")

    # Test expiration
    print(f"\nBitcoin TTL: {cache.get_ttl('bitcoin')} seconds")

    # Bulk operations
    prices = cache.bulk_get(["bitcoin", "ethereum"])
    print(f"\nBulk get: {prices}")
