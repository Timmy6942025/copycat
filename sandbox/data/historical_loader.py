"""
Historical Data Loader for Sandbox Simulation.

Loads and processes historical market data for backtesting and simulation.
Supports data from multiple sources with automatic fallback.
"""

import asyncio
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from enum import Enum

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Sources of historical market data."""
    COINGECKO = "coingecko"
    YAHOO_FINANCE = "yahoo_finance"
    POLYMARKET = "polymarket"
    CSV = "csv"
    JSON = "json"
    SIMULATED = "simulated"


class DataFormat(Enum):
    """Formats for historical data storage."""
    OHLCV = "ohlcv"  # Open, High, Low, Close, Volume
    TICK = "tick"    # Individual trades
    SNAPSHOT = "snapshot"  # Price snapshots


@dataclass
class HistoricalDataPoint:
    """Single data point in historical time series."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    market_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "market_id": self.market_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalDataPoint":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data.get("volume", 0.0),
            market_id=data.get("market_id", ""),
        )


@dataclass
class HistoricalDataSeries:
    """Time series of historical market data."""
    market_id: str
    data_source: DataSource
    data_format: DataFormat
    data_points: List[HistoricalDataPoint] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbol: str = ""

    def __len__(self) -> int:
        return len(self.data_points)

    def __iter__(self) -> Generator[HistoricalDataPoint, None, None]:
        for point in self.data_points:
            yield point

    @property
    def is_empty(self) -> bool:
        return len(self.data_points) == 0

    def get_date_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range of the data series."""
        if not self.data_points:
            return None, None
        return (
            self.data_points[0].timestamp,
            self.data_points[-1].timestamp,
        )

    def filter_by_date(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> "HistoricalDataSeries":
        """Filter data series by date range."""
        filtered_points = [
            point for point in self.data_points
            if (start_date is None or point.timestamp >= start_date)
            and (end_date is None or point.timestamp <= end_date)
        ]
        return HistoricalDataSeries(
            market_id=self.market_id,
            data_source=self.data_source,
            data_format=self.data_format,
            data_points=filtered_points,
            symbol=self.symbol,
        )

    def resample(self, interval_hours: int = 1) -> "HistoricalDataSeries":
        """Resample data to specified interval in hours."""
        if not self.data_points:
            return self

        resampled: List[HistoricalDataPoint] = []
        current_bucket: Optional[HistoricalDataPoint] = None

        for point in sorted(self.data_points, key=lambda x: x.timestamp):
            bucket_time = point.timestamp.replace(
                minute=0, second=0, microsecond=0
            )
            bucket_time = bucket_time - timedelta(
                minutes=bucket_time.minute % interval_hours,
                seconds=0,
            )

            if current_bucket is None or current_bucket.timestamp != bucket_time:
                if current_bucket is not None:
                    resampled.append(current_bucket)
                current_bucket = HistoricalDataPoint(
                    timestamp=bucket_time,
                    open=point.close,
                    high=point.close,
                    low=point.close,
                    close=point.close,
                    volume=point.volume,
                    market_id=point.market_id,
                )
            else:
                current_bucket.high = max(current_bucket.high, point.close)
                current_bucket.low = min(current_bucket.low, point.close)
                current_bucket.close = point.close
                current_bucket.volume += point.volume

        if current_bucket is not None:
            resampled.append(current_bucket)

        return HistoricalDataSeries(
            market_id=self.market_id,
            data_source=self.data_source,
            data_format=self.data_format,
            data_points=resampled,
            symbol=self.symbol,
        )


@dataclass
class HistoricalLoaderConfig:
    """Configuration for historical data loader."""
    # Data storage
    data_dir: str = "./sandbox_data"
    max_series_in_memory: int = 100

    # Data source settings
    default_data_source: DataSource = DataSource.SIMULATED
    prefer_cached: bool = True

    # API settings (for external sources)
    api_rate_limit: float = 1.0  # Seconds between API calls
    api_timeout: float = 30.0  # Timeout for API requests

    # Fallback settings
    use_simulated_fallback: bool = True


class HistoricalDataLoader:
    """
    Loads and manages historical market data for backtesting.

    Supports multiple data sources with automatic fallback:
    - Local CSV/JSON files
    - External APIs (CoinGecko, Yahoo Finance)
    - Simulated data for testing

    Features:
    - Automatic data caching
    - Date range filtering
    - Data resampling
    - Memory management
    """

    def __init__(self, config: Optional[HistoricalLoaderConfig] = None):
        self.config = config or HistoricalLoaderConfig()
        self._data_cache: Dict[str, HistoricalDataSeries] = {}
        self._load_statistics: Dict[str, int] = {}

        # Ensure data directory exists
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)

    async def load_market_data(
        self,
        market_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        data_source: Optional[DataSource] = None,
        data_format: DataFormat = DataFormat.OHLCV,
        force_reload: bool = False,
    ) -> HistoricalDataSeries:
        """
        Load historical market data for a specific market.

        Args:
            market_id: Identifier for the market (e.g., "bitcoin", "AAPL")
            start_date: Start of date range (default: 30 days ago)
            end_date: End of date range (default: now)
            data_source: Source of data (default: from config)
            data_format: Format of data points
            force_reload: Skip cache and reload from source

        Returns:
            HistoricalDataSeries with all data points
        """
        data_source = data_source or self.config.default_data_source
        start_date = start_date or datetime.utcnow() - timedelta(days=30)
        end_date = end_date or datetime.utcnow()

        cache_key = f"{market_id}_{data_source.value}_{start_date.date()}_{end_date.date()}"

        # Check cache
        if not force_reload and cache_key in self._data_cache:
            logger.debug(f"Using cached data for {market_id}")
            cached = self._data_cache[cache_key]
            return cached.filter_by_date(start_date, end_date)

        # Try loading from different sources
        series = await self._try_load_from_sources(
            market_id=market_id,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
            data_format=data_format,
        )

        # Cache the result
        if not self._is_memory_full():
            self._data_cache[cache_key] = series
            self._load_statistics[cache_key] = 0

        return series

    async def _try_load_from_sources(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
        data_source: DataSource,
        data_format: DataFormat,
    ) -> HistoricalDataSeries:
        """Try loading from multiple sources in order."""
        sources_to_try = [
            DataSource.CSV,
            DataSource.JSON,
            data_source,
        ]

        if self.config.use_simulated_fallback:
            sources_to_try.append(DataSource.SIMULATED)

        last_error = None
        for source in sources_to_try:
            try:
                if source == DataSource.CSV:
                    return await self._load_from_csv(market_id, start_date, end_date)
                elif source == DataSource.JSON:
                    return await self._load_from_json(market_id, start_date, end_date)
                elif source == DataSource.COINGECKO:
                    return await self._load_from_coingecko(market_id, start_date, end_date)
                elif source == DataSource.YAHOO_FINANCE:
                    return await self._load_from_yahoo(market_id, start_date, end_date)
                elif source == DataSource.SIMULATED:
                    return self._generate_simulated_data(
                        market_id, start_date, end_date, data_format
                    )
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load from {source}: {e}")
                continue

        # If all sources failed, return simulated data
        logger.warning(f"All sources failed, using simulated data: {last_error}")
        return self._generate_simulated_data(market_id, start_date, end_date, data_format)

    async def _load_from_csv(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalDataSeries:
        """Load historical data from CSV file."""
        csv_path = Path(self.config.data_dir) / f"{market_id}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        data_points: List[HistoricalDataPoint] = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = datetime.fromisoformat(row["timestamp"])
                if start_date <= timestamp <= end_date:
                    data_points.append(HistoricalDataPoint(
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume", 0)),
                        market_id=market_id,
                    ))

        data_points.sort(key=lambda x: x.timestamp)

        return HistoricalDataSeries(
            market_id=market_id,
            data_source=DataSource.CSV,
            data_format=DataFormat.OHLCV,
            data_points=data_points,
            symbol=market_id,
        )

    async def _load_from_json(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalDataSeries:
        """Load historical data from JSON file."""
        json_path = Path(self.config.data_dir) / f"{market_id}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        data_points = [
            HistoricalDataPoint.from_dict(point)
            for point in data
            if start_date <= datetime.fromisoformat(point["timestamp"]) <= end_date
        ]

        data_points.sort(key=lambda x: x.timestamp)

        return HistoricalDataSeries(
            market_id=market_id,
            data_source=DataSource.JSON,
            data_format=DataFormat.OHLCV,
            data_points=data_points,
            symbol=market_id,
        )

    async def _load_from_coingecko(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalDataSeries:
        """Load historical data from CoinGecko API."""
        try:
            import aiohttp

            # Convert dates to Unix timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())

            # CoinGecko historical market chart endpoint
            url = f"https://api.coingecko.com/api/v3/coins/{market_id}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": start_ts,
                "to": end_ts,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"API returned {response.status}")
                    data = await response.json()

            # Parse prices (format: [timestamp, price])
            data_points = []
            for timestamp, price in data.get("prices", []):
                dt = datetime.fromtimestamp(timestamp / 1000)
                if start_date <= dt <= end_date:
                    data_points.append(HistoricalDataPoint(
                        timestamp=dt,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=0,
                        market_id=market_id,
                    ))

            return HistoricalDataSeries(
                market_id=market_id,
                data_source=DataSource.COINGECKO,
                data_format=DataFormat.OHLCV,
                data_points=data_points,
                symbol=market_id,
            )

        except ImportError:
            raise Exception("aiohttp required for CoinGecko API")

    async def _load_from_yahoo(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalDataSeries:
        """Load historical data from Yahoo Finance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(market_id)
            hist = ticker.history(start=start_date, end=end_date)

            data_points = [
                HistoricalDataPoint(
                    timestamp=index.to_pydatetime(),
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    close=row["Close"],
                    volume=row["Volume"],
                    market_id=market_id,
                )
                for index, row in hist.iterrows()
            ]

            return HistoricalDataSeries(
                market_id=market_id,
                data_source=DataSource.YAHOO_FINANCE,
                data_format=DataFormat.OHLCV,
                data_points=data_points,
                symbol=market_id,
            )

        except ImportError:
            raise Exception("yfinance required for Yahoo Finance data")

    def _generate_simulated_data(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
        data_format: DataFormat,
        initial_price: float = 0.50,
        volatility: float = 0.02,
    ) -> HistoricalDataSeries:
        """
        Generate simulated historical data for testing.

        Uses random walk with mean reversion.
        """
        import random

        data_points: List[HistoricalDataPoint] = []
        current_price = initial_price
        current_time = start_date

        while current_time <= end_date:
            # Random walk with slight upward drift
            change = random.gauss(0, volatility)
            current_price = current_price * (1 + change)

            # Ensure price stays positive
            current_price = max(0.01, current_price)

            data_points.append(HistoricalDataPoint(
                timestamp=current_time,
                open=current_price * (1 + random.uniform(-0.001, 0.001)),
                high=current_price * (1 + random.uniform(0, 0.005)),
                low=current_price * (1 - random.uniform(0, 0.005)),
                close=current_price,
                volume=random.uniform(1000, 10000),
                market_id=market_id,
            ))

            # Advance time by 1 hour
            current_time += timedelta(hours=1)

        return HistoricalDataSeries(
            market_id=market_id,
            data_source=DataSource.SIMULATED,
            data_format=data_format,
            data_points=data_points,
            symbol=market_id,
        )

    async def save_to_csv(self, series: HistoricalDataSeries, filepath: Optional[str] = None) -> str:
        """Save historical data series to CSV file."""
        if filepath is None:
            filepath = Path(self.config.data_dir) / f"{series.market_id}.csv"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "open", "high", "low", "close", "volume", "market_id"
            ])
            writer.writeheader()
            for point in series.data_points:
                writer.writerow(point.to_dict())

        logger.info(f"Saved {len(series.data_points)} points to {filepath}")
        return str(filepath)

    async def save_to_json(self, series: HistoricalDataSeries, filepath: Optional[str] = None) -> str:
        """Save historical data series to JSON file."""
        if filepath is None:
            filepath = Path(self.config.data_dir) / f"{series.market_id}.json"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = [point.to_dict() for point in series.data_points]

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(series.data_points)} points to {filepath}")
        return str(filepath)

    def clear_cache(self, market_id: Optional[str] = None):
        """Clear cached data."""
        if market_id is None:
            self._data_cache.clear()
            logger.info("Cleared all cached data")
        else:
            keys_to_remove = [k for k in self._data_cache if k.startswith(market_id)]
            for key in keys_to_remove:
                del self._data_cache[key]
            logger.info(f"Cleared cache for {market_id}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data."""
        return {
            "cached_series": len(self._data_cache),
            "total_data_points": sum(len(s) for s in self._data_cache.values()),
            "load_statistics": self._load_statistics.copy(),
        }

    def _is_memory_full(self) -> bool:
        """Check if memory cache is at capacity."""
        return len(self._data_cache) >= self.config.max_series_in_memory

    def preload_market_list(self, market_ids: List[str], days: int = 30):
        """Preload data for a list of markets."""
        for market_id in market_ids:
            asyncio.create_task(
                self.load_market_data(
                    market_id,
                    start_date=datetime.utcnow() - timedelta(days=days),
                    data_source=self.config.default_data_source,
                )
            )

    async def get_market_list(self, data_dir: Optional[str] = None) -> List[str]:
        """Get list of available markets from local files."""
        data_dir = data_dir or self.config.data_dir
        path = Path(data_dir)

        if not path.exists():
            return []

        markets = []
        for file in path.glob("*.csv"):
            markets.append(file.stem)
        for file in path.glob("*.json"):
            if file.stem not in markets:
                markets.append(file.stem)

        return sorted(markets)


# Convenience function for quick data loading
async def load_historical_data(
    market_id: str,
    days: int = 30,
    data_dir: str = "./sandbox_data",
) -> HistoricalDataSeries:
    """
    Quick function to load historical data for a market.

    Args:
        market_id: Market identifier (e.g., "bitcoin", "AAPL")
        days: Number of days of history
        data_dir: Directory for cached data

    Returns:
        HistoricalDataSeries with market data
    """
    config = HistoricalLoaderConfig(data_dir=data_dir)
    loader = HistoricalDataLoader(config)

    return await loader.load_market_data(
        market_id=market_id,
        start_date=datetime.utcnow() - timedelta(days=days),
    )


if __name__ == "__main__":
    # Example usage
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        market_id = sys.argv[1]
    else:
        market_id = "bitcoin"

    async def demo():
        loader = HistoricalDataLoader()

        print(f"Loading data for {market_id}...")
        series = await loader.load_market_data(market_id)

        print(f"Loaded {len(series)} data points")
        print(f"Date range: {series.get_date_range()}")

        if not series.is_empty:
            first = series.data_points[0]
            last = series.data_points[-1]
            print(f"Price range: ${first.close:.2f} - ${last.close:.2f}")

    asyncio.run(demo())
