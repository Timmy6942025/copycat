"""
Basket Trading Orchestrator.
Integrates basket system with existing CopyCat modules.
"""

import asyncio
from typing import Dict, List, Optional

from api_clients.polymarket.data_api import DataAPIClient

from . import (
    BasketConfig,
    BasketSignal,
    BasketTradeResult,
    Topic,
    WalletBasket,
)
from .builder import WalletBasketBuilder
from .cluster_detector import ClusterDetector
from .engine import BasketEngine
from .ranker import BasketRanker


class BasketTradingOrchestrator:
    """
    Orchestrates basket-based trading system.

    Integrates:
    - WalletBasketBuilder: Builds topic-based baskets
    - ClusterDetector: Detects copycat clusters
    - BasketRanker: Ranks wallets within baskets
    - BasketEngine: Generates consensus signals
    """

    def __init__(
        self,
        data_client: DataAPIClient,
        config: Optional[BasketConfig] = None,
    ):
        """Initialize basket trading orchestrator."""
        self.config = config or BasketConfig()
        self.data_client = data_client

        self.builder = WalletBasketBuilder(data_client, self.config)
        self.ranker = BasketRanker(self.config)
        self.engine = BasketEngine(data_client, self.config)
        self.cluster_detector = ClusterDetector(self.config)

        self.baskets: Dict[Topic, WalletBasket] = {}

    async def initialize_topic_baskets(
        self,
        topic_wallets: Dict[Topic, List[str]],
    ) -> Dict[Topic, WalletBasket]:
        """Initialize all topic baskets."""
        logger.info(f"Initializing {len(topic_wallets)} topic baskets")

        tasks = [
            self.builder.build_basket(topic, addresses)
            for topic, addresses in topic_wallets.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (topic, _), basket in zip(topic_wallets.items(), results):
            if isinstance(basket, Exception):
                logger.error(f"Failed to build {topic.value} basket: {basket}")
                continue

            if basket and basket.is_valid():
                await self.engine.add_basket(basket)
                self.baskets[topic] = basket
                logger.info(
                    f"Initialized {topic.value} basket with "
                    f"{basket.basket_size()} wallets"
                )

        return self.baskets

    async def detect_and_assign_clusters(self) -> None:
        """Detect copycat clusters across all baskets."""
        logger.info("Detecting copycat clusters across all baskets")

        for topic, basket in self.baskets.items():
            if not basket.wallets:
                continue

            wallet_trades: Dict[str, List] = {}
            for wallet in basket.wallets:
                trades = await self.data_client.get_trades(
                    user_address=wallet.wallet_address,
                    limit=200,
                )
                wallet_trades[wallet.wallet_address] = trades

            clusters = self.cluster_detector.detect_clusters(
                basket.wallets,
                wallet_trades,
            )

            if clusters:
                updated_wallets = self.cluster_detector.assign_clusters_to_basket(
                    basket.wallets,
                    clusters,
                )
                basket.wallets = updated_wallets
                logger.info(
                    f"Assigned {len(clusters)} clusters to {topic.value} basket"
                )

    async def rank_all_baskets(self) -> Dict[Topic, List]:
        """Rank wallets within all baskets."""
        logger.info("Ranking wallets within all baskets")

        ranked_baskets: Dict[Topic, List] = {}

        for topic, basket in self.baskets.items():
            ranked = self.ranker.rank_basket(basket)
            ranked_baskets[topic] = ranked
            logger.info(
                f"Ranked {len(ranked)} wallets in {topic.value} basket"
            )

        return ranked_baskets

    async def scan_for_signals(
        self,
        topics: Optional[List[Topic]] = None,
    ) -> List[BasketSignal]:
        """Scan all baskets for consensus signals."""
        signals = await self.engine.scan_for_signals(topics)

        logger.info(f"Scanned baskets, found {len(signals)} new signals")

        return signals

    async def execute_top_signals(
        self,
        limit: int = 5,
        quantity_per_signal: float = 100.0,
    ) -> List[BasketTradeResult]:
        """Execute top N strongest basket signals."""
        recent_signals = self.engine.get_recent_signals(hours=1)

        sorted_signals = sorted(
            recent_signals,
            key=lambda s: s.signal_strength,
            reverse=True,
        )

        top_signals = sorted_signals[:limit]

        logger.info(f"Executing top {len(top_signals)} signals")

        tasks = [
            self.engine.execute_signal(signal, quantity_per_signal)
            for signal in top_signals
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        trade_results: List[BasketTradeResult] = []
        for signal, result in zip(top_signals, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to execute signal: {result}")
                continue

            trade_results.append(result)

        executed_count = sum(1 for r in trade_results if r.executed)
        logger.info(f"Executed {executed_count}/{len(top_signals)} signals")

        return trade_results

    def get_basket_stats(self) -> Dict[str, Dict]:
        """Get statistics for all baskets."""
        stats = {}

        for topic, basket in self.baskets.items():
            stats[topic.value] = {
                "total_wallets": len(basket.wallets),
                "effective_wallets": basket.basket_size(),
                "active_wallets": len(basket.active_wallets()),
                "is_valid": basket.is_valid(),
                "created_at": basket.created_at.isoformat(),
                "last_updated": basket.last_updated.isoformat(),
            }

        return stats

    def get_basket(self, topic: Topic) -> Optional[WalletBasket]:
        """Get basket for specific topic."""
        return self.baskets.get(topic)

    def list_baskets(self) -> List[Topic]:
        """List all initialized basket topics."""
        return list(self.baskets.keys())


logger = __import__('logging').getLogger(__name__)
