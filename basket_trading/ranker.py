"""
Basket Ranking System.
Ranks wallets within baskets by performance metrics.
"""

from typing import List, Optional

from . import BasketConfig, BasketWallet, WalletBasket


class BasketRanker:
    """
    Ranks wallet baskets based on weighted performance scores.

    Ranking factors:
    - Weighted win rate (7d, 30d, all-time)
    - Average entry vs final price
    - Consistency (Sharpe ratio, max drawdown)
    - Trading volume
    """

    def __init__(self, config: Optional[BasketConfig] = None):
        """Initialize basket ranker."""
        self.config = config or BasketConfig()

    def rank_basket(
        self,
        basket: WalletBasket,
        now: Optional[datetime] = None,
    ) -> List[BasketWallet]:
        """Rank wallets within a basket."""
        wallets = basket.wallets.copy()

        for wallet in wallets:
            wallet.score = self.calculate_score(wallet, self.config)

        sorted_wallets = sorted(
            wallets,
            key=lambda w: w.score(self.config),
            reverse=True,
        )

        logger.info(
            f"Ranked {len(sorted_wallets)} wallets for {basket.topic.value} basket"
        )

        return sorted_wallets

    def get_top_wallets(
        self,
        basket: WalletBasket,
        top_n: int = 10,
    ) -> List[BasketWallet]:
        """Get top N wallets from basket."""
        ranked = self.rank_basket(basket)
        return ranked[:top_n]

    def calculate_score(
        self,
        wallet: BasketWallet,
        config: BasketConfig,
    ) -> float:
        """Calculate overall wallet score."""
        perf = wallet.performance

        weighted_win_rate = perf.weighted_score(config)

        consistency_score = self._calculate_consistency_score(perf)
        volume_score = self._calculate_volume_score(perf)
        entry_price_score = perf.avg_entry_vs_final_price_all_time

        overall_score = (
            weighted_win_rate * (
                config.win_rate_7d_weight +
                config.win_rate_30d_weight +
                config.win_rate_all_time_weight
            ) +
            config.avg_entry_vs_final_weight * entry_price_score +
            config.consistency_weight * consistency_score +
            config.volume_weight * volume_score
        )

        return min(1.0, max(0.0, overall_score))

    def _calculate_consistency_score(
        self,
        performance: 'WalletPerformance',
    ) -> float:
        """Calculate consistency score from risk metrics."""
        sharpe_normalized = min(1.0, performance.sharpe_ratio / 3.0)
        drawdown_penalty = performance.max_drawdown
        consistency_score = sharpe_normalized * (1 - drawdown_penalty)

        return max(0.0, min(1.0, consistency_score))

    def _calculate_volume_score(
        self,
        performance: 'WalletPerformance',
    ) -> float:
        """Calculate volume score from trading activity."""
        volume_30d = performance.volume_30d

        if volume_30d < 100:
            return 0.2
        elif volume_30d < 1000:
            return 0.5
        elif volume_30d < 10000:
            return 0.8
        else:
            return 1.0

    def filter_by_score(
        self,
        basket: WalletBasket,
        min_score: float = 0.5,
    ) -> List[BasketWallet]:
        """Filter basket wallets by minimum score."""
        return [
            w for w in basket.wallets
            if w.score(self.config) >= min_score
        ]

    def filter_eligible(
        self,
        basket: WalletBasket,
    ) -> List[BasketWallet]:
        """Get only eligible wallets from basket."""
        return [
            w for w in basket.wallets
            if w.is_eligible(self.config)
        ]


logger = __import__('logging').getLogger(__name__)
