"""
Wallet Basket Builder.
Constructs topic-based wallets baskets with filtering and ranking.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from api_clients.base import Trade
from api_clients.polymarket.data_api import DataAPIClient
from bot_filtering import BotFilter, BotFilterConfig
from trader_identification import TraderIdentificationEngine, TraderSelectionConfig

from . import (
    BasketConfig,
    BasketWallet,
    WalletBasket,
    WalletPerformance,
    Topic,
)


class WalletBasketBuilder:
    """
    Builds topic-based wallet baskets from historical trade data.

    Process:
    1. Fetch wallet trade history from Data API
    2. Calculate performance metrics (7d, 30d, all-time)
    3. Apply bot filtering to exclude automated traders
    4. Filter by age, activity, win rate criteria
    5. Group wallets by trading topic
    6. Rank wallets within each topic
    7. Return validated basket per topic
    """

    def __init__(
        self,
        data_client: DataAPIClient,
        config: Optional[BasketConfig] = None,
    ):
        """Initialize basket builder."""
        self.data_client = data_client
        self.config = config or BasketConfig()
        self.bot_filter = BotFilter(BotFilterConfig())
        self.trader_identification = TraderIdentificationEngine(TraderSelectionConfig())

    async def build_basket(
        self,
        topic: Topic,
        wallet_addresses: List[str],
        now: Optional[datetime] = None,
    ) -> WalletBasket:
        """Build a wallet basket for a specific topic."""
        now = now or datetime.utcnow()

        logger.info(f"Building {topic.value} basket from {len(wallet_addresses)} wallets")

        tasks = [self._analyze_wallet(addr, now) for addr in wallet_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        basket_wallets: List[BasketWallet] = []
        for addr, result in zip(wallet_addresses, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to analyze {addr[:8]}...: {result}")
                continue

            if result and result.is_eligible(self.config):
                basket_wallets.append(result)

        logger.info(f"Found {len(basket_wallets)} eligible wallets for {topic.value}")

        basket = WalletBasket(
            topic=topic,
            wallets=basket_wallets,
            created_at=now,
            last_updated=now,
            config=self.config,
        )

        if not basket.is_valid():
            logger.warning(
                f"Basket size {basket.basket_size()} outside valid range "
                f"[{self.config.min_basket_size}, {self.config.max_basket_size}]"
            )

        return basket

    async def build_all_baskets(
        self,
        wallet_topic_mapping: Dict[Topic, List[str]],
        now: Optional[datetime] = None,
    ) -> Dict[Topic, WalletBasket]:
        """Build baskets for all topics."""
        now = now or datetime.utcnow()
        baskets: Dict[Topic, WalletBasket] = {}

        for topic, addresses in wallet_topic_mapping.items():
            if not addresses:
                logger.info(f"No addresses provided for {topic.value}")
                continue

            basket = await self.build_basket(topic, addresses, now)
            baskets[topic] = basket

        return baskets

    async def _analyze_wallet(
        self,
        wallet_address: str,
        now: datetime,
    ) -> Optional[BasketWallet]:
        """Analyze a single wallet for basket eligibility."""
        trades = await self.data_client.get_trades(
            user_address=wallet_address,
            limit=1000,
        )

        if not trades:
            logger.info(f"No trades found for {wallet_address[:8]}...")
            return None

        performance = self._calculate_wallet_performance(wallet_address, trades, now)
        bot_result = self.bot_filter.analyze_trades(trades)

        basket_wallet = BasketWallet(
            wallet_address=wallet_address,
            topic=Topic.OTHER,
            performance=performance,
            bot_filter_result=bot_result,
        )

        return basket_wallet

    def _calculate_wallet_performance(
        self,
        wallet_address: str,
        trades: List[Trade],
        now: datetime,
    ) -> WalletPerformance:
        """Calculate performance metrics across time windows."""
        trades_sorted = sorted(trades, key=lambda t: t.timestamp)

        first_trade = trades_sorted[0]
        last_trade = trades_sorted[-1]

        trades_7d = [t for t in trades if (now - t.timestamp).days <= 7]
        trades_30d = [t for t in trades if (now - t.timestamp).days <= 30]

        win_rate_all_time = self._calculate_win_rate(trades_sorted)
        win_rate_7d = self._calculate_win_rate(trades_7d)
        win_rate_30d = self._calculate_win_rate(trades_30d)

        avg_entry_vs_final = self._calculate_avg_entry_vs_final(trades_sorted)

        return WalletPerformance(
            wallet_address=wallet_address,
            first_trade_date=first_trade.timestamp,
            last_trade_date=last_trade.timestamp,
            total_trades_all_time=len(trades_sorted),

            win_rate_all_time=win_rate_all_time,
            avg_entry_vs_final_price_all_time=avg_entry_vs_final,

            win_rate_7d=win_rate_7d,
            trades_7d=len(trades_7d),
            volume_7d=sum(t.total_value for t in trades_7d),

            win_rate_30d=win_rate_30d,
            trades_30d=len(trades_30d),
            volume_30d=sum(t.total_value for t in trades_30d),

            sharpe_ratio=self._calculate_sharpe_ratio(trades_sorted),
            max_drawdown=self._calculate_max_drawdown(trades_sorted),
            profit_factor=self._calculate_profit_factor(trades_sorted),
        )

    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0

        winning_trades = sum(1 for t in trades if self._is_winning_trade(t))
        return winning_trades / len(trades)

    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if trade is winning."""
        if trade.outcome == "YES":
            return trade.price > 0.5
        else:
            return trade.price < 0.5

    def _calculate_avg_entry_vs_final(self, trades: List[Trade]) -> float:
        """Calculate average entry price vs final price."""
        if not trades:
            return 0.0

        entry_vs_final_diffs = []

        for i, trade in enumerate(trades):
            if i + 1 < len(trades):
                next_price = trades[i + 1].price
                diff = abs(trade.price - next_price) / trade.price if trade.price > 0 else 0
                entry_vs_final_diffs.append(diff)

        return sum(entry_vs_final_diffs) / len(entry_vs_final_diffs) if entry_vs_final_diffs else 0.0

    def _calculate_sharpe_ratio(self, trades: List[Trade]) -> float:
        """Calculate Sharpe ratio from trades."""
        if len(trades) < 2:
            return 0.0

        returns = [
            (t.total_value - t.fees) / t.total_value
            for t in trades
            if t.total_value > 0
        ]

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std = variance ** 0.5 if variance > 0 else 0.001

        return mean_return / std if std > 0 else 0.0

    def _calculate_max_drawdown(self, trades: List[Trade]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0.0

        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_dd = 0.0

        for trade in trades:
            pnl = trade.total_value - trade.fees if self._is_winning_trade(trade) else -(trade.total_value + trade.fees)
            cumulative_pnl += pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            dd = peak_pnl - cumulative_pnl
            max_dd = max(max_dd, dd)

        total_value = sum(t.total_value for t in trades)
        return max_dd / total_value if total_value > 0 else 0.0

    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate profit factor from trades."""
        if not trades:
            return 0.0

        gross_profits = sum(
            (t.total_value - t.fees)
            for t in trades
            if self._is_winning_trade(t)
        )
        gross_losses = abs(sum(
            (t.total_value + t.fees)
            for t in trades
            if not self._is_winning_trade(t)
        ))

        return gross_profits / gross_losses if gross_losses > 0 else float('inf')


logger = __import__('logging').getLogger(__name__)
