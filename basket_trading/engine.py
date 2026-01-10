"""
Basket Trading Engine.
Generates consensus-based trading signals from wallet baskets.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from api_clients.base import Trade, OrderSide, MarketData
from api_clients.polymarket.data_api import DataAPIClient

from . import (
    BasketConfig,
    BasketSignal,
    BasketPosition,
    BasketWallet,
    BasketTradeResult,
    Topic,
    WalletBasket,
)
from .cluster_detector import ClusterDetector


class BasketEngine:
    """
    Generates trading signals based on basket consensus.

    Signal generation logic:
    1. Monitor basket wallets' current positions
    2. Group positions by market and outcome
    3. Calculate consensus for each (market, outcome) pair
    4. Check 80%+ threshold and price band constraints
    5. Validate spread isn't "cooked"
    6. Generate signal if all criteria met
    """

    def __init__(
        self,
        data_client: DataAPIClient,
        config: Optional[BasketConfig] = None,
    ):
        """Initialize basket engine."""
        self.data_client = data_client
        self.config = config or BasketConfig()
        self.cluster_detector = ClusterDetector(self.config)

        self.baskets: Dict[Topic, WalletBasket] = {}
        self.active_positions: Dict[str, List[BasketPosition]] = defaultdict(list)
        self.signals: List[BasketSignal] = []

    async def add_basket(self, basket: WalletBasket) -> None:
        """Add a wallet basket to monitor."""
        topic = basket.topic
        self.baskets[topic] = basket
        logger.info(f"Added {topic.value} basket with {basket.basket_size()} wallets")

    async def remove_basket(self, topic: Topic) -> None:
        """Remove a basket from monitoring."""
        if topic in self.baskets:
            del self.baskets[topic]
            logger.info(f"Removed {topic.value} basket")

    async def scan_for_signals(
        self,
        topics: Optional[List[Topic]] = None,
        now: Optional[datetime] = None,
    ) -> List[BasketSignal]:
        """Scan all baskets for consensus signals."""
        now = now or datetime.utcnow()

        topics_to_scan = topics or list(self.baskets.keys())
        logger.info(f"Scanning {len(topics_to_scan)} baskets for signals")

        new_signals = []

        for topic in topics_to_scan:
            if topic not in self.baskets:
                continue

            signals = await self._scan_basket_for_signals(topic, now)
            new_signals.extend(signals)

        self.signals.extend(new_signals)
        logger.info(f"Generated {len(new_signals)} new signals")

        return new_signals

    async def _scan_basket_for_signals(
        self,
        topic: Topic,
        now: datetime,
    ) -> List[BasketSignal]:
        """Scan a single basket for consensus signals."""
        basket = self.baskets[topic]

        active_wallets = basket.active_wallets(now)
        if len(active_wallets) < self.config.min_basket_size:
            logger.debug(
                f"Insufficient active wallets for {topic.value}: "
                f"{len(active_wallets)} < {self.config.min_basket_size}"
            )
            return []

        cluster_reps = basket.get_cluster_representatives()
        addresses = [w.wallet_address for w in cluster_reps]

        tasks = [self._get_wallet_positions(addr, now) for addr in addresses]
        positions_by_wallet = await asyncio.gather(*tasks, return_exceptions=True)

        all_positions: List[BasketPosition] = []
        for addr, positions in zip(addresses, positions_by_wallet):
            if isinstance(positions, Exception):
                logger.warning(f"Failed to get positions for {addr[:8]}...: {positions}")
                continue

            if positions:
                all_positions.extend(positions)

        if not all_positions:
            logger.debug(f"No active positions found for {topic.value} basket")
            return []

        signals = self._generate_signals_from_positions(
            topic,
            all_positions,
            cluster_reps,
            now,
        )

        return signals

    async def _get_wallet_positions(
        self,
        wallet_address: str,
        now: datetime,
    ) -> List[BasketPosition]:
        """Get current positions for a wallet."""
        positions_data = await self.data_client.get_positions(
            user_address=wallet_address,
            redeemable=False,
            limit=100,
        )

        positions = []

        for pos in positions_data:
            if not pos.current_value or pos.current_value < 1.0:
                continue

            basket_pos = BasketPosition(
                wallet_address=wallet_address,
                market_id=pos.market_id,
                market_title=pos.title,
                outcome=pos.outcome,
                side=OrderSide.BUY if pos.size > 0 else OrderSide.SELL,
                quantity=abs(pos.size),
                entry_price=pos.avg_price,
                timestamp=now,
                final_price=pos.current_price,
            )
            positions.append(basket_pos)

        return positions

    def _generate_signals_from_positions(
        self,
        topic: Topic,
        positions: List[BasketPosition],
        basket_wallets: List[BasketWallet],
        now: datetime,
    ) -> List[BasketSignal]:
        """Generate trading signals from basket positions."""
        signals = []

        positions_by_market = defaultdict(lambda: {"YES": [], "NO": []})

        for pos in positions:
            positions_by_market[pos.market_id][pos.outcome].append(pos)

        for market_id, outcome_positions in positions_by_market.items():
            for outcome, outcome_list in outcome_positions.items():
                if not outcome_list:
                    continue

                signal = self._create_signal_for_market_outcome(
                    topic,
                    market_id,
                    outcome,
                    outcome_list,
                    basket_wallets,
                    now,
                )

                if signal and signal.is_valid(self.config):
                    signals.append(signal)

        return signals

    def _create_signal_for_market_outcome(
        self,
        topic: Topic,
        market_id: str,
        outcome: str,
        positions: List[BasketPosition],
        basket_wallets: List[BasketWallet],
        now: datetime,
    ) -> Optional[BasketSignal]:
        """Create signal for specific market and outcome."""
        if len(positions) < self.config.min_basket_size:
            return None

        participating_addresses = set(p.wallet_address for p in positions)
        participating_wallets = [
            w for w in basket_wallets
            if w.wallet_address in participating_addresses
        ]

        total_wallets = len(basket_wallets)
        consensus_pct = len(participating_wallets) / total_wallets if total_wallets > 0 else 0.0

        if consensus_pct < self.config.min_consensus_pct:
            return None

        avg_entry_price = sum(p.entry_price for p in positions) / len(positions)
        price_band_low = min(p.entry_price for p in positions)
        price_band_high = max(p.entry_price for p in positions)

        price_range = (price_band_high - price_band_low) / avg_entry_price
        if price_range > self.config.max_price_band_pct:
            logger.debug(
                f"Price band too wide for {market_id}: {price_range:.1%} "
                f"> {self.config.max_price_band_pct:.1%}"
            )
            return None

        spread_pct = price_range

        signal_strength = min(1.0, consensus_pct * (1 - price_range))

        side = OrderSide.BUY if positions[0].side == OrderSide.BUY else OrderSide.SELL

        signal = BasketSignal(
            basket_topic=topic,
            market_id=market_id,
            market_title=positions[0].market_title,
            outcome=outcome,
            side=side,
            consensus_pct=consensus_pct,
            participating_wallets=len(participating_wallets),
            total_wallets=total_wallets,
            avg_entry_price=avg_entry_price,
            price_band_low=price_band_low,
            price_band_high=price_band_high,
            spread_pct=spread_pct,
            signal_strength=signal_strength,
            created_at=now,
        )

        logger.info(
            f"Signal generated for {market_id[:16]}...: "
            f"{consensus_pct:.1%} consensus, {signal_strength:.2f} strength"
        )

        return signal

    def get_recent_signals(
        self,
        hours: int = 24,
    ) -> List[BasketSignal]:
        """Get signals from last N hours."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=hours)

        return [s for s in self.signals if s.created_at >= cutoff]

    async def execute_signal(
        self,
        signal: BasketSignal,
        quantity: float,
    ) -> BasketTradeResult:
        """Execute a basket signal (paper trading)."""
        logger.info(f"Executing signal for {signal.market_id[:16]}...")

        result = BasketTradeResult(
            signal=signal,
            executed=False,
            quantity=quantity,
            execution_price=signal.avg_entry_price,
            timestamp=datetime.utcnow(),
        )

        if not signal.is_valid(self.config):
            result.rejection_reason = "Signal validation failed"
            return result

        result.executed = True
        result.order_id = f"basket_signal_{signal.market_id[:8]}_{datetime.utcnow().timestamp()}"

        logger.info(
            f"Executed basket signal: {signal.market_title} "
            f"@ {result.execution_price:.4f}, qty={quantity:.2f}"
        )

        return result


logger = __import__('logging').getLogger(__name__)
