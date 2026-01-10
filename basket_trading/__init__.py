"""
Topic-based wallet basket construction and signal generation for prediction markets.

Builds "baskets" of wallets grouped by topic (e.g., geopolitics, crypto, sports).
Signals are generated when 80%+ of basket converges on same outcome.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import logging

from api_clients.base import Trade, OrderSide
from trader_identification import TraderIdentificationResult, TraderPerformance
from bot_filtering import BotFilterResult


logger = logging.getLogger(__name__)


class Topic(Enum):
    """Trading topic categories."""
    GEOPOLITICS = "geopolitics"
    POLITICS = "politics"
    ELECTIONS = "elections"
    CRYPTO = "crypto"
    ECONOMICS = "economics"
    FINANCE = "finance"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    CLIMATE = "climate"
    HEALTH = "health"
    OTHER = "other"


@dataclass
class BasketConfig:
    """Configuration for basket construction and trading."""
    min_wallet_age_months: int = 6
    min_trades_7d: int = 3
    min_trades_30d: int = 10
    min_win_rate_7d: float = 0.45
    min_win_rate_30d: float = 0.50

    max_hft_score: float = 0.5
    max_arbitrage_score: float = 0.5
    max_pattern_score: float = 0.5
    max_micro_trades_per_day: int = 1000

    min_basket_size: int = 10
    max_basket_size: int = 50
    min_consensus_pct: float = 0.80

    max_price_band_pct: float = 0.05
    max_spread_pct: float = 0.10
    min_basket_participation_pct: float = 0.60

    win_rate_7d_weight: float = 0.40
    win_rate_30d_weight: float = 0.35
    win_rate_all_time_weight: float = 0.25
    avg_entry_vs_final_weight: float = 0.30
    consistency_weight: float = 0.20
    volume_weight: float = 0.10

    cluster_similarity_threshold: float = 0.85
    max_cluster_size: int = 3


@dataclass
class WalletPerformance:
    """Performance metrics for a wallet across time windows."""
    wallet_address: str
    first_trade_date: datetime
    last_trade_date: datetime
    total_trades_all_time: int

    win_rate_all_time: float = 0.0
    avg_entry_vs_final_price_all_time: float = 0.0

    win_rate_7d: float = 0.0
    trades_7d: int = 0
    volume_7d: float = 0.0

    win_rate_30d: float = 0.0
    trades_30d: int = 0
    volume_30d: float = 0.0

    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0

    def weighted_score(
        self,
        config: BasketConfig,
    ) -> float:
        """Calculate weighted score based on recent performance."""
        score = (
            self.win_rate_7d * config.win_rate_7d_weight +
            self.win_rate_30d * config.win_rate_30d_weight +
            self.win_rate_all_time * config.win_rate_all_time_weight
        )
        return min(1.0, max(0.0, score))

    def is_eligible(self, config: BasketConfig, now: Optional[datetime] = None) -> bool:
        """Check if wallet meets eligibility criteria."""
        now = now or datetime.utcnow()

        age_months = (now - self.first_trade_date).days / 30.44
        if age_months < config.min_wallet_age_months:
            return False

        if self.trades_7d < config.min_trades_7d:
            return False

        if self.trades_30d < config.min_trades_30d:
            return False

        if self.win_rate_7d < config.min_win_rate_7d:
            return False

        if self.win_rate_30d < config.min_win_rate_30d:
            return False

        return True


@dataclass
class BasketWallet:
    """A single wallet within a basket."""
    wallet_address: str
    topic: Topic
    performance: WalletPerformance
    bot_filter_result: Optional[BotFilterResult] = None
    cluster_id: Optional[str] = None
    is_cluster_representative: bool = False

    def score(self, config: BasketConfig) -> float:
        """Calculate overall score for ranking."""
        base_score = self.performance.weighted_score(config)

        if self.bot_filter_result and self.bot_filter_result.is_bot:
            base_score *= 0.5

        if self.cluster_id and not self.is_cluster_representative:
            base_score *= 0.3

        return base_score

    def is_eligible(self, config: BasketConfig) -> bool:
        """Check if wallet is eligible for basket inclusion."""
        if not self.performance.is_eligible(config):
            return False

        if self.bot_filter_result:
            if self.bot_filter_result.hft_score > config.max_hft_score:
                return False
            if self.bot_filter_result.arbitrage_score > config.max_arbitrage_score:
                return False
            if self.bot_filter_result.pattern_score > config.max_pattern_score:
                return False

        return True


@dataclass
class WalletBasket:
    """A basket of wallets for a specific topic."""
    topic: Topic
    wallets: List[BasketWallet] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    config: Optional[BasketConfig] = None

    def get_cluster_representatives(self) -> List[BasketWallet]:
        """Get only cluster representative wallets."""
        return [w for w in self.wallets if w.is_cluster_representative or not w.cluster_id]

    def active_wallets(self, now: Optional[datetime] = None) -> List[BasketWallet]:
        """Get wallets with recent activity (last 30 days)."""
        now = now or datetime.utcnow()
        cutoff = now - timedelta(days=30)
        return [w for w in self.wallets if w.performance.last_trade_date >= cutoff]

    def basket_size(self) -> int:
        """Get effective basket size (cluster reps only)."""
        return len(self.get_cluster_representatives())

    def is_valid(self, config: Optional[BasketConfig] = None) -> bool:
        """Check if basket meets minimum requirements."""
        config = config or self.config
        if not config:
            return False

        size = self.basket_size()
        return config.min_basket_size <= size <= config.max_basket_size


@dataclass
class BasketPosition:
    """Position that a wallet has taken on a market."""
    wallet_address: str
    market_id: str
    market_title: str
    outcome: str
    side: OrderSide
    quantity: float
    entry_price: float
    timestamp: datetime
    final_price: Optional[float] = None


@dataclass
class BasketSignal:
    """A trading signal generated by basket consensus."""
    basket_topic: Topic
    market_id: str
    market_title: str
    outcome: str
    side: OrderSide
    consensus_pct: float
    participating_wallets: int
    total_wallets: int
    avg_entry_price: float
    price_band_low: float
    price_band_high: float
    spread_pct: float
    signal_strength: float
    created_at: datetime = field(default_factory=datetime.utcnow)

    def is_valid(self, config: BasketConfig) -> bool:
        """Check if signal meets all validity criteria."""
        if self.consensus_pct < config.min_consensus_pct:
            return False

        participation_pct = self.participating_wallets / self.total_wallets if self.total_wallets > 0 else 0
        if participation_pct < config.min_basket_participation_pct:
            return False

        price_range = (self.price_band_high - self.price_band_low) / self.avg_entry_price
        if price_range > config.max_price_band_pct:
            return False

        if self.spread_pct > config.max_spread_pct:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/serialization."""
        return {
            "basket_topic": self.basket_topic.value,
            "market_id": self.market_id,
            "market_title": self.market_title,
            "outcome": self.outcome,
            "side": self.side.value,
            "consensus_pct": f"{self.consensus_pct:.1%}",
            "participating_wallets": self.participating_wallets,
            "total_wallets": self.total_wallets,
            "avg_entry_price": self.avg_entry_price,
            "price_band_low": self.price_band_low,
            "price_band_high": self.price_band_high,
            "price_range_pct": f"{((self.price_band_high - self.price_band_low) / self.avg_entry_price):.1%}",
            "spread_pct": f"{self.spread_pct:.1%}",
            "signal_strength": f"{self.signal_strength:.2f}",
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class BasketTradeResult:
    """Result of executing a basket signal."""
    signal: BasketSignal
    executed: bool
    order_id: Optional[str] = None
    quantity: float = 0.0
    execution_price: float = 0.0
    fees: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    rejection_reason: Optional[str] = None
