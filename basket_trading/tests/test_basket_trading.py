"""
Tests for basket trading module.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from basket_trading import (
    BasketConfig,
    BasketWallet,
    BasketSignal,
    Topic,
    WalletBasket,
    WalletPerformance,
)
from basket_trading.builder import WalletBasketBuilder
from basket_trading.cluster_detector import ClusterDetector
from basket_trading.engine import BasketEngine
from basket_trading.ranker import BasketRanker
from api_clients.base import Trade, OrderSide
from api_clients.polymarket.data_api import DataAPIClient
from bot_filtering import BotFilterResult


class TestBasketConfig:
    """Tests for BasketConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BasketConfig()

        assert config.min_wallet_age_months == 6
        assert config.min_consensus_pct == 0.80
        assert config.max_price_band_pct == 0.05
        assert config.max_spread_pct == 0.10

    def test_custom_config(self):
        """Test custom configuration."""
        config = BasketConfig(
            min_wallet_age_months=12,
            min_consensus_pct=0.90,
            max_price_band_pct=0.03,
        )

        assert config.min_wallet_age_months == 12
        assert config.min_consensus_pct == 0.90
        assert config.max_price_band_pct == 0.03


class TestWalletPerformance:
    """Tests for WalletPerformance dataclass."""

    def test_weighted_score_default(self):
        """Test weighted score calculation with default weights."""
        config = BasketConfig()

        perf = WalletPerformance(
            wallet_address="0x123",
            first_trade_date=datetime(2024, 1, 1),
            last_trade_date=datetime(2024, 1, 15),
            total_trades_all_time=100,
            win_rate_7d=0.60,
            win_rate_30d=0.55,
            win_rate_all_time=0.50,
            trades_7d=10,
            trades_30d=50,
        )

        score = perf.weighted_score(config)
        expected = (
            0.60 * 0.40 +
            0.55 * 0.35 +
            0.50 * 0.25
        )

        assert abs(score - expected) < 0.01

    def test_is_eligible_new_wallet(self):
        """Test eligibility for new wallet (under 6 months)."""
        config = BasketConfig()

        perf = WalletPerformance(
            wallet_address="0x123",
            first_trade_date=datetime.utcnow() - timedelta(days=100),
            last_trade_date=datetime.utcnow(),
            total_trades_all_time=50,
            win_rate_7d=0.60,
            win_rate_30d=0.55,
            trades_7d=10,
            trades_30d=50,
        )

        assert not perf.is_eligible(config)

    def test_is_eligible_old_wallet(self):
        """Test eligibility for old wallet (6+ months)."""
        config = BasketConfig()

        perf = WalletPerformance(
            wallet_address="0x123",
            first_trade_date=datetime.utcnow() - timedelta(days=200),
            last_trade_date=datetime.utcnow(),
            total_trades_all_time=100,
            win_rate_7d=0.60,
            win_rate_30d=0.55,
            trades_7d=10,
            trades_30d=50,
        )

        assert perf.is_eligible(config)


class TestBasketSignal:
    """Tests for BasketSignal dataclass."""

    def test_is_valid_high_consensus(self):
        """Test signal validity with high consensus."""
        config = BasketConfig()

        signal = BasketSignal(
            basket_topic=Topic.GEOPOLITICS,
            market_id="market_123",
            market_title="Test Market",
            outcome="YES",
            side=OrderSide.BUY,
            consensus_pct=0.90,
            participating_wallets=40,
            total_wallets=50,
            avg_entry_price=0.55,
            price_band_low=0.53,
            price_band_high=0.57,
            spread_pct=0.04,
            signal_strength=0.85,
        )

        assert signal.is_valid(config)

    def test_is_valid_low_consensus(self):
        """Test signal validity with low consensus."""
        config = BasketConfig()

        signal = BasketSignal(
            basket_topic=Topic.GEOPOLITICS,
            market_id="market_123",
            market_title="Test Market",
            outcome="YES",
            side=OrderSide.BUY,
            consensus_pct=0.70,
            participating_wallets=35,
            total_wallets=50,
            avg_entry_price=0.55,
            price_band_low=0.53,
            price_band_high=0.57,
            spread_pct=0.04,
            signal_strength=0.70,
        )

        assert not signal.is_valid(config)

    def test_is_valid_wide_price_band(self):
        """Test signal validity with wide price band."""
        config = BasketConfig()

        signal = BasketSignal(
            basket_topic=Topic.GEOPOLITICS,
            market_id="market_123",
            market_title="Test Market",
            outcome="YES",
            side=OrderSide.BUY,
            consensus_pct=0.90,
            participating_wallets=40,
            total_wallets=50,
            avg_entry_price=0.55,
            price_band_low=0.50,
            price_band_high=0.62,
            spread_pct=0.11,
            signal_strength=0.85,
        )

        assert not signal.is_valid(config)

    def test_to_dict(self):
        """Test signal conversion to dictionary."""
        signal = BasketSignal(
            basket_topic=Topic.GEOPOLITICS,
            market_id="market_123",
            market_title="Test Market",
            outcome="YES",
            side=OrderSide.BUY,
            consensus_pct=0.85,
            participating_wallets=40,
            total_wallets=50,
            avg_entry_price=0.55,
            price_band_low=0.53,
            price_band_high=0.57,
            spread_pct=0.04,
            signal_strength=0.85,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
        )

        result = signal.to_dict()

        assert result["basket_topic"] == "geopolitics"
        assert result["consensus_pct"] == "85.0%"
        assert result["signal_strength"] == "0.85"


@pytest.mark.asyncio
class TestBasketBuilder:
    """Tests for WalletBasketBuilder."""

    async def test_build_basket_empty_addresses(self, mock_data_client):
        """Test building basket with empty addresses."""
        builder = WalletBasketBuilder(mock_data_client)

        basket = await builder.build_basket(
            Topic.GEOPOLITICS,
            [],
        )

        assert basket.topic == Topic.GEOPOLITICS
        assert len(basket.wallets) == 0

    async def test_build_basket_with_addresses(self, mock_data_client, sample_trades):
        """Test building basket with sample addresses."""
        mock_data_client.get_trades.return_value = sample_trades

        builder = WalletBasketBuilder(mock_data_client)

        basket = await builder.build_basket(
            Topic.GEOPOLITICS,
            ["0x123", "0x456"],
        )

        assert basket.topic == Topic.GEOPOLITICS
        assert len(basket.wallets) > 0


class TestClusterDetector:
    """Tests for ClusterDetector."""

    def test_detect_clusters_empty_wallets(self):
        """Test cluster detection with empty wallets."""
        config = BasketConfig()
        detector = ClusterDetector(config)

        clusters = detector.detect_clusters([], {})

        assert len(clusters) == 0

    def test_calculate_wallet_similarity_identical(self):
        """Test similarity calculation for identical wallets."""
        config = BasketConfig()
        detector = ClusterDetector(config)

        trade1 = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=10.0,
            price=0.50,
            total_value=5.0,
            timestamp=datetime(2024, 1, 1),
        )

        trade2 = Trade(
            trade_id="t2",
            market_id="m1",
            trader_address="0x456",
            side=OrderSide.BUY,
            quantity=10.0,
            price=0.50,
            total_value=5.0,
            timestamp=datetime(2024, 1, 1),
        )

        similarity = detector._calculate_wallet_similarity(
            "0x123",
            "0x456",
            [trade1],
            [trade2],
        )

        assert similarity == 1.0


class TestBasketEngine:
    """Tests for BasketEngine."""

    async def test_scan_for_signals_no_baskets(self, mock_data_client):
        """Test signal scanning with no baskets."""
        engine = BasketEngine(mock_data_client)

        signals = await engine.scan_for_signals()

        assert len(signals) == 0

    def test_get_recent_signals(self):
        """Test getting recent signals."""
        engine = BasketEngine(MagicMock())

        now = datetime.utcnow()

        old_signal = BasketSignal(
            basket_topic=Topic.GEOPOLITICS,
            market_id="market_1",
            market_title="Old Market",
            outcome="YES",
            side=OrderSide.BUY,
            consensus_pct=0.90,
            participating_wallets=40,
            total_wallets=50,
            avg_entry_price=0.55,
            price_band_low=0.53,
            price_band_high=0.57,
            spread_pct=0.04,
            signal_strength=0.85,
            created_at=now - timedelta(hours=25),
        )

        recent_signal = BasketSignal(
            basket_topic=Topic.GEOPOLITICS,
            market_id="market_2",
            market_title="Recent Market",
            outcome="YES",
            side=OrderSide.BUY,
            consensus_pct=0.85,
            participating_wallets=35,
            total_wallets=50,
            avg_entry_price=0.52,
            price_band_low=0.50,
            price_band_high=0.54,
            spread_pct=0.04,
            signal_strength=0.80,
            created_at=now - timedelta(hours=1),
        )

        engine.signals = [old_signal, recent_signal]

        recent = engine.get_recent_signals(hours=24)

        assert len(recent) == 1
        assert recent[0].market_id == "market_2"


class TestBasketRanker:
    """Tests for BasketRanker."""

    def test_rank_basket(self):
        """Test basket ranking."""
        config = BasketConfig()
        ranker = BasketRanker(config)

        perf1 = WalletPerformance(
            wallet_address="0x111",
            first_trade_date=datetime(2024, 1, 1),
            last_trade_date=datetime(2024, 1, 15),
            total_trades_all_time=100,
            win_rate_7d=0.70,
            win_rate_30d=0.65,
            win_rate_all_time=0.60,
            trades_7d=10,
            trades_30d=50,
        )

        perf2 = WalletPerformance(
            wallet_address="0x222",
            first_trade_date=datetime(2024, 1, 1),
            last_trade_date=datetime(2024, 1, 15),
            total_trades_all_time=100,
            win_rate_7d=0.55,
            win_rate_30d=0.50,
            win_rate_all_time=0.45,
            trades_7d=10,
            trades_30d=50,
        )

        wallet1 = BasketWallet(
            wallet_address="0x111",
            topic=Topic.GEOPOLITICS,
            performance=perf1,
        )

        wallet2 = BasketWallet(
            wallet_address="0x222",
            topic=Topic.GEOPOLITICS,
            performance=perf2,
        )

        basket = WalletBasket(
            topic=Topic.GEOPOLITICS,
            wallets=[wallet1, wallet2],
        )

        ranked = ranker.rank_basket(basket)

        assert len(ranked) == 2
        assert ranked[0].wallet_address == "0x111"
        assert ranked[1].wallet_address == "0x222"

    def test_get_top_wallets(self):
        """Test getting top N wallets."""
        config = BasketConfig()
        ranker = BasketRanker(config)

        wallets = []
        for i in range(10):
            perf = WalletPerformance(
                wallet_address=f"0x{i:03d}",
                first_trade_date=datetime(2024, 1, 1),
                last_trade_date=datetime(2024, 1, 15),
                total_trades_all_time=100 - i * 5,
                win_rate_7d=0.70 - i * 0.05,
                win_rate_30d=0.65 - i * 0.05,
                win_rate_all_time=0.60 - i * 0.05,
                trades_7d=10,
                trades_30d=50,
            )

            wallet = BasketWallet(
                wallet_address=f"0x{i:03d}",
                topic=Topic.GEOPOLITICS,
                performance=perf,
            )
            wallets.append(wallet)

        basket = WalletBasket(
            topic=Topic.GEOPOLITICS,
            wallets=wallets,
        )

        top_5 = ranker.get_top_wallets(basket, top_n=5)

        assert len(top_5) == 5
