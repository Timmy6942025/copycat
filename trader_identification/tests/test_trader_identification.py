"""
Unit tests for Trader Identification Engine.
Tests trader analysis, performance calculation, and reputation scoring.
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass
from unittest.mock import Mock

import sys
sys.path.insert(0, '/home/timmy/copycat')

from trader_identification import (
    TraderIdentificationEngine,
    TraderSelectionConfig,
    InsiderDetectionResult,
    BotDetectionResult,
    TraderIdentificationResult,
)
from api_clients.base import (
    Trade,
    Trader,
    TraderPerformance,
    OrderSide,
)


class TestTraderIdentificationEngine:
    """Test TraderIdentificationEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a trader identification engine."""
        config = TraderSelectionConfig(
            min_win_rate=0.55,
            min_trades=10,
            max_drawdown=0.25,
            min_sharpe_ratio=0.5,
            min_profit_factor=1.0,
        )
        return TraderIdentificationEngine(config=config)

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        base_time = datetime.utcnow()
        return [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.60,
                total_value=60.0,
                fees=1.0,
                timestamp=base_time,
                outcome="YES",
            ),
            Trade(
                trade_id="t2",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.SELL,
                quantity=100.0,
                price=0.70,
                total_value=70.0,
                fees=1.5,
                timestamp=base_time + timedelta(hours=2),
                outcome="YES",
            ),
            Trade(
                trade_id="t3",
                market_id="m2",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=50.0,
                price=0.40,
                total_value=20.0,
                fees=0.5,
                timestamp=base_time + timedelta(hours=4),
                outcome="YES",
            ),
            Trade(
                trade_id="t4",
                market_id="m2",
                trader_address="0x123",
                side=OrderSide.SELL,
                quantity=50.0,
                price=0.55,
                total_value=27.5,
                fees=0.6,
                timestamp=base_time + timedelta(hours=6),
                outcome="YES",
            ),
            Trade(
                trade_id="t5",
                market_id="m3",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=75.0,
                price=0.45,
                total_value=33.75,
                fees=0.7,
                timestamp=base_time + timedelta(hours=8),
                outcome="YES",
            ),
        ]

    def test_calculate_performance(self, engine, sample_trades):
        """Test performance calculation from trades."""
        perf = engine.calculate_performance("0x123", sample_trades)

        assert perf.trader_address == "0x123"
        assert perf.total_trades == 5
        assert perf.win_rate >= 0  # At least some wins
        assert perf.total_pnl > 0  # Profitable in this sample

    def test_detect_insider_trading(self, engine, sample_trades):
        """Test insider trading detection."""
        result = engine.detect_insider_trading(sample_trades)

        assert isinstance(result, InsiderDetectionResult)
        assert 0 <= result.early_position_score <= 1
        assert 0 <= result.event_correlation_score <= 1
        assert 0 <= result.category_expertise_score <= 1

    def test_detect_bot_activity(self, engine, sample_trades):
        """Test bot activity detection."""
        result = engine.detect_bot_activity(sample_trades)

        assert isinstance(result, BotDetectionResult)
        assert 0 <= result.hft_score <= 1
        assert 0 <= result.arbitrage_score <= 1
        assert 0 <= result.pattern_score <= 1

    def test_calculate_reputation_score(self, engine):
        """Test reputation score calculation."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=100.0,
            win_rate=0.7,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            total_trades=50,
        )

        insider = InsiderDetectionResult(early_position_score=0.3)
        bot = BotDetectionResult(pattern_score=0.2)

        score = engine.calculate_reputation_score(perf, insider, bot)

        assert 0 <= score <= 1
        assert score > 0.5  # Should be positive for good trader

    def test_calculate_confidence(self, engine):
        """Test confidence calculation."""
        perf = TraderPerformance(trader_address="0x123", total_trades=100)
        insider = InsiderDetectionResult(confidence=0.8)
        bot = BotDetectionResult(confidence=0.7)

        confidence = engine.calculate_confidence(perf, insider, bot)

        assert 0 <= confidence <= 1

    def test_evaluate_suitability(self, engine):
        """Test suitability evaluation."""
        perf = TraderPerformance(
            trader_address="0x123",
            win_rate=0.65,
            max_drawdown=0.15,
            sharpe_ratio=1.0,
            profit_factor=2.0,
            total_pnl=500.0,
            avg_hold_time_hours=24.0,
            total_trades=50,
        )

        insider = InsiderDetectionResult(is_insider_likely=False)
        bot = BotDetectionResult(is_bot_likely=False)
        reputation = 0.7

        result = engine.evaluate_suitability(perf, insider, bot, reputation)

        assert "is_suitable" in result
        assert "selected" in result
        assert "rejected" in result
        assert len(result["selected"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_trader(self, engine, sample_trades):
        """Test complete trader analysis."""
        result = await engine.analyze_trader("0x123", sample_trades)

        assert isinstance(result, TraderIdentificationResult)
        assert result.trader_address == "0x123"
        assert isinstance(result.performance, TraderPerformance)


class TestTraderSelectionConfig:
    """Test TraderSelectionConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TraderSelectionConfig()

        assert config.min_win_rate == 0.55
        assert config.min_trades == 10
        assert config.max_drawdown == 0.25
        assert config.min_sharpe_ratio == 0.5
        assert config.min_profit_factor == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TraderSelectionConfig(
            min_win_rate=0.6,
            min_trades=20,
            max_drawdown=0.15,
            min_sharpe_ratio=1.0,
        )

        assert config.min_win_rate == 0.6
        assert config.min_trades == 20
        assert config.max_drawdown == 0.15


class TestInsiderDetectionResult:
    """Test InsiderDetectionResult class."""

    def test_default_result(self):
        """Test default values."""
        result = InsiderDetectionResult()

        assert result.early_position_score == 0.0
        assert result.event_correlation_score == 0.0
        assert result.category_expertise_score == 0.0
        assert result.is_insider_likely is False
        assert result.confidence == 0.0

    def test_custom_result(self):
        """Test custom values."""
        result = InsiderDetectionResult(
            early_position_score=0.8,
            event_correlation_score=0.7,
            is_insider_likely=True,
            confidence=0.9,
        )

        assert result.early_position_score == 0.8
        assert result.is_insider_likely is True


class TestBotDetectionResult:
    """Test BotDetectionResult class."""

    def test_default_result(self):
        """Test default values."""
        result = BotDetectionResult()

        assert result.is_bot_likely is False
        assert result.hft_score == 0.0
        assert result.arbitrage_score == 0.0
        assert result.pattern_score == 0.0

    def test_hft_detection(self):
        """Test HFT detection result."""
        result = BotDetectionResult(
            is_bot_likely=True,
            hft_score=0.8,
            confidence=0.95,
        )

        assert result.is_bot_likely is True
        assert result.hft_score == 0.8


class TestPerformanceMetrics:
    """Test performance metric calculations."""

    @pytest.fixture
    def engine(self):
        """Create engine with default config."""
        return TraderIdentificationEngine()

    def test_sharpe_ratio_calculation(self, engine):
        """Test Sharpe ratio calculation."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        sharpe = engine._calculate_sharpe_ratio(returns)

        assert sharpe > 0  # Positive Sharpe for positive returns

    def test_sharpe_ratio_empty(self, engine):
        """Test Sharpe ratio with empty returns."""
        sharpe = engine._calculate_sharpe_ratio([])
        assert sharpe == 0.0

    def test_sharpe_ratio_single(self, engine):
        """Test Sharpe ratio with single return."""
        sharpe = engine._calculate_sharpe_ratio([0.01])
        assert sharpe == 0.0

    def test_max_drawdown_calculation(self, engine):
        """Test max drawdown calculation."""
        base_time = datetime.utcnow()
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.50 + (i * 0.05),  # Increasing prices
                total_value=50.0 + (i * 5.0),
                fees=1.0,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(5)
        ]

        dd = engine._calculate_max_drawdown(trades)
        assert 0 <= dd <= 1  # Normalized to volume

    def test_win_loss_classification(self, engine):
        """Test win/loss classification."""
        win_trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.60,  # > 0.5 = win for YES
            total_value=60.0,
            fees=1.0,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )

        loss_trade = Trade(
            trade_id="t2",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.SELL,
            quantity=100.0,
            price=0.40,  # < 0.5 = win for NO (which is a loss here)
            total_value=40.0,
            fees=1.0,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )

        assert engine._is_winning_trade(win_trade) is True
        assert engine._is_winning_trade(loss_trade) is False

    def test_hold_time_calculation(self, engine):
        """Test hold time calculation."""
        base_time = datetime.utcnow()
        trades = [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.50,
                total_value=50.0,
                fees=1.0,
                timestamp=base_time,
            ),
            Trade(
                trade_id="t2",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.SELL,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time + timedelta(hours=2),
            ),
        ]

        hold_times = engine._calculate_hold_times(trades)
        assert len(hold_times) == 1
        assert hold_times[0] == pytest.approx(2.0, rel=0.1)  # 2 hours (already in hours)


class TestBotPatternDetection:
    """Test bot pattern detection methods."""

    @pytest.fixture
    def engine(self):
        """Create engine with default config."""
        return TraderIdentificationEngine()

    def test_consistent_position_sizing(self, engine):
        """Test detection of consistent position sizing."""
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,  # Same size
                price=0.50,
                total_value=50.0,
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(5)
        ]

        result = engine._detect_bot_patterns(trades)
        # Should detect consistent sizing
        assert "consistent_position_sizing" in result["reasons"]

    def test_varying_position_sizing(self, engine):
        """Test with varying position sizes."""
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0 + (i * 50),  # Varying sizes
                price=0.50,
                total_value=50.0 + (i * 5.0),
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(5)
        ]

        result = engine._detect_bot_patterns(trades)
        assert "consistent_position_sizing" not in result.get("reasons", [])

    def test_hft_detection(self, engine):
        """Test HFT pattern detection."""
        base_time = datetime.utcnow()
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.50,
                total_value=50.0,
                fees=1.0,
                timestamp=base_time + timedelta(seconds=i * 0.5),  # Very fast
            )
            for i in range(10)
        ]

        result = engine._detect_hft_patterns(trades)
        assert result["hft_ratio"] > 0.5
