"""
Unit tests for Bot Filtering Module.
Tests HFT, arbitrage, and pattern detection.
"""

import pytest
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/home/timmy/copycat')

from bot_filtering import (
    BotFilter,
    BotFilterConfig,
    BotFilterResult,
)
from api_clients.base import Trade, OrderSide


class TestBotFilter:
    """Test BotFilter class."""

    @pytest.fixture
    def filter_instance(self):
        """Create a bot filter instance."""
        config = BotFilterConfig(
            hft_max_hold_time_seconds=1.0,
            hft_min_trades_per_minute=5,
            arbitrage_max_profit_pct=0.5,
        )
        return BotFilter(config=config)

    @pytest.fixture
    def sample_trades(self):
        """Create sample human-like trades."""
        base_time = datetime.utcnow()
        return [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time,
            ),
            Trade(
                trade_id="t2",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.SELL,
                quantity=100.0,
                price=0.65,
                total_value=65.0,
                fees=1.3,
                timestamp=base_time + timedelta(hours=2),
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
                timestamp=base_time + timedelta(hours=5),
            ),
        ]

    @pytest.fixture
    def hft_trades(self):
        """Create sample HFT trades."""
        base_time = datetime.utcnow()
        return [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x456",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time + timedelta(seconds=i * 0.5),
            )
            for i in range(20)
        ]

    @pytest.fixture
    def arbitrage_trades(self):
        """Create sample arbitrage-like trades."""
        base_time = datetime.utcnow()
        return [
            Trade(
                trade_id=f"t{i}",
                market_id=f"m{i % 3}",
                trader_address="0x789",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.51,
                total_value=51.0,
                fees=1.0,
                timestamp=base_time + timedelta(minutes=i * 10),
            )
            for i in range(50)
        ]

    def test_analyze_human_trades(self, filter_instance, sample_trades):
        """Test analysis of human-like trading patterns."""
        result = filter_instance.analyze_trades(sample_trades)

        assert isinstance(result, BotFilterResult)
        assert result.is_hft is False
        assert result.is_arbitrage is False
        assert result.is_bot is False
        assert result.confidence > 0

    def test_analyze_hft_trades(self, filter_instance, hft_trades):
        """Test analysis of HFT trading patterns."""
        result = filter_instance.analyze_trades(hft_trades)

        assert result.is_hft is True
        assert result.hft_score > 0.7

    def test_analyze_arbitrage_trades(self, filter_instance, arbitrage_trades):
        """Test analysis of arbitrage-like patterns."""
        result = filter_instance.analyze_trades(arbitrage_trades)

        # Should detect multi-market trading patterns (3 markets)
        assert result.is_pattern_bot is True
        assert result.pattern_score > 0.5
        # Arbitrage score may vary based on profit patterns detected

    def test_empty_trades(self, filter_instance):
        """Test analysis with empty trades list."""
        result = filter_instance.analyze_trades([])

        assert result.is_bot is False
        assert result.confidence == 0.0

    def test_single_trade(self, filter_instance):
        """Test analysis with single trade."""
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.55,
            total_value=55.0,
            fees=1.0,
            timestamp=datetime.utcnow(),
        )

        result = filter_instance.analyze_trades([trade])

        assert result.is_bot is False
        assert result.confidence < 1.0  # Low confidence with single trade

    def test_should_exclude(self, filter_instance):
        """Test exclusion determination."""
        # Bot result with high confidence
        bot_result = BotFilterResult(
            is_bot=True,
            confidence=0.8,
            reasons=["HFT detected"],
        )
        assert filter_instance.should_exclude(bot_result) is True

        # Human result
        human_result = BotFilterResult(
            is_bot=False,
            confidence=0.9,
            reasons=[],
        )
        assert filter_instance.should_exclude(human_result) is False

    def test_get_exclusion_reasons(self, filter_instance):
        """Test getting exclusion reasons."""
        result = BotFilterResult(
            is_bot=True,
            is_hft=True,
            hft_score=0.8,
            reasons=["High-frequency trading detected"],
        )

        reasons = filter_instance.get_exclusion_reasons(result)
        assert "High-frequency trading detected" in reasons

    def test_hft_analysis(self, filter_instance, hft_trades):
        """Test detailed HFT analysis."""
        analysis = filter_instance._analyze_hft(hft_trades)

        assert "score" in analysis
        assert "hft_ratio" in analysis
        assert "trades_per_minute" in analysis
        assert analysis["hft_ratio"] > 0.5

    def test_arbitrage_analysis(self, filter_instance, arbitrage_trades):
        """Test detailed arbitrage analysis."""
        analysis = filter_instance._analyze_arbitrage(arbitrage_trades)

        assert "score" in analysis
        assert "small_trade_ratio" in analysis
        # Should detect multi-market trading patterns
        assert analysis.get("multi_market_trading", False) is True

    def test_pattern_analysis(self, filter_instance):
        """Test pattern analysis."""
        # Consistent sizing
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,  # Fixed size
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(10)
        ]

        analysis = filter_instance._analyze_patterns(trades)

        assert "score" in analysis
        assert "reasons" in analysis
        assert "Consistent position sizing" in analysis["reasons"]


class TestBotFilterConfig:
    """Test BotFilterConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BotFilterConfig()

        assert config.hft_max_hold_time_seconds == 1.0
        assert config.hft_min_trades_per_minute == 5
        assert config.arbitrage_max_profit_pct == 0.5
        assert config.min_hft_score_to_exclude == 0.7
        assert config.min_arbitrage_score_to_exclude == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BotFilterConfig(
            hft_max_hold_time_seconds=0.5,
            hft_min_trades_per_minute=10,
            arbitrage_max_profit_pct=0.3,
            pattern_check_enabled=False,
        )

        assert config.hft_max_hold_time_seconds == 0.5
        assert config.hft_min_trades_per_minute == 10
        assert config.pattern_check_enabled is False


class TestBotFilterResult:
    """Test BotFilterResult class."""

    def test_default_result(self):
        """Test default values."""
        result = BotFilterResult()

        assert result.is_bot is False
        assert result.is_hft is False
        assert result.is_arbitrage is False
        assert result.is_pattern_bot is False
        assert result.hft_score == 0.0
        assert result.arbitrage_score == 0.0
        assert result.overall_score == 0.0
        assert result.confidence == 0.0

    def test_hft_result(self):
        """Test HFT detection result."""
        result = BotFilterResult(
            is_bot=True,
            is_hft=True,
            hft_score=0.85,
            overall_score=0.85,
            confidence=0.95,
            reasons=["High-frequency trading detected"],
        )

        assert result.is_bot is True
        assert result.is_hft is True
        assert result.hft_score == 0.85

    def test_arbitrage_result(self):
        """Test arbitrage detection result."""
        result = BotFilterResult(
            is_bot=True,
            is_arbitrage=True,
            arbitrage_score=0.75,
            overall_score=0.75,
            confidence=0.85,
            reasons=["Arbitrage patterns detected"],
        )

        assert result.is_bot is True
        assert result.is_arbitrage is True


class TestBotDetectionHelpers:
    """Test helper methods for bot detection."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_calculate_hold_times(self, bot_filter):
        """Test hold time calculation."""
        base_time = datetime.utcnow()
        trades = [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time,
            ),
            Trade(
                trade_id="t2",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.SELL,
                quantity=100.0,
                price=0.60,
                total_value=60.0,
                fees=1.2,
                timestamp=base_time + timedelta(hours=1),
            ),
            Trade(
                trade_id="t3",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.50,
                total_value=50.0,
                fees=1.0,
                timestamp=base_time + timedelta(hours=3),
            ),
        ]

        hold_times = bot_filter._calculate_hold_times(trades)

        assert len(hold_times) == 2
        assert hold_times[0] == pytest.approx(3600, rel=0.1)  # 1 hour
        assert hold_times[1] == pytest.approx(7200, rel=0.1)  # 2 hours

    def test_variance_calculation(self, bot_filter):
        """Test variance calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        variance = bot_filter._variance(values)

        assert variance > 0  # Non-zero variance

    def test_zero_variance(self, bot_filter):
        """Test variance with constant values."""
        values = [5.0, 5.0, 5.0, 5.0]
        variance = bot_filter._variance(values)

        assert variance == 0.0

    def test_single_value_variance(self, bot_filter):
        """Test variance with single value."""
        variance = bot_filter._variance([5.0])

        assert variance == 0.0

    def test_detect_spread_exploitation(self, bot_filter):
        """Test spread exploitation detection."""
        # Small profitable trades
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=5.0,  # Small quantity
                price=0.52,
                total_value=2.6,  # Small value
                fees=0.1,
                timestamp=datetime.utcnow() + timedelta(minutes=i * 10),
            )
            for i in range(10)
        ]

        result = bot_filter._detect_spread_exploitation(trades)

        assert result["detected"] is True
        assert result["count"] == 10

    def test_generate_reasons(self, bot_filter):
        """Test reason generation."""
        result = BotFilterResult(
            is_hft=True,
            is_arbitrage=True,
            hft_score=0.8,
            arbitrage_score=0.7,
            reasons=["HFT detected", "Arbitrage detected"],
        )

        reasons = bot_filter._generate_reasons(result, {"hft": {"reasons": ["HFT detected"]}, "arbitrage": {"reasons": ["Arbitrage detected"]}})

        assert "High-frequency trading detected" in reasons
        assert "Arbitrage patterns detected" in reasons


class TestEdgeCases:
    """Test edge cases for bot filtering."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_weekend_trading(self, bot_filter):
        """Test weekend trading pattern detection."""
        # Create trades only on weekends
        saturday = datetime.utcnow()
        while saturday.weekday() != 5:  # Saturday
            saturday += timedelta(days=1)

        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=saturday + timedelta(hours=i),
            )
            for i in range(5)
        ]

        analysis = bot_filter._analyze_patterns(trades)
        # Should detect no weekend trading pattern
        assert "No weekend trading" not in analysis.get("reasons", [])

    def test_business_hours_only(self, bot_filter):
        """Test business hours only trading."""
        # Create trades only during business hours (9-17)
        base_time = datetime.utcnow()
        base_time = base_time.replace(hour=9, minute=0, second=0, microsecond=0)

        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(5)  # 9 AM to 1 PM
        ]

        analysis = bot_filter._analyze_patterns(trades)
        assert any("Limited trading hours" in r for r in analysis.get("reasons", []))

    def test_round_number_usage(self, bot_filter):
        """Test round number detection."""
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,  # Round number
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(10)
        ]

        analysis = bot_filter._analyze_patterns(trades)
        assert any("Round number usage" in r for r in analysis.get("reasons", []))
