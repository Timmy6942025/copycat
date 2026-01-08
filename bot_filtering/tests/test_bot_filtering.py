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


class TestHFTScenarios:
    """Test various HFT detection scenarios."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_moderate_hft_trades(self, bot_filter):
        """Test detection of moderate HFT activity."""
        base_time = datetime.utcnow()
        # Create trades with some HFT, some not
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
                timestamp=base_time + timedelta(seconds=i * 0.5),
            )
            for i in range(30)
        ]
        
        analysis = bot_filter._analyze_hft(trades)
        
        assert analysis["hft_ratio"] > 0.5

    def test_very_sustained_hft(self, bot_filter):
        """Test very sustained HFT activity detection."""
        base_time = datetime.utcnow()
        # Extreme HFT: 100 trades in 10 seconds
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xHFT",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time + timedelta(seconds=i * 0.1),
            )
            for i in range(100)
        ]
        
        analysis = bot_filter._analyze_hft(trades)
        
        assert analysis["trades_per_minute"] > 50  # Very high
        assert analysis["sustained_hft"] is True
        assert analysis["score"] > 0.8

    def test_no_hft_activity(self, bot_filter):
        """Test detection when no HFT activity present."""
        base_time = datetime.utcnow()
        # Slow trading: 1 trade per hour
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xSLOW",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(10)
        ]
        
        analysis = bot_filter._analyze_hft(trades)
        
        assert analysis["hft_ratio"] == 0.0
        assert analysis["sustained_hft"] is False
        assert analysis["score"] == 0.0

    def test_hft_score_calculation(self, bot_filter):
        """Test HFT score calculation components."""
        base_time = datetime.utcnow()
        # Create known pattern
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xTEST",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time + timedelta(seconds=i * 0.5),
            )
            for i in range(50)
        ]
        
        analysis = bot_filter._analyze_hft(trades)
        
        # Verify score components
        assert 0 <= analysis["score"] <= 1.0
        assert "reasons" in analysis
        assert len(analysis["reasons"]) > 0


class TestArbitrageScenarios:
    """Test various arbitrage detection scenarios."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_high_small_trade_ratio(self, bot_filter):
        """Test detection of high small-trade ratio."""
        base_time = datetime.utcnow()
        # Create many small, consistent profit trades
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id=f"m{i % 5}",
                trader_address="0xARB",
                side=OrderSide.BUY,
                quantity=50.0,
                price=0.52,  # Small profit
                total_value=26.0,
                fees=0.5,
                timestamp=base_time + timedelta(minutes=i * 5),
            )
            for i in range(50)
        ]
        
        analysis = bot_filter._analyze_arbitrage(trades)
        
        assert analysis["small_trade_ratio"] > 0.5
        assert analysis["multi_market_trading"] is True

    def test_no_arbitrage_patterns(self, bot_filter):
        """Test when no arbitrage patterns present."""
        base_time = datetime.utcnow()
        # Create varied profit/loss trades
        trades = []
        for i in range(30):
            is_profit = i % 2 == 0
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id="m1",
                    trader_address="0xVARIED",
                    side=OrderSide.BUY,
                    quantity=100.0 + (i * 10),
                    price=0.70 if is_profit else 0.30,
                    total_value=(70 + i) if is_profit else (30 + i),
                    fees=1.0,
                    timestamp=base_time + timedelta(hours=i),
                    outcome="YES",
                )
            )
        
        analysis = bot_filter._analyze_arbitrage(trades)
        
        # Should have low arbitrage score
        assert analysis["score"] < 0.5

    def test_spread_exploitation_detection(self, bot_filter):
        """Test spread exploitation pattern detection."""
        base_time = datetime.utcnow()
        # Create many small, profitable trades
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xSPREAD",
                side=OrderSide.BUY,
                quantity=2.0,
                price=0.52,
                total_value=1.04,  # Very small
                fees=0.05,
                timestamp=base_time + timedelta(minutes=i * 15),
            )
            for i in range(20)
        ]
        
        result = bot_filter._detect_spread_exploitation(trades)
        
        assert result["detected"] is True
        assert result["ratio"] > 0.3
        assert result["count"] == 20


class TestPatternDetectionScenarios:
    """Test various pattern detection scenarios."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_fixed_position_size(self, bot_filter):
        """Test detection of fixed position size."""
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xFIXED",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(15)
        ]
        
        analysis = bot_filter._analyze_patterns(trades)
        
        assert "Fixed position size" in analysis["reasons"]
        assert "Consistent position sizing" in analysis["reasons"]

    def test_varied_position_sizes(self, bot_filter):
        """Test with varied position sizes."""
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xVARIED",
                side=OrderSide.BUY,
                quantity=50.0 + (i * 25),
                price=0.55,
                total_value=(50 + i * 25) * 0.55,
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(10)
        ]
        
        analysis = bot_filter._analyze_patterns(trades)
        
        assert "Fixed position size" not in analysis.get("reasons", [])
        assert "Consistent position sizing" not in analysis.get("reasons", [])

    def test_weekend_trading_pattern(self, bot_filter):
        """Test weekend trading pattern."""
        # Create trades on weekends
        friday = datetime.utcnow()
        while friday.weekday() != 4:  # Friday
            friday += timedelta(days=1)
        
        trades = []
        for i in range(5):
            # Saturday and Sunday trades
            timestamp = friday + timedelta(days=1, hours=i)
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id="m1",
                    trader_address="0xWEEKEND",
                    side=OrderSide.BUY,
                    quantity=100.0,
                    price=0.55,
                    total_value=55.0,
                    fees=1.0,
                    timestamp=timestamp,
                )
            )
        
        analysis = bot_filter._analyze_patterns(trades)
        
        # Should not trigger "No weekend trading" since they do trade weekends
        assert "No weekend trading" not in analysis.get("reasons", [])

    def test_all_business_hours(self, bot_filter):
        """Test detection of trading only during business hours."""
        base_time = datetime.utcnow()
        base_time = base_time.replace(hour=9, minute=0, second=0, microsecond=0)
        
        trades = []
        for i in range(6):
            hour = 9 + (i * 2)
            if hour > 18:
                hour = 18 - (i % 2)
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id="m1",
                    trader_address="0xBUSINESS",
                    side=OrderSide.BUY,
                    quantity=100.0 + (i * 51),
                    price=0.52 + (i % 4) * 0.03,
                    total_value=(100 + i * 51) * (0.52 + (i % 4) * 0.03),
                    fees=0.5 + (i * 0.3),
                    timestamp=base_time.replace(hour=hour, minute=i * 7 % 60),
                )
            )
        
        analysis = bot_filter._analyze_patterns(trades)
        
        assert len([r for r in analysis["reasons"] if "trading hours" in r.lower()]) > 0 or \
               "Limited trading hours" in analysis["reasons"]

    def test_24_hour_trading(self, bot_filter):
        """Test with 24-hour trading patterns."""
        base_time = datetime.utcnow()
        
        trades = []
        for i in range(24):
            timestamp = base_time.replace(hour=i, minute=0, second=0, microsecond=0)
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id="m1",
                    trader_address="0x24HRS",
                    side=OrderSide.BUY,
                    quantity=100.0,
                    price=0.55,
                    total_value=55.0,
                    fees=1.0,
                    timestamp=timestamp,
                )
            )
        
        analysis = bot_filter._analyze_patterns(trades)
        
        assert "Limited trading hours" not in analysis.get("reasons", [])


class TestExclusionLogic:
    """Test exclusion logic edge cases."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_exclude_low_confidence(self, bot_filter):
        """Test exclusion with low confidence."""
        result = BotFilterResult(
            is_bot=True,
            confidence=0.3,  # Low confidence
            reasons=["Some patterns detected"],
        )
        
        assert bot_filter.should_exclude(result) is False

    def test_exclude_high_confidence_human(self, bot_filter):
        """Test not excluding high confidence human."""
        result = BotFilterResult(
            is_bot=False,
            confidence=0.95,
            reasons=[],
        )
        
        assert bot_filter.should_exclude(result) is False

    def test_exclude_boundary_confidence(self, bot_filter):
        """Test exclusion at boundary confidence (0.5)."""
        result = BotFilterResult(
            is_bot=True,
            confidence=0.5,
            reasons=["Patterns detected"],
        )
        
        assert bot_filter.should_exclude(result) is True


class TestCombinedDetection:
    """Test combined bot detection scenarios."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_multiple_bot_signals(self, bot_filter):
        """Test detection with multiple bot signals."""
        base_time = datetime.utcnow()
        # Create trades that trigger HFT, arbitrage, and pattern detection
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id=f"m{i % 3}",
                trader_address="0xMULTI",
                side=OrderSide.BUY,
                quantity=100.0,  # Fixed size
                price=0.52,
                total_value=52.0,
                fees=1.0,
                timestamp=base_time + timedelta(seconds=i * 0.5),
            )
            for i in range(50)
        ]
        
        result = bot_filter.analyze_trades(trades)
        
        # Should detect multiple bot signals
        assert result.is_hft or result.is_arbitrage or result.is_pattern_bot
        assert result.overall_score > 0.5

    def test_human_with_some_patterns(self, bot_filter):
        """Test human trader with some incidental patterns."""
        base_time = datetime.utcnow()
        # Create human-like but some patterns
        trades = []
        for i in range(30):
            hour = 9 + (i % 10)  # Mostly business hours
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id="m1",
                    trader_address="0xHUMAN",
                    side=OrderSide.BUY,
                    quantity=100.0 + (i * 5),
                    price=0.55 + (0.01 if i % 2 == 0 else -0.01),
                    total_value=(100 + i * 5) * (0.55 + (0.01 if i % 2 == 0 else -0.01)),
                    fees=1.0,
                    timestamp=base_time.replace(hour=hour % 24, minute=0, second=0, microsecond=0) + timedelta(days=i // 10),
                )
            )
        
        result = bot_filter.analyze_trades(trades)
        
        # Should not be flagged as bot
        assert result.is_bot is False

    def test_pattern_disabled_config(self, bot_filter):
        """Test with pattern check disabled."""
        bot_filter.config.pattern_check_enabled = False
        
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xNOPATTERN",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(10)
        ]
        
        result = bot_filter.analyze_trades(trades)
        
        # Pattern score should be 0
        assert result.pattern_score == 0.0
        assert "patterns" not in result.details


class TestConfidenceCalculation:
    """Test confidence calculation scenarios."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_confidence_increases_with_trades(self, bot_filter):
        """Test that confidence increases with more trades."""
        # Few trades = low confidence
        few_trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(5)
        ]
        
        result_few = bot_filter.analyze_trades(few_trades)
        
        # Many trades = higher confidence
        many_trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(100)
        ]
        
        result_many = bot_filter.analyze_trades(many_trades)
        
        assert result_many.confidence > result_few.confidence

    def test_max_confidence(self, bot_filter):
        """Test that confidence is capped at 1.0."""
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
                timestamp=datetime.utcnow() + timedelta(hours=i),
            )
            for i in range(200)  # More than 100
        ]
        
        result = bot_filter.analyze_trades(trades)
        
        assert result.confidence == 1.0


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def bot_filter(self):
        """Create a bot filter instance."""
        return BotFilter()

    def test_extreme_hft_bot(self, bot_filter):
        """Test detection of extreme HFT bot."""
        base_time = datetime.utcnow()
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xHFTBOT",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time + timedelta(milliseconds=i * 50),
            )
            for i in range(100)
        ]
        
        result = bot_filter.analyze_trades(trades)
        
        assert result.is_hft is True
        assert result.hft_score > 0.8
        assert result.is_bot is True
        assert bot_filter.should_exclude(result) is True

    def test_sophisticated_arbitrage_bot(self, bot_filter):
        """Test detection of sophisticated arbitrage bot."""
        base_time = datetime.utcnow()
        trades = []
        for i in range(60):
            # Multi-market, consistent timing, small profits
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id=f"m{i % 4}",
                    trader_address="0xARBBOT",
                    side=OrderSide.BUY,
                    quantity=75.0,
                    price=0.51,
                    total_value=38.25,
                    fees=0.5,
                    timestamp=base_time + timedelta(minutes=i * 5),
                )
            )
        
        result = bot_filter.analyze_trades(trades)
        
        # Should be flagged
        assert result.is_arbitrage or result.is_pattern_bot

    def test_human_trader_no_exclusion(self, bot_filter):
        """Test that genuine human traders are not excluded."""
        base_time = datetime.utcnow()
        trades = []
        for i in range(50):
            # Varied patterns like a human
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id=f"m{i % 7}",
                    trader_address="0xHUMANTRADER",
                    side=OrderSide.BUY,
                    quantity=25.0 + (i * 15) + (i % 3) * 10,
                    price=0.45 + ((i % 5) * 0.1),
                    total_value=(25 + i * 15 + (i % 3) * 10) * (0.45 + ((i % 5) * 0.1)),
                    fees=0.5 + (i % 3) * 0.25,
                    timestamp=base_time + timedelta(hours=i * 3 + (i % 4)),
                )
            )
        
        result = bot_filter.analyze_trades(trades)
        
        # Human trader should not be flagged
        assert result.is_bot is False
        assert bot_filter.should_exclude(result) is False
