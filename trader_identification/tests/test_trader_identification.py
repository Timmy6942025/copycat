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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def engine(self):
        """Create engine with default config."""
        return TraderIdentificationEngine()

    def test_empty_trades_performance(self, engine):
        """Test performance calculation with empty trades."""
        perf = engine.calculate_performance("0x123", [])
        
        assert perf.trader_address == "0x123"
        assert perf.total_trades == 0
        assert perf.total_pnl == 0
        assert perf.win_rate == 0

    def test_single_trade_performance(self, engine):
        """Test performance calculation with single trade."""
        trades = [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.60,
                total_value=60.0,
                fees=1.0,
                timestamp=datetime.utcnow(),
                outcome="YES",
            )
        ]
        
        perf = engine.calculate_performance("0x123", trades)
        
        assert perf.total_trades == 1
        assert perf.win_rate == 1.0  # Single win

    def test_empty_trades_insider_detection(self, engine):
        """Test insider detection with empty trades."""
        result = engine.detect_insider_trading([])
        
        assert result.early_position_score == 0.0
        assert result.event_correlation_score == 0.0
        assert result.is_insider_likely is False

    def test_empty_trades_bot_detection(self, engine):
        """Test bot detection with empty trades."""
        result = engine.detect_bot_activity([])
        
        assert result.hft_score == 0.0
        assert result.arbitrage_score == 0.0
        assert result.is_bot_likely is False

    def test_low_trade_count_hft(self, engine):
        """Test HFT detection with insufficient trades."""
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
                timestamp=datetime.utcnow(),
            )
        ]
        
        result = engine._detect_hft_patterns(trades)
        assert result["score"] == 0.0
        assert result["details"] == "Insufficient trades"

    def test_low_trade_count_arbitrage(self, engine):
        """Test arbitrage detection with insufficient trades."""
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
                timestamp=datetime.utcnow(),
            )
        ]
        
        result = engine._detect_arbitrage_patterns(trades)
        assert result["score"] == 0.0

    def test_low_trade_count_bot_patterns(self, engine):
        """Test bot pattern detection with insufficient trades."""
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
                timestamp=datetime.utcnow(),
            )
        ]
        
        result = engine._detect_bot_patterns(trades)
        assert result["score"] == 0.0

    def test_returns_calculation(self, engine):
        """Test returns calculation from trades."""
        trades = [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.60,
                total_value=60.0,
                fees=1.0,
                timestamp=datetime.utcnow(),
                outcome="YES",
            ),
            Trade(
                trade_id="t2",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.SELL,
                quantity=100.0,
                price=0.40,
                total_value=40.0,
                fees=1.0,
                timestamp=datetime.utcnow(),
                outcome="YES",
            ),
        ]
        
        returns = engine._calculate_returns(trades)
        assert len(returns) == 2

    def test_empty_returns(self, engine):
        """Test returns calculation with empty trades."""
        returns = engine._calculate_returns([])
        assert returns == []

    def test_early_positions_no_market_events(self, engine):
        """Test early position detection without market events."""
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
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.60,
                total_value=60.0,
                fees=1.0,
                timestamp=base_time + timedelta(hours=1),
            ),
        ]
        
        result = engine._detect_early_positions(trades)
        assert "score" in result
        assert "early_count" in result
        assert "total_count" in result

    def test_category_expertise_empty(self, engine):
        """Test category expertise with empty trades."""
        result = engine._analyze_category_expertise([])
        assert result["score"] == 0.5
        assert result["details"] == "No trades"

    def test_event_correlation_no_events(self, engine):
        """Test event correlation without event data."""
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
                timestamp=datetime.utcnow(),
            )
        ]
        
        result = engine._calculate_event_correlation(trades, None)
        assert result["score"] == 0.5
        assert "No event data" in result["details"]


class TestPerformanceScenarios:
    """Test various performance scenarios."""

    @pytest.fixture
    def engine(self):
        """Create engine with default config."""
        return TraderIdentificationEngine()

    def test_highly_profitable_trader(self, engine):
        """Test with highly profitable trader."""
        base_time = datetime.utcnow()
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
                outcome="YES",
            )
            for i in range(20)
        ]
        
        perf = engine.calculate_performance("0x123", trades)
        
        assert perf.total_trades == 20
        assert perf.win_rate == 1.0  # All wins at > 0.5
        assert perf.total_pnl > 0

    def test_losing_trader(self, engine):
        """Test with consistently losing trader."""
        base_time = datetime.utcnow()
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.45,  # < 0.5 = loss
                total_value=45.0,
                fees=1.0,
                timestamp=base_time + timedelta(hours=i),
                outcome="YES",
            )
            for i in range(20)
        ]
        
        perf = engine.calculate_performance("0x123", trades)
        
        assert perf.total_trades == 20
        assert perf.win_rate == 0.0  # All losses
        assert perf.total_pnl < 0

    def test_breakeven_trader(self, engine):
        """Test with breakeven trader (50% win rate)."""
        base_time = datetime.utcnow()
        trades = []
        for i in range(20):
            is_win = i % 2 == 0
            price = 0.60 if is_win else 0.40
            outcome = "YES"
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id="m1",
                    trader_address="0x123",
                    side=OrderSide.BUY,
                    quantity=100.0,
                    price=price,
                    total_value=price * 100,
                    fees=1.0,
                    timestamp=base_time + timedelta(hours=i),
                    outcome=outcome,
                )
            )
        
        perf = engine.calculate_performance("0x123", trades)
        
        assert perf.total_trades == 20
        assert perf.win_rate == pytest.approx(0.5, rel=0.1)

    def test_high_drawdown_trader(self, engine):
        """Test performance calculation for high drawdown trader."""
        base_time = datetime.utcnow()
        trades = []
        # Create pattern: profit, profit, big loss
        for i in range(10):
            if i % 3 == 2:  # Every 3rd trade is a loss
                price = 0.30
            else:
                price = 0.60
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id="m1",
                    trader_address="0x123",
                    side=OrderSide.BUY,
                    quantity=100.0,
                    price=price,
                    total_value=price * 100,
                    fees=1.0,
                    timestamp=base_time + timedelta(hours=i),
                    outcome="YES",
                )
            )
        
        perf = engine.calculate_performance("0x123", trades)
        
        assert perf.max_drawdown > 0

    def test_long_hold_times(self, engine):
        """Test with very long hold times."""
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
                price=0.60,
                total_value=60.0,
                fees=1.0,
                timestamp=base_time + timedelta(days=7),  # 1 week hold
            ),
        ]
        
        perf = engine.calculate_performance("0x123", trades)
        
        assert perf.avg_hold_time_hours >= 24 * 7  # At least 7 days

    def test_profit_factor_edge_cases(self, engine):
        """Test profit factor calculation edge cases."""
        # No losses = infinite profit factor
        trades = [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.60,
                total_value=60.0,
                fees=1.0,
                timestamp=datetime.utcnow(),
                outcome="YES",
            )
        ]
        
        perf = engine.calculate_performance("0x123", trades)
        
        assert perf.profit_factor == float('inf')


class TestSuitabilityEvaluation:
    """Test suitability evaluation scenarios."""

    @pytest.fixture
    def engine(self):
        """Create engine with strict config."""
        return TraderIdentificationEngine(
            config=TraderSelectionConfig(
                min_win_rate=0.60,
                min_trades=50,
                max_drawdown=0.10,
                min_sharpe_ratio=1.0,
                min_profit_factor=1.5,
                min_total_pnl=100.0,
            )
        )

    def test_suitable_trader(self, engine):
        """Test evaluation of highly suitable trader."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=500.0,
            total_pnl_pct=0.10,
            win_rate=0.70,
            profit_factor=2.5,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
            avg_hold_time_hours=24.0,
            total_trades=100,
            winning_trades=70,
            losing_trades=30,
            avg_win=50.0,
            avg_loss=25.0,
            total_volume=5000.0,
        )
        
        insider = InsiderDetectionResult(is_insider_likely=False)
        bot = BotDetectionResult(is_bot_likely=False)
        
        result = engine.evaluate_suitability(perf, insider, bot, 0.8)
        
        assert result["is_suitable"] is True
        assert len(result["selected"]) > len(result["rejected"])

    def test_unsuitable_trader_low_win_rate(self, engine):
        """Test evaluation of trader with low win rate."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=100.0,
            total_pnl_pct=0.02,
            win_rate=0.45,  # Below minimum
            profit_factor=1.2,
            sharpe_ratio=0.3,
            max_drawdown=0.15,
            avg_hold_time_hours=24.0,
            total_trades=50,
            winning_trades=22,
            losing_trades=28,
            avg_win=40.0,
            avg_loss=35.0,
            total_volume=4000.0,
        )
        
        insider = InsiderDetectionResult(is_insider_likely=False)
        bot = BotDetectionResult(is_bot_likely=False)
        
        result = engine.evaluate_suitability(perf, insider, bot, 0.4)
        
        assert result["is_suitable"] is False
        assert any("win rate" in r.lower() for r in result["rejected"])

    def test_unsuitable_trader_high_drawdown(self, engine):
        """Test evaluation of trader with high drawdown."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=200.0,
            total_pnl_pct=0.05,
            win_rate=0.65,
            profit_factor=1.8,
            sharpe_ratio=0.8,
            max_drawdown=0.30,  # Above maximum
            avg_hold_time_hours=24.0,
            total_trades=60,
            winning_trades=39,
            losing_trades=21,
            avg_win=45.0,
            avg_loss=40.0,
            total_volume=4500.0,
        )
        
        insider = InsiderDetectionResult(is_insider_likely=False)
        bot = BotDetectionResult(is_bot_likely=False)
        
        result = engine.evaluate_suitability(perf, insider, bot, 0.6)
        
        assert any("drawdown" in r.lower() for r in result["rejected"])
        # Note: may still be suitable if more selected than rejected

    def test_min_trades_not_met(self, engine):
        """Test rejection when minimum trades not met."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=500.0,
            total_pnl_pct=0.25,
            win_rate=0.70,
            profit_factor=3.0,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
            avg_hold_time_hours=24.0,
            total_trades=10,  # Below minimum of 50
            winning_trades=7,
            losing_trades=3,
            avg_win=60.0,
            avg_loss=40.0,
            total_volume=2000.0,
        )
        
        insider = InsiderDetectionResult(is_insider_likely=False)
        bot = BotDetectionResult(is_bot_likely=False)
        
        result = engine.evaluate_suitability(perf, insider, bot, 0.7)
        
        # Should note low trades
        assert any("trades" in r.lower() for r in result["rejected"])


class TestReputationScoring:
    """Test reputation scoring variations."""

    @pytest.fixture
    def engine(self):
        """Create engine with default config."""
        return TraderIdentificationEngine()

    def test_perfect_reputation_score(self, engine):
        """Test reputation score for perfect trader."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=1000.0,
            win_rate=0.90,
            sharpe_ratio=3.0,
            max_drawdown=0.02,
            total_trades=100,
        )
        
        insider = InsiderDetectionResult(early_position_score=0.0)
        bot = BotDetectionResult(pattern_score=0.0)
        
        score = engine.calculate_reputation_score(perf, insider, bot)
        
        assert score > 0.9

    def test_low_reputation_score(self, engine):
        """Test reputation score for poor trader."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=-100.0,
            win_rate=0.30,
            sharpe_ratio=-0.5,
            max_drawdown=0.40,
            total_trades=50,
        )
        
        insider = InsiderDetectionResult(early_position_score=0.8)
        bot = BotDetectionResult(pattern_score=0.8)
        
        score = engine.calculate_reputation_score(perf, insider, bot)
        
        assert score < 0.5

    def test_reputation_with_insider_concern(self, engine):
        """Test reputation impact of insider detection."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=500.0,
            win_rate=0.70,
            sharpe_ratio=1.2,
            max_drawdown=0.10,
            total_trades=80,
        )
        
        # High insider score reduces reputation
        insider = InsiderDetectionResult(early_position_score=0.9)
        bot = BotDetectionResult(pattern_score=0.1)
        
        score = engine.calculate_reputation_score(perf, insider, bot)
        
        # Should be lower due to insider concern
        assert score < 0.8

    def test_reputation_with_bot_concern(self, engine):
        """Test reputation impact of bot detection."""
        perf = TraderPerformance(
            trader_address="0x123",
            total_pnl=300.0,
            win_rate=0.65,
            sharpe_ratio=1.0,
            max_drawdown=0.15,
            total_trades=60,
        )
        
        insider = InsiderDetectionResult(early_position_score=0.1)
        # High bot pattern score reduces reputation
        bot = BotDetectionResult(pattern_score=0.9)
        
        score = engine.calculate_reputation_score(perf, insider, bot)
        
        # Should be lower due to bot concern
        assert score < 0.6


class TestConfidenceCalculation:
    """Test confidence score variations."""

    @pytest.fixture
    def engine(self):
        """Create engine with default config."""
        return TraderIdentificationEngine()

    def test_high_confidence_many_trades(self, engine):
        """Test high confidence with many trades."""
        perf = TraderPerformance(trader_address="0x123", total_trades=100)
        insider = InsiderDetectionResult(confidence=0.9)
        bot = BotDetectionResult(confidence=0.85)
        
        confidence = engine.calculate_confidence(perf, insider, bot)
        
        assert confidence > 0.7

    def test_low_confidence_few_trades(self, engine):
        """Test low confidence with few trades."""
        perf = TraderPerformance(trader_address="0x123", total_trades=5)
        insider = InsiderDetectionResult(confidence=0.2)
        bot = BotDetectionResult(confidence=0.15)
        
        confidence = engine.calculate_confidence(perf, insider, bot)
        
        assert confidence < 0.4

    def test_confidence_trade_count_capped(self, engine):
        """Test that confidence is capped at 1.0."""
        perf = TraderPerformance(trader_address="0x123", total_trades=1000)
        insider = InsiderDetectionResult(confidence=1.0)
        bot = BotDetectionResult(confidence=1.0)
        
        confidence = engine.calculate_confidence(perf, insider, bot)
        
        assert confidence <= 1.0


class TestAsyncAnalysis:
    """Test async analysis methods."""

    @pytest.fixture
    def engine(self):
        """Create engine with default config."""
        return TraderIdentificationEngine()

    @pytest.mark.asyncio
    async def test_analyze_trader_with_events(self, engine):
        """Test async trader analysis with market events."""
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
                outcome="YES",
            ),
            Trade(
                trade_id="t2",
                market_id="m1",
                trader_address="0x123",
                side=OrderSide.SELL,
                quantity=100.0,
                price=0.65,
                total_value=65.0,
                fees=1.5,
                timestamp=base_time + timedelta(hours=2),
                outcome="YES",
            ),
        ]
        
        market_events = {"election_date": "2024-11-05", "debate_date": "2024-09-10"}
        
        result = await engine.analyze_trader("0x123", trades, market_events)
        
        assert isinstance(result, TraderIdentificationResult)
        assert result.trader_address == "0x123"
        assert result.performance.total_trades == 2
        assert result.confidence_score > 0

    @pytest.mark.asyncio
    async def test_analyze_trader_empty(self, engine):
        """Test async analysis with empty trades."""
        result = await engine.analyze_trader("0x456", [])
        
        assert result.trader_address == "0x456"
        assert result.performance.total_trades == 0


class TestTradePnLCalculation:
    """Test trade P&L calculation edge cases."""

    @pytest.fixture
    def engine(self):
        """Create engine with default config."""
        return TraderIdentificationEngine()

    def test_pnl_winning_trade(self, engine):
        """Test P&L calculation for winning trade."""
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.60,  # > 0.5 = win
            total_value=60.0,
            fees=2.0,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )
        
        pnl = engine._calculate_trade_pnl(trade)
        
        assert pnl > 0  # Win minus fees

    def test_pnl_losing_trade(self, engine):
        """Test P&L calculation for losing trade."""
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.40,  # < 0.5 = loss
            total_value=40.0,
            fees=2.0,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )
        
        pnl = engine._calculate_trade_pnl(trade)
        
        assert pnl < 0  # Loss plus fees

    def test_pnl_no_fees(self, engine):
        """Test P&L calculation without fees."""
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.60,
            total_value=60.0,
            fees=0.0,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )
        
        pnl = engine._calculate_trade_pnl(trade)
        
        assert pnl == 60.0

    def test_pnl_high_fees(self, engine):
        """Test P&L with high fees."""
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x123",
            side=OrderSide.BUY,
            quantity=100.0,
            price=0.55,
            total_value=55.0,
            fees=10.0,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )
        
        pnl = engine._calculate_trade_pnl(trade)
        
        assert pnl == 45.0


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def engine(self):
        """Create engine with realistic config."""
        return TraderIdentificationEngine(
            config=TraderSelectionConfig(
                min_win_rate=0.55,
                min_trades=20,
                max_drawdown=0.25,
                min_sharpe_ratio=0.5,
                min_profit_factor=1.0,
                min_total_pnl=0.0,
                max_avg_hold_time_hours=168.0,
                min_reputation_score=0.5,
            )
        )

    @pytest.mark.asyncio
    async def test_profitable_trader_identification(self, engine):
        """Test complete trader identification for profitable trader."""
        base_time = datetime.utcnow()
        trades = []
        for i in range(30):
            price = 0.55 + (0.02 if i % 2 == 0 else 0.01)
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id=f"m{i % 5}",
                    trader_address="0xPROFITABLE",
                    side=OrderSide.BUY,
                    quantity=100.0 + (i * 10),
                    price=price,
                    total_value=price * (100 + i * 10),
                    fees=1.0,
                    timestamp=base_time + timedelta(hours=i * 2),
                    outcome="YES",
                )
            )
        
        result = await engine.analyze_trader("0xPROFITABLE", trades)
        
        assert result.is_suitable is True
        assert result.reputation_score > 0.5

    @pytest.mark.asyncio
    async def test_unprofitable_trader_rejection(self, engine):
        """Test complete trader identification for unprofitable trader."""
        base_time = datetime.utcnow()
        # Create 30 losing trades
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id=f"m{i % 5}",
                trader_address="0xLOSING",
                side=OrderSide.BUY,
                quantity=100.0,
                price=0.45,  # Always losing
                total_value=45.0,
                fees=1.0,
                timestamp=base_time + timedelta(hours=i),
                outcome="YES",
            )
            for i in range(30)
        ]
        
        result = await engine.analyze_trader("0xLOSING", trades)
        
        assert result.is_suitable is False
        assert result.performance.win_rate < engine.config.min_win_rate

    @pytest.mark.asyncio
    async def test_bot_like_trader_detection(self, engine):
        """Test identification of bot-like trading patterns."""
        base_time = datetime.utcnow()
        # Create bot-like pattern: consistent timing and sizing
        trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xBOT",
                side=OrderSide.BUY,
                quantity=100.0,  # Consistent sizing
                price=0.55,
                total_value=55.0,
                fees=1.0,
                timestamp=base_time + timedelta(seconds=i * 60),  # Every minute
            )
            for i in range(50)
        ]
        
        result = await engine.analyze_trader("0xBOT", trades)
        
        # Should be flagged as bot-like
        assert result.bot_result is not None
        assert result.bot_result.is_bot_likely or result.bot_result.pattern_score > 0.3

    @pytest.mark.asyncio
    async def test_mixed_performance_trader(self, engine):
        """Test identification of trader with mixed performance."""
        base_time = datetime.utcnow()
        # Create mixed performance: some wins, some losses
        trades = []
        for i in range(40):
            is_win = i % 3 != 0  # 2/3 wins
            price = 0.60 if is_win else 0.40
            trades.append(
                Trade(
                    trade_id=f"t{i}",
                    market_id=f"m{i % 4}",
                    trader_address="0xMIXED",
                    side=OrderSide.BUY,
                    quantity=50.0 + (i * 2),
                    price=price,
                    total_value=price * (50 + i * 2),
                    fees=0.5 + (i * 0.01),
                    timestamp=base_time + timedelta(hours=i * 2),
                    outcome="YES",
                )
            )
        
        result = await engine.analyze_trader("0xMIXED", trades)
        
        # Should have moderate win rate
        assert 0.4 < result.performance.win_rate < 0.8
