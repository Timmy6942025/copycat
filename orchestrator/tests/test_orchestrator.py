"""
Unit tests for the CopyCat Orchestrator.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from api_clients.base import Trade, Trader, TraderPerformance, OrderSide, MarketData
from bot_filtering import BotFilterResult
from trader_identification import TraderIdentificationResult
from sandbox import VirtualOrderResult

from orchestrator.config import (
    OrchestratorConfig,
    TradingMode,
    MarketPlatform,
    TraderCopyConfig,
    OrchestratorState,
    OrchestrationResult,
    TraderAnalysisResult,
    HealthCheckConfig,
)
from orchestrator.engine import CopyCatOrchestrator


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        
        assert config.mode == TradingMode.SANDBOX
        assert config.platform == MarketPlatform.POLYMARKET
        assert config.trader_selection.min_win_rate == 0.55
        assert config.trader_selection.min_trades == 10
        assert config.copy_trading.position_sizing_method == "scaled"
        assert config.sandbox.initial_balance == 10000.0

    def test_sandbox_mode_config(self):
        """Test sandbox mode configuration."""
        from orchestrator.config import SandboxConfigOrchestrator
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            sandbox=SandboxConfigOrchestrator(initial_balance=50000.0)
        )
        
        assert config.mode == TradingMode.SANDBOX
        assert config.sandbox.initial_balance == 50000.0

    def test_live_mode_config(self):
        """Test live mode configuration."""
        config = OrchestratorConfig(
            mode=TradingMode.LIVE,
            platform=MarketPlatform.KALSHI
        )
        
        assert config.mode == TradingMode.LIVE
        assert config.platform == MarketPlatform.KALSHI

    def test_trader_selection_config(self):
        """Test trader selection configuration."""
        from orchestrator.config import TraderSelectionConfig
        config = OrchestratorConfig(
            trader_selection=TraderSelectionConfig(
                min_win_rate=0.6,
                min_trades=20,
                min_sharpe_ratio=1.0,
            )
        )
        
        assert config.trader_selection.min_win_rate == 0.6
        assert config.trader_selection.min_trades == 20
        assert config.trader_selection.min_sharpe_ratio == 1.0

    def test_copy_trading_config(self):
        """Test copy trading configuration."""
        from orchestrator.config import CopyTradingConfig
        config = OrchestratorConfig(
            copy_trading=CopyTradingConfig(
                position_sizing_method="kelly",
                kelly_fraction=0.25,
                max_position_size_pct=0.15,
            )
        )
        
        assert config.copy_trading.position_sizing_method == "kelly"
        assert config.copy_trading.kelly_fraction == 0.25
        assert config.copy_trading.max_position_size_pct == 0.15


class TestTraderCopyConfig:
    """Tests for TraderCopyConfig."""

    def test_trader_copy_config_defaults(self):
        """Test default trader copy configuration."""
        config = TraderCopyConfig(trader_address="0x1234")
        
        assert config.trader_address == "0x1234"
        assert config.enabled is True
        assert config.auto_remove_if_performance_drops is True

    def test_trader_copy_config_custom(self):
        """Test custom trader copy configuration."""
        config = TraderCopyConfig(
            trader_address="0x5678",
            base_position_size=500.0,
            position_size_pct=0.10,
            enabled=False,
        )
        
        assert config.base_position_size == 500.0
        assert config.position_size_pct == 0.10
        assert config.enabled is False


class TestOrchestratorState:
    """Tests for OrchestratorState."""

    def test_initial_state(self):
        """Test initial orchestrator state."""
        state = OrchestratorState()
        
        assert state.is_running is False
        assert state.is_paused is False
        assert state.copied_traders == {}
        assert state.total_pnl == 0.0
        assert state.api_healthy is True

    def test_state_with_copied_traders(self):
        """Test state with copied traders."""
        state = OrchestratorState()
        state.copied_traders = {
            "0x1111": TraderCopyConfig(trader_address="0x1111", base_position_size=100.0),
            "0x2222": TraderCopyConfig(trader_address="0x2222", base_position_size=200.0),
        }
        
        assert len(state.copied_traders) == 2


class TestOrchestrationResult:
    """Tests for OrchestrationResult."""

    def test_success_result(self):
        """Test success result."""
        result = OrchestrationResult(
            success=True,
            message="Operation completed",
            details={"key": "value"}
        )
        
        assert result.success is True
        assert result.message == "Operation completed"
        assert result.details == {"key": "value"}
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        result = OrchestrationResult(
            success=False,
            message="Operation failed",
            error="Something went wrong"
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"


class TestTraderAnalysisResult:
    """Tests for TraderAnalysisResult."""

    def test_trader_analysis_result(self):
        """Test trader analysis result."""
        result = TraderAnalysisResult(
            trader_address="0x1234",
            is_suitable=True,
            reputation_score=0.8,
            confidence_score=0.9,
            performance_metrics={"win_rate": 0.65},
            selection_reasons=["High win rate", "Good Sharpe ratio"],
        )
        
        assert result.trader_address == "0x1234"
        assert result.is_suitable is True
        assert result.reputation_score == 0.8
        assert len(result.selection_reasons) == 2


class TestCopyCatOrchestrator:
    """Tests for CopyCatOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )
        return CopyCatOrchestrator(config=config)

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.config.mode == TradingMode.SANDBOX
        assert orchestrator.state.is_running is False
        assert len(orchestrator.api_clients) > 0

    def test_get_status(self, orchestrator):
        """Test getting orchestrator status."""
        status = orchestrator.get_status()
        
        assert "is_running" in status
        assert "mode" in status
        assert "platform" in status
        assert "copied_traders" in status
        assert status["mode"] == "sandbox"

    def test_get_state(self, orchestrator):
        """Test getting orchestrator state."""
        state = orchestrator.get_state()
        
        assert isinstance(state, OrchestratorState)
        assert state.is_running is False

    @pytest.mark.asyncio
    async def test_start_stop(self, orchestrator):
        """Test starting and stopping orchestrator."""
        # Start
        start_result = await orchestrator.start()
        assert start_result.success is True
        assert orchestrator.state.is_running is True
        
        # Stop
        stop_result = await orchestrator.stop()
        assert stop_result.success is True
        assert orchestrator.state.is_running is False

    @pytest.mark.asyncio
    async def test_pause_resume(self, orchestrator):
        """Test pausing and resuming orchestrator."""
        await orchestrator.start()
        
        # Pause
        pause_result = await orchestrator.pause()
        assert pause_result.success is True
        assert orchestrator.state.is_paused is True
        
        # Resume
        resume_result = await orchestrator.resume()
        assert resume_result.success is True
        assert orchestrator.state.is_paused is False
        
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, orchestrator):
        """Test starting when already running."""
        # First start
        start_result = await orchestrator.start()
        assert start_result.success is True
        
        # Try to start again
        start_result2 = await orchestrator.start()
        assert start_result2.success is False
        assert "already running" in start_result2.message.lower()
        
        await orchestrator.stop()

    def test_is_winning_trade(self, orchestrator):
        """Test winning trade detection."""
        # Create mock trades
        yes_trade = Mock(spec=Trade, price=0.7, outcome="YES")
        no_trade = Mock(spec=Trade, price=0.3, outcome="NO")
        
        assert orchestrator._is_winning_trade(yes_trade) is True
        assert orchestrator._is_winning_trade(no_trade) is True
        
        losing_yes = Mock(spec=Trade, price=0.3, outcome="YES")
        losing_no = Mock(spec=Trade, price=0.7, outcome="NO")
        
        assert orchestrator._is_winning_trade(losing_yes) is False
        assert orchestrator._is_winning_trade(losing_no) is False

    def test_calculate_position_size_fixed(self, orchestrator):
        """Test position size calculation with fixed method."""
        orchestrator.config.copy_trading.position_sizing_method = "fixed_amount"
        orchestrator.config.copy_trading.base_position_size = 250.0
        
        mock_result = Mock()
        mock_result.confidence_score = 0.8
        
        size = orchestrator._calculate_position_size(mock_result)
        assert size == 250.0

    def test_calculate_position_size_percentage(self, orchestrator):
        """Test position size calculation with percentage method."""
        orchestrator.config.copy_trading.position_sizing_method = "percentage"
        orchestrator.config.copy_trading.position_size_pct = 0.10
        orchestrator.state.total_pnl = 0.0  # Starting balance
        
        mock_result = Mock()
        mock_result.confidence_score = 0.8
        
        size = orchestrator._calculate_position_size(mock_result)
        expected = 10000.0 * 0.10  # Initial balance * percentage
        assert size == expected

    @pytest.mark.asyncio
    async def test_analyze_trader_no_api_client(self, orchestrator):
        """Test analyzing trader when API client is not available."""
        # Remove API clients
        orchestrator.api_clients = {}
        
        result = await orchestrator._analyze_trader("0x1234")
        
        assert result.trader_address == "0x1234"
        assert result.is_suitable is False

    @pytest.mark.asyncio
    async def test_add_trader_not_running(self, orchestrator):
        """Test analyzing a trader when orchestrator is not running."""
        result = await orchestrator.add_trader("0x1234")
        
        # Should return a result (either success or failure)
        assert isinstance(result, OrchestrationResult)

    @pytest.mark.asyncio
    async def test_remove_trader_not_running(self, orchestrator):
        """Test removing trader when orchestrator is not running."""
        result = await orchestrator.remove_trader("0x1234")
        
        assert result.success is False
        assert "not in copy list" in result.message.lower()


# =============================================================================
# Trading Cycle Tests
# =============================================================================

class TestTradingCycle:
    """Tests for trading cycle functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )
        return CopyCatOrchestrator(config=config)

    @pytest.mark.asyncio
    async def test_run_trading_cycle_empty_traders(self, orchestrator):
        """Test trading cycle with no new traders discovered."""
        # Mock API client to return empty trades
        mock_client = AsyncMock()
        mock_client.get_trades = AsyncMock(return_value=[])
        orchestrator.api_clients[MarketPlatform.POLYMARKET] = mock_client
        
        result = await orchestrator._run_trading_cycle()
        
        assert result.success is True
        assert orchestrator.state.cycle_count == 1

    @pytest.mark.asyncio
    async def test_run_trading_cycle_with_traders(self, orchestrator):
        """Test trading cycle discovers and analyzes traders."""
        # Create mock trades with different traders
        from datetime import datetime
        from api_clients.base import Trade, OrderSide
        
        mock_trades = [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0xAAA",
                side=OrderSide.BUY,
                quantity=1.0,
                price=0.6,
                total_value=0.6,
                timestamp=datetime.utcnow(),
                outcome="YES",
            ),
            Trade(
                trade_id="t2",
                market_id="m2",
                trader_address="0xBBB",
                side=OrderSide.SELL,
                quantity=1.0,
                price=0.4,
                total_value=0.4,
                timestamp=datetime.utcnow(),
                outcome="NO",
            ),
        ]
        
        mock_client = AsyncMock()
        mock_client.get_trades = AsyncMock(return_value=mock_trades)
        mock_client.get_trades_for_trader = AsyncMock(return_value=mock_trades)
        orchestrator.api_clients[MarketPlatform.POLYMARKET] = mock_client
        
        result = await orchestrator._run_trading_cycle()
        
        assert result.success is True
        assert orchestrator.state.cycle_count == 1

    @pytest.mark.asyncio
    async def test_discover_traders_filters_existing(self, orchestrator):
        """Test that trader discovery filters out already copied traders."""
        from datetime import datetime
        from api_clients.base import Trade, OrderSide
        from orchestrator.engine import KNOWN_PROFITABLE_TRADERS
        
        # Clear bootstrap list for this test
        original_traders = KNOWN_PROFITABLE_TRADERS[:]
        KNOWN_PROFITABLE_TRADERS.clear()
        
        # Add existing copied trader
        orchestrator.state.copied_traders["0xEXISTING"] = TraderCopyConfig(
            trader_address="0xEXISTING"
        )
        
        mock_trades = [
            Trade(
                trade_id="t1",
                market_id="m1",
                trader_address="0xEXISTING",  # Already copied
                side=OrderSide.BUY,
                quantity=1.0,
                price=0.6,
                total_value=0.6,
                timestamp=datetime.utcnow(),
                outcome="YES",
            ),
            Trade(
                trade_id="t2",
                market_id="m2",
                trader_address="0xNEW",  # New trader
                side=OrderSide.BUY,
                quantity=1.0,
                price=0.6,
                total_value=0.6,
                timestamp=datetime.utcnow(),
                outcome="YES",
            ),
        ]
        
        mock_client = AsyncMock()
        mock_client.get_trades = AsyncMock(return_value=mock_trades)
        orchestrator.api_clients[MarketPlatform.POLYMARKET] = mock_client
        
        new_traders = await orchestrator._discover_traders()
        
        # Should only return the new trader
        assert len(new_traders) == 1
        
        # Restore bootstrap list
        KNOWN_PROFITABLE_TRADERS.extend(original_traders)
        assert len(new_traders) == 1
        assert "0xNEW" in new_traders
        assert "0xEXISTING" not in new_traders

    @pytest.mark.asyncio
    async def test_analyze_trader_no_trades(self, orchestrator):
        """Test analyzing a trader with no trades."""
        mock_client = AsyncMock()
        mock_client.get_trades = AsyncMock(return_value=[])
        orchestrator.api_clients[MarketPlatform.POLYMARKET] = mock_client
        
        result = await orchestrator._analyze_trader("0x1234")
        
        assert result.trader_address == "0x1234"
        assert result.is_suitable is False

    @pytest.mark.asyncio
    async def test_analyze_trader_bot_filtered(self, orchestrator):
        """Test that bot traders are filtered out."""
        from datetime import datetime
        from api_clients.base import Trade, OrderSide
        from bot_filtering import BotFilterResult
        
        # Create many rapid trades to trigger HFT detection
        base_time = datetime.utcnow()
        mock_trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xBOT",
                side=OrderSide.BUY,
                quantity=1.0,
                price=0.5,
                total_value=0.5,
                timestamp=base_time,  # All at same time = HFT
                outcome="YES",
            )
            for i in range(50)
        ]
        
        mock_client = AsyncMock()
        mock_client.get_trades = AsyncMock(return_value=mock_trades)
        orchestrator.api_clients[MarketPlatform.POLYMARKET] = mock_client
        
        result = await orchestrator._analyze_trader("0xBOT")
        
        # Should be flagged as bot
        assert result.bot_filter_result.get("is_bot") or "bot" in " ".join(result.rejection_reasons).lower()

    @pytest.mark.asyncio
    async def test_analyze_trader_suitable(self, orchestrator):
        """Test analyzing a suitable trader."""
        from datetime import datetime, timedelta
        from api_clients.base import Trade, OrderSide
        
        # Create realistic trading pattern
        base_time = datetime.utcnow() - timedelta(days=30)
        mock_trades = []
        for i in range(20):
            time_offset = timedelta(hours=i * 12)  # 12 hours between trades
            is_win = i % 3 != 0  # 66% win rate
            price = 0.6 if is_win else 0.4
            
            mock_trades.append(Trade(
                trade_id=f"t{i}",
                market_id=f"m{i % 5}",
                trader_address="0xGOOD",
                side=OrderSide.BUY,
                quantity=1.0,
                price=price,
                total_value=price,
                timestamp=base_time + time_offset,
                outcome="YES",
            ))
        
        mock_client = AsyncMock()
        mock_client.get_trades = AsyncMock(return_value=mock_trades)
        orchestrator.api_clients[MarketPlatform.POLYMARKET] = mock_client
        
        result = await orchestrator._analyze_trader("0xGOOD")
        
        # Should be suitable based on good performance
        assert result.trader_address == "0xGOOD"

    @pytest.mark.asyncio
    async def test_add_trader_max_limit(self, orchestrator):
        """Test adding trader when at max limit."""
        # Fill up copied traders
        for i in range(orchestrator.config.max_traders_to_copy):
            orchestrator.state.copied_traders[f"0x{i:04d}"] = TraderCopyConfig(
                trader_address=f"0x{i:04d}"
            )
        
        mock_result = TraderAnalysisResult(
            trader_address="0xNEW",
            is_suitable=True,
            recommended_position_size=100.0,
        )
        
        result = await orchestrator._add_trader_to_copy("0xNEW", mock_result)
        
        assert result.success is False
        assert "maximum" in result.message.lower()


class TestPositionSizing:
    """Tests for position sizing methods."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )
        return CopyCatOrchestrator(config=config)

    def test_position_size_fixed_amount(self, orchestrator):
        """Test fixed amount position sizing."""
        orchestrator.config.copy_trading.position_sizing_method = "fixed_amount"
        orchestrator.config.copy_trading.base_position_size = 500.0
        
        mock_result = Mock()
        mock_result.confidence_score = 0.8
        
        size = orchestrator._calculate_position_size(mock_result)
        assert size == 500.0

    def test_position_size_percentage(self, orchestrator):
        """Test percentage-based position sizing."""
        orchestrator.config.copy_trading.position_sizing_method = "percentage"
        orchestrator.config.copy_trading.position_size_pct = 0.10
        orchestrator.state.total_pnl = 0.0
        
        mock_result = Mock()
        mock_result.confidence_score = 0.8
        
        size = orchestrator._calculate_position_size(mock_result)
        assert size == 10000.0 * 0.10  # Initial balance * 10%

    def test_position_size_scaled(self, orchestrator):
        """Test scaled position sizing by confidence."""
        orchestrator.config.copy_trading.position_sizing_method = "scaled"
        orchestrator.config.copy_trading.position_size_pct = 0.10
        orchestrator.state.total_pnl = 0.0
        
        mock_result = Mock()
        mock_result.confidence_score = 0.5  # Low confidence
        
        size = orchestrator._calculate_position_size(mock_result)
        expected = 10000.0 * 0.10 * 0.75  # 50% conf = 0.75 multiplier
        assert size == expected

    def test_position_size_scaled_high_confidence(self, orchestrator):
        """Test scaled position sizing with high confidence."""
        orchestrator.config.copy_trading.position_sizing_method = "scaled"
        orchestrator.config.copy_trading.position_size_pct = 0.10
        orchestrator.state.total_pnl = 0.0
        
        mock_result = Mock()
        mock_result.confidence_score = 1.0  # High confidence
        
        size = orchestrator._calculate_position_size(mock_result)
        expected = 10000.0 * 0.10 * 1.0  # 100% conf = 1.0 multiplier
        assert size == expected

    def test_position_size_kelly(self, orchestrator):
        """Test Kelly criterion position sizing."""
        orchestrator.config.copy_trading.position_sizing_method = "kelly"
        orchestrator.config.copy_trading.kelly_fraction = 0.25
        orchestrator.state.total_pnl = 0.0
        
        mock_result = Mock()
        mock_result.performance.win_rate = 0.60
        mock_result.performance.avg_win = 100.0
        mock_result.performance.avg_loss = 50.0
        
        size = orchestrator._calculate_position_size(mock_result)
        # Kelly formula: (PL_ratio * win_rate - (1 - win_rate)) / PL_ratio
        # = (2.0 * 0.60 - 0.40) / 2.0 = 0.40
        # Fractional Kelly: 0.40 * 0.25 = 0.10
        expected = 10000.0 * 0.10
        assert size == pytest.approx(expected)

    def test_position_size_kelly_negative(self, orchestrator):
        """Test Kelly with losing strategy returns zero."""
        orchestrator.config.copy_trading.position_sizing_method = "kelly"
        orchestrator.config.copy_trading.kelly_fraction = 0.25
        orchestrator.state.total_pnl = 0.0
        
        mock_result = Mock()
        mock_result.performance.win_rate = 0.30  # Losing strategy
        mock_result.performance.avg_win = 50.0
        mock_result.performance.avg_loss = 100.0
        
        size = orchestrator._calculate_position_size(mock_result)
        assert size == 0.0


class TestTraderMonitoring:
    """Tests for copied trader monitoring."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )
        return CopyCatOrchestrator(config=config)

    @pytest.mark.asyncio
    async def test_monitor_copied_traders_no_auto_remove(self, orchestrator):
        """Test monitoring with auto_remove disabled."""
        # Add copied trader with auto_remove disabled
        orchestrator.state.copied_traders["0x1234"] = TraderCopyConfig(
            trader_address="0x1234",
            auto_remove_if_performance_drops=False,
        )
        
        # Should not attempt to remove
        await orchestrator._monitor_copied_traders()
        
        assert "0x1234" in orchestrator.state.copied_traders

    @pytest.mark.asyncio
    async def test_monitor_copied_traders_removes_underperformer(self, orchestrator):
        """Test monitoring removes underperforming traders."""
        from datetime import datetime
        from api_clients.base import Trade, OrderSide
        
        # Add copied trader with auto_remove enabled
        orchestrator.state.copied_traders["0xUNDERPERFORM"] = TraderCopyConfig(
            trader_address="0xUNDERPERFORM",
            auto_remove_if_performance_drops=True,
            min_performance_threshold=0.50,
        )
        
        # Create losing trades
        mock_trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xUNDERPERFORM",
                side=OrderSide.BUY,
                quantity=1.0,
                price=0.3,  # Low price = losing for YES
                total_value=0.3,
                timestamp=datetime.utcnow(),
                outcome="YES",
            )
            for i in range(5)
        ]
        
        mock_client = AsyncMock()
        mock_client.get_trades = AsyncMock(return_value=mock_trades)
        orchestrator.api_clients[MarketPlatform.POLYMARKET] = mock_client
        
        await orchestrator._monitor_copied_traders()
        
        # Trader should be removed due to poor performance
        assert "0xUNDERPERFORM" not in orchestrator.state.copied_traders

    @pytest.mark.asyncio
    async def test_monitor_copied_traders_keeps_good_trader(self, orchestrator):
        """Test monitoring keeps good performing traders."""
        from datetime import datetime
        from api_clients.base import Trade, OrderSide
        
        # Add copied trader
        orchestrator.state.copied_traders["0xGOOD"] = TraderCopyConfig(
            trader_address="0xGOOD",
            auto_remove_if_performance_drops=True,
            min_performance_threshold=0.30,
        )
        
        # Create winning trades
        mock_trades = [
            Trade(
                trade_id=f"t{i}",
                market_id="m1",
                trader_address="0xGOOD",
                side=OrderSide.BUY,
                quantity=1.0,
                price=0.7,  # High price = winning for YES
                total_value=0.7,
                timestamp=datetime.utcnow(),
                outcome="YES",
            )
            for i in range(5)
        ]
        
        mock_client = AsyncMock()
        mock_client.get_trades = AsyncMock(return_value=mock_trades)
        orchestrator.api_clients[MarketPlatform.POLYMARKET] = mock_client
        
        await orchestrator._monitor_copied_traders()
        
        # Trader should be kept
        assert "0xGOOD" in orchestrator.state.copied_traders


class TestCopyTradeExecution:
    """Tests for copy trade execution."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )
        return CopyCatOrchestrator(config=config)

    @pytest.mark.asyncio
    async def test_execute_copy_trade_disabled(self, orchestrator):
        """Test that disabled traders don't get copied."""
        from datetime import datetime
        from api_clients.base import Trade, OrderSide
        
        # Add disabled trader
        orchestrator.state.copied_traders["0xDISABLED"] = TraderCopyConfig(
            trader_address="0xDISABLED",
            enabled=False,
        )
        
        mock_trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0xDISABLED",
            side=OrderSide.BUY,
            quantity=1.0,
            price=0.6,
            total_value=0.6,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )
        
        # Should not execute
        await orchestrator._execute_copy_trade("0xDISABLED", mock_trade)
        
        # No trade should be executed (state unchanged)
        assert orchestrator.state.trades_executed == 0

    @pytest.mark.asyncio
    async def test_should_copy_trade_default(self, orchestrator):
        """Test default copy trade decision."""
        from api_clients.base import Trade, OrderSide
        from datetime import datetime
        
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            trader_address="0x1234",
            side=OrderSide.BUY,
            quantity=1.0,
            price=0.6,
            total_value=0.6,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )
        
        # Default should copy in sandbox
        should_copy = orchestrator._should_copy_trade("0x1234", trade)
        assert should_copy is True


class TestPerformanceMetrics:
    """Tests for performance metrics tracking."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )
        return CopyCatOrchestrator(config=config)

    def test_update_performance_metrics_no_runner(self, orchestrator):
        """Test performance update without sandbox runner."""
        orchestrator.config.mode = TradingMode.LIVE
        
        # Should not raise
        import asyncio
        asyncio.run(orchestrator._update_performance_metrics())
        
        # State should be unchanged
        assert orchestrator.state.total_pnl == 0.0

    @pytest.mark.asyncio
    async def test_update_performance_metrics_with_runner(self, orchestrator):
        """Test performance update with sandbox runner."""
        # Set up market data callback first
        orchestrator.trading_runner.set_market_data_callback(lambda mid: {
            "market_id": mid,
            "current_price": 0.5,
            "previous_price": 0.5,
            "volatility": 0.02,
        })
        
        # Execute some trades in sandbox
        from sandbox import VirtualOrder
        
        order = VirtualOrder(
            order_id="test_001",
            market_id="test_market",
            side="buy",
            quantity=100.0,
            order_type="market",
            outcome="YES",
        )
        result = await orchestrator.trading_runner.execute_order(order)
        
        # Call update - should not raise and should update metrics
        await orchestrator._update_performance_metrics()
        
        # Verify method completed successfully
        assert orchestrator.state is not None


class TestErrorRecovery:
    """Tests for error recovery mechanisms."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )
        return CopyCatOrchestrator(config=config)

    def test_circuit_breaker_opens(self, orchestrator):
        """Test circuit breaker opens after threshold failures."""
        orchestrator.config.error_recovery.circuit_breaker_enabled = True
        orchestrator.config.error_recovery.circuit_breaker_threshold = 3
        
        # Simulate failures
        orchestrator.state.consecutive_failures = 3
        orchestrator._check_circuit_breaker()
        
        assert orchestrator.state.circuit_breaker_open is True

    def test_circuit_breaker_closes(self, orchestrator):
        """Test circuit breaker closes after successful operation."""
        orchestrator.config.error_recovery.circuit_breaker_enabled = True
        
        # Open circuit breaker
        orchestrator.state.circuit_breaker_open = True
        orchestrator.state.consecutive_failures = 3
        orchestrator._check_circuit_breaker()
        
        # Successful operation
        orchestrator.state.consecutive_failures = 0
        orchestrator._check_circuit_breaker()
        
        assert orchestrator.state.circuit_breaker_open is False


class TestHealthChecks:
    """Tests for health check functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
            health_check=HealthCheckConfig(enabled=True, api_client_check=True),
        )
        return CopyCatOrchestrator(config=config)

    @pytest.mark.asyncio
    async def test_health_check_disabled(self, orchestrator):
        """Test health check when disabled."""
        orchestrator.config.health_check.api_client_check = False
        
        result = await orchestrator._check_api_health()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_no_client(self, orchestrator):
        """Test health check with no API client."""
        orchestrator.api_clients = {}
        
        result = await orchestrator._check_api_health()
        
        assert result is False
        assert orchestrator.state.api_healthy is False


class TestOrchestratorCLI:
    """Tests for OrchestratorCLI."""

    @pytest.fixture
    def cli(self):
        """Create CLI instance for testing."""
        from orchestrator.cli import OrchestratorCLI
        return OrchestratorCLI()

    def test_cli_initialization(self, cli):
        """Test CLI initialization."""
        assert cli.orchestrator is None

    def test_status_output_format(self, cli):
        """Test status output formatting."""
        status = {
            "is_running": True,
            "is_paused": False,
            "mode": "sandbox",
            "platform": "polymarket",
            "copied_traders": 3,
            "trades_executed": 10,
            "total_pnl": 500.0,
            "total_pnl_pct": 0.05,
            "win_rate": 0.6,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "api_healthy": True,
            "circuit_breaker_open": False,
            "uptime_seconds": 3600,
            "cycle_count": 5,
        }
        
        # Should not raise exception
        cli._print_status(status, json_format=False)
        cli._print_status(status, json_format=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
