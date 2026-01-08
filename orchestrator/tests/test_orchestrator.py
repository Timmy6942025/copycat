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
