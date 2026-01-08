"""
CopyCat Orchestrator Engine.
Main orchestration class that coordinates all trading modules.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from api_clients.base import (
    Trade, Trader, TraderPerformance, MarketData, OrderSide, OrderType,
    MarketAPIClient, Position
)
from api_clients import PolymarketAPIClient, KalshiAPIClient
from api_clients.mock import MockMarketAPIClient
from trader_identification import TraderIdentificationEngine, TraderSelectionConfig as TISelectionConfig
from bot_filtering import BotFilter, BotFilterConfig
from sandbox import (
    SandboxRunner,
    SandboxConfig,
    VirtualOrder,
    VirtualOrderResult,
)

from .config import (
    MarketPlatform,
    APIClientConfig,
    TraderSelectionConfig,
    BotFilterOrchestratorConfig,
    CopyTradingConfig,
    SandboxConfigOrchestrator,
    HealthCheckConfig,
    ErrorRecoveryConfig,
    OrchestratorConfig,
    TraderCopyConfig,
    OrchestratorState,
    OrchestrationResult,
    TraderAnalysisResult,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CopyCatOrchestrator:
    """
    Main orchestrator for CopyCat trading system.
    
    Coordinates the entire trading pipeline:
    1. API Clients → Fetch market data and trader information
    2. Trader Identification → Analyze and score traders
    3. Bot Filtering → Filter out automated trading strategies
    4. Copy Trading → Execute trades in sandbox or live mode
    5. Performance Tracking → Monitor and report results
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the orchestrator with configuration."""
        self.config = config or OrchestratorConfig()
        self.state = OrchestratorState()
        
        # Initialize API clients
        self.api_clients: Dict[MarketPlatform, MarketAPIClient] = {}
        self._init_api_clients()
        
        # Initialize module engines
        self._init_module_engines()
        
        # Initialize sandbox/live runner
        self._init_trading_runner()
        
        # Health check and recovery
        self._health_check_task: Optional[asyncio.Task] = None
        self._cycle_task: Optional[asyncio.Task] = None
        
        logger.info(f"CopyCat Orchestrator initialized in {self.config.mode.value} mode")

    def _init_api_clients(self):
        """Initialize API clients based on configuration."""
        if self.config.mode.value == "sandbox":
            # Sandbox mode: use real API clients for real data, but with sandbox execution
            # Polymarket Gamma API is free for market data (no auth required)
            # Falls back to mock client if APIs fail
            try:
                polymarket_client = PolymarketAPIClient()
                self.api_clients[MarketPlatform.POLYMARKET] = polymarket_client
                logger.info("Initialized Polymarket API client for sandbox mode (real data)")
            except Exception as e:
                logger.warning(f"Failed to init Polymarket client, using mock: {e}")
                mock_client = MockMarketAPIClient(
                    platform="polymarket",
                    initial_balance=self.config.sandbox.initial_balance,
                    num_markets=50,
                    num_traders=100,
                )
                self.api_clients[MarketPlatform.POLYMARKET] = mock_client
                logger.info("Using mock API client (real APIs unavailable)")
        elif not self.config.api_clients:
            # Default: use Polymarket
            self.api_clients[MarketPlatform.POLYMARKET] = PolymarketAPIClient()
            self.api_clients[MarketPlatform.KALSHI] = KalshiAPIClient()
        else:
            for client_config in self.config.api_clients:
                if client_config.platform == MarketPlatform.POLYMARKET:
                    client = PolymarketAPIClient(api_key=client_config.api_key)
                elif client_config.platform == MarketPlatform.KALSHI:
                    client = KalshiAPIClient(api_key=client_config.api_key)
                else:
                    logger.warning(f"Unknown platform: {client_config.platform}")
                    continue
                
                client.set_rate_limit(client_config.rate_limit_rps)
                self.api_clients[client_config.platform] = client
        
        logger.info(f"Initialized {len(self.api_clients)} API client(s)")

    def _init_module_engines(self):
        """Initialize trader identification and bot filtering engines."""
        # Convert orchestrator config to module-specific configs
        trader_selection_config = TISelectionConfig(
            min_win_rate=self.config.trader_selection.min_win_rate,
            min_trades=self.config.trader_selection.min_trades,
            max_drawdown=self.config.trader_selection.max_drawdown,
            min_sharpe_ratio=self.config.trader_selection.min_sharpe_ratio,
            min_profit_factor=self.config.trader_selection.min_profit_factor,
            min_total_pnl=self.config.trader_selection.min_total_pnl,
            max_avg_hold_time_hours=self.config.trader_selection.max_avg_hold_time_hours,
            min_reputation_score=self.config.trader_selection.min_reputation_score,
        )
        self.trader_engine = TraderIdentificationEngine(config=trader_selection_config)
        
        # Bot filter config
        bot_filter_config = BotFilterConfig(
            hft_max_hold_time_seconds=self.config.bot_filter.hft_max_hold_time_seconds,
            hft_min_trades_per_minute=self.config.bot_filter.hft_min_trades_per_minute,
            arbitrage_max_profit_pct=self.config.bot_filter.arbitrage_max_profit_pct,
            arbitrage_min_trade_frequency=self.config.bot_filter.arbitrage_min_trade_frequency,
            min_hft_score_to_exclude=self.config.bot_filter.min_hft_score_to_exclude,
            min_arbitrage_score_to_exclude=self.config.bot_filter.min_arbitrage_score_to_exclude,
            min_pattern_score_to_exclude=self.config.bot_filter.min_pattern_score_to_exclude,
        )
        self.bot_filter = BotFilter(config=bot_filter_config)
        
        logger.info("Module engines initialized")

    def _init_trading_runner(self):
        """Initialize the trading runner (sandbox with real data or live trading)."""
        if self.config.mode.value == "sandbox":
            # Sandbox runner with real market data from APIs
            sandbox_config = SandboxConfig(
                initial_balance=self.config.sandbox.initial_balance,
                simulate_slippage=self.config.sandbox.simulate_slippage,
                slippage_model=self.config.sandbox.slippage_model,
                simulate_fees=self.config.sandbox.simulate_fees,
                fee_model=self.config.sandbox.fee_model,
                simulate_fill_probability=self.config.sandbox.simulate_fill_probability,
                results_storage_path=self.config.sandbox.results_storage_path,
            )
            self.trading_runner = SandboxRunner(config=sandbox_config)
            
            # Get real market data from API client
            self._setup_market_data_callback()
            
            logger.info("Sandbox runner initialized with real market data from Polymarket API")
        else:
            # Live trading would initialize real API trading
            self.trading_runner = None
            logger.info("Live trading mode - no sandbox runner")

    def _setup_market_data_callback(self):
        """Set up callback to fetch real market data from API for sandbox execution."""
        if not self.trading_runner:
            return
        
        async def get_real_market_data(market_id: str):
            """Fetch real market data from API client."""
            client = self._get_api_client()
            if not client:
                return self._get_default_market_data(market_id)
            
            try:
                # Get real market data from API
                market_data = await client.get_market_data(market_id)
                if market_data:
                    return {
                        "market_id": market_data.market_id,
                        "current_price": market_data.current_price,
                        "previous_price": market_data.previous_price,
                        "mid_price": market_data.mid_price,
                        "best_bid": market_data.best_bid,
                        "best_ask": market_data.best_ask,
                        "spread": market_data.spread,
                        "volatility": market_data.volatility,
                    }
                
                return self._get_default_market_data(market_id)
                
            except Exception as e:
                logger.warning(f"Error fetching market data for {market_id}: {e}")
                return self._get_default_market_data(market_id)
        
        self.trading_runner.set_market_data_callback(get_real_market_data)
        logger.info("Real market data callback configured")

    def _get_default_market_data(self, market_id: str) -> MarketData:
        """Get default market data when API is unavailable."""
        return MarketData(
            market_id=market_id,
            current_price=0.5,
            previous_price=0.5,
            mid_price=0.5,
            best_bid=0.49,
            best_ask=0.51,
            spread=0.02,
            volume_24h=0.0,
            liquidity=0.0,
            volatility=0.1,
            last_updated=datetime.utcnow(),
        )

    def _get_api_client(self) -> MarketAPIClient:
        """Get the primary API client based on platform configuration."""
        return self.api_clients.get(self.config.platform)

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> OrchestrationResult:
        """Start the orchestrator."""
        if self.state.is_running:
            return OrchestrationResult(
                success=False,
                message="Orchestrator is already running"
            )
        
        try:
            self.state.is_running = True
            self.state.start_time = datetime.utcnow()
            
            # Run initial health check
            health_ok = await self._check_api_health()
            if not health_ok and self.config.mode.value == "live":
                return OrchestrationResult(
                    success=False,
                    message="API health check failed - cannot start in live mode",
                    error="API health check failed"
                )
            
            # Start background tasks
            if self.config.health_check.enabled:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Start trading cycle loop
            self._cycle_task = asyncio.create_task(self._trading_cycle_loop())
            
            logger.info("CopyCat Orchestrator started")
            return OrchestrationResult(
                success=True,
                message="Orchestrator started successfully"
            )
            
        except Exception as e:
            logger.error(f"Error starting orchestrator: {e}")
            return OrchestrationResult(
                success=False,
                message="Failed to start orchestrator",
                error=str(e)
            )

    async def stop(self) -> OrchestrationResult:
        """Stop the orchestrator."""
        try:
            self.state.is_running = False
            self.state.is_paused = False
            
            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._cycle_task:
                self._cycle_task.cancel()
                try:
                    await self._cycle_task
                except asyncio.CancelledError:
                    pass
            
            # Generate final report
            if self.config.mode.value == "sandbox":
                self._generate_performance_report()
            
            logger.info("CopyCat Orchestrator stopped")
            return OrchestrationResult(
                success=True,
                message="Orchestrator stopped successfully"
            )
            
        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")
            return OrchestrationResult(
                success=False,
                message="Failed to stop orchestrator",
                error=str(e)
            )

    async def pause(self) -> OrchestrationResult:
        """Pause the orchestrator."""
        if not self.state.is_running:
            return OrchestrationResult(
                success=False,
                message="Orchestrator is not running"
            )
        
        self.state.is_paused = True
        logger.info("Orchestrator paused")
        return OrchestrationResult(
            success=True,
            message="Orchestrator paused"
        )

    async def resume(self) -> OrchestrationResult:
        """Resume the orchestrator."""
        if not self.state.is_running:
            return OrchestrationResult(
                success=False,
                message="Orchestrator is not running"
            )
        
        self.state.is_paused = False
        logger.info("Orchestrator resumed")
        return OrchestrationResult(
            success=True,
            message="Orchestrator resumed"
        )

    # =========================================================================
    # Health Checks & Error Recovery
    # =========================================================================

    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while self.state.is_running:
            try:
                await asyncio.sleep(self.config.health_check.check_interval_seconds)
                if self.state.is_paused:
                    continue
                
                await self._check_api_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_api_health(self) -> bool:
        """Check API client health."""
        if not self.config.health_check.api_client_check:
            return True
        
        client = self._get_api_client()
        if not client:
            self.state.api_healthy = False
            return False
        
        try:
            # Simple health check - try to get markets
            timeout = self.config.health_check.api_client_timeout_seconds
            markets = await asyncio.wait_for(
                client.get_markets(limit=1),
                timeout=timeout
            )
            self.state.api_healthy = True
            self.state.consecutive_failures = 0
            return True
            
        except asyncio.TimeoutError:
            logger.warning("API health check timed out")
            self.state.api_healthy = False
            self.state.consecutive_failures += 1
            self._check_circuit_breaker()
            return False
            
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            self.state.api_healthy = False
            self.state.consecutive_failures += 1
            self._check_circuit_breaker()
            return False

    def _check_circuit_breaker(self):
        """Check and update circuit breaker state."""
        if not self.config.error_recovery.circuit_breaker_enabled:
            return
        
        threshold = self.config.error_recovery.circuit_breaker_threshold
        if self.state.consecutive_failures >= threshold:
            self.state.circuit_breaker_open = True
            logger.warning(f"Circuit breaker opened after {self.state.consecutive_failures} failures")
        elif self.state.circuit_breaker_open and self.state.consecutive_failures == 0:
            self.state.circuit_breaker_open = False
            logger.info("Circuit breaker closed")

    # =========================================================================
    # Trading Cycle
    # =========================================================================

    async def _trading_cycle_loop(self):
        """Background task for periodic trading cycles."""
        while self.state.is_running:
            try:
                await asyncio.sleep(self.config.trader_data_refresh_interval_seconds)
                if self.state.is_paused or self.state.circuit_breaker_open:
                    continue
                
                await self._run_trading_cycle()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                self.state.errors_encountered += 1

    async def _run_trading_cycle(self) -> OrchestrationResult:
        """Run a single trading cycle."""
        self.state.cycle_count += 1
        self.state.last_cycle_time = datetime.utcnow()
        
        logger.info(f"Running trading cycle {self.state.cycle_count}")
        
        try:
            # Step 1: Discover new traders to analyze
            new_traders = await self._discover_traders()
            
            # Step 2: Analyze and filter traders
            analyzed_count = 0
            for trader_address in new_traders[:self.config.max_traders_to_analyze_per_cycle]:
                result = await self._analyze_trader(trader_address)
                if result.is_suitable:
                    await self._add_trader_to_copy(trader_address, result)
                analyzed_count += 1
            
            self.state.traders_analyzed += analyzed_count
            
            # Step 3: Monitor copied traders
            await self._monitor_copied_traders()
            
            # Step 4: Execute copy trades
            await self._execute_copy_trades()
            
            # Step 5: Update performance metrics
            await self._update_performance_metrics()
            
            logger.info(f"Trading cycle {self.state.cycle_count} completed")
            return OrchestrationResult(
                success=True,
                message=f"Cycle {self.state.cycle_count} completed",
                details={
                    "traders_analyzed": analyzed_count,
                    "copied_traders": len(self.state.copied_traders),
                }
            )
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.state.errors_encountered += 1
            return OrchestrationResult(
                success=False,
                message=f"Cycle {self.state.cycle_count} failed",
                error=str(e)
            )

    # =========================================================================
    # Trader Discovery & Analysis
    # =========================================================================

    async def _discover_traders(self) -> List[str]:
        """Discover new traders to analyze."""
        client = self._get_api_client()
        if not client:
            return []
        
        try:
            # Get recent trades to discover active traders
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=self.config.trade_history_days)
            
            trades = await client.get_trades(
                start_time=start_time,
                end_time=end_time,
                limit=self.config.max_traders_to_analyze_per_cycle
            )
            
            # Extract unique trader addresses
            trader_addresses = list(set(t.trader_address for t in trades))
            
            # Filter out already copied traders
            existing_traders = set(self.state.copied_traders.keys())
            new_traders = [t for t in trader_addresses if t not in existing_traders]
            
            return new_traders
            
        except Exception as e:
            logger.error(f"Error discovering traders: {e}")
            return []

    async def _analyze_trader(self, trader_address: str) -> TraderAnalysisResult:
        """Analyze a single trader for copy trading suitability."""
        client = self._get_api_client()
        if not client:
            return TraderAnalysisResult(trader_address=trader_address)
        
        try:
            # Fetch trader data
            trades = await client.get_trades(
                trader_address=trader_address,
                limit=100
            )
            
            if not trades:
                return TraderAnalysisResult(trader_address=trader_address)
            
            # Step 1: Bot filtering
            bot_result = self.bot_filter.analyze_trades(trades)
            
            if bot_result.is_bot and bot_result.confidence >= 0.5:
                return TraderAnalysisResult(
                    trader_address=trader_address,
                    is_suitable=False,
                    rejection_reasons=[f"Bot detected: {', '.join(bot_result.reasons)}"],
                    bot_filter_result={
                        "is_bot": bot_result.is_bot,
                        "hft_score": bot_result.hft_score,
                        "arbitrage_score": bot_result.arbitrage_score,
                    }
                )
            
            # Step 2: Trader identification and scoring
            trader_result = await self.trader_engine.analyze_trader(
                trader_address=trader_address,
                trades=trades
            )
            
            # Step 3: Calculate recommended position size
            position_size = self._calculate_position_size(trader_result)
            
            return TraderAnalysisResult(
                trader_address=trader_address,
                is_suitable=trader_result.is_suitable,
                reputation_score=trader_result.reputation_score,
                confidence_score=trader_result.confidence_score,
                performance_metrics={
                    "total_pnl": trader_result.performance.total_pnl,
                    "win_rate": trader_result.performance.win_rate,
                    "sharpe_ratio": trader_result.performance.sharpe_ratio,
                    "max_drawdown": trader_result.performance.max_drawdown,
                    "total_trades": trader_result.performance.total_trades,
                },
                bot_filter_result={
                    "is_bot": bot_result.is_bot,
                    "hft_score": bot_result.hft_score,
                    "arbitrage_score": bot_result.arbitrage_score,
                    "pattern_score": bot_result.pattern_score,
                },
                selection_reasons=trader_result.selection_reasons,
                rejection_reasons=trader_result.rejection_reasons,
                recommended_position_size=position_size,
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trader {trader_address}: {e}")
            return TraderAnalysisResult(
                trader_address=trader_address,
                rejection_reasons=[f"Analysis error: {str(e)}"]
            )

    def _calculate_position_size(self, trader_result) -> float:
        """Calculate recommended position size for a trader."""
        method = self.config.copy_trading.position_sizing_method
        base_size = self.config.copy_trading.base_position_size
        pct = self.config.copy_trading.position_size_pct
        
        if method == "fixed_amount":
            return base_size
        elif method == "percentage":
            # Get current portfolio value from state or use initial balance
            portfolio_value = self.state.total_pnl + self.config.sandbox.initial_balance
            return portfolio_value * pct
        elif method == "scaled":
            # Scale by trader reputation score
            portfolio_value = self.state.total_pnl + self.config.sandbox.initial_balance
            base = portfolio_value * pct
            # Scale by confidence (0.5 to 1.0 multiplier)
            multiplier = 0.5 + (trader_result.confidence_score / 2)
            return base * multiplier
        elif method == "kelly":
            # Kelly Criterion (simplified)
            win_rate = trader_result.performance.win_rate
            profit_loss_ratio = (
                trader_result.performance.avg_win / trader_result.performance.avg_loss
                if trader_result.performance.avg_loss > 0 else 1.0
            )
            kelly = ((profit_loss_ratio * win_rate) - (1 - win_rate)) / profit_loss_ratio
            kelly = max(0, kelly) * self.config.copy_trading.kelly_fraction
            
            portfolio_value = self.state.total_pnl + self.config.sandbox.initial_balance
            return portfolio_value * kelly
        else:
            return base_size

    # =========================================================================
    # Copy Trading Management
    # =========================================================================

    async def _add_trader_to_copy(
        self,
        trader_address: str,
        analysis_result: TraderAnalysisResult
    ) -> OrchestrationResult:
        """Add a trader to copy."""
        if len(self.state.copied_traders) >= self.config.max_traders_to_copy:
            return OrchestrationResult(
                success=False,
                message=f"Maximum traders ({self.config.max_traders_to_copy}) already copying"
            )
        
        copy_config = TraderCopyConfig(
            trader_address=trader_address,
            position_sizing_method=self.config.copy_trading.position_sizing_method,
            base_position_size=analysis_result.recommended_position_size,
            position_size_pct=self.config.copy_trading.position_size_pct,
            max_position_size_pct=self.config.copy_trading.max_position_size_pct,
        )
        
        self.state.copied_traders[trader_address] = copy_config
        self.state.traders_added += 1
        
        logger.info(f"Added trader {trader_address[:8]}... to copy list")
        return OrchestrationResult(
            success=True,
            message=f"Added trader {trader_address[:8]}..."
        )

    async def _remove_trader_from_copy(self, trader_address: str) -> OrchestrationResult:
        """Remove a trader from copy list."""
        if trader_address in self.state.copied_traders:
            del self.state.copied_traders[trader_address]
            self.state.traders_removed += 1
            logger.info(f"Removed trader {trader_address[:8]}... from copy list")
            return OrchestrationResult(
                success=True,
                message=f"Removed trader {trader_address[:8]}..."
            )
        
        return OrchestrationResult(
            success=False,
            message=f"Trader {trader_address[:8]}... not in copy list"
        )

    async def _monitor_copied_traders(self):
        """Monitor performance of copied traders and remove underperformers."""
        client = self._get_api_client()
        if not client:
            return
        
        for trader_address, copy_config in list(self.state.copied_traders.items()):
            if not copy_config.auto_remove_if_performance_drops:
                continue
            
            try:
                # Get recent performance
                trades = await client.get_trades(
                    trader_address=trader_address,
                    limit=20
                )
                
                if not trades:
                    continue
                
                # Calculate recent win rate
                wins = sum(1 for t in trades if self._is_winning_trade(t))
                recent_win_rate = wins / len(trades)
                
                # Remove if performance dropped below threshold
                if recent_win_rate < copy_config.min_performance_threshold:
                    logger.info(f"Removing {trader_address[:8]}... - win rate {recent_win_rate:.1%}")
                    await self._remove_trader_from_copy(trader_address)
                    
            except Exception as e:
                logger.error(f"Error monitoring trader {trader_address[:8]}...: {e}")

    async def _execute_copy_trades(self):
        """Execute trades for copied traders."""
        client = self._get_api_client()
        if not client:
            return
        
        for trader_address in self.state.copied_traders:
            try:
                # Get recent trades from this trader
                recent_trades = await client.get_trades(
                    trader_address=trader_address,
                    limit=10
                )
                
                for trade in recent_trades:
                    # Skip if already executed (in production, track this)
                    if self._should_copy_trade(trader_address, trade):
                        await self._execute_copy_trade(trader_address, trade)
                        
            except Exception as e:
                logger.error(f"Error executing trades for {trader_address[:8]}...: {e}")

    def _should_copy_trade(self, trader_address: str, trade: Trade) -> bool:
        """Determine if a trade should be copied."""
        # In production, track which trades have been copied
        # For now, always copy in sandbox mode
        return True

    async def _execute_copy_trade(self, trader_address: str, trade: Trade):
        """Execute a copy trade."""
        if self.config.mode.value == "sandbox" and self.trading_runner:
            copy_config = self.state.copied_traders.get(trader_address)
            if not copy_config or not copy_config.enabled:
                return
            
            # Create virtual order
            order = VirtualOrder(
                market_id=trade.market_id,
                side=trade.side,
                quantity=copy_config.base_position_size or self.config.copy_trading.base_position_size,
                outcome=trade.outcome,
                source_trader_id=trader_address,
            )
            
            # Execute in sandbox
            result = await self.trading_runner.execute_order(order)
            
            if result.status.value == "filled":
                self.state.trades_executed += 1
                logger.info(f"Copied trade: {trade.market_id} {trade.side.value} @ {result.average_price:.3f}")
            
        # Live mode would use real API orders

    # =========================================================================
    # Performance Tracking
    # =========================================================================

    async def _update_performance_metrics(self):
        """Update overall performance metrics."""
        if self.config.mode.value == "sandbox" and self.trading_runner:
            try:
                metrics = self.trading_runner.get_performance_metrics()
                
                self.state.total_pnl = metrics.total_pnl
                self.state.total_pnl_pct = metrics.total_pnl_pct
                self.state.win_rate = metrics.win_rate
                self.state.sharpe_ratio = metrics.sharpe_ratio
                self.state.max_drawdown = metrics.max_drawdown
                
            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")

    def _generate_performance_report(self):
        """Generate and save performance report."""
        if self.config.mode.value == "sandbox" and self.trading_runner:
            try:
                report_path = self.trading_runner.save_report("orchestrator_report.md")
                logger.info(f"Performance report saved to {report_path}")
            except Exception as e:
                logger.error(f"Error generating report: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    async def add_trader(self, trader_address: str) -> OrchestrationResult:
        """Manually add a trader to analyze and copy."""
        result = await self._analyze_trader(trader_address)
        if result.is_suitable:
            return await self._add_trader_to_copy(trader_address, result)
        else:
            return OrchestrationResult(
                success=False,
                message=f"Trader {trader_address[:8]}... is not suitable for copying",
                details={"rejection_reasons": result.rejection_reasons}
            )

    async def remove_trader(self, trader_address: str) -> OrchestrationResult:
        """Remove a trader from copy list."""
        return await self._remove_trader_from_copy(trader_address)

    def get_state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        return self.state

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status summary."""
        return {
            "is_running": self.state.is_running,
            "is_paused": self.state.is_paused,
            "mode": self.config.mode.value,
            "platform": self.config.platform.value,
            "copied_traders": len(self.state.copied_traders),
            "trades_executed": self.state.trades_executed,
            "total_pnl": self.state.total_pnl,
            "total_pnl_pct": self.state.total_pnl_pct,
            "win_rate": self.state.win_rate,
            "sharpe_ratio": self.state.sharpe_ratio,
            "max_drawdown": self.state.max_drawdown,
            "api_healthy": self.state.api_healthy,
            "circuit_breaker_open": self.state.circuit_breaker_open,
            "uptime_seconds": (
                (datetime.utcnow() - self.state.start_time).total_seconds()
                if self.state.start_time else 0
            ),
            "cycle_count": self.state.cycle_count,
        }

    # Helper methods

    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade is winning."""
        return trade.price > 0.5 if trade.outcome == "YES" else trade.price < 0.5
