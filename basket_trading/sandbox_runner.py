"""
Basket Trading Sandbox Runner.
Paper trading simulation for basket-based trading.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from api_clients.polymarket.data_api import DataAPIClient
from sandbox.runner import SandboxRunner
from sandbox.models import VirtualOrder, OrderSide, OrderType

from . import (
    BasketConfig,
    BasketSignal,
    BasketTradeResult,
    Topic,
)
from .orchestrator import BasketTradingOrchestrator


class BasketSandboxRunner:
    """
    Paper trading runner for basket-based trading.

    Integrates basket signal generation with sandbox
    execution simulation.
    """

    def __init__(
        self,
        data_client: DataAPIClient,
        config: Optional[BasketConfig] = None,
        initial_balance: float = 10000.0,
    ):
        """Initialize basket sandbox runner."""
        self.config = config or BasketConfig()
        self.data_client = data_client

        self.orchestrator = BasketTradingOrchestrator(data_client, self.config)

        sandbox_config = {
            'mode': 'sandbox',
            'initial_balance': initial_balance,
            'max_orders_per_day': 50,
            'max_position_size_pct': 0.10,
            'max_total_exposure_pct': 0.50,
            'simulate_slippage': True,
            'simulate_fees': True,
        }

        from sandbox.config import SandboxConfig as SandboxConfigData
        self.sandbox_config = SandboxConfigData(**sandbox_config)
        self.sandbox_runner = SandboxRunner(self.sandbox_config)

        self.execution_history: List[BasketTradeResult] = []
        self.portfolio_value = initial_balance

    async def initialize(self) -> None:
        """Initialize basket trading system."""
        logger.info("Initializing basket trading sandbox runner")

        logger.info("Starting sandbox runner")
        await self.sandbox_runner.start()

    async def run_basket_trading(
        self,
        topic_wallets: Dict[Topic, List[str]],
        scan_interval_seconds: int = 60,
    ) -> None:
        """Run basket trading loop."""
        await self.initialize()

        logger.info(f"Initializing baskets for {len(topic_wallets)} topics")
        await self.orchestrator.initialize_topic_baskets(topic_wallets)
        await self.orchestrator.detect_and_assign_clusters()
        await self.orchestrator.rank_all_baskets()

        logger.info("Starting basket trading loop")

        while True:
            try:
                signals = await self.orchestrator.scan_for_signals()

                if signals:
                    logger.info(f"Processing {len(signals)} basket signals")

                    results = await self.orchestrator.execute_top_signals(
                        limit=3,
                        quantity_per_signal=self._calculate_position_size(),
                    )

                    for result in results:
                        await self._execute_basket_trade(result)
                        self.execution_history.append(result)

                await self._update_portfolio()

                await asyncio.sleep(scan_interval_seconds)

            except Exception as e:
                logger.error(f"Error in basket trading loop: {e}")
                await asyncio.sleep(10)

    async def _execute_basket_trade(
        self,
        result: BasketTradeResult,
    ) -> None:
        """Execute basket trade in sandbox."""
        if not result.executed:
            logger.warning(f"Trade not executed: {result.rejection_reason}")
            return

        signal = result.signal

        order = VirtualOrder(
            market_id=signal.market_id,
            side=OrderSide.BUY if signal.side == OrderSide.BUY else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=result.quantity,
            outcome=signal.outcome,
            created_at=datetime.utcnow(),
        )

        execution_result = await self.sandbox_runner.execute_order(order)

        logger.info(
            f"Executed basket trade: {signal.market_title} "
            f"@ {execution_result.average_price:.4f}, "
            f"qty={result.quantity:.2f}"
        )

    def _calculate_position_size(self) -> float:
        """Calculate position size based on portfolio."""
        portfolio_summary = self.sandbox_runner.get_summary()
        position_size = portfolio_summary.total_value * self.sandbox_config.max_position_size_pct

        return max(
            self.sandbox_config.min_order_size,
            position_size,
        )

    async def _update_portfolio(self) -> None:
        """Update portfolio value tracking."""
        summary = self.sandbox_runner.get_summary()
        self.portfolio_value = summary.total_value

        logger.debug(
            f"Portfolio value: ${self.portfolio_value:.2f}, "
            f"Positions: {summary.position_count}"
        )

    def get_execution_history(
        self,
        hours: int = 24,
    ) -> List[BasketTradeResult]:
        """Get basket trade execution history."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=hours)

        return [
            r for r in self.execution_history
            if r.timestamp >= cutoff
        ]

    def get_performance_summary(self) -> Dict:
        """Get performance summary of basket trading."""
        history = self.get_execution_history(hours=24 * 30)

        if not history:
            return {
                "total_trades": 0,
                "successful_trades": 0,
                "pnl": 0.0,
                "win_rate": 0.0,
                "portfolio_value": self.portfolio_value,
            }

        total_trades = len(history)
        successful_trades = sum(1 for r in history if r.executed)

        pnl = sum(r.quantity * (r.execution_price - r.signal.avg_entry_price) for r in history if r.executed)
        win_rate = successful_trades / total_trades if total_trades > 0 else 0.0

        portfolio_summary = self.sandbox_runner.get_summary()

        return {
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "pnl": pnl,
            "win_rate": win_rate,
            "portfolio_value": portfolio_summary.total_value,
            "positions_count": portfolio_summary.position_count,
            "exposure_pct": portfolio_summary.exposure_pct,
        }

    async def stop(self) -> None:
        """Stop basket trading runner."""
        logger.info("Stopping basket trading sandbox runner")
        await self.sandbox_runner.stop()


logger = __import__('logging').getLogger(__name__)
