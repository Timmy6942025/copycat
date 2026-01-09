#!/usr/bin/env python3
"""
CopyCat Live Trading Demo with Micro Mode.

This script demonstrates how to use CopyCat for live trading with
Micro Mode settings optimized for small accounts ($10-$50).

WARNING: This executes REAL trades with REAL money.
Always test in SANDBOX mode first!

Usage:
    python demo_live_trading_micro.py --mode sandbox    # Test mode
    python demo_live_trading_micro.py --mode live       # Live trading
"""

import asyncio
import logging
import argparse
from datetime import datetime
from typing import Optional

from api_clients import PolymarketAPIClient
from api_clients.mock import MockMarketAPIClient

from orchestrator.config import OrchestratorConfig, TradingMode, MarketPlatform
from orchestrator.config_micro import create_micro_config, MicroModeLevel, TradingMode as MicroTradingMode
from orchestrator.live_trading_micro import (
    MicroLiveTradingConfig,
    MicroLiveTradingRunner,
    create_micro_live_config,
    create_micro_live_runner,
)
from orchestrator.notification_service import (
    NotificationService, NotificationConfig, NotificationEvent, NotificationPriority
)
from orchestrator.mode_transition import create_mode_manager, TradingModeLevel
from orchestrator.circuit_breaker import create_circuit_breaker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CopyCatLiveTrader:
    """
    Complete CopyCat live trading setup with Micro Mode.
    
    Features:
    - Micro Mode position sizing (NANO/MICRO/MINI)
    - Circuit breaker protection
    - Discord notifications
    - Automatic mode transitions
    """
    
    def __init__(
        self,
        initial_balance: float,
        wallet_address: str,
        micro_mode: str,
        discord_webhook_url: Optional[str] = None,
        platform: str = "polymarket",
    ):
        self.initial_balance = initial_balance
        self.wallet_address = wallet_address
        self.micro_mode = micro_mode
        self.discord_webhook_url = discord_webhook_url
        self.platform = platform
        
        # Components
        self.api_client: Optional = None
        self.notification_service: Optional[NotificationService] = None
        self.trading_runner: Optional[MicroLiveTradingRunner] = None
        self.mode_manager = None
        self.circuit_breaker = None
        
        # State
        self.is_running = False
        
    async def initialize(self, sandbox: bool = True):
        """Initialize all components."""
        logger.info(f"Initializing CopyCat Live Trader (sandbox={sandbox})...")
        
        # Step 1: Initialize API client
        self.api_client = self._create_api_client(sandbox)
        
        # Step 2: Initialize notification service
        if self.discord_webhook_url:
            await self._init_notification_service()
        
        # Step 3: Initialize trading runner
        await self._init_trading_runner(sandbox)
        
        # Step 4: Initialize mode manager
        self._init_mode_manager()
        
        # Step 5: Initialize circuit breaker
        self._init_circuit_breaker()
        
        logger.info("CopyCat Live Trader initialized successfully")
    
    def _create_api_client(self, sandbox: bool):
        """Create API client."""
        if sandbox:
            return MockMarketAPIClient(
                platform=self.platform,
                initial_balance=self.initial_balance,
                num_markets=50,
                num_traders=100,
            )
        else:
            return PolymarketAPIClient()
    
    async def _init_notification_service(self):
        """Initialize Discord notification service."""
        config = NotificationConfig(
            discord_webhook_url=self.discord_webhook_url,
            milestone_channels=[20, 50, 100, 200, 500, 1000],
            notify_on_trade=True,
            notify_on_trade_error=True,
            notify_on_circuit_breaker=True,
            notify_on_mode_transition=True,
            notify_on_startup_shutdown=True,
            batch_trade_notifications=True,
            enabled=True,
        )
        
        self.notification_service = NotificationService(config)
        await self.notification_service.start()
        
        logger.info("Notification service initialized")
    
    async def _init_trading_runner(self, sandbox: bool):
        """Initialize the trading runner."""
        mode = MicroTradingMode.SANDBOX if sandbox else MicroTradingMode.LIVE
        
        config = create_micro_live_config(
            initial_balance=self.initial_balance,
            wallet_address=self.wallet_address,
            micro_mode=self.micro_mode,
            enable_notifications=bool(self.discord_webhook_url),
            discord_webhook_url=self.discord_webhook_url,
        )
        
        self.trading_runner = await create_micro_live_runner(
            config=config,
            api_client=self.api_client,
            notification_service=self.notification_service,
        )
        
        # Set up callbacks
        self.trading_runner.set_trade_callback(self._on_trade_executed)
        self.trading_runner.set_circuit_breaker_callback(self._on_circuit_breaker_change)
        
        logger.info(f"Trading runner initialized in {config.micro_mode.value} mode")
    
    def _init_mode_manager(self):
        """Initialize the mode transition manager."""
        # Create a mock orchestrator for the mode manager
        initial_balance = self.initial_balance
        position_size_pct = self.trading_runner.config.position_size_pct if self.trading_runner else 0.5
        kelly_fraction = self.trading_runner.config.kelly_fraction if self.trading_runner else 0.5
        
        class MockOrchestrator:
            def __init__(self):
                self.state = type('State', (), {
                    'total_pnl': 0.0,
                    'cycle_count': 0,
                    'max_drawdown': 0.0,
                })()
                self.config = type('Config', (), {
                    'sandbox': type('Sandbox', (), {'initial_balance': initial_balance})(),
                    'copy_trading': type('Trading', (), {
                        'position_size_pct': position_size_pct,
                        'kelly_fraction': kelly_fraction,
                    })(),
                    'trader_selection': type('Selection', (), {
                        'growth_min_total_pnl': 25.0,
                        'growth_min_growth_rate': 0.005,
                        'growth_max_drawdown': 0.50,
                    })(),
                    'boost_mode': type('Boost', (), {'enabled': True})(),
                })()
        
        mock_orchestrator = MockOrchestrator()
        
        self.mode_manager = create_mode_manager(
            orchestrator=mock_orchestrator,
            balance=self.initial_balance,
            on_transition=self._on_mode_transition,
        )
        
        logger.info(f"Mode manager initialized: {self.mode_manager.current_mode.value}")
    
    def _init_circuit_breaker(self):
        """Initialize the circuit breaker."""
        mode_level = self.trading_runner.config.get_mode_level()
        
        self.circuit_breaker = create_circuit_breaker(
            orchestrator=None,  # Will be updated manually
            mode=mode_level,
            on_state_change=self._on_circuit_breaker_change,
        )
        
        logger.info(f"Circuit breaker initialized for {mode_level.value} mode")
    
    async def _on_trade_executed(self, trade_info: dict):
        """Handle trade execution callback."""
        logger.info(
            f"Trade executed: {trade_info['side'].upper()} "
            f"{trade_info['quantity']:.2f} {trade_info['market_id'][:20]}... "
            f"@ ${trade_info['price']:.4f}"
        )
    
    async def _on_mode_transition(self, transition_record):
        """Handle mode transition callback."""
        logger.info(
            f"MODE TRANSITION: {transition_record.from_mode.value.upper()} → "
            f"{transition_record.to_mode.value.upper()} "
            f"(balance: ${transition_record.balance:.2f})"
        )
        
        # Update trading runner mode
        new_mode = transition_record.to_mode
        if hasattr(self.trading_runner.config, 'micro_mode'):
            mode_map = {
                TradingModeLevel.NANO: "nano",
                TradingModeLevel.MICRO: "micro",
                TradingModeLevel.MINI: "mini",
                TradingModeLevel.BALANCED: "balanced",
            }
            self.trading_runner.config.micro_mode = LiveMicroTradingMode(mode_map.get(new_mode, "nano"))
        
        # Update circuit breaker mode
        self.circuit_breaker.set_mode(new_mode)
    
    async def _on_circuit_breaker_change(self, record):
        """Handle circuit breaker state change."""
        logger.warning(
            f"CIRCUIT BREAKER: {record.previous_state.value.upper()} → "
            f"{record.new_state.value.upper()} "
            f"(drawdown: {record.drawdown_pct:.1%})"
        )
    
    async def start(self):
        """Start live trading."""
        if self.is_running:
            logger.warning("Already running")
            return
        
        # Start notification service
        if self.notification_service:
            await self.notification_service.start()
        
        # Start trading runner
        success = await self.trading_runner.start()
        if not success:
            logger.error("Failed to start trading runner")
            return
        
        self.is_running = True
        logger.info("CopyCat Live Trader started")
        
        # Send startup notification
        if self.notification_service:
            await self.notification_service.notify_event(
                event=NotificationEvent.STARTUP,
                title="🚀 CopyCat Live Trading Started",
                description=f"Started with ${self.initial_balance:.2f} in {self.micro_mode.upper()} mode",
                priority=NotificationPriority.LOW,
            )
    
    async def stop(self):
        """Stop live trading."""
        if not self.is_running:
            return
        
        # Send shutdown notification
        if self.notification_service:
            await self.notification_service.notify_event(
                event=NotificationEvent.SHUTDOWN,
                title="🛑 CopyCat Live Trading Stopped",
                description=f"Stopped after trading cycle",
                priority=NotificationPriority.LOW,
            )
        
        # Stop trading runner
        await self.trading_runner.stop()
        
        # Stop notification service
        if self.notification_service:
            await self.notification_service.stop()
        
        self.is_running = False
        logger.info("CopyCat Live Trader stopped")
    
    async def run_cycle(self):
        """Run a single trading cycle (for testing)."""
        if not self.is_running:
            logger.error("Not running")
            return
        
        # Check mode transitions
        if self.mode_manager:
            transition = await self.mode_manager.check_transition()
            if transition:
                logger.info(f"Mode transition: {transition.from_mode.value} → {transition.to_mode.value}")
        
        # Check circuit breaker
        if self.circuit_breaker:
            position_multiplier = await self.circuit_breaker.check()
            if position_multiplier < 1.0:
                logger.warning(f"Circuit breaker active: position multiplier {position_multiplier:.2f}")
        
        # Get portfolio summary
        summary = self.trading_runner.get_portfolio_summary()
        logger.info(
            f"Portfolio: ${summary['balance']:.2f}, "
            f"Positions: {summary['positions_count']}, "
            f"Circuit Breaker: {summary.get('circuit_breaker_state', 'N/A')}"
        )
        
        return summary
    
    def get_status(self) -> dict:
        """Get current status."""
        runner = self.trading_runner.runner if hasattr(self.trading_runner, 'runner') else self.trading_runner
        
        status = {
            "is_running": self.is_running,
            "mode": self.micro_mode,
            "balance": self.trading_runner.state.balance if self.trading_runner else 0,
            "positions": len(runner.positions) if hasattr(runner, 'positions') else 0,
            "circuit_breaker": self.trading_runner.get_circuit_breaker_status() if self.trading_runner else {},
        }
        
        if self.mode_manager:
            status["mode_status"] = self.mode_manager.get_status()
        
        return status


# Alias for backwards compatibility
LiveMicroTradingMode = MicroLiveTradingConfig


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CopyCat Live Trading Demo")
    parser.add_argument(
        "--mode", choices=["sandbox", "live"], default="sandbox",
        help="Trading mode (sandbox or live)"
    )
    parser.add_argument(
        "--balance", type=float, default=10.0,
        help="Initial balance"
    )
    parser.add_argument(
        "--micro-mode", choices=["nano", "micro", "mini", "balanced"],
        default="nano",
        help="Micro mode level"
    )
    parser.add_argument(
        "--wallet", type=str, default="",
        help="Wallet address for live trading"
    )
    parser.add_argument(
        "--discord", type=str, default="",
        help="Discord webhook URL for notifications"
    )
    parser.add_argument(
        "--cycles", type=int, default=3,
        help="Number of trading cycles to run"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CopyCat Live Trading Demo")
    print("=" * 60)
    print(f"Mode:         {args.mode}")
    print(f"Balance:      ${args.balance:.2f}")
    print(f"Micro Mode:   {args.micro_mode}")
    print(f"Wallet:       {args.wallet[:16] if args.wallet else 'Not set'}...")
    print(f"Notifications: {'Enabled' if args.discord else 'Disabled'}")
    print(f"Cycles:       {args.cycles}")
    print("=" * 60)
    
    if args.mode == "live" and not args.wallet:
        print("ERROR: Wallet address required for live trading")
        print("Use: --wallet 0x...")
        return
    
    if args.mode == "live":
        print("\n⚠️  WARNING: LIVE TRADING MODE - REAL MONEY AT RISK! ⚠️")
        print("Make sure you understand the risks before proceeding.\n")
    
    # Create trader
    trader = CopyCatLiveTrader(
        initial_balance=args.balance,
        wallet_address=args.wallet,
        micro_mode=args.micro_mode,
        discord_webhook_url=args.discord or None,
    )
    
    try:
        # Initialize
        await trader.initialize(sandbox=(args.mode == "sandbox"))
        
        # Start
        await trader.start()
        
        # Run cycles
        for i in range(args.cycles):
            print(f"\n--- Cycle {i + 1}/{args.cycles} ---")
            await trader.run_cycle()
            await asyncio.sleep(2)  # Wait between cycles
        
        # Show final status
        print("\n" + "=" * 60)
        print("Final Status:")
        print("-" * 60)
        status = trader.get_status()
        print(f"Running: {status['is_running']}")
        print(f"Mode: {status['mode']}")
        print(f"Balance: ${status['balance']:.2f}")
        print(f"Positions: {status['positions']}")
        if status.get('circuit_breaker'):
            print(f"Circuit Breaker: {status['circuit_breaker'].get('state', 'N/A')}")
        print("=" * 60)
        
    finally:
        # Stop
        await trader.stop()
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
