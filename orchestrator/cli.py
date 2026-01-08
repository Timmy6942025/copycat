"""
CopyCat Orchestrator CLI.
Command-line interface for controlling the trading orchestrator.
"""

import asyncio
import argparse
import logging
import sys
import uuid
from datetime import datetime
from typing import List, Optional

from api_clients.base import Trade, OrderSide

from .config import (
    OrchestratorConfig,
    TradingMode,
    MarketPlatform,
    TraderCopyConfig,
)
from .engine import CopyCatOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrchestratorCLI:
    """Command-line interface for CopyCat Orchestrator."""

    def __init__(self):
        self.orchestrator: Optional[CopyCatOrchestrator] = None

    def run(self):
        """Run the CLI."""
        parser = argparse.ArgumentParser(
            description="CopyCat - Polymarket/Kalshi Copy Trading Bot",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python -m orchestrator.cli start                    # Start in sandbox mode
  python -m orchestrator.cli start --mode live        # Start in live mode
  python -m orchestrator.cli status                   # Show current status
  python -m orchestrator.cli add-trader 0x...         # Add trader to copy
  python -m orchestrator.cli list-copied              # List copied traders
  python -m orchestrator.cli remove-trader 0x...      # Remove trader
  python -m orchestrator.cli stop                     # Stop orchestrator
            """
        )

        parser.add_argument(
            "command",
            choices=[
                "start", "stop", "status", "pause", "resume",
                "add-trader", "remove-trader", "list-copied",
                "run-once", "demo"
            ],
            help="Command to execute"
        )

        # Mode options
        parser.add_argument(
            "--mode",
            choices=["sandbox", "live"],
            default="sandbox",
            help="Trading mode (default: sandbox)"
        )

        parser.add_argument(
            "--platform",
            choices=["polymarket", "kalshi"],
            default="polymarket",
            help="Trading platform (default: polymarket)"
        )

        parser.add_argument(
            "--balance",
            type=float,
            default=10000.0,
            help="Initial balance for sandbox mode (default: 10000)"
        )

        # Trader management
        parser.add_argument(
            "--trader",
            type=str,
            help="Trader address for add-trader/remove-trader commands"
        )

        # Output options
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output in JSON format"
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )

        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Execute command
        asyncio.run(self._execute_command(args))

    async def _execute_command(self, args):
        """Execute the CLI command."""
        command = args.command

        if command == "start":
            await self._cmd_start(args)
        elif command == "stop":
            await self._cmd_stop()
        elif command == "status":
            await self._cmd_status(args)
        elif command == "pause":
            await self._cmd_pause()
        elif command == "resume":
            await self._cmd_resume()
        elif command == "add-trader":
            await self._cmd_add_trader(args)
        elif command == "remove-trader":
            await self._cmd_remove_trader(args)
        elif command == "list-copied":
            await self._cmd_list_copied(args)
        elif command == "run-once":
            await self._cmd_run_once(args)
        elif command == "demo":
            await self._cmd_demo(args)

    async def _cmd_start(self, args):
        """Start the orchestrator."""
        if self.orchestrator and self.orchestrator.state.is_running:
            print("Error: Orchestrator is already running. Use 'stop' first.")
            return

        # Create configuration
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX if args.mode == "sandbox" else TradingMode.LIVE,
            platform=MarketPlatform.POLYMARKET if args.platform == "polymarket" else MarketPlatform.KALSHI,
        )

        # Update sandbox config with provided balance
        if args.mode == "sandbox":
            from .config import SandboxConfigOrchestrator
            config.sandbox = SandboxConfigOrchestrator(initial_balance=args.balance)

        # Initialize and start
        self.orchestrator = CopyCatOrchestrator(config=config)
        result = await self.orchestrator.start()

        if result.success:
            print(f"✓ CopyCat Orchestrator started in {args.mode} mode")
            status = self.orchestrator.get_status()
            self._print_status(status, args.json)
        else:
            print(f"✗ Failed to start: {result.message}")
            if result.error:
                print(f"  Error: {result.error}")

    async def _cmd_stop(self):
        """Stop the orchestrator."""
        if not self.orchestrator:
            print("Error: Orchestrator is not running")
            return

        result = await self.orchestrator.stop()
        if result.success:
            print("✓ CopyCat Orchestrator stopped")
        else:
            print(f"✗ Failed to stop: {result.message}")

    async def _cmd_status(self, args):
        """Show orchestrator status."""
        if not self.orchestrator:
            print("CopyCat Orchestrator is not running")
            return

        status = self.orchestrator.get_status()
        self._print_status(status, args.json)

    async def _cmd_pause(self):
        """Pause the orchestrator."""
        if not self.orchestrator:
            print("Error: Orchestrator is not running")
            return

        result = await self.orchestrator.pause()
        if result.success:
            print("✓ Orchestrator paused")
        else:
            print(f"✗ Failed to pause: {result.message}")

    async def _cmd_resume(self):
        """Resume the orchestrator."""
        if not self.orchestrator:
            print("Error: Orchestrator is not running")
            return

        result = await self.orchestrator.resume()
        if result.success:
            print("✓ Orchestrator resumed")
        else:
            print(f"✗ Failed to resume: {result.message}")

    async def _cmd_add_trader(self, args):
        """Add a trader to copy."""
        if not args.trader:
            print("Error: --trader argument required")
            return

        if not self.orchestrator:
            # Start orchestrator if not running
            config = OrchestratorConfig()
            self.orchestrator = CopyCatOrchestrator(config=config)

        result = await self.orchestrator.add_trader(args.trader)
        if result.success:
            print(f"✓ Added trader {args.trader[:8]}... to copy list")
        else:
            print(f"✗ Failed to add trader: {result.message}")
            if hasattr(result, 'rejection_reasons') and result.rejection_reasons:
                for reason in result.rejection_reasons:
                    print(f"  - {reason}")

    async def _cmd_remove_trader(self, args):
        """Remove a trader from copy list."""
        if not args.trader:
            print("Error: --trader argument required")
            return

        if not self.orchestrator:
            print("Error: Orchestrator is not running")
            return

        result = await self.orchestrator.remove_trader(args.trader)
        if result.success:
            print(f"✓ Removed trader {args.trader[:8]}... from copy list")
        else:
            print(f"✗ Failed to remove trader: {result.message}")

    async def _cmd_list_copied(self, args):
        """List copied traders."""
        if not self.orchestrator:
            print("CopyCat Orchestrator is not running")
            return

        state = self.orchestrator.get_state()
        copied_traders = state.copied_traders

        if not copied_traders:
            print("No traders currently being copied")
            return

        print(f"\nCopied Traders ({len(copied_traders)}):")
        print("-" * 80)

        for addr, config in copied_traders.items():
            print(f"  {addr[:16]}... | Position: ${config.base_position_size:.2f} | Enabled: {config.enabled}")

        print("-" * 80)

    async def _cmd_run_once(self, args):
        """Run a single trading cycle."""
        if not self.orchestrator:
            config = OrchestratorConfig()
            self.orchestrator = CopyCatOrchestrator(config=config)

        print("Running single trading cycle...")
        result = await self.orchestrator._run_trading_cycle()

        if result.success:
            print(f"✓ Trading cycle completed")
            if result.details:
                print(f"  Traders analyzed: {result.details.get('traders_analyzed', 0)}")
                print(f"  Copied traders: {result.details.get('copied_traders', 0)}")
        else:
            print(f"✗ Trading cycle failed: {result.message}")
            if result.error:
                print(f"  Error: {result.error}")

    async def _cmd_demo(self, args):
        """Run a demonstration of the orchestrator."""
        print("\n" + "=" * 60)
        print("CopyCat Orchestrator Demo")
        print("=" * 60)

        # Create configuration
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )

        print("\n1. Initializing orchestrator...")
        orchestrator = CopyCatOrchestrator(config=config)
        print("   ✓ Orchestrator initialized")

        print("\n2. Starting orchestrator...")
        result = await orchestrator.start()
        print(f"   ✓ {result.message}")

        print("\n3. Running single trading cycle...")
        cycle_result = await orchestrator._run_trading_cycle()
        print(f"   ✓ {cycle_result.message}")
        if cycle_result.details:
            for key, value in cycle_result.details.items():
                print(f"     - {key}: {value}")

        print("\n4. Checking status...")
        status = orchestrator.get_status()
        self._print_status(status, json_format=False)

        print("\n5. Stopping orchestrator...")
        stop_result = await orchestrator.stop()
        print(f"   ✓ {stop_result.message}")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60 + "\n")

    def _print_status(self, status: dict, json_format: bool = False):
        """Print orchestrator status."""
        if json_format:
            import json
            print(json.dumps(status, indent=2, default=str))
        else:
            print("\nCopyCat Orchestrator Status:")
            print("-" * 40)
            print(f"  Running:      {'Yes' if status['is_running'] else 'No'}")
            print(f"  Paused:       {'Yes' if status['is_paused'] else 'No'}")
            print(f"  Mode:         {status['mode']}")
            print(f"  Platform:     {status['platform']}")
            print(f"  Copied Traders: {status['copied_traders']}")
            print(f"  Trades Executed: {status['trades_executed']}")
            print(f"  Total P&L:     ${status['total_pnl']:+,.2f} ({status['total_pnl_pct']:+.2%})")
            print(f"  Win Rate:      {status['win_rate']:.1%}")
            print(f"  Sharpe Ratio:  {status['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:  {status['max_drawdown']:.2%}")
            print(f"  API Healthy:   {'Yes' if status['api_healthy'] else 'No'}")
            print(f"  Circuit Open:  {'Yes' if status['circuit_breaker_open'] else 'No'}")
            print(f"  Uptime:        {status['uptime_seconds']:.0f}s")
            print(f"  Cycles:        {status['cycle_count']}")
            print("-" * 40)


def main():
    """Main entry point."""
    cli = OrchestratorCLI()
    cli.run()


if __name__ == "__main__":
    main()
