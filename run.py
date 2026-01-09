#!/usr/bin/env python3
"""
CopyCat - Simple Entry Point

Usage:
    python run.py                    # Run sandbox demo (quick test)
    python run.py sandbox            # Run in sandbox mode
    python run.py status             # Check status
    python run.py stop               # Stop the bot
    python run.py speed              # Run with Speed Mode (all 8 optimizations)
    python run.py add 0x...          # Add trader to copy
    python run.py list               # List copied traders
    python run.py logs               # View recent logs

Options:
    --balance 100    Starting balance (default: 100)
    --mode balanced  Speed mode: conservative/balanced/aggressive
    --platform polymarket  Platform: polymarket
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from datetime import datetime
from orchestrator.config import (
    OrchestratorConfig, TradingMode, MarketPlatform, SelectionMode,
    TraderSelectionConfig, BotFilterOrchestratorConfig, CopyTradingConfig,
    SandboxConfigOrchestrator,
)
from orchestrator.config_speed import create_speed_mode_config, SpeedModeEngine
from orchestrator.engine import CopyCatOrchestrator

# Default settings
DEFAULT_BALANCE = 100.0
DEFAULT_MODE = "sandbox"


def get_venv_python():
    """Get Python interpreter from virtual environment."""
    venv_python = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def run_command(args):
    """Run a shell command and print output."""
    python = get_venv_python()
    cmd = f"{python} -m orchestrator.cli {' '.join(args)}" if args else f"{python} -m orchestrator.cli demo"
    os.system(cmd)


async def run_sandbox(balance=100.0, speed_mode=None):
    """Run the bot in sandbox mode."""
    print("\n" + "=" * 60)
    print("CopyCat - Sandbox Mode")
    print("=" * 60)
    
    if speed_mode:
        print(f"\n🚀 Running with Speed Mode: {speed_mode}")
        config = create_speed_mode_config(
            initial_balance=balance,
            speed_mode=speed_mode,
        )
        engine = SpeedModeEngine(config)
        await engine.start()
        
        print("\n✓ Bot running! Press Ctrl+C to stop.\n")
        try:
            while True:
                await asyncio.sleep(60)
                status = engine.get_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] P&L: ${status['total_pnl']:+,.2f} | Trades: {status['trades_executed']}")
        except KeyboardInterrupt:
            print("\n\nStopping...")
            await engine.stop()
            print("✓ Bot stopped")
    else:
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
            trader_selection=TraderSelectionConfig(
                mode=SelectionMode.GROWTH,
            ),
            sandbox=SandboxConfigOrchestrator(initial_balance=balance),
        )
        orchestrator = CopyCatOrchestrator(config)
        result = await orchestrator.start()
        
        if result.success:
            print(f"\n✓ Bot started with ${balance:,.2f} balance")
            print("  Press Ctrl+C to stop\n")
            
            try:
                while True:
                    await asyncio.sleep(60)
                    status = orchestrator.get_status()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] P&L: ${status['total_pnl']:+,.2f} | Trades: {status['trades_executed']}")
            except KeyboardInterrupt:
                print("\n\nStopping...")
                await orchestrator.stop()
                print("✓ Bot stopped")
        else:
            print(f"✗ Failed to start: {result.message}")


def main():
    """Main entry point."""
    args = sys.argv[1:]
    
    if not args or args[0] in ["-h", "--help", "help"]:
        print(__doc__)
        return
    
    command = args[0]
    
    if command == "sandbox":
        balance = 100.0
        speed_mode = None
        
        # Parse options
        i = 1
        while i < len(args):
            if args[i] == "--balance" and i + 1 < len(args):
                balance = float(args[i + 1])
                i += 2
            elif args[i] == "--mode" and i + 1 < len(args):
                speed_mode = args[i + 1]
                i += 2
            else:
                i += 1
        
        asyncio.run(run_sandbox(balance, speed_mode))
    
    elif command == "speed":
        # Speed mode - all 8 optimizations
        balance = 100.0
        speed_mode = "balanced"
        
        i = 1
        while i < len(args):
            if args[i] == "--balance" and i + 1 < len(args):
                balance = float(args[i + 1])
                i += 2
            elif args[i] == "--mode" and i + 1 < len(args):
                speed_mode = args[i + 1]
                i += 2
            else:
                i += 1
        
        asyncio.run(run_sandbox(balance, speed_mode))
    
    elif command in ["status", "stop", "list"]:
        # Pass through to CLI
        run_command(args)
    
    elif command == "add":
        # Add trader
        if len(args) < 2:
            print("Usage: python run.py add 0xTRADER_ADDRESS")
            print("Example: python run.py add 0x1234567890abcdef")
            return
        run_command(args)
    
    elif command == "logs":
        # Show logs (last 50 lines)
        python = get_venv_python()
        print("\nLast 50 log entries:")
        print("-" * 40)
        os.system(f"{python} -m orchestrator.cli status 2>&1 | tail -50")
    
    elif command == "demo":
        # Run demo
        run_command(["demo"])
    
    else:
        print(f"Unknown command: {command}")
        print("\nUse: python run.py --help")


if __name__ == "__main__":
    main()
