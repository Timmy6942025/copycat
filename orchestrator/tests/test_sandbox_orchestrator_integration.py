"""Integration Test: Sandbox-Orchestrator Connection."""

import asyncio
import sys
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.engine import CopyCatOrchestrator
from orchestrator.config import (
    OrchestratorConfig, TradingMode, MarketPlatform,
    SandboxConfigOrchestrator, CopyTradingConfig, TraderCopyConfig,
)
from sandbox import SandboxRunner, SandboxConfig, VirtualOrder
from api_clients.base import Trade, OrderSide


async def test_orchestrator_sandbox_initialization():
    print("\n" + "=" * 60)
    print("Test 1: Orchestrator Sandbox Initialization")
    print("=" * 60)
    
    config = OrchestratorConfig(
        mode=TradingMode.SANDBOX,
        platform=MarketPlatform.POLYMARKET,
        sandbox=SandboxConfigOrchestrator(initial_balance=10000.0),
        copy_trading=CopyTradingConfig(base_position_size=100.0),
    )
    
    orchestrator = CopyCatOrchestrator(config)
    
    assert orchestrator.trading_runner is not None
    assert orchestrator.state.is_running == False
    
    print(f"Initial balance: ${config.sandbox.initial_balance:,.2f}")
    print(f"Sandbox runner: {type(orchestrator.trading_runner).__name__}")
    
    return True


async def test_sandbox_runner_standalone():
    print("\n" + "=" * 60)
    print("Test 2: Sandbox Runner Standalone")
    print("=" * 60)
    
    runner = SandboxRunner(config=SandboxConfig(initial_balance=10000.0))
    
    runner.set_market_data_callback(lambda mid: {
        "market_id": mid,
        "current_price": 0.5,
        "previous_price": 0.5,
        "volatility": 0.02,
    })
    
    order = VirtualOrder(
        order_id="test_order_001",
        market_id="test_market",
        side="buy",
        quantity=100.0,
        order_type="market",
        outcome="YES",
    )
    
    result = await runner.execute_order(order)
    
    assert result is not None
    assert result.status in ["FILLED", "PENDING", "NO_FILL"]
    
    print(f"Order status: {result.status}")
    print(f"Filled quantity: {result.filled_quantity}")
    
    summary = runner.get_portfolio_summary()
    print(f"Portfolio value: ${summary.total_value:,.2f}")
    
    return True


async def test_orchestrator_copy_trade_execution():
    print("\n" + "=" * 60)
    print("Test 3: Orchestrator Copy Trade Execution")
    print("=" * 60)
    
    config = OrchestratorConfig(
        mode=TradingMode.SANDBOX,
        sandbox=SandboxConfigOrchestrator(initial_balance=10000.0),
        copy_trading=CopyTradingConfig(base_position_size=50.0),
    )
    
    with patch('orchestrator.engine.PolymarketAPIClient') as mock_client_class:
        mock_client_class.return_value = AsyncMock()
        orchestrator = CopyCatOrchestrator(config)
        
        mock_trade = Trade(
            trade_id="mock_trade_001",
            market_id="test_market",
            trader_address="0x1234567890abcdef",
            side=OrderSide.BUY,
            quantity=1.0,
            price=0.55,
            total_value=0.55,
            timestamp=datetime.utcnow(),
            outcome="YES",
        )
        
        orchestrator.state.copied_traders["0x1234567890abcdef"] = TraderCopyConfig(
            trader_address="0x1234567890abcdef",
            base_position_size=50.0,
            enabled=True,
        )
        
        await orchestrator._execute_copy_trade("0x1234567890abcdef", mock_trade)
        
        print("Copy trade executed successfully")
        print(f"Traders in copy list: {len(orchestrator.state.copied_traders)}")
        
    return True


async def test_performance_metrics():
    print("\n" + "=" * 60)
    print("Test 4: Performance Metrics Integration")
    print("=" * 60)
    
    runner = SandboxRunner(config=SandboxConfig(initial_balance=10000.0))
    runner.set_market_data_callback(lambda mid: {
        "market_id": mid,
        "current_price": 0.5,
        "previous_price": 0.5,
        "volatility": 0.02,
    })
    
    for i in range(5):
        order = VirtualOrder(
            order_id=f"test_order_{i:03d}",
            market_id=f"market_{i % 3}",
            side="buy",
            quantity=100.0,
            order_type="market",
            outcome="YES",
        )
        await runner.execute_order(order)
        await runner.update_market_prices({f"market_{i % 3}": 0.5 + (i * 0.01)})
    
    metrics = runner.get_performance_metrics()
    
    print(f"Total trades: {metrics.total_trades}")
    print(f"Total P&L: ${metrics.total_pnl:,.2f}")
    print(f"Win rate: {metrics.win_rate:.1%}")
    
    assert metrics.total_trades == 5
    
    return True


async def test_orchestrator_status():
    print("\n" + "=" * 60)
    print("Test 5: Orchestrator Status Reporting")
    print("=" * 60)
    
    config = OrchestratorConfig(mode=TradingMode.SANDBOX)
    
    with patch('orchestrator.engine.PolymarketAPIClient') as mock_client_class:
        mock_client_class.return_value = AsyncMock()
        orchestrator = CopyCatOrchestrator(config)
        
        status = orchestrator.get_status()
        
        expected_keys = [
            'is_running', 'is_paused', 'mode', 'platform',
            'copied_traders', 'trades_executed', 'total_pnl',
            'total_pnl_pct', 'win_rate', 'sharpe_ratio',
            'max_drawdown', 'api_healthy', 'cycle_count',
        ]
        
        for key in expected_keys:
            assert key in status, f"Status missing key: {key}"
        
        print("All expected status keys present")
        
    return True


async def test_sandbox_real_time_data():
    print("\n" + "=" * 60)
    print("Test 6: Sandbox Real-Time Data Integration")
    print("=" * 60)
    
    from sandbox.data_providers import DataProviderConfig
    
    runner = SandboxRunner(
        config=SandboxConfig(initial_balance=10000.0),
        data_provider_config=DataProviderConfig(
            enable_crypto=True,
            enable_stocks=True,
            enable_prediction=True,
        ),
    )
    
    runner.enable_realtime_data(enable_crypto=True, enable_stocks=True, enable_prediction=True)
    
    market_data = runner._get_market_data("bitcoin")
    
    assert market_data is not None
    assert "current_price" in market_data
    
    print("Real-time data enabled")
    print("Market data retrieved")
    
    return True


async def run_all_tests():
    print("\n" + "=" * 60)
    print("SANDBOX-ORCHESTRATOR INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Orchestrator Sandbox Initialization", test_orchestrator_sandbox_initialization),
        ("Sandbox Runner Standalone", test_sandbox_runner_standalone),
        ("Orchestrator Copy Trade Execution", test_orchestrator_copy_trade_execution),
        ("Performance Metrics Integration", test_performance_metrics),
        ("Orchestrator Status Reporting", test_orchestrator_status),
        ("Sandbox Real-Time Data Integration", test_sandbox_real_time_data),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"Test failed: {e}")
            results.append((name, "FAIL"))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, status in results if status == "PASS")
    failed = sum(1 for _, status in results if status == "FAIL")
    
    for name, status in results:
        print(f"{'PASS' if status == 'PASS' else 'FAIL'} - {name}")
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
