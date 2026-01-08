"""
Adaptive Scaling System for CopyCat.

Automatically adjusts trader limits based on:
1. Portfolio growth (more $$ = more diversification)
2. Drawdown protection (reduce exposure on losses)
3. Trade frequency (scan more when quiet)

Usage:
    from orchestrator.config_adaptive import AdaptiveScaler, AdaptiveConfig
"""

import logging
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.config import (
    TradingMode, MarketPlatform, TraderSelectionConfig,
    BotFilterOrchestratorConfig, CopyTradingConfig, SandboxConfigOrchestrator,
    OrchestratorConfig, SelectionMode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class AdaptiveLimits:
    max_traders_to_copy: int = 10
    max_traders_to_analyze: int = 100
    position_size_pct: float = 0.05
    scaling_reason: str = "baseline"
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AdaptiveConfig:
    base_max_traders: int = 10
    base_analyze_count: int = 100
    base_position_pct: float = 0.05
    strategy: ScalingStrategy = ScalingStrategy.BALANCED
    max_traders_cap: int = 30
    drawdown_tier1_pct: float = 0.10
    drawdown_tier2_pct: float = 0.20
    drawdown_tier3_pct: float = 0.30
    min_trades_per_day: int = 5
    max_trades_per_day: int = 50
    scan_boost_multiplier: float = 2.0
    min_change_interval_seconds: int = 600


class AdaptiveScaler:
    def __init__(self, config: AdaptiveConfig, initial_balance: float = 100.0):
        self.config = config
        self.limits = AdaptiveLimits(
            max_traders_to_copy=config.base_max_traders,
            max_traders_to_analyze=config.base_analyze_count,
            position_size_pct=config.base_position_pct,
        )
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_drawdown = 0.0
        self.trades_last_24h = 0
        self.total_pnl = 0.0
        self.is_profitable = False
        self.last_scaled = datetime.utcnow()
        logger.info(f"AdaptiveScaler initialized with {config.strategy.value} strategy")

    def update_state(self, current_balance: float, trades_last_24h: int, total_pnl: float):
        self.current_balance = current_balance
        self.trades_last_24h = trades_last_24h
        self.total_pnl = total_pnl
        self.is_profitable = total_pnl > 0
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance

    def _growth_multiplier(self) -> float:
        ratio = self.current_balance / self.initial_balance
        if self.config.strategy == ScalingStrategy.CONSERVATIVE:
            return 1.0 + min((ratio - 1) * 0.5, 0.3)
        elif self.config.strategy == ScalingStrategy.BALANCED:
            return 1.0 + min((ratio - 1) * 1.0, 0.5)
        else:
            return 1.0 + min((ratio - 1) * 2.0, 1.0)

    def _drawdown_multiplier(self) -> float:
        dd = self.current_drawdown
        if dd < self.config.drawdown_tier1_pct:
            return 1.0
        elif dd < self.config.drawdown_tier2_pct:
            return 0.85
        elif dd < self.config.drawdown_tier3_pct:
            return 0.65
        else:
            return 0.45

    def _frequency_multiplier(self) -> float:
        trades = self.trades_last_24h
        if trades < self.config.min_trades_per_day:
            boost = self.config.scan_boost_multiplier * (1 - trades / self.config.min_trades_per_day)
            return 1.0 + boost
        elif trades > self.config.max_trades_per_day:
            return 0.8
        return 1.0

    def _performance_multiplier(self) -> float:
        if self.config.strategy == ScalingStrategy.CONSERVATIVE:
            return 1.0
        if self.is_profitable:
            pnl_ratio = min(self.total_pnl / self.initial_balance, 0.5)
            return 1.0 + pnl_ratio * 0.3
        else:
            loss_ratio = min(-self.total_pnl / self.initial_balance, 0.3)
            return 1.0 - loss_ratio * 0.3

    def get_effective_limits(self, force: bool = False) -> AdaptiveLimits:
        now = datetime.utcnow()
        if not force and (now - self.last_scaled).total_seconds() < self.config.min_change_interval_seconds:
            return self.limits

        gm = self._growth_multiplier()
        dm = self._drawdown_multiplier()
        fm = self._frequency_multiplier()
        pm = self._performance_multiplier()
        combined = gm * dm * fm * pm

        new_traders = int(self.config.base_max_traders * combined)
        new_traders = max(3, min(new_traders, self.config.max_traders_cap))
        new_analyze = int(self.config.base_analyze_count * combined * fm)
        new_analyze = max(50, min(new_analyze, 500))
        new_position = self.config.base_position_pct * dm
        new_position = max(0.02, min(new_position, 0.15))

        if dm < 0.9:
            reason = f"drawdown({self.current_drawdown:.0%})"
        elif gm > 1.1:
            reason = f"growth({(gm-1)*100:.0f}%+)"
        elif fm > 1.2:
            reason = "low_activity"
        else:
            reason = "baseline"

        if new_traders != self.limits.max_traders_to_copy or new_analyze != self.limits.max_traders_to_analyze or abs(new_position - self.limits.position_size_pct) > 0.005:
            self.limits = AdaptiveLimits(
                max_traders_to_copy=new_traders,
                max_traders_to_analyze=new_analyze,
                position_size_pct=new_position,
                scaling_reason=reason,
                last_update=now,
            )
            self.last_scaled = now

        return self.limits

    def get_status(self) -> Dict[str, Any]:
        """Get adaptive scaler status."""
        strategy = self.config.strategy
        strategy_value = strategy.value if hasattr(strategy, 'value') else strategy
        return {
            "strategy": strategy_value,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown,
            "is_profitable": self.is_profitable,
            "limits": {
                "max_traders_to_copy": self.limits.max_traders_to_copy,
                "max_traders_to_analyze": self.limits.max_traders_to_analyze,
                "position_size_pct": self.limits.position_size_pct,
                "scaling_reason": self.limits.scaling_reason,
            },
        }


def create_adaptive_config(
    initial_balance: float = 100.0,
    mode: TradingMode = TradingMode.SANDBOX,
    strategy: ScalingStrategy = ScalingStrategy.BALANCED,
) -> OrchestratorConfig:
    trader_selection = TraderSelectionConfig(
        mode=SelectionMode.GROWTH,
        growth_min_total_pnl=100.0,
        growth_max_drawdown=0.50,
        growth_min_active_days=3,
    )
    copy_trading = CopyTradingConfig(
        position_sizing_method="scaled",
        position_size_pct=0.08,
        base_position_size=25.0,
    )
    return OrchestratorConfig(
        mode=mode,
        platform=MarketPlatform.POLYMARKET,
        trader_selection=trader_selection,
        copy_trading=copy_trading,
        sandbox=SandboxConfigOrchestrator(initial_balance=initial_balance),
        bot_filter=BotFilterOrchestratorConfig(),
        max_traders_to_copy=10,
        max_traders_to_analyze_per_cycle=100,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE SCALING SYSTEM")
    print("=" * 60)

    config = AdaptiveConfig(base_max_traders=10, strategy=ScalingStrategy.BALANCED)
    scaler = AdaptiveScaler(config, initial_balance=100.0)

    scenarios = [
        ("Starting ($100)", 100.0, 5, 0, 0.0),
        ("Big profit ($200)", 200.0, 12, 100, 0.0),
        ("Small drawdown (10%)", 180.0, 10, 80, 0.1),
        ("Big drawdown (20%)", 160.0, 5, 60, 0.2),
        ("Huge drawdown (30%)", 140.0, 2, 40, 0.3),
        ("Recovery", 180.0, 15, 80, 0.1),
    ]

    print("\nScenario Analysis:")
    print("-" * 60)

    for name, balance, trades, pnl, drawdown in scenarios:
        scaler.update_state(balance, trades, pnl)
        scaler.current_drawdown = drawdown
        limits = scaler.get_effective_limits(force=True)

        print(f"\n{name}:")
        print(f"  Balance: ${balance:,.2f} | Drawdown: {drawdown:.0%} | Trades: {trades}/day")
        print(f"  → Traders: {limits.max_traders_to_copy}")
        print(f"  → Analyze: {limits.max_traders_to_analyze}")
        print(f"  → Position: {limits.position_size_pct:.1%}")
        print(f"  → Reason: {limits.scaling_reason}")

    print("\n" + "=" * 60)
    print("ADAPTIVE BEHAVIOR:")
    print("  • Scales UP when portfolio grows")
    print("  • Scales DOWN on drawdown (risk protection)")
    print("  • Boosts scanning when few trades")
    print("  • Reduces scanning when busy")
    print("=" * 60)
