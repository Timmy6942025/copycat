"""
Backtesting Validation for Micro Mode.

Validates micro mode configurations against historical performance.

Usage:
    from orchestrator.micro_validator import MicroModeValidator
    
    validator = MicroModeValidator()
    result = await validator.validate_config(balance=10.0, config=micro_config)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.config import (
    OrchestratorConfig, TradingMode, TraderSelectionConfig, CopyTradingConfig,
)
from orchestrator.config_micro import MicroModeLevel, create_micro_config
from orchestrator.mode_transition import TradingModeLevel
from sandbox.backtest import BacktestEngine, BacktestConfig, BacktestResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationCriteria:
    """Validation criteria for micro mode."""
    min_monthly_return: float = 0.10  # 10%
    max_drawdown: float = 0.20  # 20%
    min_win_rate: float = 0.50  # 50%
    min_sharpe_ratio: float = 0.5
    min_profit_factor: float = 1.0
    min_trades: int = 10
    simulation_days: int = 30


@dataclass
class ValidationResult:
    """Result of micro mode validation."""
    is_valid: bool
    mode: str
    balance: float
    criteria: Dict[str, Any]
    results: Dict[str, Any]
    passed_criteria: List[str]
    failed_criteria: List[str]
    comparison: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    backtest_result: Optional[BacktestResult] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModeComparison:
    """Comparison between different modes."""
    mode_a: str
    mode_b: str
    winner: str
    metrics_compared: Dict[str, Dict[str, float]]
    recommendation: str


class MicroModeValidator:
    """
    Validates micro mode configurations against historical data.
    
    Features:
    - Runs backtests with micro mode configurations
    - Validates against success criteria
    - Compares performance across modes
    - Provides recommendations
    """
    
    def __init__(
        self,
        criteria: Optional[ValidationCriteria] = None,
        backtest_config: Optional[BacktestConfig] = None,
    ):
        self.criteria = criteria or ValidationCriteria()
        self.backtest_config = backtest_config or BacktestConfig(
            starting_balance=100.0,
            output_path="./backtest_results"
        )
        self.backtest_engine = BacktestEngine(self.backtest_config)
        self._validation_history: List[ValidationResult] = []
    
    async def validate_config(
        self,
        balance: float,
        config: Dict[str, Any],
        mode: str = "nano",
    ) -> ValidationResult:
        """
        Validate a micro mode configuration.
        
        Args:
            balance: Starting balance
            config: Configuration dictionary
            mode: Mode name (nano, micro, mini)
            
        Returns:
            ValidationResult with pass/fail details
        """
        logger.info(f"Validating {mode} mode config for ${balance} balance")
        
        # Create orchestrator config from dict
        orchestrator_config = self._create_config(balance, config)
        
        # Run backtest
        backtest_result = await self._run_backtest(orchestrator_config, balance)
        
        # Validate results
        criteria_results = self._evaluate_criteria(backtest_result)
        
        passed = [k for k, v in criteria_results.items() if v]
        failed = [k for k, v in criteria_results.items() if not v]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(failed, backtest_result)
        
        result = ValidationResult(
            is_valid=len(failed) == 0,
            mode=mode,
            balance=balance,
            criteria=dict(criteria_results),
            results={
                "monthly_return": backtest_result.total_pnl_pct * 12 if backtest_result.total_pnl_pct else 0,
                "drawdown": getattr(backtest_result.metrics, 'max_drawdown', 0) if backtest_result.metrics else 0,
                "win_rate": getattr(backtest_result.metrics, 'win_rate', 0) if backtest_result.metrics else 0,
                "sharpe_ratio": getattr(backtest_result.metrics, 'sharpe_ratio', 0) if backtest_result.metrics else 0,
                "profit_factor": getattr(backtest_result.metrics, 'profit_factor', 0) if backtest_result.metrics else 0,
                "total_trades": len(backtest_result.trades),
                "ending_balance": backtest_result.ending_balance,
            },
            passed_criteria=passed,
            failed_criteria=failed,
            recommendations=recommendations,
            backtest_result=backtest_result,
        )
        
        self._validation_history.append(result)
        
        logger.info(
            f"Validation {'PASSED' if result.is_valid else 'FAILED'} for {mode} mode: "
            f"{len(passed)}/{len(criteria_results)} criteria passed"
        )
        
        return result
    
    def _create_config(
        self,
        balance: float,
        config: Dict[str, Any]
    ) -> OrchestratorConfig:
        """Create OrchestratorConfig from dictionary."""
        copy_trading_dict = config.get("copy_trading", {})
        trader_selection_dict = config.get("trader_selection", {})
        boost_dict = config.get("boost_mode", {})
        
        return OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            
            copy_trading=CopyTradingConfig(
                position_sizing_method=copy_trading_dict.get("position_sizing_method", "kelly"),
                position_size_pct=copy_trading_dict.get("position_size_pct", 0.50),
                kelly_fraction=copy_trading_dict.get("kelly_fraction", 0.50),
                max_position_size_pct=copy_trading_dict.get("max_position_size_pct", 0.55),
            ),
            
            trader_selection=TraderSelectionConfig(
                mode=trader_selection_dict.get("mode", "growth"),
                growth_min_total_pnl=trader_selection_dict.get("growth_min_total_pnl", 25.0),
                growth_min_growth_rate=trader_selection_dict.get("growth_min_growth_rate", 0.005),
                growth_max_drawdown=trader_selection_dict.get("growth_max_drawdown", 0.50),
            ),
            
            max_traders_to_copy=config.get("max_traders_to_copy", 5),
            trader_data_refresh_interval_seconds=config.get("trading_cycle", {}).get("refresh_interval_seconds", 10),
        )
    
    async def _run_backtest(
        self,
        config: OrchestratorConfig,
        balance: float
    ) -> BacktestResult:
        """Run backtest with configuration."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.criteria.simulation_days)
        
        try:
            result = await self.backtest_engine.run_backtest(
                start_date=start_date,
                end_date=end_date,
                trader_selection_config={
                    "min_reputation_score": 0.5,
                    "max_insider_score": 0.3,
                    "max_bot_score": 0.2,
                },
                copy_trading_config={
                    "position_sizing_method": config.copy_trading.position_sizing_method,
                    "position_size_pct": config.copy_trading.position_size_pct,
                    "kelly_fraction": config.copy_trading.kelly_fraction,
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                starting_balance=balance,
                ending_balance=balance,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                metrics=None,
                trades=[],
                report="",
            )
    
    def _evaluate_criteria(self, result: BacktestResult) -> Dict[str, bool]:
        """Evaluate backtest results against criteria."""
        if not result.metrics:
            return {
                "monthly_return": False,
                "drawdown": False,
                "win_rate": False,
                "sharpe_ratio": False,
                "profit_factor": False,
                "trades": False,
            }
        
        metrics = result.metrics
        
        # Calculate approximate monthly return
        days = (result.end_date - result.start_date).days or 1
        monthly_return = result.total_pnl_pct * (365 / days)
        
        return {
            "monthly_return": monthly_return >= self.criteria.min_monthly_return,
            "drawdown": getattr(metrics, 'max_drawdown', 1.0) <= self.criteria.max_drawdown,
            "win_rate": getattr(metrics, 'win_rate', 0) >= self.criteria.min_win_rate,
            "sharpe_ratio": getattr(metrics, 'sharpe_ratio', 0) >= self.criteria.min_sharpe_ratio,
            "profit_factor": getattr(metrics, 'profit_factor', 0) >= self.criteria.min_profit_factor,
            "trades": len(result.trades) >= self.criteria.min_trades,
        }
    
    def _generate_recommendations(
        self,
        failed_criteria: List[str],
        result: BacktestResult
    ) -> List[str]:
        """Generate recommendations based on failed criteria."""
        recommendations = []
        
        if "monthly_return" in failed_criteria:
            recommendations.append(
                "Consider increasing position sizes or selecting more aggressive traders"
            )
        
        if "drawdown" in failed_criteria:
            recommendations.append(
                "Reduce position sizes or enable hedging to limit drawdown"
            )
        
        if "win_rate" in failed_criteria:
            recommendations.append(
                "Filter for traders with higher win rates or longer track records"
            )
        
        if "sharpe_ratio" in failed_criteria:
            recommendations.append(
                "Look for traders with more consistent returns (lower volatility)"
            )
        
        if "trades" in failed_criteria:
            recommendations.append(
                "Extend simulation period or reduce trader qualification thresholds"
            )
        
        return recommendations
    
    async def compare_modes(
        self,
        balance: float,
        modes: List[str] = None
    ) -> List[ValidationResult]:
        """
        Compare performance across multiple modes.
        
        Args:
            balance: Starting balance
            modes: List of modes to compare (default: nano, micro, mini)
            
        Returns:
            List of ValidationResults for each mode
        """
        modes = modes or ["nano", "micro", "mini"]
        results = []
        
        for mode in modes:
            config = create_micro_config(
                initial_balance=balance,
                mode=TradingMode.SANDBOX,
                micro_mode=mode,
            )
            
            # Convert to dict
            config_dict = {
                "mode": mode,
                "copy_trading": {
                    "position_sizing_method": "kelly",
                    "position_size_pct": config.orchestrator_config.copy_trading.position_size_pct,
                    "kelly_fraction": config.orchestrator_config.copy_trading.kelly_fraction,
                },
                "trader_selection": {
                    "mode": "growth",
                    "growth_min_total_pnl": config.orchestrator_config.trader_selection.growth_min_total_pnl,
                    "growth_min_growth_rate": config.orchestrator_config.trader_selection.growth_min_growth_rate,
                    "growth_max_drawdown": config.orchestrator_config.trader_selection.growth_max_drawdown,
                },
                "max_traders_to_copy": config.orchestrator_config.max_traders_to_copy,
            }
            
            result = await self.validate_config(balance, config_dict, mode)
            results.append(result)
        
        return results
    
    def recommend_mode(
        self,
        validation_results: List[ValidationResult]
    ) -> Tuple[str, str]:
        """
        Recommend best mode based on validation results.
        
        Returns:
            Tuple of (recommended_mode, reason)
        """
        valid_results = [r for r in validation_results if r.is_valid]
        
        if not valid_results:
            return "nano", "No modes passed validation - defaulting to most conservative"
        
        # Score each valid result
        scores = []
        for result in valid_results:
            score = 0
            
            # Prefer higher returns
            monthly_return = result.results.get("monthly_return", 0)
            if monthly_return > 0.15:
                score += 3
            elif monthly_return > 0.10:
                score += 2
            else:
                score += 1
            
            # Prefer lower drawdown
            drawdown = result.results.get("drawdown", 1)
            if drawdown < 0.10:
                score += 2
            elif drawdown < 0.15:
                score += 1
            
            # Bonus for more trades (diversification)
            trades = result.results.get("total_trades", 0)
            if trades > 20:
                score += 1
            
            scores.append((result.mode, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[0][0], f"Highest score ({scores[0][1]}) among valid modes"
    
    def get_status(self) -> Dict[str, Any]:
        """Get validator status."""
        total = len(self._validation_history)
        passed = sum(1 for r in self._validation_history if r.is_valid)
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "last_validation": self._validation_history[-1].isoformat() if self._validation_history else None,
        }
    
    def get_validation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent validation history."""
        recent = self._validation_history[-limit:]
        return [
            {
                "mode": r.mode,
                "balance": r.balance,
                "is_valid": r.is_valid,
                "passed_criteria": r.passed_criteria,
                "failed_criteria": r.failed_criteria,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in recent
        ]


# =============================================================================
# Factory Functions
# =============================================================================

def create_validation_criteria(
    min_monthly_return: float = 0.10,
    max_drawdown: float = 0.20,
    min_win_rate: float = 0.50,
    min_sharpe_ratio: float = 0.5,
    simulation_days: int = 30,
) -> ValidationCriteria:
    """Factory function to create ValidationCriteria."""
    return ValidationCriteria(
        min_monthly_return=min_monthly_return,
        max_drawdown=max_drawdown,
        min_win_rate=min_win_rate,
        min_sharpe_ratio=min_sharpe_ratio,
        simulation_days=simulation_days,
    )


def create_validator(
    criteria: ValidationCriteria = None,
) -> MicroModeValidator:
    """Factory function to create MicroModeValidator."""
    return MicroModeValidator(criteria=criteria)


if __name__ == "__main__":
    print("=" * 60)
    print("MICRO MODE VALIDATOR - Example Usage")
    print("=" * 60)
    
    validator = create_validator()
    
    status = validator.get_status()
    print(f"\nValidator Status:")
    print(f"  Total Validations: {status['total_validations']}")
    print(f"  Pass Rate: {status['pass_rate']:.1f}%")
    
    print("\n" + "=" * 60)
    print("Example: Compare modes for $10 balance")
    print("-" * 60)
    
    async def run_comparison():
        results = await validator.compare_modes(balance=10.0)
        
        for result in results:
            print(f"\n{result.mode.upper()} Mode:")
            print(f"  Valid: {result.is_valid}")
            print(f"  Monthly Return: {result.results.get('monthly_return', 0)*100:.1f}%")
            print(f"  Drawdown: {result.results.get('drawdown', 0)*100:.1f}%")
            print(f"  Win Rate: {result.results.get('win_rate', 0)*100:.1f}%")
            print(f"  Trades: {result.results.get('total_trades', 0)}")
            
            if result.failed_criteria:
                print(f"  Failed: {result.failed_criteria}")
        
        # Get recommendation
        mode, reason = validator.recommend_mode(results)
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION: {mode.upper()}")
        print(f"Reason: {reason}")
        
        return results
    
    results = asyncio.run(run_comparison())
    
    print("\n" + "=" * 60)
