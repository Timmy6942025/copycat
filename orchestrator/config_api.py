"""
API Endpoint for Auto-Select Mode.

Provides REST endpoint to recommend optimal trading mode based on balance and risk tolerance.

Usage:
    POST /api/config/auto-select
    {
        "balance": 25.0,
        "risk_tolerance": "medium"  # low, medium, high
    }
    
    Response:
    {
        "success": true,
        "recommended_mode": "mini",
        "config": {...},
        "expected_monthly_return": "20-30%",
        "risk_level": "medium_high"
    }
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, request, jsonify

from orchestrator.config import (
    OrchestratorConfig, TradingMode, CopyTradingConfig, TraderSelectionConfig,
    BoostModeConfig, BotFilterOrchestratorConfig, SandboxConfigOrchestrator,
)
from orchestrator.config_micro import MicroModeLevel
from orchestrator.mode_transition import TradingModeLevel, create_default_thresholds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
config_bp = Blueprint('config', __name__, url_prefix='/api/config')


@dataclass
class RiskTolerance(Enum):
    """Risk tolerance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModeRecommendation:
    """Result of mode recommendation."""
    success: bool
    recommended_mode: str
    position_size_pct: float
    kelly_fraction: float
    max_drawdown_threshold: float
    expected_monthly_return: str
    risk_level: str
    config: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class ConfigValidator:
    """Validates and generates configuration recommendations."""
    
    def __init__(self):
        self.thresholds = create_default_thresholds()
    
    def recommend_mode(
        self,
        balance: float,
        risk_tolerance: str = "medium",
        current_pnl: float = 0.0,
    ) -> ModeRecommendation:
        """
        Recommend optimal mode based on balance and risk tolerance.
        
        Args:
            balance: Current account balance
            risk_tolerance: "low", "medium", or "high"
            current_pnl: Current profit/loss (optional)
            
        Returns:
            ModeRecommendation with details
        """
        warnings = []
        
        # Validate inputs
        if balance <= 0:
            return ModeRecommendation(
                success=False,
                recommended_mode="nano",
                position_size_pct=0.0,
                kelly_fraction=0.0,
                max_drawdown_threshold=0.0,
                expected_monthly_return="N/A",
                risk_level="N/A",
                config={},
                error="Balance must be positive"
            )
        
        # Adjust risk tolerance based on current P&L
        effective_risk = risk_tolerance
        if current_pnl < 0:
            loss_pct = abs(current_pnl) / balance if balance > 0 else 0
            if loss_pct > 0.10:
                effective_risk = "low"
                warnings.append(f"Reducing risk tolerance to 'low' due to {loss_pct:.1%} drawdown")
        
        # Find matching threshold
        recommendation = None
        for threshold in self.thresholds:
            if threshold.min_balance <= balance < threshold.max_balance:
                recommendation = threshold
                break
        
        if not recommendation:
            # Default to CONSERVATIVE for very large balances
            recommendation = self.thresholds[-1]
        
        # Adjust based on risk tolerance
        if effective_risk == "low":
            adjusted_position = recommendation.position_size_pct * 0.7
            adjusted_kelly = recommendation.kelly_fraction * 0.7
        elif effective_risk == "high":
            adjusted_position = min(recommendation.position_size_pct * 1.2, 0.80)
            adjusted_kelly = min(recommendation.kelly_fraction * 1.2, 0.75)
        else:
            adjusted_position = recommendation.position_size_pct
            adjusted_kelly = recommendation.kelly_fraction
        
        # Calculate expected returns and risk level
        expected_return = self._estimate_monthly_return(adjusted_position)
        risk_level = self._get_risk_level(adjusted_position)
        
        # Generate config
        config = self._generate_config(balance, recommendation, adjusted_position, adjusted_kelly)
        
        return ModeRecommendation(
            success=True,
            recommended_mode=recommendation.target_mode.value,
            position_size_pct=adjusted_position,
            kelly_fraction=adjusted_kelly,
            max_drawdown_threshold=recommendation.max_drawdown_threshold,
            expected_monthly_return=expected_return,
            risk_level=risk_level,
            config=config,
            warnings=warnings,
        )
    
    def _estimate_monthly_return(self, position_pct: float) -> str:
        """Estimate monthly return based on position size."""
        if position_pct >= 0.70:
            return "25-50%"
        elif position_pct >= 0.50:
            return "15-30%"
        elif position_pct >= 0.35:
            return "10-20%"
        elif position_pct >= 0.25:
            return "8-15%"
        else:
            return "5-10%"
    
    def _get_risk_level(self, position_pct: float) -> str:
        """Get risk level based on position size."""
        if position_pct >= 0.70:
            return "very_high"
        elif position_pct >= 0.50:
            return "high"
        elif position_pct >= 0.35:
            return "medium_high"
        elif position_pct >= 0.25:
            return "medium"
        else:
            return "low"
    
    def _generate_config(
        self,
        balance: float,
        threshold: Any,
        position_pct: float,
        kelly_fraction: float,
    ) -> Dict[str, Any]:
        """Generate full configuration dictionary."""
        # Trader selection based on mode
        if threshold.target_mode in [TradingModeLevel.NANO, TradingModeLevel.MICRO]:
            min_pnl = 10.0
            min_growth_rate = 0.002
            max_drawdown = 0.75
        elif threshold.target_mode in [TradingModeLevel.MINI, TradingModeLevel.BALANCED]:
            min_pnl = 25.0
            min_growth_rate = 0.005
            max_drawdown = 0.50
        else:
            min_pnl = 100.0
            min_growth_rate = 0.010
            max_drawdown = 0.30
        
        return {
            "mode": threshold.target_mode.value,
            "copy_trading": {
                "position_sizing_method": "kelly",
                "position_size_pct": round(position_pct, 2),
                "kelly_fraction": round(kelly_fraction, 2),
                "max_position_size_pct": round(min(position_pct * 1.1, 0.80), 2),
                "max_total_exposure_pct": 0.95 if position_pct >= 0.50 else 0.85,
            },
            "trader_selection": {
                "mode": "growth",
                "growth_min_total_pnl": min_pnl,
                "growth_min_growth_rate": min_growth_rate,
                "growth_max_drawdown": max_drawdown,
                "growth_min_active_days": 1,
            },
            "boost_mode": {
                "enabled": threshold.enable_boost_mode,
                "position_multiplier": 3.0 if threshold.enable_boost_mode else 1.0,
                "max_boost_position_pct": 0.75 if threshold.enable_boost_mode else 0.40,
            },
            "bot_filter": {
                "min_hft_score_to_exclude": 0.9,
                "min_arbitrage_score_to_exclude": 0.9,
            },
            "trading_cycle": {
                "refresh_interval_seconds": 10,
                "max_traders_to_copy": 5,
                "max_traders_to_analyze_per_cycle": 200,
            },
            "sandbox": {
                "initial_balance": balance,
                "simulate_slippage": True,
                "simulate_fees": True,
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration dictionary."""
        errors = []
        warnings = []
        
        # Validate position size
        position_pct = config.get("position_size_pct", 0)
        if position_pct <= 0:
            errors.append("Position size must be positive")
        elif position_pct > 1.0:
            errors.append("Position size cannot exceed 100%")
        elif position_pct > 0.80:
            warnings.append("Position size above 80% is very aggressive")
        
        # Validate Kelly fraction
        kelly = config.get("kelly_fraction", 0)
        if kelly <= 0:
            errors.append("Kelly fraction must be positive")
        elif kelly > 1.0:
            errors.append("Kelly fraction cannot exceed 1.0")
        elif kelly > 0.50:
            warnings.append("Kelly fraction above 0.5 is aggressive")
        
        # Validate balance
        balance = config.get("initial_balance", 0)
        if balance <= 0:
            errors.append("Initial balance must be positive")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


# Create validator instance
validator = ConfigValidator()


# =============================================================================
# API Routes
# =============================================================================

@config_bp.route('/auto-select', methods=['POST'])
def api_auto_select_mode():
    """
    Auto-select optimal trading mode based on balance and risk tolerance.
    
    Request body:
    {
        "balance": 25.0,           # Required: current balance
        "risk_tolerance": "medium",  # Optional: "low", "medium", "high"
        "current_pnl": 5.0          # Optional: current profit/loss
    }
    
    Response:
    {
        "success": true,
        "recommended_mode": "mini",
        "position_size_pct": 0.50,
        "kelly_fraction": 0.50,
        "max_drawdown_threshold": 0.20,
        "expected_monthly_return": "20-30%",
        "risk_level": "medium_high",
        "config": {...},
        "warnings": []
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({
            "success": False,
            "error": "Request body must be JSON"
        }), 400
    
    # Extract parameters
    balance = data.get('balance')
    if balance is None:
        return jsonify({
            "success": False,
            "error": "Missing required field: balance"
        }), 400
    
    try:
        balance = float(balance)
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "Balance must be a number"
        }), 400
    
    risk_tolerance = data.get('risk_tolerance', 'medium')
    if risk_tolerance not in ['low', 'medium', 'high']:
        return jsonify({
            "success": False,
            "error": "risk_tolerance must be 'low', 'medium', or 'high'"
        }), 400
    
    current_pnl = float(data.get('current_pnl', 0))
    
    # Get recommendation
    recommendation = validator.recommend_mode(
        balance=balance,
        risk_tolerance=risk_tolerance,
        current_pnl=current_pnl,
    )
    
    if not recommendation.success:
        return jsonify({
            "success": False,
            "error": recommendation.error
        }), 400
    
    return jsonify({
        "success": True,
        "recommended_mode": recommendation.recommended_mode,
        "position_size_pct": recommendation.position_size_pct,
        "kelly_fraction": recommendation.kelly_fraction,
        "max_drawdown_threshold": recommendation.max_drawdown_threshold,
        "expected_monthly_return": recommendation.expected_monthly_return,
        "risk_level": recommendation.risk_level,
        "config": recommendation.config,
        "warnings": recommendation.warnings,
    })


@config_bp.route('/validate', methods=['POST'])
def api_validate_config():
    """
    Validate a configuration before applying.
    
    Request body:
    {
        "config": {
            "position_size_pct": 0.50,
            "kelly_fraction": 0.50,
            "initial_balance": 25.0
        }
    }
    
    Response:
    {
        "valid": true,
        "errors": [],
        "warnings": []
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({
            "success": False,
            "error": "Request body must be JSON"
        }), 400
    
    config = data.get('config')
    if not config:
        return jsonify({
            "success": False,
            "error": "Missing required field: config"
        }), 400
    
    result = validator.validate_config(config)
    
    return jsonify({
        "success": True,
        "valid": result["valid"],
        "errors": result["errors"],
        "warnings": result["warnings"],
    })


@config_bp.route('/milestones', methods=['GET'])
def api_get_milestones():
    """
    Get list of balance milestones.
    
    Response:
    {
        "success": true,
        "milestones": [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    }
    """
    return jsonify({
        "success": True,
        "milestones": [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    })


@config_bp.route('/modes', methods=['GET'])
def api_get_modes():
    """
    Get list of available modes with descriptions.
    
    Response:
    {
        "success": true,
        "modes": [
            {"id": "nano", "name": "NANO", "balance_range": "$0-$15", ...},
            ...
        ]
    }
    """
    modes = []
    
    for threshold in create_default_thresholds():
        mode_info = {
            "id": threshold.target_mode.value,
            "name": threshold.target_mode.value.upper(),
            "balance_range": f"${threshold.min_balance:,.0f}-${threshold.max_balance:,.0f}" 
                           if threshold.max_balance != float('inf') 
                           else f"${threshold.min_balance:,.0f}+",
            "position_size_pct": threshold.position_size_pct,
            "kelly_fraction": threshold.kelly_fraction,
            "max_drawdown_threshold": threshold.max_drawdown_threshold,
            "enable_boost_mode": threshold.enable_boost_mode,
            "enable_hedging": threshold.enable_hedging,
            "enable_momentum_filter": threshold.enable_momentum_filter,
        }
        
        # Add expected performance
        if threshold.position_size_pct >= 0.70:
            mode_info["expected_monthly_return"] = "25-50%"
            mode_info["risk_level"] = "very_high"
        elif threshold.position_size_pct >= 0.50:
            mode_info["expected_monthly_return"] = "15-30%"
            mode_info["risk_level"] = "high"
        elif threshold.position_size_pct >= 0.35:
            mode_info["expected_monthly_return"] = "10-20%"
            mode_info["risk_level"] = "medium_high"
        elif threshold.position_size_pct >= 0.25:
            mode_info["expected_monthly_return"] = "8-15%"
            mode_info["risk_level"] = "medium"
        else:
            mode_info["expected_monthly_return"] = "5-10%"
            mode_info["risk_level"] = "low"
        
        modes.append(mode_info)
    
    return jsonify({
        "success": True,
        "modes": modes
    })


# =============================================================================
# Factory and Utility Functions
# =============================================================================

def create_orchestrator_from_recommendation(
    recommendation: ModeRecommendation,
    mode: str = "sandbox",
) -> OrchestratorConfig:
    """
    Create an OrchestratorConfig from a mode recommendation.
    
    Args:
        recommendation: ModeRecommendation from auto-select
        mode: "sandbox" or "live"
        
    Returns:
        Configured OrchestratorConfig
    """
    config_dict = recommendation.config
    
    # Build orchestrator config
    orchestrator_config = OrchestratorConfig(
        mode=TradingMode.SANDBOX if mode == "sandbox" else TradingMode.LIVE,
        platform=config_dict.get("platform", "polymarket"),
        
        # Copy trading
        copy_trading=CopyTradingConfig(
            position_sizing_method=config_dict.get("copy_trading", {}).get("position_sizing_method", "kelly"),
            position_size_pct=config_dict.get("copy_trading", {}).get("position_size_pct", 0.50),
            kelly_fraction=config_dict.get("copy_trading", {}).get("kelly_fraction", 0.50),
            max_position_size_pct=config_dict.get("copy_trading", {}).get("max_position_size_pct", 0.55),
            max_total_exposure_pct=config_dict.get("copy_trading", {}).get("max_total_exposure_pct", 0.85),
        ),
        
        # Trader selection
        trader_selection=TraderSelectionConfig(
            mode=config_dict.get("trader_selection", {}).get("mode", "growth"),
            growth_min_total_pnl=config_dict.get("trader_selection", {}).get("growth_min_total_pnl", 25.0),
            growth_min_growth_rate=config_dict.get("trader_selection", {}).get("growth_min_growth_rate", 0.005),
            growth_max_drawdown=config_dict.get("trader_selection", {}).get("growth_max_drawdown", 0.50),
            growth_min_active_days=1,
        ),
        
        # Boost mode
        boost_mode=BoostModeConfig(
            enabled=config_dict.get("boost_mode", {}).get("enabled", True),
            position_multiplier=config_dict.get("boost_mode", {}).get("position_multiplier", 3.0),
            max_boost_position_pct=config_dict.get("boost_mode", {}).get("max_boost_position_pct", 0.75),
        ),
        
        # Bot filter
        bot_filter=BotFilterOrchestratorConfig(
            min_hft_score_to_exclude=config_dict.get("bot_filter", {}).get("min_hft_score_to_exclude", 0.9),
            min_arbitrage_score_to_exclude=config_dict.get("bot_filter", {}).get("min_arbitrage_score_to_exclude", 0.9),
        ),
        
        # Sandbox
        sandbox=SandboxConfigOrchestrator(
            initial_balance=config_dict.get("sandbox", {}).get("initial_balance", 100.0),
        ),
        
        # Constraints
        max_traders_to_copy=config_dict.get("trading_cycle", {}).get("max_traders_to_copy", 5),
        max_traders_to_analyze_per_cycle=config_dict.get("trading_cycle", {}).get("max_traders_to_analyze_per_cycle", 200),
        trader_data_refresh_interval_seconds=config_dict.get("trading_cycle", {}).get("refresh_interval_seconds", 10),
    )
    
    return orchestrator_config


def register_blueprint(app):
    """Register config blueprint with Flask app."""
    app.register_blueprint(config_bp)


if __name__ == "__main__":
    # Example usage
    import sys
    
    print("=" * 60)
    print("Config Auto-Select API - Example Outputs")
    print("=" * 60)
    
    # Test different balances
    test_cases = [
        (10, "medium", 0),
        (25, "medium", 5),
        (50, "low", -2),
        (100, "high", 20),
        (500, "medium", 100),
        (2000, "low", 500),
    ]
    
    for balance, risk, pnl in test_cases:
        recommendation = validator.recommend_mode(balance, risk, pnl)
        
        print(f"\nBalance: ${balance}, Risk: {risk}, P&L: ${pnl:+}")
        print(f"  Mode: {recommendation.recommended_mode}")
        print(f"  Position: {recommendation.position_size_pct:.0%}")
        print(f"  Kelly: {recommendation.kelly_fraction:.2f}")
        print(f"  Expected: {recommendation.expected_monthly_return}")
        print(f"  Risk: {recommendation.risk_level}")
        if recommendation.warnings:
            print(f"  Warnings: {recommendation.warnings}")
    
    print("\n" + "=" * 60)
    print("To use with Flask, add to dashboard/app.py:")
    print("  from orchestrator.config_api import register_blueprint")
    print("  register_blueprint(app)")
    print("=" * 60)
