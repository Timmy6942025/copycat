"""
CopyCat Web Dashboard.
Flask-based web interface for monitoring and controlling the copy trading bot.
"""

import asyncio
import logging
import os
import secrets
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect,
    url_for,
    session,
    abort,
)
from threading import Thread

from orchestrator.engine import CopyCatOrchestrator
from orchestrator.config import OrchestratorConfig, TradingMode, MarketPlatform


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================


def get_secret_key() -> str:
    """Get or generate dashboard secret key from environment or generate secure default."""
    secret_key = os.environ.get("DASHBOARD_SECRET_KEY")
    if secret_key:
        return secret_key
    # Generate secure random key for this session if not set
    return secrets.token_hex(32)


def get_api_keys() -> dict:
    """Load and validate API keys from environment variables."""
    return {
        "polymarket": os.environ.get("POLYMARKET_API_KEY", ""),
        "kalshi": os.environ.get("KALSHI_API_KEY", ""),
    }


def get_webhook_urls() -> dict:
    """Load webhook URLs from environment variables."""
    return {
        "discord": os.environ.get("DISCORD_WEBHOOK_URL", ""),
        "slack": os.environ.get("SLACK_WEBHOOK_URL", ""),
    }


# Session configuration
SESSION_EXPIRY_HOURS = int(os.environ.get("SESSION_EXPIRY_HOURS", "24"))

# Rate limiting (simple in-memory implementation)
rate_limit_storage = {}


def rate_limit(max_requests: int = 100, per_seconds: int = 3600):
    """Simple rate limiting decorator for API endpoints."""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr or "unknown"
            current_time = datetime.utcnow()

            key = f"{client_ip}:{request.endpoint}"

            if key not in rate_limit_storage:
                rate_limit_storage[key] = []

            # Clean old requests
            rate_limit_storage[key] = [
                t
                for t in rate_limit_storage[key]
                if (current_time - t).total_seconds() < per_seconds
            ]

            if len(rate_limit_storage[key]) >= max_requests:
                return jsonify(
                    {
                        "success": False,
                        "error": "Rate limit exceeded. Please try again later.",
                    }
                ), 429

            rate_limit_storage[key].append(current_time)
            return f(*args, **kwargs)

        return decorated_function

    return decorator


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_trader_address(address: str) -> tuple[bool, str]:
    """Validate trader address format."""
    if not address:
        return False, "Trader address is required"

    address = address.strip()

    if len(address) != 42:
        return False, "Invalid address length (expected 42 characters)"

    if not address.startswith("0x"):
        return False, "Address must start with 0x"

    # Check hex characters
    try:
        int(address[2:], 16)
    except ValueError:
        return False, "Address contains invalid hex characters"

    return True, ""


def validate_numeric_param(
    value: float,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> tuple[bool, str]:
    """Validate numeric parameter."""
    if value is None:
        return False, f"{name} is required"

    if not isinstance(value, (int, float)):
        return False, f"{name} must be a number"

    if min_val is not None and value < min_val:
        return False, f"{name} must be >= {min_val}"

    if max_val is not None and value > max_val:
        return False, f"{name} must be <= {max_val}"

    return True, ""


# =============================================================================
# FLASK APP SETUP
# =============================================================================

# Create Flask app with secure configuration
app = Flask(__name__)
app.secret_key = get_secret_key()
app.permanent_session_lifetime = timedelta(hours=SESSION_EXPIRY_HOURS)


# Security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Disable caching for sensitive endpoints
    if request.endpoint and "api_" in request.endpoint:
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
    return response


# Global orchestrator instance
orchestrator = None
orchestrator_thread = None


def get_orchestrator() -> CopyCatOrchestrator:
    """Get or create the global orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        config = OrchestratorConfig(
            mode=TradingMode.SANDBOX,
            platform=MarketPlatform.POLYMARKET,
        )
        orchestrator = CopyCatOrchestrator(config)
    return orchestrator


def run_orchestrator_async(orchestrator: CopyCatOrchestrator):
    """Run the orchestrator in an async event loop."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(orchestrator.start())
        loop.close()
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")


# =============================================================================
# Routes
# =============================================================================


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/status")
@rate_limit(max_requests=60, per_seconds=60)
def api_status():
    """Get current system status."""
    try:
        orch = get_orchestrator()
        status = orch.get_status()
        return jsonify(
            {
                "success": True,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/start", methods=["POST"])
@rate_limit(max_requests=10, per_seconds=60)
def api_start():
    """Start the orchestrator."""
    try:
        orch = get_orchestrator()

        if orch.state.is_running:
            return jsonify(
                {
                    "success": False,
                    "message": "Orchestrator is already running",
                }
            )

        # Start in background thread
        global orchestrator_thread
        orchestrator_thread = Thread(
            target=run_orchestrator_async, args=(orch,), daemon=True
        )
        orchestrator_thread.start()

        return jsonify(
            {
                "success": True,
                "message": "Orchestrator started",
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop the orchestrator."""
    try:
        orch = get_orchestrator()

        if not orch.state.is_running:
            return jsonify(
                {
                    "success": False,
                    "message": "Orchestrator is not running",
                }
            )

        # Stop the orchestrator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(orch.stop())
        loop.close()

        return jsonify(
            {
                "success": True,
                "message": "Orchestrator stopped",
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/pause", methods=["POST"])
def api_pause():
    """Pause the orchestrator."""
    try:
        orch = get_orchestrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orch.pause())
        loop.close()

        return jsonify(
            {
                "success": result.success,
                "message": result.message,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/resume", methods=["POST"])
def api_resume():
    """Resume the orchestrator."""
    try:
        orch = get_orchestrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orch.resume())
        loop.close()

        return jsonify(
            {
                "success": result.success,
                "message": result.message,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/traders")
def api_traders():
    """Get list of copied traders."""
    try:
        orch = get_orchestrator()
        traders = []

        for address, config in orch.state.copied_traders.items():
            traders.append(
                {
                    "address": address,
                    "enabled": config.enabled,
                    "position_size": config.base_position_size,
                }
            )

        return jsonify(
            {
                "success": True,
                "traders": traders,
                "count": len(traders),
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/traders/add", methods=["POST"])
@rate_limit(max_requests=20, per_seconds=60)
def api_add_trader():
    """Add a new trader to copy."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(
                {
                    "success": False,
                    "error": "Invalid JSON data",
                }
            )

        address = data.get("address", "").strip()

        # Validate trader address format
        is_valid, error_msg = validate_trader_address(address)
        if not is_valid:
            return jsonify(
                {
                    "success": False,
                    "error": error_msg,
                }
            )

        orch = get_orchestrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orch.add_trader(address))
        loop.close()

        return jsonify(
            {
                "success": result.success,
                "message": result.message,
                "details": result.details,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/traders/remove", methods=["POST"])
@rate_limit(max_requests=20, per_seconds=60)
def api_remove_trader():
    """Remove a trader from copy list."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(
                {
                    "success": False,
                    "error": "Invalid JSON data",
                }
            )

        address = data.get("address", "").strip()

        # Validate trader address format
        is_valid, error_msg = validate_trader_address(address)
        if not is_valid:
            return jsonify(
                {
                    "success": False,
                    "error": error_msg,
                }
            )

        orch = get_orchestrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orch.remove_trader(address))
        loop.close()

        return jsonify(
            {
                "success": result.success,
                "message": result.message,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/performance")
def api_performance():
    """Get performance metrics."""
    try:
        orch = get_orchestrator()

        metrics = {
            "total_pnl": orch.state.total_pnl,
            "total_pnl_pct": orch.state.total_pnl_pct,
            "win_rate": orch.state.win_rate,
            "sharpe_ratio": orch.state.sharpe_ratio,
            "max_drawdown": orch.state.max_drawdown,
            "trades_executed": orch.state.trades_executed,
        }

        return jsonify(
            {
                "success": True,
                "metrics": metrics,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


@app.route("/api/health")
def api_health():
    """Get health check status."""
    try:
        orch = get_orchestrator()

        health = {
            "api_healthy": orch.state.api_healthy,
            "circuit_breaker_open": orch.state.circuit_breaker_open,
            "consecutive_failures": orch.state.consecutive_failures,
            "is_running": orch.state.is_running,
            "is_paused": orch.state.is_paused,
            "cycle_count": orch.state.cycle_count,
        }

        return jsonify(
            {
                "success": True,
                "health": health,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        )


# =============================================================================
# Error Handlers
# =============================================================================


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error="Internal server error"), 500


# =============================================================================
# Main Entry Point
# =============================================================================


def create_app():
    """Create and configure the Flask app."""
    return app


def run_server(host="0.0.0.0", port=5000, debug=False):
    """Run the dashboard server."""
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_server(debug=True)
