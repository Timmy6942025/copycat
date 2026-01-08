"""
CopyCat Web Dashboard.
Flask-based web interface for monitoring and controlling the copy trading bot.
"""

import asyncio
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for
from threading import Thread

from orchestrator.engine import CopyCatOrchestrator
from orchestrator.config import OrchestratorConfig, TradingMode, MarketPlatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = 'copycat-dashboard-secret-key'

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

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get current system status."""
    try:
        orch = get_orchestrator()
        status = orch.get_status()
        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start the orchestrator."""
    try:
        orch = get_orchestrator()
        
        if orch.state.is_running:
            return jsonify({
                'success': False,
                'message': 'Orchestrator is already running',
            })
        
        # Start in background thread
        global orchestrator_thread
        orchestrator_thread = Thread(target=run_orchestrator_async, args=(orch,), daemon=True)
        orchestrator_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Orchestrator started',
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop the orchestrator."""
    try:
        orch = get_orchestrator()
        
        if not orch.state.is_running:
            return jsonify({
                'success': False,
                'message': 'Orchestrator is not running',
            })
        
        # Stop the orchestrator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(orch.stop())
        loop.close()
        
        return jsonify({
            'success': True,
            'message': 'Orchestrator stopped',
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/pause', methods=['POST'])
def api_pause():
    """Pause the orchestrator."""
    try:
        orch = get_orchestrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orch.pause())
        loop.close()
        
        return jsonify({
            'success': result.success,
            'message': result.message,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/resume', methods=['POST'])
def api_resume():
    """Resume the orchestrator."""
    try:
        orch = get_orchestrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orch.resume())
        loop.close()
        
        return jsonify({
            'success': result.success,
            'message': result.message,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/traders')
def api_traders():
    """Get list of copied traders."""
    try:
        orch = get_orchestrator()
        traders = []
        
        for address, config in orch.state.copied_traders.items():
            traders.append({
                'address': address,
                'enabled': config.enabled,
                'position_size': config.base_position_size,
            })
        
        return jsonify({
            'success': True,
            'traders': traders,
            'count': len(traders),
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/traders/add', methods=['POST'])
def api_add_trader():
    """Add a new trader to copy."""
    try:
        data = request.get_json()
        address = data.get('address', '').strip()
        
        if not address:
            return jsonify({
                'success': False,
                'error': 'Trader address is required',
            })
        
        orch = get_orchestrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orch.add_trader(address))
        loop.close()
        
        return jsonify({
            'success': result.success,
            'message': result.message,
            'details': result.details,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/traders/remove', methods=['POST'])
def api_remove_trader():
    """Remove a trader from copy list."""
    try:
        data = request.get_json()
        address = data.get('address', '').strip()
        
        if not address:
            return jsonify({
                'success': False,
                'error': 'Trader address is required',
            })
        
        orch = get_orchestrator()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orch.remove_trader(address))
        loop.close()
        
        return jsonify({
            'success': result.success,
            'message': result.message,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/performance')
def api_performance():
    """Get performance metrics."""
    try:
        orch = get_orchestrator()
        
        metrics = {
            'total_pnl': orch.state.total_pnl,
            'total_pnl_pct': orch.state.total_pnl_pct,
            'win_rate': orch.state.win_rate,
            'sharpe_ratio': orch.state.sharpe_ratio,
            'max_drawdown': orch.state.max_drawdown,
            'trades_executed': orch.state.trades_executed,
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/health')
def api_health():
    """Get health check status."""
    try:
        orch = get_orchestrator()
        
        health = {
            'api_healthy': orch.state.api_healthy,
            'circuit_breaker_open': orch.state.circuit_breaker_open,
            'consecutive_failures': orch.state.consecutive_failures,
            'is_running': orch.state.is_running,
            'is_paused': orch.state.is_paused,
            'cycle_count': orch.state.cycle_count,
        }
        
        return jsonify({
            'success': True,
            'health': health,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500


# =============================================================================
# Main Entry Point
# =============================================================================

def create_app():
    """Create and configure the Flask app."""
    return app


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the dashboard server."""
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_server(debug=True)
