"""
CopyCat Web Dashboard.
Flask-based web interface for monitoring and controlling the copy trading bot.
"""

from .app import app, run_server, create_app

__all__ = ['app', 'run_server', 'create_app']
