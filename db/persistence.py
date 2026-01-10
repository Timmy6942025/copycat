"""
SQLite Persistence Layer for CopyCat Trading System.

Provides data persistence for positions, trades, configurations, and analytics.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import logging


logger = logging.getLogger(__name__)


class SQLitePersistence:
    """
    SQLite database manager for CopyCat.

    Handles:
    - Database creation and migrations
    - Position tracking
    - Trade history
    - Configuration persistence
    - Analytics data

    Features:
    - Thread-safe with context managers
    - Automatic backups
    - Query optimization with indexes
    """

    def __init__(self, db_path: str = "copycat_data.db"):
        """Initialize persistence layer."""
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Create database and tables if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript(_SCHEMA_SCRIPT)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection (context manager)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def backup(self, backup_path: Optional[str] = None) -> str:
        """Create a database backup."""
        if backup_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"copycat_backup_{timestamp}.db"

        with self._get_connection() as conn:
            conn.backup(sqlite3.connect(backup_path))

        logger.info(f"Database backup created: {backup_path}")
        return backup_path

    # =============================================================================
    # POSITION MANAGEMENT
    # =============================================================================

    def save_position(
        self,
        market_id: str,
        outcome: str,
        quantity: float,
        avg_price: float,
        current_price: float,
        trader_address: Optional[str] = None,
    ) -> int:
        """Save or update a position."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO positions (
                    market_id, outcome, quantity, avg_price, current_price,
                    trader_address, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    market_id,
                    outcome,
                    quantity,
                    avg_price,
                    current_price,
                    trader_address,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM positions").fetchall()
            return [dict(row) for row in rows]

    def close_position(self, market_id: str) -> bool:
        """Close and record a position as closed."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE positions SET closed_at = ? WHERE market_id = ?
            """,
                (datetime.utcnow().isoformat(), market_id),
            )
            conn.commit()
            return conn.rowcount > 0

    # =============================================================================
    # TRADE HISTORY
    # =============================================================================

    def record_trade(
        self,
        trade_id: str,
        market_id: str,
        outcome: str,
        side: str,
        quantity: float,
        price: float,
        fees: float,
        slippage: float,
        pnl: float,
        trader_address: Optional[str] = None,
    ) -> int:
        """Record a completed trade."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades (
                    trade_id, market_id, outcome, side, quantity, price,
                    fees, slippage, pnl, trader_address, executed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade_id,
                    market_id,
                    outcome,
                    side,
                    quantity,
                    price,
                    fees,
                    slippage,
                    pnl,
                    trader_address,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_trade_history(
        self, limit: int = 100, trader_address: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trade history."""
        with self._get_connection() as conn:
            if trader_address:
                rows = conn.execute(
                    """
                    SELECT * FROM trades
                    WHERE trader_address = ?
                    ORDER BY executed_at DESC
                    LIMIT ?
                """,
                    (trader_address, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM trades
                    ORDER BY executed_at DESC
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]

    # =============================================================================
    # CONFIGURATION PERSISTENCE
    # =============================================================================

    def save_config(self, config_name: str, config_data: Dict[str, Any]):
        """Save a configuration."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO configurations (name, data, updated_at)
                VALUES (?, ?, ?)
            """,
                (config_name, json.dumps(config_data), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load a configuration."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM configurations WHERE name = ?", (config_name,)
            ).fetchone()
            return json.loads(row["data"]) if row else None

    def get_all_configs(self) -> List[Dict[str, Any]]:
        """Get all saved configurations."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM configurations").fetchall()
            return [dict(row) for row in rows]

    # =============================================================================
    # ANALYTICS & METRICS
    # =============================================================================

    def save_metrics(self, metrics_type: str, metrics_data: Dict[str, Any]):
        """Save analytics metrics."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO metrics (type, data, recorded_at)
                VALUES (?, ?, ?)
            """,
                (metrics_type, json.dumps(metrics_data), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_metrics(self, metrics_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics of a specific type."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM metrics
                WHERE type = ?
                ORDER BY recorded_at DESC
                LIMIT ?
            """,
                (metrics_type, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        with self._get_connection() as conn:
            # Total trades
            total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

            # Total P&L
            total_pnl = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades"
            ).fetchone()[0]

            # Win rate
            winning_trades = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE pnl > 0"
            ).fetchone()[0]
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
            }

    # =============================================================================
    # COPIED TRADERS
    # =============================================================================

    def add_copied_trader(self, address: str, config: Dict[str, Any]):
        """Add a trader to copy list."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO copied_traders (address, config, added_at)
                VALUES (?, ?, ?)
            """,
                (address, json.dumps(config), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def remove_copied_trader(self, address: str) -> bool:
        """Remove a trader from copy list."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM copied_traders WHERE address = ?", (address,))
            conn.commit()
            return conn.rowcount > 0

    def get_copied_traders(self) -> List[Dict[str, Any]]:
        """Get all copied traders."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM copied_traders").fetchall()
            return [dict(row) for row in rows]


# Database schema
_SCHEMA_SCRIPT = """
-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT UNIQUE NOT NULL,
    outcome TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_price REAL NOT NULL,
    current_price REAL NOT NULL,
    trader_address TEXT,
    opened_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    closed_at TEXT
);

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    market_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    fees REAL NOT NULL,
    slippage REAL NOT NULL,
    pnl REAL NOT NULL,
    trader_address TEXT,
    executed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Configurations table
CREATE TABLE IF NOT EXISTS configurations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Copied traders table
CREATE TABLE IF NOT EXISTS copied_traders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT UNIQUE NOT NULL,
    config TEXT NOT NULL,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_trades_trader ON trades(trader_address);
CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(type);
CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market_id);
"""


# Global persistence instance
persistence = SQLitePersistence()
