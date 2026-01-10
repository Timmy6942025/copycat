"""
Structured Logging Configuration for CopyCat Trading System.

Provides log rotation, structured output, and consistent logging across the application.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any


class StructuredLogFormatter(logging.Formatter):
    """Formatter that outputs structured JSON log entries."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        return json.dumps(log_entry)


class CopyCatLogger:
    """
    Centralized logger configuration for CopyCat.

    Features:
    - Log rotation (10MB files, 5 backups)
    - Structured JSON output
    - Separate handlers for console and file
    - Consistent formatting across modules
    """

    _instance: Optional["CopyCatLogger"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not CopyCatLogger._initialized:
            self.log_dir = "logs"
            self._setup_logging()
            CopyCatLogger._initialized = True

    def _setup_logging(self):
        """Configure logging with rotation and structured output."""
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear existing handlers
        root_logger.handlers.clear()

        # JSON formatter for file output
        json_formatter = StructuredLogFormatter()

        # Console formatter (human-readable)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Rotating file handler (10MB, 5 backups)
        log_file = os.path.join(self.log_dir, "copycat.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Create module-specific loggers
        self._setup_module_loggers()

    def _setup_module_loggers(self):
        """Configure module-specific logger levels."""
        # Silence verbose third-party loggers
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific module."""
        return logging.getLogger(name)

    def log_with_data(
        self,
        logger: logging.Logger,
        level: int,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ):
        """Log a message with structured data."""
        extra = {"extra_data": data} if data else {}
        logger.log(level, message, extra=extra, exc_info=exc_info)


# Global logger instance
copycat_logger = CopyCatLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module."""
    return copycat_logger.get_logger(name)


def log_trade_event(
    logger: logging.Logger,
    event_type: str,
    trader_address: str,
    market_id: str,
    amount: float,
    side: str,
    result: str,
    pnl: float = 0.0,
):
    """Log a trade event with structured data."""
    copycat_logger.log_with_data(
        logger,
        logging.INFO,
        f"Trade event: {event_type}",
        {
            "event_type": event_type,
            "trader": trader_address,
            "market": market_id,
            "amount": amount,
            "side": side,
            "result": result,
            "pnl": pnl,
        },
    )


def log_risk_event(
    logger: logging.Logger,
    event_type: str,
    position_info: Dict[str, Any],
    action_taken: str,
):
    """Log a risk management event."""
    copycat_logger.log_with_data(
        logger,
        logging.WARNING,
        f"Risk event: {event_type}",
        {"event_type": event_type, "position": position_info, "action": action_taken},
    )
