"""
Notification Service for CopyCat.

Provides Discord webhook notifications for trading events, milestones, and alerts.

Usage:
    from orchestrator.notification_service import (
        NotificationService, NotificationConfig, NotificationEvent
    )
    
    config = NotificationConfig(
        discord_webhook_url="https://discord.com/api/webhooks/...",
        milestone_channels=[20, 50, 100],
    )
    notifier = NotificationService(config)
    
    # Send milestone notification
    await notifier.notify_milestone(25.0, 15.0)  # $25 reached from $15
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque
import json

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationEvent(Enum):
    """Types of notification events."""
    MILESTONE = "milestone"
    TRADE_EXECUTED = "trade_executed"
    TRADE_ERROR = "trade_error"
    TRADER_ADDED = "trader_added"
    TRADER_REMOVED = "trader_removed"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"
    MODE_TRANSITION = "mode_transition"
    DRAWDOWN_ALERT = "drawdown_alert"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    HEALTH_CHECK_FAILED = "health_check_failed"
    API_ERROR = "api_error"
    DAILY_SUMMARY = "daily_summary"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = 0      # Daily summaries, minor events
    MEDIUM = 1   # Trades, trader changes
    HIGH = 2     # Errors, circuit breaker, drawdown alerts
    CRITICAL = 3 # Major milestones, emergency stops


# Color coding for Discord embeds (hex colors)
NOTIFICATION_COLORS = {
    NotificationPriority.LOW: 0x3498db,       # Blue
    NotificationPriority.MEDIUM: 0x2ecc71,    # Green
    NotificationPriority.HIGH: 0xf39c12,      # Orange
    NotificationPriority.CRITICAL: 0xe74c3c,  # Red
}

# Milestone thresholds
MILESTONE_THRESHOLDS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


@dataclass
class NotificationConfig:
    """Configuration for notification service."""
    # Webhook URLs
    discord_webhook_url: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    
    # Milestone tracking
    milestone_channels: List[float] = field(default_factory=lambda: [20, 50, 100, 200, 500, 1000])
    notify_on_milestone: bool = True
    
    # Event notifications
    notify_on_trade: bool = True
    notify_on_trade_error: bool = True
    notify_on_trader_change: bool = True
    notify_on_circuit_breaker: bool = True
    notify_on_mode_transition: bool = True
    notify_on_drawdown_alert: bool = True
    notify_on_startup_shutdown: bool = True
    notify_on_daily_summary: bool = False
    daily_summary_time: str = "18:00"  # UTC time
    
    # Priority thresholds
    min_priority_for_trade: NotificationPriority = NotificationPriority.LOW
    min_priority_for_error: NotificationPriority = NotificationPriority.MEDIUM
    min_priority_for_milestone: NotificationPriority = NotificationPriority.LOW
    
    # Rate limiting
    rate_limit_requests: int = 5
    rate_limit_window_seconds: float = 2.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    
    # Batch settings
    batch_trade_notifications: bool = True
    batch_interval_seconds: float = 60.0
    max_batched_trades: int = 10
    
    # Formatting
    username: str = "CopyCat Bot"
    avatar_url: Optional[str] = None
    enabled: bool = True


@dataclass
class MilestoneRecord:
    """Record of a milestone achievement."""
    balance: float
    milestone: float
    previous_balance: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    growth_pct: float = 0.0


class RateLimiter:
    """Rate limiter for webhook requests."""
    
    def __init__(self, max_requests: int, time_window_seconds: float):
        self.max_requests = max_requests
        self.time_window = time_window_seconds
        self.requests: deque = deque()
    
    def can_send(self) -> bool:
        """Check if we can send a request."""
        now = time.time()
        
        # Remove old requests outside the time window
        while self.requests and now - self.requests[0] > self.time_window:
            self.requests.popleft()
        
        return len(self.requests) < self.max_requests
    
    def wait_if_needed(self) -> float:
        """Wait if we're rate limited. Returns wait time in seconds."""
        if not self.can_send():
            wait_time = self.time_window - (time.time() - self.requests[0])
            if wait_time > 0:
                time.sleep(min(wait_time, 1.0))  # Cap wait time
            return wait_time
        
        self.requests.append(time.time())
        return 0.0
    
    def record_request(self):
        """Record a request."""
        self.requests.append(time.time())
    
    def handle_response_headers(self, headers: Dict[str, str]):
        """Parse rate limit headers from Discord response."""
        remaining = headers.get('X-RateLimit-Remaining', str(self.max_requests))
        reset_after = headers.get('X-RateLimit-Reset-After', '0')
        
        if int(remaining) == 0:
            wait_time = float(reset_after)
            logger.warning(f"Rate limit warning. Reset in {wait_time}s")
            return wait_time
        return 0.0


class DiscordWebhookClient:
    """Async Discord webhook client with rate limiting and retry logic."""
    
    def __init__(self, webhook_url: str, username: str = "CopyCat Bot", 
                 avatar_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url
        self.rate_limiter = RateLimiter(5, 2.0)  # Discord's rate limit
        
        # Metrics
        self._requests_sent = 0
        self._requests_failed = 0
        self._rate_limit_hits = 0
    
    def _create_payload(self, content: str = None, embeds: List[Dict] = None) -> Dict[str, Any]:
        """Create webhook payload."""
        payload = {}
        
        if content:
            payload["content"] = content
        
        if self.username:
            payload["username"] = self.username
        
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url
        
        if embeds:
            payload["embeds"] = embeds
        
        payload["tts"] = False
        
        return payload
    
    def _create_embed(
        self,
        title: str,
        description: str,
        color: int,
        fields: List[Dict] = None,
        footer: Dict = None,
        timestamp: bool = True
    ) -> Dict[str, Any]:
        """Create a Discord embed."""
        embed = {
            "title": title,
            "description": description,
            "color": color,
        }
        
        if fields:
            embed["fields"] = fields
        
        if footer:
            embed["footer"] = footer
        
        if timestamp:
            embed["timestamp"] = datetime.utcnow().isoformat()
        
        return embed
    
    async def send(
        self,
        content: str = None,
        embed: Dict = None,
        embeds: List[Dict] = None,
        priority: NotificationPriority = NotificationPriority.MEDIUM
    ) -> bool:
        """Send a webhook message with retry logic."""
        if not self.webhook_url:
            return False
        
        payload = self._create_payload(content, embeds or ([embed] if embed else None))
        
        last_exception = None
        
        for attempt in range(3):  # Max 3 retries
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        # Handle rate limiting
                        if response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get('Retry-After', 1))
                            logger.warning(f"Rate limited. Retry after {retry_after}s")
                            self._rate_limit_hits += 1
                            await asyncio.sleep(retry_after)
                            continue  # Retry without counting as failure
                        
                        # Handle other errors
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"Webhook error {response.status}: {error_text}")
                            self._requests_failed += 1
                            return False
                        
                        # Success
                        self._requests_sent += 1
                        self.rate_limiter.record_request()
                        
                        # Check for rate limit warnings
                        wait_time = self.rate_limiter.handle_response_headers(dict(response.headers))
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)
                        
                        return True
                        
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Webhook timeout (attempt {attempt + 1}): {e}")
                
            except aiohttp.ClientError as e:
                last_exception = e
                logger.warning(f"Webhook client error (attempt {attempt + 1}): {e}")
            
            # Exponential backoff
            await asyncio.sleep(1.0 * (2 ** attempt))
        
        logger.error(f"Webhook failed after 3 attempts: {last_exception}")
        self._requests_failed += 1
        return False
    
    def get_metrics(self) -> Dict[str, int]:
        """Get client metrics."""
        return {
            "requests_sent": self._requests_sent,
            "requests_failed": self._requests_failed,
            "rate_limit_hits": self._rate_limit_hits,
        }


class NotificationService:
    """
    Central notification service for CopyCat.
    
    Manages Discord (and extensible to Slack/Telegram) notifications for:
    - Milestone achievements ($20, $50, $100, etc.)
    - Trade executions and errors
    - Trader additions/removals
    - Circuit breaker events
    - Mode transitions
    - Drawdown alerts
    - System health events
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        
        # Initialize Discord client
        self.discord_client: Optional[DiscordWebhookClient] = None
        if self.config.discord_webhook_url:
            self.discord_client = DiscordWebhookClient(
                webhook_url=self.config.discord_webhook_url,
                username=self.config.username,
                avatar_url=self.config.avatar_url
            )
        
        # Milestone tracking
        self._reached_milestones: Set[float] = set()
        self._pending_milestones: List[MilestoneRecord] = []
        
        # Trade batching
        self._pending_trades: List[Dict] = []
        self._batch_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self._event_callbacks: Dict[NotificationEvent, List[Callable]] = {}
        
        # Metrics
        self._notifications_sent = 0
        self._notifications_failed = 0
        
        logger.info(f"NotificationService initialized (enabled: {self.config.enabled})")
    
    async def start(self):
        """Start the notification service."""
        if not self.config.enabled:
            logger.info("NotificationService is disabled")
            return
        
        # Start trade batching task
        if self.config.batch_trade_notifications:
            self._batch_task = asyncio.create_task(self._trade_batch_processor())
        
        logger.info("NotificationService started")
        
        # Send startup notification
        if self.config.notify_on_startup_shutdown:
            await self.notify_event(
                event=NotificationEvent.STARTUP,
                title="🚀 CopyCat Started",
                description="CopyCat trading bot has been started",
                priority=NotificationPriority.LOW
            )
    
    async def stop(self):
        """Stop the notification service and send shutdown notification."""
        # Cancel batch task
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        # Send remaining batched trades
        if self._pending_trades:
            await self._send_batched_trades()
        
        # Send shutdown notification
        if self.config.notify_on_startup_shutdown:
            await self.notify_event(
                event=NotificationEvent.SHUTDOWN,
                title="🛑 CopyCat Stopped",
                description="CopyCat trading bot has been stopped",
                priority=NotificationPriority.LOW
            )
        
        logger.info("NotificationService stopped")
    
    # =========================================================================
    # Milestone Notifications
    # =========================================================================
    
    async def check_milestones(self, current_balance: float) -> List[MilestoneRecord]:
        """
        Check if any milestones have been reached and send notifications.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            List of new milestone records
        """
        new_milestones = []
        
        if not self.config.notify_on_milestone:
            return new_milestones
        
        for milestone in self.config.milestone_channels:
            if current_balance >= milestone and milestone not in self._reached_milestones:
                self._reached_milestones.add(milestone)
                
                # Find previous milestone for growth calculation
                previous_milestone = max(
                    [m for m in self._reached_milestones if m < milestone],
                    default=0
                )
                growth_pct = ((current_balance - previous_milestone) / previous_milestone * 100) if previous_milestone > 0 else 0
                
                record = MilestoneRecord(
                    balance=current_balance,
                    milestone=milestone,
                    previous_balance=previous_milestone,
                    growth_pct=growth_pct
                )
                new_milestones.append(record)
                
                # Send notification
                await self.notify_milestone(record)
        
        return new_milestones
    
    async def notify_milestone(self, record: MilestoneRecord):
        """Send milestone achievement notification."""
        if not self.config.notify_on_milestone:
            return
        
        # Calculate emoji based on milestone
        if record.milestone >= 1000:
            emoji = "💎"
        elif record.milestone >= 500:
            emoji = "🏆"
        elif record.milestone >= 100:
            emoji = "🎯"
        elif record.milestone >= 50:
            emoji = "📈"
        elif record.milestone >= 20:
            emoji = "✅"
        else:
            emoji = "🎉"
        
        # Determine color based on milestone size
        if record.milestone >= 1000:
            color = 0xe74c3c  # Red - Critical
        elif record.milestone >= 100:
            color = 0xf39c12  # Orange - High
        else:
            color = 0x2ecc71  # Green - Medium
        
        embed = self.discord_client._create_embed(
            title=f"{emoji} Milestone Reached!",
            description=f"Account balance has reached **${record.milestone:,.0f}**",
            color=color,
            fields=[
                {"name": "💰 Current Balance", "value": f"${record.balance:,.2f}", "inline": True},
                {"name": "📊 Previous Milestone", "value": f"${record.previous_milestone:,.2f}", "inline": True},
                {"name": "🚀 Growth", "value": f"+{record.growth_pct:.1f}% from last milestone", "inline": True},
            ],
            footer={"text": f"Milestone #{len(self._reached_milestones)}"}
        )
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.MEDIUM)
        
        if success:
            self._notifications_sent += 1
            logger.info(f"Milestone notification sent: ${record.milestone}")
        else:
            self._notifications_failed += 1
    
    # =========================================================================
    # Trade Notifications
    # =========================================================================
    
    async def notify_trade(
        self,
        market_id: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = None,
        trader_address: str = None
    ):
        """Queue a trade for batched notification."""
        if not self.config.notify_on_trade:
            return
        
        trade_info = {
            "market_id": market_id,
            "side": side,
            "quantity": quantity,
            "price": price,
            "pnl": pnl,
            "trader_address": trader_address[:8] + "..." if trader_address else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._pending_trades.append(trade_info)
    
    async def _trade_batch_processor(self):
        """Background task to process batched trade notifications."""
        while True:
            try:
                await asyncio.sleep(self.config.batch_interval_seconds)
                
                if self._pending_trades:
                    await self._send_batched_trades()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trade batch processor: {e}")
    
    async def _send_batched_trades(self):
        """Send batched trade notifications."""
        if not self._pending_trades or not self.discord_client:
            return
        
        trades_to_send = self._pending_trades[:self.config.max_batched_trades]
        self._pending_trades = self._pending_trades[self.config.max_batched_trades:]
        
        # Create embed for batched trades
        fields = []
        for trade in trades_to_send:
            pnl_str = f" P&L: ${trade['pnl']:+.2f}" if trade['pnl'] else ""
            fields.append({
                "name": f"{trade['side'].upper()} {trade['market_id'][:20]}...",
                "value": f"${trade['quantity']:.2f} @ ${trade['price']:.3f}{pnl_str}",
                "inline": True
            })
        
        embed = self.discord_client._create_embed(
            title=f"📊 {len(trades_to_send)} Trades Executed",
            description=f"Recent copy trading activity",
            color=NOTIFICATION_COLORS[NotificationPriority.LOW],
            fields=fields[:25],  # Discord limit
            footer={"text": f"Showing {len(trades_to_send)} of {len(trades_to_send)} trades"}
        )
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.LOW)
        
        if success:
            self._notifications_sent += 1
        else:
            self._notifications_failed += 1
    
    async def notify_trade_error(self, market_id: str, error: str):
        """Send trade error notification."""
        if not self.config.notify_on_trade_error:
            return
        
        embed = self.discord_client._create_embed(
            title="⚠️ Trade Error",
            description=f"Failed to execute trade on market",
            color=NOTIFICATION_COLORS[NotificationPriority.HIGH],
            fields=[
                {"name": "Market", "value": market_id[:30], "inline": True},
                {"name": "Error", "value": error[:500], "inline": False},
            ]
        )
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.HIGH)
        
        if success:
            self._notifications_sent += 1
        else:
            self._notifications_failed += 1
    
    # =========================================================================
    # Trader Notifications
    # =========================================================================
    
    async def notify_trader_added(self, trader_address: str, position_size: float):
        """Send notification when a trader is added to copy."""
        if not self.config.notify_on_trader_change:
            return
        
        embed = self.discord_client._create_embed(
            title="👤 Trader Added",
            description=f"Started copying a new trader",
            color=NOTIFICATION_COLORS[NotificationPriority.MEDIUM],
            fields=[
                {"name": "Trader", "value": f"`{trader_address[:16]}...`", "inline": True},
                {"name": "Position Size", "value": f"${position_size:.2f}", "inline": True},
            ]
        )
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.MEDIUM)
        
        if success:
            self._notifications_sent += 1
    
    async def notify_trader_removed(self, trader_address: str, reason: str):
        """Send notification when a trader is removed from copy."""
        if not self.config.notify_on_trader_change:
            return
        
        embed = self.discord_client._create_embed(
            title="👋 Trader Removed",
            description=f"Stopped copying a trader",
            color=NOTIFICATION_COLORS[NotificationPriority.MEDIUM],
            fields=[
                {"name": "Trader", "value": f"`{trader_address[:16]}...`", "inline": True},
                {"name": "Reason", "value": reason[:200], "inline": False},
            ]
        )
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.MEDIUM)
        
        if success:
            self._notifications_sent += 1
    
    # =========================================================================
    # Circuit Breaker Notifications
    # =========================================================================
    
    async def notify_circuit_breaker_opened(self, reason: str, consecutive_failures: int):
        """Send notification when circuit breaker opens."""
        if not self.config.notify_on_circuit_breaker:
            return
        
        embed = self.discord_client._create_embed(
            title="🛑 Circuit Breaker OPENED",
            description="Trading has been paused due to repeated failures",
            color=NOTIFICATION_COLORS[NotificationPriority.CRITICAL],
            fields=[
                {"name": "Reason", "value": reason, "inline": False},
                {"name": "Consecutive Failures", "value": str(consecutive_failures), "inline": True},
            ]
        )
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.CRITICAL)
        
        if success:
            self._notifications_sent += 1
        else:
            self._notifications_failed += 1
    
    async def notify_circuit_breaker_closed(self):
        """Send notification when circuit breaker closes."""
        if not self.config.notify_on_circuit_breaker:
            return
        
        embed = self.discord_client._create_embed(
            title="✅ Circuit Breaker CLOSED",
            description="Trading has been resumed",
            color=NOTIFICATION_COLORS[NotificationPriority.MEDIUM],
            fields=[
                {"name": "Status", "value": "Trading active", "inline": True},
            ]
        )
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.MEDIUM)
        
        if success:
            self._notifications_sent += 1
    
    # =========================================================================
    # Mode Transition Notifications
    # =========================================================================
    
    async def notify_mode_transition(
        self,
        from_mode: str,
        to_mode: str,
        balance: float,
        reason: str = None
    ):
        """Send notification when trading mode transitions."""
        if not self.config.notify_on_mode_transition:
            return
        
        embed = self.discord_client._create_embed(
            title="🔄 Mode Transition",
            description=f"Trading mode has changed",
            color=NOTIFICATION_COLORS[NotificationPriority.MEDIUM],
            fields=[
                {"name": "From", "value": from_mode.upper(), "inline": True},
                {"name": "To", "value": to_mode.upper(), "inline": True},
                {"name": "Balance", "value": f"${balance:,.2f}", "inline": True},
            ]
        )
        
        if reason:
            embed["fields"].append({"name": "Reason", "value": reason, "inline": False})
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.MEDIUM)
        
        if success:
            self._notifications_sent += 1
    
    # =========================================================================
    # Drawdown Alerts
    # =========================================================================
    
    async def notify_drawdown_alert(self, current_drawdown: float, threshold: float):
        """Send drawdown alert notification."""
        if not self.config.notify_on_drawdown_alert:
            return
        
        embed = self.discord_client._create_embed(
            title="📉 Drawdown Alert",
            description=f"Account drawdown has exceeded threshold",
            color=NOTIFICATION_COLORS[NotificationPriority.HIGH],
            fields=[
                {"name": "Current Drawdown", "value": f"{current_drawdown:.1%}", "inline": True},
                {"name": "Threshold", "value": f"{threshold:.1%}", "inline": True},
                {"name": "Status", "value": "Position sizing reduced" if current_drawdown < threshold * 1.2 else "Emergency stop recommended", "inline": False},
            ]
        )
        
        success = await self.discord_client.send(embed=embed, priority=NotificationPriority.HIGH)
        
        if success:
            self._notifications_sent += 1
        else:
            self._notifications_failed += 1
    
    # =========================================================================
    # Generic Event Notifications
    # =========================================================================
    
    async def notify_event(
        self,
        event: NotificationEvent,
        title: str,
        description: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        fields: List[Dict] = None,
        content: str = None
    ):
        """Send a generic event notification."""
        if not self.config.enabled:
            return
        
        # Determine if we should notify based on priority and event type
        should_notify = self._should_notify_event(event, priority)
        if not should_notify:
            return
        
        if not self.discord_client:
            return
        
        color = NOTIFICATION_COLORS.get(priority, NOTIFICATION_COLORS[NotificationPriority.MEDIUM])
        
        embed = self.discord_client._create_embed(
            title=title,
            description=description,
            color=color,
            fields=fields or []
        )
        
        success = await self.discord_client.send(
            content=content,
            embed=embed,
            priority=priority
        )
        
        if success:
            self._notifications_sent += 1
        else:
            self._notifications_failed += 1
        
        # Call registered callbacks
        await self._call_event_callbacks(event, {"title": title, "description": description})
    
    def _should_notify_event(self, event: NotificationEvent, priority: NotificationPriority) -> bool:
        """Determine if an event should trigger a notification."""
        # Check if event type notifications are enabled
        event_config = {
            NotificationEvent.TRADE_EXECUTED: self.config.notify_on_trade,
            NotificationEvent.TRADE_ERROR: self.config.notify_on_trade_error,
            NotificationEvent.TRADER_ADDED: self.config.notify_on_trader_change,
            NotificationEvent.TRADER_REMOVED: self.config.notify_on_trader_change,
            NotificationEvent.CIRCUIT_BREAKER_OPENED: self.config.notify_on_circuit_breaker,
            NotificationEvent.CIRCUIT_BREAKER_CLOSED: self.config.notify_on_circuit_breaker,
            NotificationEvent.MODE_TRANSITION: self.config.notify_on_mode_transition,
            NotificationEvent.DRAWDOWN_ALERT: self.config.notify_on_drawdown_alert,
            NotificationEvent.STARTUP: self.config.notify_on_startup_shutdown,
            NotificationEvent.SHUTDOWN: self.config.notify_on_startup_shutdown,
        }
        
        enabled = event_config.get(event, True)
        
        if not enabled:
            return False
        
        # Check priority thresholds
        priority_config = {
            NotificationEvent.TRADE_EXECUTED: self.config.min_priority_for_trade,
            NotificationEvent.TRADE_ERROR: self.config.min_priority_for_error,
        }
        
        min_priority = priority_config.get(event, NotificationPriority.LOW)
        
        return priority.value >= min_priority.value
    
    # =========================================================================
    # Event Callbacks
    # =========================================================================
    
    def add_event_callback(self, event: NotificationEvent, callback: Callable):
        """Add a callback for an event type."""
        if event not in self._event_callbacks:
            self._event_callbacks[event] = []
        self._event_callbacks[event].append(callback)
    
    async def _call_event_callbacks(self, event: NotificationEvent, data: Dict):
        """Call all registered callbacks for an event."""
        if event not in self._event_callbacks:
            return
        
        for callback in self._event_callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    # =========================================================================
    # Metrics and Status
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get notification service status."""
        return {
            "enabled": self.config.enabled,
            "discord_configured": self.discord_client is not None,
            "reached_milestones": sorted(list(self._reached_milestones)),
            "pending_trades": len(self._pending_trades),
            "notifications_sent": self._notifications_sent,
            "notifications_failed": self._notifications_failed,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics."""
        metrics = {
            "sent": self._notifications_sent,
            "failed": self._notifications_failed,
            "success_rate": (
                self._notifications_sent / 
                (self._notifications_sent + self._notifications_failed) * 100
                if (self._notifications_sent + self._notifications_failed) > 0 else 100
            ),
            "reached_milestones": list(self._reached_milestones),
        }
        
        if self.discord_client:
            metrics["discord"] = self.discord_client.get_metrics()
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self._notifications_sent = 0
        self._notifications_failed = 0
        self._reached_milestones = set()
        self._pending_trades = []
        
        if self.discord_client:
            self.discord_client._requests_sent = 0
            self.discord_client._requests_failed = 0
            self.discord_client._rate_limit_hits = 0


# =============================================================================
# Factory Functions
# =============================================================================

def create_notification_config(
    discord_webhook_url: Optional[str] = None,
    milestone_channels: List[float] = None,
    notify_trades: bool = True,
    notify_errors: bool = True,
    batch_trades: bool = True,
) -> NotificationConfig:
    """
    Factory function to create NotificationConfig.
    
    Args:
        discord_webhook_url: Discord webhook URL
        milestone_channels: List of balance milestones to notify on
        notify_trades: Whether to notify on trades
        notify_errors: Whether to notify on errors
        batch_trades: Whether to batch trade notifications
        
    Returns:
        Configured NotificationConfig
    """
    return NotificationConfig(
        discord_webhook_url=discord_webhook_url,
        milestone_channels=milestone_channels or [20, 50, 100, 200, 500, 1000],
        notify_on_trade=notify_trades,
        notify_on_trade_error=notify_errors,
        notify_on_milestone=True,
        batch_trade_notifications=batch_trades,
    )


async def test_notification_service(webhook_url: str) -> bool:
    """
    Test the notification service with a sample webhook.
    
    Args:
        webhook_url: Discord webhook URL to test
        
    Returns:
        True if test succeeded, False otherwise
    """
    config = create_notification_config(discord_webhook_url=webhook_url)
    service = NotificationService(config)
    
    try:
        # Test milestone notification
        await service.notify_event(
            event=NotificationEvent.STARTUP,
            title="🧪 Test Notification",
            description="This is a test notification from CopyCat",
            priority=NotificationPriority.LOW,
        )
        
        status = service.get_status()
        print(f"Test notification sent successfully!")
        print(f"Status: {status}")
        
        return True
        
    except Exception as e:
        print(f"Test notification failed: {e}")
        return False
    
    finally:
        await service.stop()


if __name__ == "__main__":
    import sys
    
    # Example usage
    webhook_url = sys.argv[1] if len(sys.argv) > 1 else None
    
    if webhook_url:
        print(f"Testing notification service with webhook...")
        success = asyncio.run(test_notification_service(webhook_url))
        sys.exit(0 if success else 1)
    else:
        print("Usage: python notification_service.py <webhook_url>")
        print("Example: python notification_service.py https://discord.com/api/webhooks/...")
        sys.exit(1)
