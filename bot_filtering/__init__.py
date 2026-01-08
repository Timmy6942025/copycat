"""
Bot Filtering Module.
Filters out arbitrage bots and HFT strategies from trader analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from api_clients.base import Trade, OrderSide


@dataclass
class BotFilterConfig:
    """Configuration for bot filtering."""
    # HFT detection
    hft_max_hold_time_seconds: float = 1.0
    hft_min_trades_per_minute: int = 5

    # Arbitrage detection
    arbitrage_max_profit_pct: float = 0.5  # Max profit for single trade to be considered arbitrage
    arbitrage_min_trade_frequency: int = 100  # Min trades per day to indicate arbitrage

    # Pattern detection
    pattern_check_enabled: bool = True
    consistent_timing_threshold: float = 0.5  # Variance threshold for timing consistency
    fixed_size_threshold: float = 0.1  # Variance threshold for position sizing

    # Exclusion thresholds
    min_hft_score_to_exclude: float = 0.7
    min_arbitrage_score_to_exclude: float = 0.7
    min_pattern_score_to_exclude: float = 0.7


@dataclass
class BotFilterResult:
    """Result of bot filtering analysis."""
    is_bot: bool = False
    is_hft: bool = False
    is_arbitrage: bool = False
    is_pattern_bot: bool = False
    hft_score: float = 0.0
    arbitrage_score: float = 0.0
    pattern_score: float = 0.0
    overall_score: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class BotFilter:
    """
    Bot filtering module.
    Identifies and filters out arbitrage bots, HFT strategies, and other automated traders.
    """

    def __init__(self, config: Optional[BotFilterConfig] = None):
        """Initialize bot filter."""
        self.config = config or BotFilterConfig()

    def analyze_trades(self, trades: List[Trade]) -> BotFilterResult:
        """Analyze trades for bot patterns."""
        if not trades:
            return BotFilterResult(confidence=0.0)

        result = BotFilterResult()
        details = {}

        # HFT analysis
        hft_result = self._analyze_hft(trades)
        result.hft_score = hft_result["score"]
        result.is_hft = result.hft_score >= self.config.min_hft_score_to_exclude
        details["hft"] = hft_result

        # Arbitrage analysis
        arbitrage_result = self._analyze_arbitrage(trades)
        result.arbitrage_score = arbitrage_result["score"]
        result.is_arbitrage = result.arbitrage_score >= self.config.min_arbitrage_score_to_exclude
        details["arbitrage"] = arbitrage_result

        # Pattern analysis
        if self.config.pattern_check_enabled:
            pattern_result = self._analyze_patterns(trades)
            result.pattern_score = pattern_result["score"]
            result.is_pattern_bot = result.pattern_score >= self.config.min_pattern_score_to_exclude
            details["patterns"] = pattern_result

        # Overall assessment
        result.is_bot = result.is_hft or result.is_arbitrage or result.is_pattern_bot

        # Calculate overall score
        result.overall_score = max(result.hft_score, result.arbitrage_score, result.pattern_score)

        # Calculate confidence based on trade count
        result.confidence = min(1.0, len(trades) / 100)

        # Generate reasons
        result.reasons = self._generate_reasons(result, details)
        result.details = details

        return result

    def _analyze_hft(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze for high-frequency trading patterns."""
        if len(trades) < 2:
            return {"score": 0.0, "reason": "Insufficient trades for HFT analysis"}

        # Calculate hold times
        hold_times = self._calculate_hold_times(trades)
        if not hold_times:
            return {"score": 0.0, "reason": "Could not calculate hold times"}

        # Count HFT trades
        hft_count = sum(1 for t in hold_times if t <= self.config.hft_max_hold_time_seconds)
        hft_ratio = hft_count / len(hold_times)

        # Calculate trades per minute
        if len(trades) >= 2:
            time_span = (trades[-1].timestamp - trades[0].timestamp).total_seconds() / 60
            trades_per_minute = len(trades) / max(time_span, 1/60)
        else:
            trades_per_minute = 0

        # Check for sustained HFT activity
        sustained_hft = trades_per_minute >= self.config.hft_min_trades_per_minute

        # Calculate score
        score = 0.0
        reasons = []

        if hft_ratio > 0.5:
            score += 0.5
            reasons.append(f"High HFT ratio: {hft_ratio:.1%} of trades have <1s hold time")
        elif hft_ratio > 0.2:
            score += 0.2
            reasons.append(f"Moderate HFT ratio: {hft_ratio:.1%} of trades have <1s hold time")

        if sustained_hft:
            score += 0.3
            reasons.append(f"Sustained HFT activity: {trades_per_minute:.1f} trades/minute")

        if hft_ratio > 0.8 and sustained_hft:
            score += 0.2
            reasons.append("Extreme HFT patterns detected")

        return {
            "score": min(1.0, score),
            "hft_ratio": hft_ratio,
            "trades_per_minute": trades_per_minute,
            "hft_count": hft_count,
            "sustained_hft": sustained_hft,
            "reasons": reasons,
        }

    def _analyze_arbitrage(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze for arbitrage trading patterns."""
        if len(trades) < 2:
            return {"score": 0.0, "reason": "Insufficient trades for arbitrage analysis"}

        # Calculate trade statistics
        pnls = [self._calculate_trade_pnl(t) for t in trades]
        total_value = sum(t.total_value for t in trades)

        # Check for small, consistent profits (arbitrage indicator)
        small_profits = sum(1 for p in pnls if 0 < p < self.config.arbitrage_max_profit_pct * 100)
        small_losses = sum(1 for p in pnls if -self.config.arbitrage_max_profit_pct * 100 < p < 0)
        small_trade_ratio = (small_profits + small_losses) / len(pnls) if pnls else 0

        # Check for cross-market patterns (if we have market IDs)
        market_trades = defaultdict(list)
        for trade in trades:
            market_trades[trade.market_id].append(trade)

        # Arbitrage often trades multiple markets simultaneously
        multi_market_trading = len(market_trades) > 1

        # Check for timing patterns (arbitrage often has consistent timing)
        hold_times = self._calculate_hold_times(trades)
        if hold_times:
            time_variance = self._variance(hold_times)
            consistent_timing = time_variance < 1.0
        else:
            consistent_timing = False

        # Calculate score
        score = 0.0
        reasons = []

        if small_trade_ratio > 0.7:
            score += 0.5
            reasons.append(f"High small-trade ratio: {small_trade_ratio:.1%}")
        elif small_trade_ratio > 0.5:
            score += 0.2
            reasons.append(f"Moderate small-trade ratio: {small_trade_ratio:.1%}")

        if multi_market_trading:
            score += 0.2
            reasons.append(f"Multi-market trading: {len(market_trades)} markets")

        if consistent_timing:
            score += 0.2
            reasons.append("Consistent timing patterns")

        # Check for spread exploitation
        spread_exploitation = self._detect_spread_exploitation(trades)
        if spread_exploitation["detected"]:
            score += 0.3
            reasons.append("Spread exploitation patterns detected")

        return {
            "score": min(1.0, score),
            "small_trade_ratio": small_trade_ratio,
            "multi_market_trading": multi_market_trading,
            "consistent_timing": consistent_timing,
            "spread_exploitation": spread_exploitation,
            "reasons": reasons,
        }

    def _analyze_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze for general bot patterns."""
        if len(trades) < 2:
            return {"score": 0.0, "reason": "Insufficient trades for pattern analysis"}

        score = 0.0
        reasons = []

        # Check position sizing consistency
        quantities = [t.quantity for t in trades]
        if quantities:
            size_variance = self._variance(quantities)
            if size_variance < self.config.fixed_size_threshold:
                score += 0.25
                reasons.append("Consistent position sizing")

        # Check timing consistency
        hold_times = self._calculate_hold_times(trades)
        if hold_times:
            time_variance = self._variance(hold_times)
            if time_variance < self.config.consistent_timing_threshold:
                score += 0.25
                reasons.append("Consistent timing patterns")

        # Check for round number usage
        round_trades = sum(1 for t in trades if t.quantity % 10 == 0)
        round_ratio = round_trades / len(trades)
        if round_ratio > 0.8:
            score += 0.15
            reasons.append(f"Round number usage: {round_ratio:.1%}")

        # Check for no trading during off-hours
        trading_hours = [t.timestamp.hour for t in trades]
        unique_hours = set(trading_hours)
        if len(unique_hours) < 6 and min(trading_hours) >= 8 and max(trading_hours) <= 20:
            score += 0.15
            reasons.append("Limited trading hours (business hours only)")

        # Check for weekend trading (bots often don't trade weekends)
        weekend_trades = sum(1 for t in trades if t.timestamp.weekday() >= 5)
        weekend_ratio = weekend_trades / len(trades)
        if weekend_ratio == 0 and len(trades) > 10:
            score += 0.1
            reasons.append("No weekend trading (potential automation)")

        # Check for exact same quantity
        if len(set(quantities)) == 1:
            score += 0.1
            reasons.append("Fixed position size")

        return {
            "score": min(1.0, score),
            "reasons": reasons,
            "size_variance": self._variance(quantities) if quantities else 0,
            "time_variance": self._variance(hold_times) if hold_times else 0,
        }

    def _detect_spread_exploitation(self, trades: List[Trade]) -> Dict[str, Any]:
        """Detect spread exploitation patterns."""
        # Look for trades that capture small spreads consistently
        small_captures = 0
        total_trades = len(trades)

        for trade in trades:
            # If trade value is small and profitable, might be spread capture
            if trade.total_value < 10 and self._calculate_trade_pnl(trade) > 0:
                small_captures += 1

        ratio = small_captures / total_trades if total_trades > 0 else 0

        return {
            "detected": ratio > 0.3,
            "ratio": ratio,
            "count": small_captures,
        }

    def _calculate_hold_times(self, trades: List[Trade]) -> List[float]:
        """Calculate hold times in seconds."""
        if len(trades) < 2:
            return []

        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        hold_times = []

        for i in range(1, len(sorted_trades)):
            time_diff = (sorted_trades[i].timestamp - sorted_trades[i-1].timestamp).total_seconds()
            hold_times.append(time_diff)

        return hold_times

    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade (simplified)."""
        # Simplified: assume profitable if price > 0.5 for YES
        is_win = trade.price > 0.5 if trade.outcome == "YES" else trade.price < 0.5
        return trade.total_value - trade.fees if is_win else -(trade.total_value + trade.fees)

    def _variance(self, values: List[float]) -> float:
        """Calculate variance of a list."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)

    def _generate_reasons(self, result: BotFilterResult, details: Dict) -> List[str]:
        """Generate human-readable reasons for the filter result."""
        reasons = []

        if result.is_hft:
            reasons.append("High-frequency trading detected")
            reasons.extend(details.get("hft", {}).get("reasons", []))

        if result.is_arbitrage:
            reasons.append("Arbitrage patterns detected")
            reasons.extend(details.get("arbitrage", {}).get("reasons", []))

        if result.is_pattern_bot:
            reasons.append("Bot-like patterns detected")
            reasons.extend(details.get("patterns", {}).get("reasons", []))

        if not reasons:
            reasons.append("No bot patterns detected")

        return reasons

    def should_exclude(self, result: BotFilterResult) -> bool:
        """Determine if a trader should be excluded based on bot analysis."""
        return result.is_bot and result.confidence >= 0.5

    def get_exclusion_reasons(self, result: BotFilterResult) -> List[str]:
        """Get reasons for exclusion."""
        return result.reasons
