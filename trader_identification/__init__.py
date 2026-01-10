"""
Trader Identification Engine.
Identifies and scores profitable traders for copy trading.
"""

import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from api_clients.base import (
    Trade,
    Trader,
    TraderPerformance,
    OrderSide,
)


@dataclass
class TraderSelectionConfig:
    """Configuration for trader selection criteria."""

    min_win_rate: float = 0.55
    min_trades: int = 10
    max_drawdown: float = 0.25
    min_sharpe_ratio: float = 0.5
    min_profit_factor: float = 1.0
    min_total_pnl: float = 0.0
    max_avg_hold_time_hours: float = 168.0  # 1 week
    min_reputation_score: float = 0.5


@dataclass
class InsiderDetectionResult:
    """Result of insider detection analysis."""

    early_position_score: float = 0.0
    event_correlation_score: float = 0.0
    category_expertise_score: float = 0.0
    is_insider_likely: bool = False
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BotDetectionResult:
    """Result of bot detection analysis."""

    is_bot_likely: bool = False
    hft_score: float = 0.0
    arbitrage_score: float = 0.0
    pattern_score: float = 0.0
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraderIdentificationResult:
    """Complete trader identification result."""

    trader_address: str
    performance: TraderPerformance
    insider_result: Optional[InsiderDetectionResult] = None
    bot_result: Optional[BotDetectionResult] = None
    reputation_score: float = 0.0
    confidence_score: float = 0.0
    is_suitable: bool = False
    selection_reasons: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class TraderIdentificationEngine:
    """
    Engine for identifying and scoring traders for copy trading.
    Analyzes trader performance, detects insiders, and filters bots.
    """

    def __init__(self, config: Optional[TraderSelectionConfig] = None):
        """Initialize trader identification engine."""
        self.config = config or TraderSelectionConfig()

    async def analyze_trader(
        self,
        trader_address: str,
        trades: List[Trade],
        market_events: Optional[Dict[str, Any]] = None,
    ) -> TraderIdentificationResult:
        """Analyze a single trader."""
        # Calculate performance metrics
        performance = self.calculate_performance(trader_address, trades)

        # Detect potential insider trading
        insider_result = self.detect_insider_trading(trades, market_events)

        # Detect potential bot activity
        bot_result = self.detect_bot_activity(trades)

        # Calculate reputation score
        reputation_score = self.calculate_reputation_score(
            performance, insider_result, bot_result
        )

        # Calculate overall confidence
        confidence_score = self.calculate_confidence(
            performance, insider_result, bot_result
        )

        # Determine if trader is suitable for copying
        result = self.evaluate_suitability(
            performance, insider_result, bot_result, reputation_score
        )

        return TraderIdentificationResult(
            trader_address=trader_address,
            performance=performance,
            insider_result=insider_result,
            bot_result=bot_result,
            reputation_score=reputation_score,
            confidence_score=confidence_score,
            is_suitable=result["is_suitable"],
            selection_reasons=result["selected"],
            rejection_reasons=result["rejected"],
        )

    def calculate_performance(
        self,
        trader_address: str,
        trades: List[Trade],
    ) -> TraderPerformance:
        """Calculate performance metrics from trades."""
        if not trades:
            return TraderPerformance(trader_address=trader_address)

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Basic trade counts
        total_trades = len(trades)
        winning_trades = [t for t in trades if self._is_winning_trade(t)]
        losing_trades = [t for t in trades if not self._is_winning_trade(t)]

        # Calculate P&L
        total_pnl = sum(self._calculate_trade_pnl(t) for t in trades)
        total_volume = sum(t.total_value for t in trades)

        # Win rate - safe division with zero check
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        # Profit factor - return 0 instead of inf for safety (infinite profit factor means no losses)
        gross_profits = sum(self._calculate_trade_pnl(t) for t in winning_trades)
        gross_losses = abs(sum(self._calculate_trade_pnl(t) for t in losing_trades))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0.0

        # Calculate returns for risk metrics
        returns = self._calculate_returns(trades)

        # Sharpe ratio (simplified - assume 0% risk-free rate)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(trades)

        # Average hold time
        hold_times = self._calculate_hold_times(sorted_trades)
        avg_hold_time_hours = statistics.mean(hold_times) if hold_times else 0

        # Average win/loss
        avg_win = gross_profits / len(winning_trades) if winning_trades else 0
        avg_loss = gross_losses / len(losing_trades) if losing_trades else 0

        return TraderPerformance(
            trader_address=trader_address,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / total_volume if total_volume > 0 else 0,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_hold_time_hours=avg_hold_time_hours,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_volume=total_volume,
        )

    def detect_insider_trading(
        self,
        trades: List[Trade],
        market_events: Optional[Dict[str, Any]] = None,
    ) -> InsiderDetectionResult:
        """Detect potential insider trading patterns."""
        if not trades:
            return InsiderDetectionResult()

        result = InsiderDetectionResult()
        details = {}

        # Early position detection
        early_positions = self._detect_early_positions(trades, market_events)
        result.early_position_score = early_positions.get("score", 0.0)
        details["early_positions"] = early_positions

        # Event correlation
        event_correlation = self._calculate_event_correlation(trades, market_events)
        result.event_correlation_score = event_correlation.get("score", 0.0)
        details["event_correlation"] = event_correlation

        # Category expertise
        category_expertise = self._analyze_category_expertise(trades)
        result.category_expertise_score = category_expertise.get("score", 0.0)
        details["category_expertise"] = category_expertise

        # Determine if likely insider
        insider_threshold = 0.7
        result.is_insider_likely = (
            result.early_position_score >= insider_threshold
            or result.event_correlation_score >= insider_threshold
        )

        # Confidence based on trade count
        confidence = min(1.0, len(trades) / 100)  # More trades = higher confidence
        result.confidence = confidence
        result.details = details

        return result

    def detect_bot_activity(self, trades: List[Trade]) -> BotDetectionResult:
        """Detect potential bot activity."""
        if not trades:
            return BotDetectionResult()

        result = BotDetectionResult()
        details = {}

        # HFT detection
        hft_analysis = self._detect_hft_patterns(trades)
        result.hft_score = hft_analysis.get("score", 0.0)
        details["hft"] = hft_analysis

        # Arbitrage detection
        arbitrage_analysis = self._detect_arbitrage_patterns(trades)
        result.arbitrage_score = arbitrage_analysis.get("score", 0.0)
        details["arbitrage"] = arbitrage_analysis

        # Pattern detection
        pattern_analysis = self._detect_bot_patterns(trades)
        result.pattern_score = pattern_analysis.get("score", 0.0)
        details["patterns"] = pattern_analysis

        # Determine if likely bot
        bot_threshold = 0.7
        result.is_bot_likely = (
            result.hft_score >= bot_threshold
            or result.arbitrage_score >= bot_threshold
            or result.pattern_score >= bot_threshold
        )

        # Confidence
        confidence = min(1.0, len(trades) / 100)
        result.confidence = confidence
        result.details = details

        return result

    def calculate_reputation_score(
        self,
        performance: TraderPerformance,
        insider_result: InsiderDetectionResult,
        bot_result: BotDetectionResult,
    ) -> float:
        """Calculate overall reputation score."""
        weights = {
            "performance": 0.40,
            "insider": 0.30,
            "bot_filter": 0.20,
            "consistency": 0.10,
        }

        # Performance score (0-1)
        performance_score = min(1.0, performance.win_rate)

        # Insider bonus (higher is better, but not too high to avoid obvious insiders)
        insider_score = 1.0 - min(1.0, insider_result.early_position_score * 0.5)

        # Bot filter (lower score = more likely human)
        bot_score = 1.0 - bot_result.pattern_score

        # Consistency score (based on Sharpe ratio and drawdown)
        consistency_score = (
            min(1.0, performance.sharpe_ratio / 3.0)
            if performance.sharpe_ratio > 0
            else 0.5
        )
        consistency_score = consistency_score * (1 - performance.max_drawdown)

        # Calculate weighted score
        reputation_score = (
            weights["performance"] * performance_score
            + weights["insider"] * insider_score
            + weights["bot_filter"] * bot_score
            + weights["consistency"] * consistency_score
        )

        return max(0.0, min(1.0, reputation_score))

    def calculate_confidence(
        self,
        performance: TraderPerformance,
        insider_result: InsiderDetectionResult,
        bot_result: BotDetectionResult,
    ) -> float:
        """Calculate confidence in the analysis."""
        # Base confidence on trade count
        trade_confidence = min(1.0, performance.total_trades / 100)

        # Adjust for insider detection confidence
        insider_confidence = insider_result.confidence

        # Adjust for bot detection confidence
        bot_confidence = bot_result.confidence

        # Combined confidence
        confidence = (
            trade_confidence * 0.5 + insider_confidence * 0.25 + bot_confidence * 0.25
        )

        return confidence

    def evaluate_suitability(
        self,
        performance: TraderPerformance,
        insider_result: InsiderDetectionResult,
        bot_result: BotDetectionResult,
        reputation_score: float,
    ) -> Dict[str, Any]:
        """Evaluate if trader is suitable for copying."""
        selected = []
        rejected = []

        # Check minimum trades
        if performance.total_trades >= self.config.min_trades:
            selected.append(
                f"Has {performance.total_trades} trades (minimum: {self.config.min_trades})"
            )
        else:
            rejected.append(
                f"Only {performance.total_trades} trades (minimum: {self.config.min_trades})"
            )

        # Check win rate
        if performance.win_rate >= self.config.min_win_rate:
            selected.append(
                f"Win rate {performance.win_rate:.1%} (minimum: {self.config.min_win_rate:.1%})"
            )
        else:
            rejected.append(
                f"Win rate {performance.win_rate:.1%} (minimum: {self.config.min_win_rate:.1%})"
            )

        # Check drawdown
        if performance.max_drawdown <= self.config.max_drawdown:
            selected.append(
                f"Max drawdown {performance.max_drawdown:.1%} (maximum: {self.config.max_drawdown:.1%})"
            )
        else:
            rejected.append(
                f"Max drawdown {performance.max_drawdown:.1%} (maximum: {self.config.max_drawdown:.1%})"
            )

        # Check Sharpe ratio
        if performance.sharpe_ratio >= self.config.min_sharpe_ratio:
            selected.append(
                f"Sharpe ratio {performance.sharpe_ratio:.2f} (minimum: {self.config.min_sharpe_ratio:.2f})"
            )
        else:
            rejected.append(
                f"Sharpe ratio {performance.sharpe_ratio:.2f} (minimum: {self.config.min_sharpe_ratio:.2f})"
            )

        # Check profit factor
        if performance.profit_factor >= self.config.min_profit_factor:
            selected.append(
                f"Profit factor {performance.profit_factor:.2f} (minimum: {self.config.min_profit_factor:.2f})"
            )
        else:
            rejected.append(
                f"Profit factor {performance.profit_factor:.2f} (minimum: {self.config.min_profit_factor:.2f})"
            )

        # Check total P&L
        if performance.total_pnl >= self.config.min_total_pnl:
            selected.append(
                f"Total P&L ${performance.total_pnl:,.2f} (minimum: ${self.config.min_total_pnl:,.2f})"
            )
        else:
            rejected.append(
                f"Total P&L ${performance.total_pnl:,.2f} (minimum: ${self.config.min_total_pnl:,.2f})"
            )

        # Check average hold time
        if performance.avg_hold_time_hours <= self.config.max_avg_hold_time_hours:
            selected.append(
                f"Avg hold time {performance.avg_hold_time_hours:.1f}h (maximum: {self.config.max_avg_hold_time_hours:.1f}h)"
            )
        else:
            rejected.append(
                f"Avg hold time {performance.avg_hold_time_hours:.1f}h (maximum: {self.config.max_avg_hold_time_hours:.1f}h)"
            )

        # Check reputation score
        if reputation_score >= self.config.min_reputation_score:
            selected.append(
                f"Reputation score {reputation_score:.2f} (minimum: {self.config.min_reputation_score:.2f})"
            )
        else:
            rejected.append(
                f"Reputation score {reputation_score:.2f} (minimum: {self.config.min_reputation_score:.2f})"
            )

        # Check bot detection
        if not bot_result.is_bot_likely:
            selected.append("Passed bot detection")
        else:
            rejected.append(
                f"Failed bot detection (HFT: {bot_result.hft_score:.2f}, Arbitrage: {bot_result.arbitrage_score:.2f})"
            )

        # Check insider detection (optional - we might want insiders)
        if insider_result.is_insider_likely:
            selected.append(
                "Detected insider trading patterns (may indicate information advantage)"
            )

        is_suitable = len(selected) > len(rejected) and not bot_result.is_bot_likely

        return {"is_suitable": is_suitable, "selected": selected, "rejected": rejected}

    # Helper methods

    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade is winning."""
        # Simplified: assume trades above 0.5 are winning for YES positions
        return trade.price > 0.5 if trade.outcome == "YES" else trade.price < 0.5

    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade."""
        # Simplified calculation
        return (
            trade.total_value - trade.fees
            if self._is_winning_trade(trade)
            else -(trade.total_value + trade.fees)
        )

    def _calculate_returns(self, trades: List[Trade]) -> List[float]:
        """Calculate returns from trades."""
        if not trades:
            return []
        return [
            self._calculate_trade_pnl(t) / t.total_value if t.total_value > 0 else 0
            for t in trades
        ]

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0.001

        return mean_return / std_return if std_return > 0 else 0.0

    def _calculate_max_drawdown(self, trades: List[Trade]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0.0

        cumulative_pnl = 0
        peak_pnl = 0
        max_dd = 0

        for trade in trades:
            cumulative_pnl += self._calculate_trade_pnl(trade)
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = peak_pnl - cumulative_pnl
            max_dd = max(max_dd, drawdown)

        # Normalize by total volume
        total_volume = sum(t.total_value for t in trades)
        return max_dd / total_volume if total_volume > 0 else 0.0

    def _calculate_hold_times(self, trades: List[Trade]) -> List[float]:
        """Calculate hold times for trades."""
        if len(trades) < 2:
            return []

        hold_times = []
        for i in range(1, len(trades)):
            time_diff = (trades[i].timestamp - trades[i - 1].timestamp).total_seconds()
            hold_times.append(time_diff / 3600)  # Convert to hours

        return hold_times

    def _detect_early_positions(
        self,
        trades: List[Trade],
        market_events: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Detect early position entry patterns."""
        # Simplified: check if trader often enters before price moves
        early_count = 0
        total_count = 0

        for i in range(1, len(trades)):
            prev_price = trades[i - 1].price if i > 0 else 0.5
            curr_price = trades[i].price

            if prev_price != 0.5 and curr_price != prev_price:
                # Price moved - check if trader entered before move
                if trades[i].quantity > 0:
                    early_count += 1
                total_count += 1

        score = early_count / total_count if total_count > 0 else 0.5

        return {"score": score, "early_count": early_count, "total_count": total_count}

    def _calculate_event_correlation(
        self,
        trades: List[Trade],
        market_events: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate correlation with market events."""
        # Simplified: check if trades happen around event dates
        if not market_events:
            return {"score": 0.5, "details": "No event data available"}

        # Placeholder for event correlation logic
        return {"score": 0.5, "details": "Event correlation not implemented"}

    def _analyze_category_expertise(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze trader expertise by market category."""
        if not trades:
            return {"score": 0.5, "details": "No trades"}

        # Group trades by market
        market_groups = defaultdict(list)
        for trade in trades:
            market_groups[trade.market_id].append(trade)

        # Calculate performance by market
        market_scores = {}
        for market_id, market_trades in market_groups.items():
            wins = sum(1 for t in market_trades if self._is_winning_trade(t))
            market_scores[market_id] = (
                len(market_trades),
                wins / len(market_trades) if market_trades else 0,
            )

        # Score based on specialization
        avg_score = (
            sum(score for _, score in market_scores.values()) / len(market_scores)
            if market_scores
            else 0.5
        )

        return {"score": avg_score, "market_scores": market_scores}

    def _detect_hft_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Detect high-frequency trading patterns."""
        if len(trades) < 2:
            return {"score": 0.0, "details": "Insufficient trades"}

        # Check for very short hold times
        hold_times = self._calculate_hold_times(trades)
        short_holds = sum(1 for t in hold_times if t < 1.0 / 3600)  # Less than 1 second
        hft_ratio = short_holds / len(hold_times) if hold_times else 0

        # Check for consistent timing patterns
        # Simplified: assume bot-like if >50% of holds are < 1 second
        score = min(1.0, hft_ratio * 2)

        return {"score": score, "hft_ratio": hft_ratio, "short_holds": short_holds}

    def _detect_arbitrage_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Detect arbitrage trading patterns."""
        if len(trades) < 2:
            return {"score": 0.0, "details": "Insufficient trades"}

        # Check for very small profits consistently
        pnls = [self._calculate_trade_pnl(t) for t in trades]
        small_profits = sum(1 for p in pnls if 0 < p < 1.0)  # Small wins
        small_loss = sum(1 for p in pnls if -1.0 < p < 0)  # Small losses

        arbitrage_ratio = (small_profits + small_loss) / len(pnls) if pnls else 0

        # High ratio of small trades might indicate arbitrage
        score = min(1.0, arbitrage_ratio)

        return {"score": score, "arbitrage_ratio": arbitrage_ratio}

    def _detect_bot_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Detect general bot patterns."""
        if len(trades) < 2:
            return {"score": 0.0, "details": "Insufficient trades"}

        score = 0.0
        reasons = []

        # Check for consistent position sizing
        quantities = [t.quantity for t in trades]
        if len(set(quantities)) == 1:
            score += 0.3
            reasons.append("consistent_position_sizing")

        # Check for consistent timing
        hold_times = self._calculate_hold_times(trades)
        if hold_times:
            time_variance = (
                statistics.variance(hold_times) if len(hold_times) > 1 else 0
            )
            if time_variance < 1.0:  # Very consistent timing
                score += 0.3
                reasons.append("consistent_timing")

        # Check for no trading during off-hours
        trading_hours = [t.timestamp.hour for t in trades]
        if 0 <= min(trading_hours) and max(trading_hours) <= 23:
            if len(set(trading_hours)) < 6:
                score += 0.2
                reasons.append("limited_trading_hours")

        # Check for pattern in trade sizes
        if quantities:
            size_variance = (
                statistics.variance(quantities) if len(quantities) > 1 else 0
            )
            if size_variance == 0:
                score += 0.2
                reasons.append("fixed_trade_sizes")

        return {"score": min(1.0, score), "reasons": reasons}
