"""
Event-Based Focus System.

Prioritizes copying traders based on EVENT TYPES.

Usage:
    from orchestrator.config_events import EventFocusConfig, EventFocusEngine
    
    config = EventFocusConfig(
        enabled=True,
        event_priorities={
            "presidential_election": 1.0,   # Highest priority
            "sports_finals": 0.9,
            "economic_reports": 0.8,
            "crypto_news": 0.6,
            "random_opinion": 0.3,
        },
        min_event_confidence=0.7,
    )
"""

import sys
import os
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventCategory(Enum):
    """Categories of prediction market events."""
    POLITICS = "politics"
    SPORTS = "sports"
    ECONOMICS = "economics"
    CRYPTO = "crypto"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    OTHER = "other"


@dataclass
class EventPriorityConfig:
    """Priority configuration for an event category."""
    priority: float = 0.5        # 0-1 priority score
    min_trader_score: float = 0.3  # Minimum trader score for this category
    max_position_pct: float = 0.15  # Max position size for this category
    auto_approve: bool = True    # Auto-approve trades in this category


@dataclass
class EventFocusConfig:
    """Configuration for event-based focus system."""
    enabled: bool = True
    
    # Event priorities (category -> config)
    event_priorities: Dict[str, EventPriorityConfig] = field(default_factory=lambda: {
        "presidential_election": EventPriorityConfig(priority=1.0, min_trader_score=0.5, max_position_pct=0.15),
        "governor_race": EventPriorityConfig(priority=0.9, min_trader_score=0.4, max_position_pct=0.12),
        "sports_finals": EventPriorityConfig(priority=0.85, min_trader_score=0.4, max_position_pct=0.12),
        "sports_playoffs": EventPriorityConfig(priority=0.75, min_trader_score=0.35, max_position_pct=0.10),
        "economic_report": EventPriorityConfig(priority=0.80, min_trader_score=0.4, max_position_pct=0.10),
        "fed_decision": EventPriorityConfig(priority=0.90, min_trader_score=0.45, max_position_pct=0.12),
        "crypto_major": EventPriorityConfig(priority=0.70, min_trader_score=0.35, max_position_pct=0.10),
        "crypto_news": EventPriorityConfig(priority=0.55, min_trader_score=0.3, max_position_pct=0.08),
        "entertainment_award": EventPriorityConfig(priority=0.60, min_trader_score=0.3, max_position_pct=0.08),
        "entertainment_other": EventPriorityConfig(priority=0.40, min_trader_score=0.25, max_position_pct=0.05),
        "science_discovery": EventPriorityConfig(priority=0.50, min_trader_score=0.3, max_position_pct=0.08),
        "other": EventPriorityConfig(priority=0.30, min_trader_score=0.2, max_position_pct=0.05),
    })
    
    # Category-level defaults
    category_defaults: Dict[str, EventPriorityConfig] = field(default_factory=lambda: {
        "politics": EventPriorityConfig(priority=0.8, min_trader_score=0.4, max_position_pct=0.10),
        "sports": EventPriorityConfig(priority=0.7, min_trader_score=0.35, max_position_pct=0.10),
        "economics": EventPriorityConfig(priority=0.75, min_trader_score=0.4, max_position_pct=0.10),
        "crypto": EventPriorityConfig(priority=0.6, min_trader_score=0.3, max_position_pct=0.08),
        "entertainment": EventPriorityConfig(priority=0.5, min_trader_score=0.3, max_position_pct=0.08),
        "science": EventPriorityConfig(priority=0.5, min_trader_score=0.3, max_position_pct=0.08),
        "other": EventPriorityConfig(priority=0.3, min_trader_score=0.2, max_position_pct=0.05),
    })
    
    # Recognition
    keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "presidential_election": ["president", "election", "democrat", "republican", "nominee"],
        "governor_race": ["governor", "state election", "mayor"],
        "sports_finals": ["finals", "championship", "super bowl", "world series", "nba finals"],
        "sports_playoffs": ["playoffs", "conference", "wild card", "divisional"],
        "economic_report": ["gdp", "unemployment", "inflation", "jobs report", "pce"],
        "fed_decision": ["federal reserve", "fed rate", "interest rate", "monetary policy"],
        "crypto_major": ["bitcoin etf", "sec approval", "major upgrade", "hard fork"],
        "crypto_news": ["crypto", "bitcoin", "ethereum", "altcoin", "defi", "nft"],
        "entertainment_award": ["oscar", "emmy", "grammy", "golden globe", "tony"],
        "entertainment_other": ["movie", "tv show", "celebrity", "music award", "streaming"],
        "science_discovery": ["spacex", "nasa", "climate", "medical breakthrough", "vaccine"],
    })
    
    # Scoring
    event_score_weight: float = 0.3  # Weight for event priority in overall score
    trader_score_weight: float = 0.7  # Weight for trader quality
    
    # Filtering
    min_event_priority: float = 0.4  # Minimum event priority to copy
    block_categories: List[str] = field(default_factory=list)  # Categories to completely block


@dataclass
class EventAnalysis:
    """Analysis of a market event."""
    market_id: str
    event_type: str
    category: EventCategory
    priority: float
    keywords_found: List[str]
    confidence: float
    is_suitable: bool
    max_position_pct: float
    rejection_reason: Optional[str] = None


class EventFocusEngine:
    """
    Analyzes and filters markets based on event type.
    
    Key behavior:
    - Categorizes markets by event type
    - Applies different rules per category
    - Prioritizes high-confidence event categories
    - Blocks or limits certain categories
    """
    
    def __init__(self, config: Optional[EventFocusConfig] = None):
        self.config = config or EventFocusConfig()
        self.analyzed_events: Dict[str, EventAnalysis] = {}
        logger.info(f"EventFocusEngine initialized (enabled={self.config.enabled})")
    
    def analyze_market(
        self,
        market_id: str,
        market_data: Dict[str, Any],
    ) -> EventAnalysis:
        """
        Analyze a market and determine its event type.
        
        Args:
            market_id: Market identifier
            market_data: Dict with:
                - title: Market title
                - description: Market description
                - category: Optional category tag
                - tags: List of tags
        
        Returns:
            EventAnalysis with classification
        """
        if not self.config.enabled:
            return EventAnalysis(
                market_id=market_id,
                event_type="unknown",
                category=EventCategory.OTHER,
                priority=0.5,
                keywords_found=[],
                confidence=0.5,
                is_suitable=True,
                max_position_pct=0.10,
            )
        
        title = market_data.get("title", "").lower()
        description = market_data.get("description", "").lower()
        category = market_data.get("category", "").lower()
        tags = [t.lower() for t in market_data.get("tags", [])]
        
        text = f"{title} {description} {category} {' '.join(tags)}"
        
        # Detect event type
        event_type, keywords = self._detect_event_type(text)
        
        # Get category
        category_enum = self._get_category(event_type, category)
        
        # Get priority config
        priority_config = self._get_priority_config(event_type, category_enum)
        
        # Calculate confidence based on keyword matches
        confidence = self._calculate_confidence(text, keywords)
        
        # Check if blocked
        is_suitable = True
        rejection_reason = None
        
        if category_enum.value in self.config.block_categories:
            is_suitable = False
            rejection_reason = f"Category {category_enum.value} is blocked"
        
        if priority_config.priority < self.config.min_event_priority:
            is_suitable = False
            rejection_reason = f"Event priority {priority_config.priority:.2f} below minimum {self.config.min_event_priority:.2f}"
        
        analysis = EventAnalysis(
            market_id=market_id,
            event_type=event_type,
            category=category_enum,
            priority=priority_config.priority,
            keywords_found=keywords,
            confidence=confidence,
            is_suitable=is_suitable,
            max_position_pct=priority_config.max_position_pct,
            rejection_reason=rejection_reason,
        )
        
        self.analyzed_events[market_id] = analysis
        
        logger.debug(
            f"Event analysis: {market_id[:10]}... | "
            f"Type: {event_type} | "
            f"Category: {category_enum.value} | "
            f"Priority: {priority_config.priority:.2f}"
        )
        
        return analysis
    
    def _detect_event_type(self, text: str) -> Tuple[str, List[str]]:
        """Detect event type from market text."""
        for event_type, keywords in self.config.keywords.items():
            matches = [kw for kw in keywords if kw.lower() in text]
            if matches:
                return event_type, matches
        
        return "other", []
    
    def _get_category(self, event_type: str, category: str) -> EventCategory:
        """Map event type to category."""
        category_map = {
            "presidential_election": EventCategory.POLITICS,
            "governor_race": EventCategory.POLITICS,
            "sports_finals": EventCategory.SPORTS,
            "sports_playoffs": EventCategory.SPORTS,
            "economic_report": EventCategory.ECONOMICS,
            "fed_decision": EventCategory.ECONOMICS,
            "crypto_major": EventCategory.CRYPTO,
            "crypto_news": EventCategory.CRYPTO,
            "entertainment_award": EventCategory.ENTERTAINMENT,
            "entertainment_other": EventCategory.ENTERTAINMENT,
            "science_discovery": EventCategory.SCIENCE,
        }
        
        if event_type in category_map:
            return category_map[event_type]
        
        # Try to detect from category string
        category_lower = category.lower()
        if "polit" in category_lower:
            return EventCategory.POLITICS
        elif "sport" in category_lower:
            return EventCategory.SPORTS
        elif "econom" in category_lower or "finance" in category_lower:
            return EventCategory.ECONOMICS
        elif "crypto" in category_lower or "bitcoin" in category_lower:
            return EventCategory.CRYPTO
        elif "entertain" in category_lower or "movie" in category_lower:
            return EventCategory.ENTERTAINMENT
        elif "science" in category_lower:
            return EventCategory.SCIENCE
        
        return EventCategory.OTHER
    
    def _get_priority_config(
        self,
        event_type: str,
        category: EventCategory,
    ) -> EventPriorityConfig:
        """Get priority config for event type or category."""
        # Try event type first
        if event_type in self.config.event_priorities:
            return self.config.event_priorities[event_type]
        
        # Fall back to category
        if category.value in self.config.category_defaults:
            return self.config.category_defaults[category.value]
        
        # Default
        return EventPriorityConfig()
    
    def _calculate_confidence(self, text: str, keywords: List[str]) -> float:
        """Calculate confidence of event detection."""
        if not keywords:
            return 0.3
        
        # More keywords = higher confidence
        keyword_score = min(len(keywords) / 5, 1.0)
        
        # Check for specificity
        specific_terms = ["will", "by", "before", "date", "exact"]
        specificity_score = sum(1 for term in specific_terms if term in text) / len(specific_terms)
        
        return min((keyword_score * 0.7 + specificity_score * 0.3 + 0.2), 1.0)
    
    def filter_markets(
        self,
        markets: Dict[str, Dict[str, Any]],
    ) -> Dict[str, EventAnalysis]:
        """Filter multiple markets."""
        results = {}
        
        for market_id, data in markets.items():
            analysis = self.analyze_market(market_id, data)
            results[market_id] = analysis
        
        # Log summary
        suitable = sum(1 for a in results.values() if a.is_suitable)
        logger.info(
            f"Event focus filtered {len(markets)} markets: "
            f"{suitable} suitable, {len(markets) - suitable} filtered"
        )
        
        return results
    
    def get_suitable_markets(
        self,
        markets: Dict[str, Dict[str, Any]],
        limit: int = 50,
    ) -> List[Tuple[str, EventAnalysis]]:
        """Get suitable markets sorted by priority."""
        results = self.filter_markets(markets)
        
        suitable = [
            (mid, analysis) for mid, analysis in results.items()
            if analysis.is_suitable
        ]
        
        suitable.sort(key=lambda x: x[1].priority, reverse=True)
        
        return suitable[:limit]
    
    def calculate_trader_event_score(
        self,
        trader_events: List[EventAnalysis],
        trader_score: float,
    ) -> float:
        """
        Calculate combined trader + event score.
        
        Args:
            trader_events: List of EventAnalysis for trader's markets
            trader_score: Base trader quality score
        
        Returns:
            Combined score 0-1
        """
        if not self.config.enabled or not trader_events:
            return trader_score
        
        # Average event priority
        avg_event_priority = sum(e.priority for e in trader_events) / len(trader_events)
        
        # Combined score
        combined = (
            trader_score * self.config.trader_score_weight +
            avg_event_priority * self.config.event_score_weight
        )
        
        return min(max(combined, 0), 1.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        if not self.config.enabled:
            return {"enabled": False}
        
        category_counts = {}
        for analysis in self.analyzed_events.values():
            cat = analysis.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "enabled": True,
            "markets_analyzed": len(self.analyzed_events),
            "by_category": category_counts,
            "event_priorities": {k: v.priority for k, v in self.config.event_priorities.items()},
        }


def create_event_focus_config(
    high_priority_events: List[str] = None,
    block_categories: List[str] = None,
    min_priority: float = 0.4,
) -> EventFocusConfig:
    """Factory function to create event focus config."""
    if high_priority_events is None:
        high_priority_events = ["presidential_election", "fed_decision", "sports_finals"]
    
    if block_categories is None:
        block_categories = []
    
    config = EventFocusConfig(
        enabled=True,
        min_event_priority=min_priority,
        block_categories=block_categories,
    )
    
    # Boost high priority events
    for event in high_priority_events:
        if event in config.event_priorities:
            config.event_priorities[event] = EventPriorityConfig(
                priority=1.0,
                min_trader_score=0.3,
                max_position_pct=0.15,
            )
    
    return config


if __name__ == "__main__":
    print("=" * 60)
    print("EVENT-BASED FOCUS SYSTEM DEMO")
    print("=" * 60)
    
    # Create engine
    engine = EventFocusEngine(
        create_event_focus_config(
            high_priority_events=["presidential_election", "fed_decision"],
            min_priority=0.4,
        )
    )
    
    # Mock markets
    mock_markets = {
        "market_001": {
            "title": "Will Biden win the 2024 Presidential Election?",
            "description": "Predict if Joe Biden will win the 2024 US Presidential Election",
            "category": "Politics",
            "tags": ["election", "president", "democrat"],
        },
        "market_002": {
            "title": "Will Bitcoin exceed $100k in 2024?",
            "description": "Predict if Bitcoin will reach $100,000 USD in 2024",
            "category": "Crypto",
            "tags": ["bitcoin", "crypto", "price"],
        },
        "market_003": {
            "title": "Who will win the Super Bowl LVIII?",
            "description": "Predict which team will win Super Bowl LVIII",
            "category": "Sports",
            "tags": ["super bowl", "nfl", "football"],
        },
        "market_004": {
            "title": "Will the Fed raise rates in March?",
            "description": "Predict if the Federal Reserve will raise interest rates in March 2024",
            "category": "Economics",
            "tags": ["fed", "rates", "monetary policy"],
        },
        "market_005": {
            "title": "Which movie will win Best Picture at the Oscars?",
            "description": "Predict which film will win Best Picture at the 2024 Oscars",
            "category": "Entertainment",
            "tags": ["oscar", "movie", "award"],
        },
        "market_006": {
            "title": "Random question about something",
            "description": "Just a random prediction market",
            "category": "Other",
            "tags": ["random"],
        },
    }
    
    # Analyze markets
    results = engine.filter_markets(mock_markets)
    
    print("\nMarket Analysis Results:")
    print("-" * 60)
    
    for market_id, analysis in results.items():
        status = "✓" if analysis.is_suitable else "✗"
        print(f"\n{status} {market_id}: {mock_markets[market_id]['title'][:50]}...")
        print(f"   Event Type: {analysis.event_type}")
        print(f"   Category: {analysis.category.value}")
        print(f"   Priority: {analysis.priority:.2f}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Max Position: {analysis.max_position_pct:.1%}")
        if analysis.rejection_reason:
            print(f"   Rejected: {analysis.rejection_reason}")
    
    # Get suitable markets
    suitable = engine.get_suitable_markets(mock_markets)
    print(f"\n✓ Suitable Markets: {len(suitable)}")
    
    print("\n" + "=" * 60)
    print("EVENT FOCUS BENEFITS:")
    print("  • Prioritizes high-conviction events")
    print("  • Applies different rules per category")
    print("  • Blocks low-quality categories")
    print("  • Boosts positions on top events")
    print("=" * 60)
