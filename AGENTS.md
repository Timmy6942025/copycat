# PROJECT KNOWLEDGE BASE

**Generated:** Fri Jan 09 2026
**Commit:** Current working tree
**Branch:** main

## OVERVIEW

AI-powered copy trading bot for prediction markets (Polymarket). Identifies profitable traders, filters out bots, and automatically copies trades in sandbox or live mode. Optimized for small accounts ($10-$100) with aggressive growth strategies.

**Stack:** Python 3.11+, asyncio, pytest, aiohttp, pandas, numpy, rich, Flask

## STRUCTURE

```
./
├── orchestrator/          # Main coordination (23 files)
├── sandbox/              # Paper trading simulation (12 files + subdirs)
├── api_clients/          # Market APIs (4 files + polymarket/)
├── trader_identification/ # Trader analysis (2 files)
├── bot_filtering/        # Bot detection (2 files)
├── live_trading/         # Real-money trading (2 files)
└── dashboard/            # Web UI (Flask, 2 files)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Main entry point | `run.py` | CLI for sandbox/speed/status |
| Core coordination | `orchestrator/engine.py` | ⚠️ 1,284 lines, god class |
| Sandbox execution | `sandbox/runner.py` | Virtual trading simulation |
| API integration | `api_clients/polymarket/` | CLOB, Gamma, Data APIs |
| Trader analysis | `trader_identification/__init__.py` | Growth-based selection |
| Bot detection | `bot_filtering/__init__.py` | HFT, arbitrage detection |
| Web dashboard | `dashboard/app.py` | Flask REST API on :5000 |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| CopyCatOrchestrator | Class | orchestrator/engine.py | High | Main coordinator |
| SandboxRunner | Class | sandbox/runner.py | High | Simulation engine |
| PolymarketAPIClient | Class | api_clients/polymarket/__init__.py | High | Unified API client |
| TraderIdentificationEngine | Class | trader_identification/__init__.py | Medium | Trader scoring |
| BotFilter | Class | bot_filtering/__init__.py | Medium | Bot detection |
| VirtualPortfolioManager | Class | sandbox/portfolio.py | Medium | Position tracking |
| PerformanceTracker | Class | sandbox/analytics.py | Medium | Metrics calculation |

## CONVENTIONS

**Async-first architecture**: All I/O uses async/await (377+ async functions)
**Dataclass configs**: All configuration uses `@dataclass` with `field(default_factory=...)`
**Enum-based types**: Modes, platforms, order types use enums for type safety
**Dependency injection**: Components accept config objects via constructors
**Extensive type hints**: Nearly all functions have type annotations
**Logging pattern**: `logger = logging.getLogger(__name__)` in each module
**No explicit config files**: Uses Python dataclasses, not JSON/YAML/TOML

## ANTI-PATTERNS (THIS PROJECT)

❌ **God class**: `orchestrator/engine.py` (1,284 lines) handles EVERYTHING
❌ **Config proliferation**: 14+ overlapping config files in `orchestrator/config_*.py`
❌ **Hardcoded values**: Trader addresses, mode thresholds, milestones in code
❌ **State mutation everywhere**: Direct `self.state.copied_traders[addr] = config`
❌ **Data model duplication**: `OrderSide`, `OrderType`, `OrderStatus` defined in multiple places
❌ **No gitignore**: Risk of committing `venv/`, `__pycache__/`, `.env`
❌ **Root file clutter**: 7 `demo_*.py`, 3 `test_*.py`, 4 `config_*.py` at root

## UNIQUE STYLES

**Multiple entry points**: `run.py` (CLI), `copycat.sh` (shell), `Makefile` (targets)
**Parallel analysis**: `asyncio.gather()` for 10-50x speedup in trader discovery
**Micro mode**: Auto-transition strategy for $10-$100 accounts (NANO→MICRO→MINI→BALANCED)
**Speed mode**: 8 optimizations unified (tiered copying, momentum, events, hedging, optimizer, allocation, bootstrap, adaptive)
**Bootstrap traders**: 26 known profitable addresses hardcoded for quick-start
**Circuit breaker**: Auto-stop after 5 consecutive failures with 60s timeout
**Real-time data**: Sandbox uses callback pattern to fetch real market data per order
**No CI/CD**: Manual testing workflow (Makefile only)
**Lack of linting tools**: `black`/`ruff` referenced in Makefile but NOT in requirements.txt

## COMMANDS

```bash
# Development
make test               # Run all tests (pytest)
make test-quick         # Quick tests without traceback
make format             # Format with black (if installed)
make lint               # Lint with ruff (if installed)
make install-deps        # pip install -r requirements.txt

# Running
make sandbox            # Run in sandbox mode
make speed              # Run with Speed Mode (all 8 optimizations)
python run.py sandbox --balance 100
python -m dashboard.app  # Start web UI at http://localhost:5000

# Management
make status             # Check bot status
make stop               # Stop the bot
python run.py add 0x...  # Add trader to copy
python run.py list      # List copied traders
```

## NOTES

**CRITICAL ARCHITECTURAL ISSUES**:
- `orchestrator/engine.py` is a god class preventing maintainability
- Configuration system is fragmented across 14+ files
- Data models duplicated between `api_clients/base.py` and `sandbox/models/__init__.py`
- No centralized validation or error handling
- Tight coupling: everything depends on orchestrator

**STRENGTHS**:
- Zero anti-patterns in source code (no TODO/FIXME/HACK comments)
- 28% test coverage (480+ tests, ~90% coverage)
- Modern Python patterns (dataclasses, async, type hints)
- Modular architecture with clear domain boundaries

**GETTING STARTED**:
- See `demo_growth_first.py` for quick start example
- See `example_usage.py` for basic usage patterns
- See `README.md` (842 lines) for comprehensive documentation
- See `plan.md` for detailed project roadmap (1,546 lines)

**DEPENDENCIES**:
- Core: python>=3.9, numpy, pandas, aiohttp, yfinance, rich
- Dev: pytest>=7.0.0, pytest-asyncio>=0.20.0
- Web: flask>=2.3.0
- API: No API key required for Polymarket (free tier)

**TRADING MODES**:
- **Micro Mode**: $10-$100 accounts with 4 levels (NANO, MICRO, MINI, BALANCED)
- **Speed Mode**: 8 unified optimizations for aggressive growth
- **Conservative/Balanced/Aggressive**: Root-level presets in `config_*.py` files

**TESTING**:
- pytest with pytest-asyncio
- No pytest.ini, uses defaults
- 480+ tests across 8 modules
- Module-specific test directories (e.g., `orchestrator/tests/`, `sandbox/tests/`)

**ENVIRONMENT VARIABLES** (`.env.example`):
- `POLYMARKET_API_KEY`, `KALSHI_API_KEY` (optional)
- `DISCORD_WEBHOOK_URL`, `SLACK_WEBHOOK_URL`
- `SMTP_*` for email notifications
- `SANDBOX_INITIAL_BALANCE` (default: 10000.0)
- `LOG_LEVEL` (default: INFO)
