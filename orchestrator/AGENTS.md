# PROJECT KNOWLEDGE BASE - ORCHESTRATOR

**Generated:** Fri Jan 09 2026
**Commit:** Current working tree
**Branch:** main

## OVERVIEW

Main coordination layer managing all trading operations, API clients, trader identification, bot filtering, copy trading, sandbox/live modes, health checks, circuit breakers, mode transitions, and notifications.

## STRUCTURE

```
orchestrator/
├── engine.py                 # CopyCatOrchestrator (1,284 lines, GOD CLASS)
├── config.py                 # Base config (332 lines, 18 classes)
├── config_*.py               # 12 feature configs (speed, micro, tiered, momentum, events, hedging, optimizer, allocation, bootstrap, adaptive, aggressive, api)
├── mode_transition.py        # Auto mode transitions (NANO→MICRO→MINI→BALANCED)
├── notification_service.py    # Discord/Slack notifications (968 lines)
├── circuit_breaker.py        # Failure detection & recovery
├── auto_micro_mode.py        # Micro mode automation
├── sandbox_micro.py           # Sandbox micro mode runner
├── live_trading_micro.py     # Live trading micro mode runner
├── micro_validator.py        # Config validation for micro mode
├── cli.py                     # CLI interface (start/stop/status/demo)
└── tests/                     # 56 tests (test_orchestrator.py, test_sandbox_orchestrator_integration.py)
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Main coordinator | engine.py | ⚠️ GOD CLASS - 1,284 lines, 21 async functions |
| Base configuration | config.py | OrchestratorConfig, OrchestratorState, 18 dataclasses |
| Speed mode config | config_speed.py | 8 optimizations unified |
| Micro mode config | config_micro.py | NANO/MICRO/MINI/BALANCED levels |
| Mode transitions | mode_transition.py | Auto-transition with hysteresis (10% up, 20% down) |
| Notifications | notification_service.py | Discord/Slack webhooks, 968 lines |
| CLI | cli.py | start, stop, status, demo commands |
| Circuit breaker | circuit_breaker.py | 5-failure threshold, 60s timeout |
| Bootstrap traders | engine.py:57-83 | 26 hardcoded profitable addresses |

## CONVENTIONS

**Factory functions**: `create_micro_config()`, `create_speed_mode_config()`, `create_aggressive_config()`
**Circuit breaker**: Auto-stop after 5 consecutive failures with 60s cooldown
**Bootstrap traders**: 26 known profitable addresses hardcoded (lines 57-83 in engine.py)
**Mode transitions**: Balance-based with hysteresis (10% upgrade threshold, 20% downgrade threshold)
**Callback pattern**: Market data injected into sandbox via `inject_market_data()` callbacks
**Config factories**: Each feature config has factory function returning `OrchestratorConfig`
**State mutation**: Direct `self.state.copied_traders[addr] = config` throughout engine.py
**Health checks**: Periodic API client, trader data, and portfolio health validation
**Notification events**: Milestone, trade, error, recovery, mode_transition events

## ANTI-PATTERNS

❌ **God class**: `engine.py` (1,284 lines) handles EVERYTHING - API clients, trader identification, bot filtering, copy trading, sandbox/live modes, health checks, circuit breakers, mode transitions, boost mode, position sizing, market data callbacks
❌ **Config proliferation**: 12 overlapping config files (config_speed, config_micro, config_tiered, config_momentum, config_events, config_hedging, config_optimizer, config_allocation, config_bootstrap, config_adaptive, config_aggressive, config_api)
❌ **Hardcoded values**: Bootstrap traders (lines 57-83), mode thresholds, milestones directly in code
❌ **State mutation everywhere**: Direct `self.state.copied_traders[addr] = config` without encapsulation
❌ **No centralized validation**: Validation scattered across multiple validator files (micro_validator.py, etc.)
❌ **Tight coupling**: Every other module depends on orchestrator.engine
❌ **High complexity**: 21 async functions, 21 classes in single god class
