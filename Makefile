# CopyCat - Makefile
# Simple commands for common tasks

VENV_PYTHON = ./venv/bin/python
PYTHON = python3
PYTHONPATH = $(CURDIR)
export PYTHONPATH

# Default target
.PHONY: help
help:
	@echo "CopyCat - Prediction Market Copy Trading Bot"
	@echo ""
	@echo "Usage: make [command]"
	@echo ""
	@echo "Quick Start:"
	@echo "  make run              # Run sandbox demo"
	@echo "  make sandbox          # Run in sandbox mode"
	@echo "  make speed            # Run with Speed Mode (recommended)"
	@echo ""
	@echo "Management:"
	@echo "  make status           # Check bot status"
	@echo "  make stop             # Stop the bot"
	@echo "  make logs             # View recent logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test             # Run all tests"
	@echo "  make test-quick       # Quick test run"
	@echo "  make demo             # Run demo script"
	@echo ""
	@echo "Development:"
	@echo "  make format           # Format code"
	@echo "  make lint             # Lint code"
	@echo "  make install-deps     # Install dependencies"
	@echo ""

# Quick start commands
.PHONY: run sandbox speed status stop logs demo test test-quick

run: sandbox
	@echo "Use 'make sandbox' to run"

sandbox:
	@echo "Starting CopyCat in sandbox mode..."
	@$(VENV_PYTHON) run.py sandbox

speed:
	@echo "Starting CopyCat with Speed Mode (all 8 optimizations)..."
	@$(VENV_PYTHON) run.py speed

status:
	@$(VENV_PYTHON) -m orchestrator.cli status

stop:
	@$(VENV_PYTHON) -m orchestrator.cli stop

logs:
	@echo "Showing recent activity..."
	@$(VENV_PYTHON) -m orchestrator.cli status

demo:
	@echo "Running demo..."
	@$(VENV_PYTHON) -m orchestrator.cli demo

# Testing
test:
	@echo "Running all tests..."
	@$(VENV_PYTHON) -m pytest -q

test-quick:
	@echo "Running quick tests..."
	@$(VENV_PYTHON) -m pytest -q --tb=no

# Development
format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then black .; else echo "black not installed"; fi

lint:
	@echo "Linting code..."
	@if command -v ruff >/dev/null 2>&1; then ruff check .; else echo "ruff not installed"; fi

install-deps:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

# Advanced
add-trader:
	@if [ -z "$(TRADER)" ]; then echo "Usage: make add-trader TRADER=0x..."; exit 1; fi
	@$(VENV_PYTHON) -m orchestrator.cli add-trader --trader $(TRADER)

remove-trader:
	@if [ -z "$(TRADER)" ]; then echo "Usage: make remove-trader TRADER=0x..."; exit 1; fi
	@$(VENV_PYTHON) -m orchestrator.cli remove-trader --trader $(TRADER)

list-traders:
	@$(VENV_PYTHON) -m orchestrator.cli list-copied

# Speed mode with options
speed-conservative:
	@echo "Running Speed Mode (conservative)..."
	@$(VENV_PYTHON) run.py speed --mode conservative

speed-balanced:
	@echo "Running Speed Mode (balanced)..."
	@$(VENV_PYTHON) run.py speed --mode balanced

speed-aggressive:
	@echo "Running Speed Mode (aggressive)..."
	@$(VENV_PYTHON) run.py speed --mode aggressive

# Custom balance
sandbox-10:
	@$(VENV_PYTHON) run.py sandbox --balance 10

sandbox-100:
	@$(VENV_PYTHON) run.py sandbox --balance 100

sandbox-1000:
	@$(VENV_PYTHON) run.py sandbox --balance 1000
