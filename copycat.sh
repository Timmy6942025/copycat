#!/bin/bash
# CopyCat - Quick Start Script
# Just run: ./copycat.sh sandbox

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
PYTHONPATH="$SCRIPT_DIR"
export PYTHONPATH

case "$1" in
    sandbox)
        echo "Starting CopyCat in sandbox mode..."
        $VENV_PYTHON run.py sandbox
        ;;
    speed)
        echo "Starting CopyCat with Speed Mode..."
        $VENV_PYTHON run.py speed
        ;;
    status)
        $VENV_PYTHON -m orchestrator.cli status
        ;;
    stop)
        $VENV_PYTHON -m orchestrator.cli stop
        ;;
    demo)
        $VENV_PYTHON -m orchestrator.cli demo
        ;;
    add)
        if [ -z "$2" ]; then
            echo "Usage: ./copycat.sh add 0xTRADER_ADDRESS"
            exit 1
        fi
        $VENV_PYTHON -m orchestrator.cli add-trader --trader "$2"
        ;;
    remove)
        if [ -z "$2" ]; then
            echo "Usage: ./copycat.sh remove 0xTRADER_ADDRESS"
            exit 1
        fi
        $VENV_PYTHON -m orchestrator.cli remove-trader --trader "$2"
        ;;
    list)
        $VENV_PYTHON -m orchestrator.cli list-copied
        ;;
    test)
        $VENV_PYTHON -m pytest -q
        ;;
    help|*)
        echo "CopyCat - Prediction Market Copy Trading Bot"
        echo ""
        echo "Usage: ./copycat.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  sandbox     Start in sandbox mode (paper trading)"
        echo "  speed       Start with Speed Mode (recommended)"
        echo "  status      Check bot status"
        echo "  stop        Stop the bot"
        echo "  demo        Run demo"
        echo "  add 0x...   Add trader to copy"
        echo "  remove 0x.. Remove trader from copy"
        echo "  list        List copied traders"
        echo "  test        Run tests"
        echo ""
        echo "Options:"
        echo "  ./copycat.sh sandbox --balance 100"
        echo "  ./copycat.sh speed --mode balanced"
        ;;
esac
