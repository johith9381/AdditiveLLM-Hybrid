#!/usr/bin/env bash
# Stop the AdditiveLLM persistent dashboard background process
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$ROOT_DIR/dashboard.pid"

cd "$ROOT_DIR"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[Dashboard] Terminating process $PID..."
        kill "$PID"
        sleep 1
        if ps -p "$PID" > /dev/null 2>&1; then
            kill -9 "$PID"
        fi
        rm -f "$PID_FILE"
        echo "[Dashboard] Service stopped successfully."
    else
        echo "[Dashboard] Process $PID not found. Cleaning up stale PID file."
        rm -f "$PID_FILE"
    fi
else
    # Fallback: check if anything is listening on port 8000
    PORT_PID=$(lsof -ti :8000 2>/dev/null || true)
    if [ -n "$PORT_PID" ]; then
        echo "[Dashboard] Terminating process $PORT_PID listening on port 8000..."
        kill -9 "$PORT_PID"
        echo "[Dashboard] Service stopped."
    else
        echo "[Dashboard] No running dashboard process found."
    fi
fi
