#!/usr/bin/env bash
# Stop AdditiveLLM Production Service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$BASE_DIR/logs"
PID_FILE="$LOG_DIR/website.pid"
USER_LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

echo "========================================================================"
echo "  STOPPING ADDITIVE LLM WEB SERVICE"
echo "========================================================================"

# 1. Unload launchd service if registered
if [ -f "$USER_LAUNCH_AGENTS/com.antigravity.additivellm.plist" ]; then
    launchctl unload "$USER_LAUNCH_AGENTS/com.antigravity.additivellm.plist" 2>/dev/null || true
    echo "[Service] Unloaded macOS launchd daemon."
fi

# 2. Terminate PID from PID file if present
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "[Service] Sending termination signal to PID $PID..."
    kill -15 "$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
fi

# 3. Clean up any remaining process listening on port 8080
PORT_PID=$(lsof -ti :8080 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
    echo "[Service] Terminating lingering process on port 8080 (PID: $PORT_PID)..."
    kill -9 $PORT_PID 2>/dev/null || true
fi

sleep 1

# 4. Verify port 8080 is released
if lsof -ti :8080 >/dev/null 2>&1; then
    echo "[Service] WARNING: Port 8080 is still busy. Please check manually."
else
    echo "[Service] SUCCESS: Service stopped and port 8080 is released."
fi
