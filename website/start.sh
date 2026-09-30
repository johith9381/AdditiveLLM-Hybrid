#!/usr/bin/env bash
# Start AdditiveLLM Production Service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$BASE_DIR/logs"
PID_FILE="$LOG_DIR/website.pid"
PLIST_FILE="$SCRIPT_DIR/service/com.antigravity.additivellm.plist"
USER_LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "  STARTING ADDITIVE LLM ALWAYS-ON PRODUCTION WEB SERVICE"
echo "========================================================================"

# 1. Quick check: Is backend ALREADY responding to health checks?
if curl -s -m 2 http://127.0.0.1:8080/api/health 2>/dev/null | grep -q '"status":"healthy"'; then
    echo "[Service] AdditiveLLM backend is ALREADY LIVE AND HEALTHY at http://localhost:8080"
    exit 0
fi

# 2. Check if an existing stale process is holding port 8080 and clear it
EXISTING_PORT_PID=$(lsof -ti :8080 2>/dev/null || true)
if [ -n "$EXISTING_PORT_PID" ]; then
    echo "[Service] Port 8080 is occupied by PID $EXISTING_PORT_PID. Cleaning up stale process..."
    kill -15 $EXISTING_PORT_PID 2>/dev/null || kill -9 $EXISTING_PORT_PID 2>/dev/null || true
    sleep 1
fi

# 3. Attempt launchd registration if user LaunchAgents directory exists and writable
if [ -d "$USER_LAUNCH_AGENTS" ]; then
    if cp "$PLIST_FILE" "$USER_LAUNCH_AGENTS/" 2>/dev/null; then
        launchctl unload "$USER_LAUNCH_AGENTS/com.antigravity.additivellm.plist" 2>/dev/null || true
        launchctl load "$USER_LAUNCH_AGENTS/com.antigravity.additivellm.plist" 2>/dev/null || true
        echo "[Service] Registered with macOS launchd daemon ($USER_LAUNCH_AGENTS/com.antigravity.additivellm.plist)"
    fi
fi

# 4. Launch daemon process
echo "[Service] Starting Uvicorn backend process..."
nohup "$BASE_DIR/venv/bin/python" -m uvicorn website.backend:app --host 0.0.0.0 --port 8080 > "$LOG_DIR/website.log" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "[Service] Loading 4-Expert models and tokenizers (PID: $NEW_PID)..."

# 5. Wait up to 30 seconds for health check to confirm server initialization
for i in {1..30}; do
    if curl -s -m 2 http://127.0.0.1:8080/api/health 2>/dev/null | grep -q '"status":"healthy"'; then
        echo "========================================================================"
        echo "  [SUCCESS] AdditiveLLM Web Service is LIVE at http://localhost:8080"
        echo "  - Process PID:    $NEW_PID (saved to $PID_FILE)"
        echo "  - Logs:           $LOG_DIR/website.log"
        echo "========================================================================"
        exit 0
    fi
    sleep 1
done

# If health check didn't respond in 30 seconds, report log tail
echo "[Service] WARNING: Health check wait timed out. Checking server log tail:"
tail -n 15 "$LOG_DIR/website.log"
