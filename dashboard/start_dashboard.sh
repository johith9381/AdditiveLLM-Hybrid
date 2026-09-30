#!/usr/bin/env bash
# Start the AdditiveLLM persistent dashboard background process
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$ROOT_DIR/dashboard.pid"
LOG_FILE="$ROOT_DIR/dashboard.log"
PORT=8000

cd "$ROOT_DIR"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[Dashboard] Service is already running with PID $PID on http://localhost:$PORT"
        exit 0
    else
        echo "[Dashboard] Removing stale PID file."
        rm -f "$PID_FILE"
    fi
fi

echo "[Dashboard] Launching persistent dashboard service on port $PORT..."
nohup ./venv/bin/uvicorn dashboard.app:app --host 127.0.0.1 --port $PORT > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

# Wait up to 5 seconds for health check
echo "[Dashboard] Verifying service health..."
for i in {1..5}; do
    sleep 1
    if curl -s "http://127.0.0.1:$PORT/api/health" | grep -q "healthy"; then
        echo "============================================================"
        echo "SUCCESS: AdditiveLLM Dashboard is running in the background!"
        echo "Process ID:    $NEW_PID"
        echo "Dashboard URL: http://localhost:$PORT"
        echo "Log File:      $LOG_FILE"
        echo "============================================================"
        exit 0
    fi
done

echo "[Dashboard] Warning: Health check did not respond within 5 seconds. Check $LOG_FILE for details."
