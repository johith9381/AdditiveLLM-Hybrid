#!/usr/bin/env bash
# Detached background training script for Hybrid AdditiveLLM
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

mkdir -p logs checkpoints results

export PYTHONUNBUFFERED=1

echo "[Hybrid Training] Launching background training process..."
nohup ./venv/bin/python -u src/trainer.py --epochs 3 --batch-size 32 --max-train-samples 12000 --max-val-samples 3000 > logs/training.log 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > logs/training.pid

echo "============================================================"
echo "Hybrid AdditiveLLM training running in background!"
echo "PID:      $TRAIN_PID"
echo "Log file: logs/training.log"
echo "============================================================"
