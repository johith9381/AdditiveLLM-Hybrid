#!/usr/bin/env bash
# ==============================================================================
# AdditiveLLM Master Reproduction & Execution Pipeline
# Fully verified on Apple Silicon Mac (M5 unified memory architecture)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/venv/bin/python"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo "  ADDITIVE LLM 2.0: HYBRID MULTI-EXPERT PIPELINE"
echo "  Zero-Leakage Group Splitting | 4-Expert Gating | Real-Time Inference"
echo "========================================================================"

case "$1" in
    audit)
        echo "\n>>> Running zero-leakage data partition & leakage audit..."
        "$PYTHON" src/data_pipeline.py
        "$PYTHON" src/verify_leakage.py
        ;;
    rf-baseline)
        echo "\n>>> Training and evaluating Random Forest baseline..."
        "$PYTHON" src/baseline_rf.py
        ;;
    train)
        echo "\n>>> Training 4-Expert HybridAdditiveLLM with LoRA & Physics Aux Head..."
        "$PYTHON" src/trainer.py --epochs 3 --batch-size 32 --max-train-samples 12000 --max-val-samples 3000
        ;;
    evaluate)
        echo "\n>>> Running full multi-tier multi-format evaluation protocol..."
        "$PYTHON" src/evaluate.py
        ;;
    ablations)
        echo "\n>>> Running expert branch ablation, loss sweep, and reliability analysis..."
        "$PYTHON" src/ablation_and_loss_sweep.py
        ;;
    robustness)
        echo "\n>>> Running 40-case robustness benchmark & offline fallback test..."
        "$PYTHON" src/open_domain_extractor.py
        ;;
    website-start)
        echo "\n>>> Starting always-on persistent web application daemon..."
        bash website/start.sh
        ;;
    website-stop)
        echo "\n>>> Stopping web application daemon..."
        bash website/stop.sh
        ;;
    all)
        echo "\n>>> Running end-to-end pipeline..."
        "$PYTHON" src/verify_leakage.py
        "$PYTHON" src/trainer.py --epochs 3 --batch-size 32
        "$PYTHON" src/evaluate.py
        "$PYTHON" src/ablation_and_loss_sweep.py
        "$PYTHON" src/open_domain_extractor.py
        bash website/start.sh
        echo "\n>>> Pipeline complete! All artifacts generated in results/."
        ;;
    *)
        echo "Usage: ./run_pipeline.sh {audit|rf-baseline|train|evaluate|ablations|robustness|website-start|website-stop|all}"
        exit 1
        ;;
esac
