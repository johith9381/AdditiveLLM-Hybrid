#!/usr/bin/env bash
# ==============================================================================
# AdditiveLLM Clean Deliverable Packager
# Excludes notebooks, virtualenvs, cache files, and creates pristine deliverable zip
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

OUTPUT_ZIP="$BASE_DIR/AdditiveLLM_Hybrid_Release.zip"

echo "========================================================================"
echo "  PACKAGING CLEAN DELIVERABLE ARCHIVE"
echo "========================================================================"

# Remove previous zip if present
rm -f "$OUTPUT_ZIP"

# Create zip excluding notebooks, venv, pycache, .DS_Store, intermediate checkpoints
zip -r "$OUTPUT_ZIP" \
    report/ \
    src/ \
    scripts/ \
    website/ \
    data/splits_group_safe/ \
    data/templates/ \
    data/README.md \
    checkpoints/pghp_ensemble/ \
    checkpoints/v2_hybrid_tierc/ \
    checkpoints/checkpoint_best.pt \
    checkpoints/physics_first_checkpoint.pt \
    results/ \
    requirements.txt \
    .env.example \
    README.md \
    -x "*.pyc" \
    -x "*__pycache__*" \
    -x "*.DS_Store" \
    -x "*.sock" \
    -x "notebooks/*" \
    -x "venv/*" \
    -x "data/raw/*" \
    -x "data/splits/*" \
    -x "logs/chrome_profile/*" \
    -x "models/checkpoints/*" \
    -x "checkpoints/checkpoint_epoch_*.pt" \
    -x "checkpoints/checkpoint_latest.pt"

echo "========================================================================"
echo "[Packager] Successfully packaged clean deliverable to:"
echo "  $OUTPUT_ZIP"
ls -lh "$OUTPUT_ZIP"
echo "========================================================================"
