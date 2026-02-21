#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Build processed dataset artifacts (YOLO-ready directories, manifests, etc.)
# Calls: python -m src.datasets.build_processed
# -----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_CFG="${1:-configs/datasets/D2.yaml}"
OUT_DIR="${2:-data/processed}"

echo "[INFO] dataset_config=$DATASET_CFG"
echo "[INFO] output_dir=$OUT_DIR"

$PYTHON_BIN -m src.datasets.build_processed \
  --dataset-config "$DATASET_CFG" \
  --output-dir "$OUT_DIR"

echo "[OK] Processed dataset prepared."