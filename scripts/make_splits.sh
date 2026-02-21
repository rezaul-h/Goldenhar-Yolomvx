#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Generate split manifests for multiple runs/seeds
# Calls: python -m src.datasets.build_splits
# -----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_CFG="${1:-configs/datasets/D2.yaml}"
SPLIT_NAME="${2:-85_15}"
RUNS="${3:-4}"

# comma-separated seeds (length must be >= RUNS)
SEEDS_CSV="${4:-42,43,44,45}"

echo "[INFO] dataset_config=$DATASET_CFG"
echo "[INFO] split_name=$SPLIT_NAME"
echo "[INFO] runs=$RUNS"
echo "[INFO] seeds=$SEEDS_CSV"

$PYTHON_BIN -m src.datasets.build_splits \
  --dataset-config "$DATASET_CFG" \
  --split-name "$SPLIT_NAME" \
  --runs "$RUNS" \
  --seeds "$SEEDS_CSV"

echo "[OK] Split manifests generated under data/splits/."