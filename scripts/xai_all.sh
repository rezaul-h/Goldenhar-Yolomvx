#!/usr/bin/env bash
set -euo pipefail

# =========================
# Run CAM methods and compute XAI metrics
# =========================
# Expected behavior:
# 1) generate CAM overlays / heatmaps into outputs/figures/xai/<method>/...
# 2) compute quantitative metrics into outputs/metrics/raw + aggregated
#
# Replace python calls if you use a different XAI pipeline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/figures/xai" "$OUTPUT_ROOT/metrics/raw" "$OUTPUT_ROOT/metrics/aggregated"

CAM_CFG="configs/xai/cam_methods.yaml"
XAI_METRICS_CFG="configs/xai/xai_metrics.yaml"

if [[ ! -f "$CAM_CFG" ]]; then
  echo "[ERROR] Missing CAM config: $CAM_CFG"
  exit 1
fi
if [[ ! -f "$XAI_METRICS_CFG" ]]; then
  echo "[ERROR] Missing XAI metrics config: $XAI_METRICS_CFG"
  exit 1
fi

echo "[INFO] CAM methods config : $CAM_CFG"
echo "[INFO] XAI metrics config : $XAI_METRICS_CFG"

# -------------------------
# XAI COMMANDS (replace if needed)
# -------------------------
python -m src.xai.cams --config "$CAM_CFG" --output-root "$OUTPUT_ROOT"
python -m src.xai.xai_metrics --config "$XAI_METRICS_CFG" --output-root "$OUTPUT_ROOT"

echo "[DONE] XAI generation + metrics completed."