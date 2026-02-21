#!/usr/bin/env bash
set -euo pipefail

# =========================
# Focused runner:
#  YOLO-MvX on D2 split 85/15 across runs 1..4
# =========================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"

MODEL="yolomvx"
DATASET="D2"
SPLIT="85_15"

MODEL_CFG="configs/train/yolomvx.yaml"
DATASET_CFG="configs/datasets/D2.yaml"

RUNS=("1" "2" "3" "4")
SEEDS=("42" "43" "44" "45")

mkdir -p "$OUTPUT_ROOT/logs/${MODEL}/${DATASET}/${SPLIT}"

if [[ ! -f "$MODEL_CFG" ]]; then
  echo "[ERROR] Missing model config: $MODEL_CFG"
  exit 1
fi
if [[ ! -f "$DATASET_CFG" ]]; then
  echo "[ERROR] Missing dataset config: $DATASET_CFG"
  exit 1
fi

for idx in "${!RUNS[@]}"; do
  run="${RUNS[$idx]}"
  seed="${SEEDS[$idx]}"
  split_manifest="data/splits/split_${SPLIT}/run${run}_seed${seed}.yaml"

  if [[ ! -f "$split_manifest" ]]; then
    echo "[WARN] Missing split manifest: $split_manifest"
    continue
  fi

  log_file="$OUTPUT_ROOT/logs/${MODEL}/${DATASET}/${SPLIT}/run${run}.log"
  echo "[RUN] $MODEL $DATASET $SPLIT run=$run seed=$seed"
  python -m src.train.train_detector \
    --model-config "$MODEL_CFG" \
    --data-config "$DATASET_CFG" \
    --split-manifest "$split_manifest" \
    --run "$run" \
    --seed "$seed" \
    2>&1 | tee "$log_file"
done

echo "[DONE] YOLO-MvX on D2 (85/15) finished."