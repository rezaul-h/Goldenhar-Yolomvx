#!/usr/bin/env bash
set -euo pipefail

# =========================
# Run baselines only
# (YOLOv9/10/11 + DETR + Swin-T)
# across D1/D2, all splits, runs 1..4
# =========================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"

DATASETS=("D1" "D2")
SPLITS=("75_25" "80_20" "85_15")
RUNS=("1" "2" "3" "4")
SEEDS=("42" "43" "44" "45")

MODELS=("yolov9" "yolov10" "yolov11" "detr" "swin_t")

declare -A MODEL_CFG
MODEL_CFG["yolov9"]="configs/train/yolov9.yaml"
MODEL_CFG["yolov10"]="configs/train/yolov10.yaml"
MODEL_CFG["yolov11"]="configs/train/yolov11.yaml"
MODEL_CFG["detr"]="configs/train/detr.yaml"
MODEL_CFG["swin_t"]="configs/train/swin_t.yaml"

declare -A DATASET_CFG
DATASET_CFG["D1"]="configs/datasets/D1.yaml"
DATASET_CFG["D2"]="configs/datasets/D2.yaml"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/checkpoints" "$OUTPUT_ROOT/metrics/raw"

for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for idx in "${!RUNS[@]}"; do
      run="${RUNS[$idx]}"
      seed="${SEEDS[$idx]}"
      split_manifest="data/splits/split_${split}/run${run}_seed${seed}.yaml"

      if [[ ! -f "$split_manifest" ]]; then
        echo "[WARN] Missing split manifest: $split_manifest"
        continue
      fi

      for model in "${MODELS[@]}"; do
        cfg="${MODEL_CFG[$model]}"
        dcfg="${DATASET_CFG[$dataset]}"

        if [[ ! -f "$cfg" || ! -f "$dcfg" ]]; then
          echo "[WARN] Missing config(s) for model=$model dataset=$dataset"
          continue
        fi

        log_dir="$OUTPUT_ROOT/logs/${model}/${dataset}/${split}"
        mkdir -p "$log_dir"
        log_file="$log_dir/run${run}.log"

        echo "[RUN] baseline=$model dataset=$dataset split=$split run=$run seed=$seed"
        python -m src.train.train_detector \
          --model-config "$cfg" \
          --data-config "$dcfg" \
          --split-manifest "$split_manifest" \
          --run "$run" \
          --seed "$seed" \
          2>&1 | tee "$log_file"
      done
    done
  done
done

echo "[DONE] Baseline experiments completed."