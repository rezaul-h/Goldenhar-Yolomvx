#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Train multiple models across multiple runs.
# Calls: python -m src.train.train_detector
# -----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_CFG="${1:-configs/datasets/D2.yaml}"
SPLIT_DIR="${2:-data/splits/D2/85_15}"
OUTPUT_ROOT="${3:-outputs}"

# models list (space-separated model config paths)
# Example:
#   configs/train/yolov9.yaml configs/train/yolov10.yaml configs/train/yolomvx.yaml
shift 3 || true
MODEL_CFGS=("$@")

if [[ ${#MODEL_CFGS[@]} -eq 0 ]]; then
  echo "[ERROR] Provide model config paths. Example:"
  echo "  bash scripts/03_train_all.sh configs/datasets/D2.yaml data/splits/D2/85_15 outputs configs/train/yolomvx.yaml"
  exit 1
fi

echo "[INFO] dataset_config=$DATASET_CFG"
echo "[INFO] split_dir=$SPLIT_DIR"
echo "[INFO] output_root=$OUTPUT_ROOT"
echo "[INFO] model_cfgs=${MODEL_CFGS[*]}"

# Infer run files: run<k>_seed<seed>.yaml
mapfile -t SPLITS < <(ls -1 "${SPLIT_DIR}"/run*_seed*.yaml 2>/dev/null || true)
if [[ ${#SPLITS[@]} -eq 0 ]]; then
  echo "[ERROR] No split manifests found in ${SPLIT_DIR}"
  exit 1
fi

for split_manifest in "${SPLITS[@]}"; do
  base="$(basename "$split_manifest")"
  run="$(echo "$base" | sed -E 's/^run([0-9]+)_seed([0-9]+)\.yaml$/\1/')"
  seed="$(echo "$base" | sed -E 's/^run([0-9]+)_seed([0-9]+)\.yaml$/\2/')"

  for model_cfg in "${MODEL_CFGS[@]}"; do
    echo "------------------------------------------------------------"
    echo "[INFO] TRAIN: model_cfg=$model_cfg | run=$run | seed=$seed"
    $PYTHON_BIN -m src.train.train_detector \
      --model-config "$model_cfg" \
      --dataset-config "$DATASET_CFG" \
      --split-manifest "$split_manifest" \
      --run "$run" \
      --seed "$seed" \
      --output-root "$OUTPUT_ROOT"
  done
done

echo "[OK] Training completed for all specified models and runs."