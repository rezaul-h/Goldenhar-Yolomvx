#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Evaluate trained models across runs (and export raw metrics JSON files)
# Calls: python -m src.eval.evaluate_detector
# -----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_ROOT="${1:-outputs}"
DATASET="${2:-D2}"
SPLIT="${3:-85_15}"

# model names (folder names under outputs/checkpoints and outputs/metrics/raw)
# Example: "yolov9 yolov10 yolomvx"
shift 3 || true
MODELS=("$@")

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "[ERROR] Provide model names. Example:"
  echo "  bash scripts/04_eval_all.sh outputs D2 85_15 yolomvx yolov9"
  exit 1
fi

echo "[INFO] output_root=$OUTPUT_ROOT dataset=$DATASET split=$SPLIT models=${MODELS[*]}"

for model in "${MODELS[@]}"; do
  ckpt_base="${OUTPUT_ROOT}/checkpoints/${model}/${DATASET}/${SPLIT}"
  if [[ ! -d "$ckpt_base" ]]; then
    echo "[WARN] No checkpoints found for: $model ($ckpt_base). Skipping."
    continue
  fi

  for run_dir in "${ckpt_base}"/run*; do
    [[ -d "$run_dir" ]] || continue
    run="$(basename "$run_dir" | sed -E 's/^run([0-9]+)$/\1/')"
    best_ckpt="${run_dir}/best.pt"

    if [[ ! -f "$best_ckpt" ]]; then
      echo "[WARN] Missing best.pt: $best_ckpt. Skipping run=$run."
      continue
    fi

    echo "------------------------------------------------------------"
    echo "[INFO] EVAL: model=$model run=$run ckpt=$best_ckpt"
    $PYTHON_BIN -m src.eval.evaluate_detector \
      --model "$model" \
      --weights "$best_ckpt" \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --run "$run" \
      --output-root "$OUTPUT_ROOT"
  done
done

echo "[OK] Evaluation completed."