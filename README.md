# Goldenhar-YOLOMvX

This repository provides an experimental pipeline for **multi-class object detection** on **Goldenhar-CFID**, covering (i) **dataset processing and leakage-aware splits**, (ii) **model training across multiple seeded runs**, (iii) **comprehensive evaluation** (P/R, mAP@0.5, mAP@0.5:0.95, efficiency), and (iv) **explainability (XAI)** with both qualitative overlays and quantitative XAI metrics.

All experiments are driven by versioned YAML configurations, and outputs are exported in standardized machine-readable formats to facilitate statistical aggregation and reporting.

---

## 1. Scope and experimental design

### 1.1 Datasets
Two dataset configurations are supported:

- **D1**: Original Goldenhar-CFID dataset configuration (`configs/datasets/D1.yaml`).
- **D2**: Augmented dataset configuration (`configs/datasets/D2.yaml`), explicitly enforcing:
  - **train-only augmentation** (to avoid evaluation leakage),
  - **no-leakage validation checks**, and
  - **required label files / no empty labels** (configurable).

The class taxonomy is fixed to **7 clinical categories**:
1. Cleft Lip  
2. Epibulbar Dermoid  
3. Eyelid Coloboma  
4. Facial Asymmetry  
5. Malocclusion  
6. Microtia  
7. Vertebral Abnormalities  

### 1.2 Splits and runs
The pipeline supports three split regimes:
- **75/25**, **80/20**, **85/15** (configured under each dataset YAML),
with **4 seeded runs** by default:
- seeds: **[42, 43, 44, 45]**

Split manifests are exported as YAML under `data/splits/...` for traceability.

### 1.3 Models
Supported model families (via `configs/train/*.yaml`):
- **YOLO-MvX** (torch-based training path)
- **YOLOv9 / YOLOv10 / YOLOv11** (adapter-based path; optional dependency if used)
- **DETR** and **Swin-T** baselines (torch-based path)

---

## 2. Repository layout (high-level)

```
configs/
  datasets/        # D1, D2 dataset definitions + class taxonomy + paths
  train/           # model training configs (yolomvx, yolov9/10/11, detr, swin_t)
  eval/            # metrics, efficiency, statistical testing configuration
  reports/         # report generation configuration
  xai/             # CAM methods + XAI metric configuration

data/
  processed/       # YOLO-ready (images/labels) processed datasets
  splits/          # split manifests: runK_seedS.yaml

outputs/
  checkpoints/     # best.pt, last.pt, history.json per run
  logs/            # run logs
  metrics/
    raw/           # per-run metrics JSON (model/dataset/split/runK.json)
    aggregated/    # aggregated metrics (mean ± CI, statistical tests)
  reports/         # tables/figures/latex assets (publication-ready)
  figures/xai/     # XAI overlays (per method, per run)

src/
  datasets/        # preprocessing, augmentation, split building, loaders
  train/           # training engine + checkpoints + schedulers
  eval/            # evaluation + efficiency + stats
  reports/         # tables/plots builders
  xai/             # CAMs, rollout, occlusion, XAI metrics
  utils/           # IO, bbox, meters, device helpers

scripts/           # CLI entrypoints for end-to-end experiments
notebooks/         # audit, error analysis, and XAI inspection notebooks
```

---

## 3. Installation and environment

### 3.1 Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 (Optional) adapter-based YOLO baselines
If you plan to run YOLOv9/10/11 through an external training backend (e.g., Ultralytics), enable the optional dependency in `requirements.txt` and ensure the corresponding config fields (e.g., `data_yaml`) are set in `configs/train/yolov*.yaml`.

---

## 4. Data preparation (processed datasets)

This repository assumes **YOLO-format** directory structure:
- `images/` and `labels/` under a processed root, as specified in dataset configs.

### 4.1 Build processed dataset
```bash
bash scripts/build_processed.sh configs/datasets/D2.yaml data/processed
```

Internally, this calls:
```bash
python -m src.datasets.build_processed --dataset-config configs/datasets/D2.yaml --output-dir data/processed
```

---

## 5. Split generation (leakage-aware manifests)

### 5.1 Generate split manifests
```bash
bash scripts/make_splits.sh configs/datasets/D2.yaml 85_15 4 42,43,44,45
```

Internally, this calls:
```bash
python -m src.datasets.build_splits --dataset-config configs/datasets/D2.yaml --split-name 85_15 --runs 4 --seeds 42,43,44,45
```

Split manifests are exported as:
```
data/splits/<dataset>/<split>/runK_seedS.yaml
```
(or an equivalent structure depending on your configured base directory).

---

## 6. Training

### 6.1 Single run training (recommended for debugging)
```bash
python -m src.train.train_detector \
  --model-config configs/train/yolomvx.yaml \
  --dataset-config configs/datasets/D2.yaml \
  --split-manifest data/splits/D2/85_15/run1_seed42.yaml \
  --run 1 \
  --seed 42 \
  --output-root outputs
```

### 6.2 Batch training across runs/models
```bash
bash scripts/run_all_experiments.sh configs/datasets/D2.yaml data/splits/D2/85_15 outputs \
  configs/train/yolomvx.yaml configs/train/yolov9.yaml configs/train/yolov10.yaml configs/train/yolov11.yaml \
  configs/train/detr.yaml configs/train/swin_t.yaml
```

**Training artifacts** (per run) are stored under:
- `outputs/checkpoints/<model>/<dataset>/<split>/runK/`
  - `best.pt`, `last.pt`, `history.json`

---

## 7. Evaluation and statistical aggregation

### 7.1 Evaluate all runs for selected models
```bash
bash scripts/eval_all.sh outputs D2 85_15 yolomvx yolov9 yolov10 yolov11 detr swin_t
```

This produces per-run raw metric JSON files under:
- `outputs/metrics/raw/<model>/<dataset>/<split>/runK.json`

### 7.2 Aggregation (mean ± CI and tests)
Statistical aggregation and reporting are driven by:
- `configs/eval/metrics.yaml`
- `configs/eval/stats.yaml`
- `configs/eval/efficiency.yaml`

Depending on your configured entrypoints, aggregated artifacts are exported to:
- `outputs/metrics/aggregated/`

---

## 8. Reporting

Report generation is controlled via:
- `configs/reports/report.yaml`

Typical outputs include:
- formatted CSV tables,
- LaTeX-ready tables,
- learning curves and summary plots.

Run:
```bash
python -m src.reports.build_report --config configs/reports/report.yaml --output-root outputs
```

Outputs are exported to:
- `outputs/reports/`
- `outputs/tables/`
- `outputs/figures/`

---

## 9. Explainability (XAI)

### 9.1 Methods
Implemented XAI modules include:
- **CAM family** (Grad-CAM, Grad-CAM++, Score-CAM) for CNN-like feature maps (`src/xai/cams.py`)
- **Attention rollout** for transformer-style attention matrices (`src/xai/attention_rollout.py`)
- **Occlusion sensitivity** (model-agnostic perturbation) (`src/xai/occlusion.py`)

### 9.2 Quantitative XAI metrics
Quantitative evaluation utilities are implemented in:
- `src/xai/xai_metrics.py`

The configuration files are:
- `configs/xai/cam_methods.yaml`
- `configs/xai/xai_metrics.yaml`

### 9.3 Batch XAI execution
```bash
bash scripts/xai_all.sh
```

Expected exports:
- qualitative overlays: `outputs/figures/xai/<method>/...`
- quantitative metrics: `outputs/metrics/raw/...` and optionally `outputs/metrics/aggregated/...`

---

## 10. Checklist

- **Fixed seeds per run**: default seeds `[42, 43, 44, 45]` (configurable).
- **Split manifests are exported** as YAML to ensure deterministic partitioning.
- **Train-only augmentation** is enforced for D2 by configuration.
- **All metrics are exported per-run** as JSON for transparent aggregation.
- **Run logs and checkpoints** are saved with deterministic naming.
- **Configuration-first workflow**: no hidden hyperparameters. 

---

## 11. Notebooks

`notebooks/` contains:
- `01_data_audit.ipynb`: dataset integrity checks (labels, class distribution, leakage hints)
- `02_error_analysis.ipynb`: failure modes, per-class confusions, qualitative inspection
- `03_xai_inspection.ipynb`: XAI sanity checks and overlay inspection

---

## 12. License
See `LICENSE`.

---

## 13. Contact and contributions
For issues, please open a GitHub issue describing:
- dataset (D1/D2), split (75_25/80_20/85_15), run and seed,
- model config used,
- exact command executed,
- relevant log file under `outputs/logs/...`.
- contact me: rezaulh603@gmail.com

Pull requests are welcome for:
- adding COCO-style evaluation (pycocotools),
- decoding + NMS for detector-specific XAI scoring (YOLO-MvX),
- additional statistical tests and multiple-comparison corrections.
