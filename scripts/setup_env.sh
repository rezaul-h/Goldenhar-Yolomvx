#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Project setup helper (lightweight, reproducible)
# -----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[INFO] Using Python: $($PYTHON_BIN --version)"
echo "[INFO] Upgrading pip..."
$PYTHON_BIN -m pip install --upgrade pip

if [[ -f "requirements.txt" ]]; then
  echo "[INFO] Installing requirements.txt"
  $PYTHON_BIN -m pip install -r requirements.txt
else
  echo "[WARN] requirements.txt not found; skipping."
fi

echo "[OK] Environment setup complete."