#!/bin/bash -l
set -euo pipefail

PROJECT_DIR="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
LOG_DIR="${PROJECT_DIR}/codes/logs/task4/task45_30k"

mkdir -p "${LOG_DIR}" "${PROJECT_DIR}/codes/logs/task4/desilike_config" "${PROJECT_DIR}/codes/logs/task4/mplconfig" "${PROJECT_DIR}/codes/logs/task4/xdg_cache"

source ~/.bashrc >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
conda activate desilike

export PYTHONUNBUFFERED=1
export DESILIKE_CONFIG_DIR="${PROJECT_DIR}/codes/logs/task4/desilike_config"
export MPLCONFIGDIR="${PROJECT_DIR}/codes/logs/task4/mplconfig"
export XDG_CACHE_HOME="${PROJECT_DIR}/codes/logs/task4/xdg_cache"
export HDF5_USE_FILE_LOCKING=FALSE

cd "${PROJECT_DIR}"

python -u codes/task4/task45_run_free_sn0_kmax_scan_30k_parallel.py \
  --max-parallel 3 \
  --min-ncall 30000 \
  --initial-min-live 120 \
  --initial-min-ess 4000 \
  --retry-min-live 120 \
  --retry-min-ess 6000 \
  --max-ncalls 100000
