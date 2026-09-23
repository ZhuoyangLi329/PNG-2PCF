#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/task43_outputs/ezmock_lightcone_covariance_x1000_s50_350_common50"
RUNTIME_DIR="${OUTPUT_ROOT}/logs/persistent"
mkdir -p "${RUNTIME_DIR}"

exec 9>"${RUNTIME_DIR}/finalizer.lock"
if ! flock -n 9; then
  echo "[$(date -u +%FT%TZ)] another finalizer holds ${RUNTIME_DIR}/finalizer.lock" >&2
  exit 3
fi

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="/pscratch/sd/l/lzy/desi-clustering:${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT_ROOT}"
exec python -u codes/task43/task43_finalize_ezmock_covariance.py --wait --poll-seconds 60 --overwrite

