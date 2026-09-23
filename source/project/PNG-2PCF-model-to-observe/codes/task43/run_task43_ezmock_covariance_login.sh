#!/usr/bin/env bash
set -euo pipefail

# Authoritative serial login-node launcher for the 1000-row Task43 covariance.
# Usage: run_task43_ezmock_covariance_login.sh START STOP [all|lightcone|lightcone_xi|xi|pk] [wait-for-lightcone:0|1]

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"
THREADS="${TASK43_EZCOV_THREADS:-8}"
START="${1:-0}"
STOP="${2:-1000}"
STAGES="${3:-all}"
WAIT_FOR_LIGHTCONE="${4:-0}"

if (( THREADS < 1 || THREADS > 12 )); then
  echo "TASK43_EZCOV_THREADS must be in [1,12], got ${THREADS}" >&2
  exit 2
fi

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONPATH="${DESI_CLUSTERING_ROOT}:${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS} inter_op_parallelism_threads=1"

cd "${PROJECT_ROOT}"
EXTRA_ARGS=()
if [[ "${WAIT_FOR_LIGHTCONE}" == "1" ]]; then
  EXTRA_ARGS+=(--wait-for-lightcone)
fi
exec python -u codes/task43/task43_run_ezmock_covariance_login.py \
  --start "${START}" --stop "${STOP}" --threads "${THREADS}" --stages "${STAGES}" "${EXTRA_ARGS[@]}"
