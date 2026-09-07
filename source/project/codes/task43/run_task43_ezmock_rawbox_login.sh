#!/usr/bin/env bash
set -euo pipefail

# Serial CPU-only raw-box target measurements for EZmock calibration.
# Usage:
#   bash codes/task43/run_task43_ezmock_rawbox_login.sh ph000
#   bash codes/task43/run_task43_ezmock_rawbox_login.sh all

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
KIRISAME_PYTHON="/global/homes/l/lzy/anaconda3/envs/kirisame/bin/python"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"
THREADS="${TASK43_RAWBOX_THREADS:-8}"
MESH_SIZE="${TASK43_RAWBOX_MESH_SIZE:-400}"

if (( THREADS < 1 || THREADS > 8 )); then
  echo "TASK43_RAWBOX_THREADS must be in [1, 8], got ${THREADS}" >&2
  exit 2
fi

TARGET="${1:-ph000}"
if [[ "${TARGET}" == "all" ]]; then
  PHASES=()
  for INDEX in $(seq 0 24); do PHASES+=("$(printf 'ph%03d' "${INDEX}")"); done
else
  PHASES=("${TARGET}")
fi

cd "${PROJECT_ROOT}"
for PHASE in "${PHASES[@]}"; do
  "${KIRISAME_PYTHON}" -u codes/task43/task43_build_ezmock_rawbox_catalog.py --phase "${PHASE}"
done

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONPATH="${DESI_CLUSTERING_ROOT}:${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS} inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

for PHASE in "${PHASES[@]}"; do
  python -u codes/task43/task43_measure_ezmock_rawbox_pk_jaxpower.py \
    --phase "${PHASE}" --meshsize "${MESH_SIZE}" --threads "${THREADS}"
  python -u codes/task43/task43_measure_ezmock_rawbox_xi_fcfc.py \
    --phase "${PHASE}" --threads "${THREADS}"
done

if [[ "${TARGET}" == "all" ]]; then
  python -u codes/task43/task43_summarize_ezmock_rawbox_clustering.py --meshsize "${MESH_SIZE}"
else
  python -u codes/task43/task43_summarize_ezmock_rawbox_clustering.py \
    --meshsize "${MESH_SIZE}" --allow-partial \
    --output-prefix "outputs/task43_outputs/ezmock_rawbox_z0p725_mmin1p4e13/summary/task43_ezmock_rawbox_z0p725_mmin1p4e13_partial_${TARGET}"
fi
