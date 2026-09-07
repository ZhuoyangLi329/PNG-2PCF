#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_x25.jsonl"
THREADS="${TASK43_RSD_PK0_THREADS:-8}"
TARGET="${1:-ph000}"

if (( THREADS < 1 || THREADS > 8 )); then
  echo "TASK43_RSD_PK0_THREADS must be in [1, 8], got ${THREADS}" >&2
  exit 2
fi

if [[ "${TARGET}" == "all" ]]; then
  PHASES=()
  for INDEX in $(seq 0 24); do PHASES+=("$(printf 'ph%03d' "${INDEX}")"); done
else
  PHASES=("${TARGET}")
fi

cd "${PROJECT_ROOT}"
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
  python -u codes/task43/task43_measure_rsd_rawbox_pk0_jaxpower.py \
    --manifest "${MANIFEST}" --phase "${PHASE}" --threads "${THREADS}"
done
