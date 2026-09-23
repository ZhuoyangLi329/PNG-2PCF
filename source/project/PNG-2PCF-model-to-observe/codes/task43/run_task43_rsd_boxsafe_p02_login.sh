#!/bin/bash
# Boxsafe RSD P0+P2 quadrupole measurement (Task 4.3.2), login-node serial.
# 8 CPUs, one phase at a time, window (smooth) only on ph000.
set -eo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODES="${PROJECT_ROOT}/codes/task43"
LOG_DIR="${PROJECT_ROOT}/codes/logs/task43/rsd_p02_increment"
mkdir -p "${LOG_DIR}"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="/pscratch/sd/l/lzy/desi-clustering:${CODES}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6 inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

stage="${1:-help}"
case "${stage}" in
  measure)
    for index in $(seq 0 24); do
      phase="$(printf 'ph%03d' "${index}")"
      extra=()
      if (( index == 0 )); then
        extra=(--window-method smooth)
      fi
      echo "[p02-measure] ${phase}" >&2
      taskset -c 6-11 python -u "${CODES}/task43_measure_rsd_lightcone_p02_jaxpower.py" \
        --phase "${phase}" "${extra[@]}" 2>&1 | tee -a "${LOG_DIR}/measure.log"
    done
    ;;
  fit)
    taskset -c 6-11 python -u "${CODES}/task43_rsd_boxsafe_p02_increment.py" 2>&1 | tee "${LOG_DIR}/fit.log"
    ;;
  *)
    echo "usage: $0 {measure|fit}" >&2
    exit 1
    ;;
esac
