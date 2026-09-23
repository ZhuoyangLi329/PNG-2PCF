#!/usr/bin/env bash
# Login-node pipeline for c302 fNL_cov=100 covariance, fit, and PDF-only plot.
set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODES="${PROJECT_ROOT}/codes/task43"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${CODES}:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8 inter_op_parallelism_threads=1"
cd "${PROJECT_ROOT}"

case "${1:-}" in
  covariance)
    taskset -c 6-13 python -u "${CODES}/task43_build_pngbase_c302_fnl100_covariance.py" all
    ;;
  smoke)
    taskset -c 6-13 python -u "${CODES}/task43_run_pngbase_pseudolc_joint.py" --cosmology c302 --covariance-mode c302_fnl100 --threads 8 --smoke
    ;;
  fit)
    taskset -c 6-13 python -u "${CODES}/task43_run_pngbase_pseudolc_joint.py" --cosmology c302 --covariance-mode c302_fnl100 --threads 8
    ;;
  plot)
    taskset -c 6-13 python -u "${CODES}/task43_plot_pngbase_pseudolc_joint.py" --cosmology c302 --covariance-mode c302_fnl100
    ;;
  audit)
    taskset -c 6-13 python -u "${CODES}/task43_audit_pngbase_c302_fnl100_covariance.py"
    ;;
  all)
    "$0" covariance
    "$0" smoke
    "$0" fit
    "$0" plot
    "$0" audit
    ;;
  *)
    echo "usage: $0 {covariance|smoke|fit|plot|audit|all}" >&2
    exit 2
    ;;
esac
