#!/usr/bin/env bash
# Login-node entry point for the Task 4.3 smin=40 BAO-mask sensitivity test.

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
PYTHON="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
THREADS="${TASK43_THREADS:-8}"
if [[ "${THREADS}" -lt 1 || "${THREADS}" -gt 8 ]]; then
  echo "TASK43_THREADS must be in [1,8]" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
cd "${PROJECT_ROOT}"

case "${1:-}" in
  build-real-cov)
    set +u
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    set -u
    taskset -c 6-7 python -u codes/task43/task43_build_real_joint_cov_s30_v1.py
    ;;
  smoke)
    taskset -c 6-13 "${PYTHON}" -u codes/task43/task43_run_lightcone_joint_baomask_smin40_v1.py --smoke --threads "${THREADS}"
    ;;
  fit)
    taskset -c 6-13 "${PYTHON}" -u codes/task43/task43_run_lightcone_joint_baomask_smin40_v1.py --threads "${THREADS}"
    ;;
  plot)
    taskset -c 6-7 "${PYTHON}" -u codes/task43/task43_plot_lightcone_smin40_vs50_v1.py
    ;;
  plot-contours)
    taskset -c 6-7 "${PYTHON}" -u codes/task43/task43_plot_lightcone_smin40_contours_v1.py
    ;;
  *)
    echo "usage: $0 {build-real-cov|smoke|fit|plot|plot-contours}" >&2
    exit 2
    ;;
esac
