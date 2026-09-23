#!/usr/bin/env bash
set -euo pipefail

# CPU-only Task43 RascalC kmin/FullDiscrete runner for a login node.
#
# Examples:
#   bash codes/task43/run_task43_rascalc_kmin_cpu_login.sh build_xi
#   bash codes/task43/run_task43_rascalc_kmin_cpu_login.sh smoke_discrete
#   bash codes/task43/run_task43_rascalc_kmin_cpu_login.sh smoke_ab
#
# The physical data normalization is always read from the full ph000 catalog.
# TASK43_RASCALC_MAX_RANDOM changes only the RascalC Monte-Carlo geometry sample.

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
RASCALC_SOURCE="/global/common/software/desi/users/mrash/RascalC"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RASCALC_SOURCE}:${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

TASK43_RASCALC_NTHREADS="${TASK43_RASCALC_NTHREADS:-8}"
TASK43_RASCALC_CPUSET="${TASK43_RASCALC_CPUSET:-6-13}"
TASK43_RASCALC_MAX_RANDOM="${TASK43_RASCALC_MAX_RANDOM:-300000}"
TASK43_RASCALC_NLOOPS="${TASK43_RASCALC_NLOOPS:-128}"
TASK43_RASCALC_LOOPS_PER_SAMPLE="${TASK43_RASCALC_LOOPS_PER_SAMPLE:-8}"
TASK43_RASCALC_SEED="${TASK43_RASCALC_SEED:-20260709}"
TASK43_RASCALC_RANDOM_SEED="${TASK43_RASCALC_RANDOM_SEED:-20260709}"

if [[ "${TASK43_RASCALC_NTHREADS}" -lt 1 || "${TASK43_RASCALC_NTHREADS}" -gt 8 ]]; then
  echo "TASK43_RASCALC_NTHREADS must be in 1..8" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

run_covariance() {
  local xi_model="$1"
  taskset -c "${TASK43_RASCALC_CPUSET}" python -u codes/task43/task43_run_rascalc_lightcone_kmin.py \
    --xi-model "${xi_model}" \
    --max-random "${TASK43_RASCALC_MAX_RANDOM}" \
    --nthreads "${TASK43_RASCALC_NTHREADS}" \
    --n-loops "${TASK43_RASCALC_NLOOPS}" \
    --loops-per-sample "${TASK43_RASCALC_LOOPS_PER_SAMPLE}" \
    --seed "${TASK43_RASCALC_SEED}" \
    --random-seed "${TASK43_RASCALC_RANDOM_SEED}"
}

MODE="${1:-smoke_discrete}"
case "${MODE}" in
  build_xi)
    taskset -c "${TASK43_RASCALC_CPUSET}" python -u codes/task43/task43_build_rascalc_kmin_xi.py
    ;;
  smoke_discrete)
    run_covariance full_discrete
    ;;
  smoke_boxcut)
    run_covariance continuous_boxcut
    ;;
  smoke_lowk)
    run_covariance continuous_lowk
    ;;
  smoke_ab)
    run_covariance full_discrete
    run_covariance continuous_boxcut
    run_covariance continuous_lowk
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
