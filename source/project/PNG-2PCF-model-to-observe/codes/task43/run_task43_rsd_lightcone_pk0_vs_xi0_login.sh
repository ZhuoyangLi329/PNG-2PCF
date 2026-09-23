#!/usr/bin/env bash
set -euo pipefail

# Task 4.3.2 RSD lightcone P0, with the frozen Task 4.3 real-space P(k)
# estimator/window/kmin contract.  Usage:
#   bash codes/task43/run_task43_rsd_lightcone_pk0_vs_xi0_login.sh smoke
#   bash codes/task43/run_task43_rsd_lightcone_pk0_vs_xi0_login.sh measure_phase 0
#   bash codes/task43/run_task43_rsd_lightcone_pk0_vs_xi0_login.sh measure_all
#   bash codes/task43/run_task43_rsd_lightcone_pk0_vs_xi0_login.sh covariance
#   bash codes/task43/run_task43_rsd_lightcone_pk0_vs_xi0_login.sh summarize
#   bash codes/task43/run_task43_rsd_lightcone_pk0_vs_xi0_login.sh fit

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_x25.jsonl"
THREADS="${TASK43_RSD_LIGHTCONE_PK0_THREADS:-8}"

if (( THREADS < 1 || THREADS > 8 )); then
  echo "TASK43_RSD_LIGHTCONE_PK0_THREADS must be in [1, 8], got ${THREADS}" >&2
  exit 2
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

MODE="${1:-smoke}"
MEASURE_TAG="${TASK43_RSD_LIGHTCONE_PK0_TAG:-x25_fkpP010000}"
COV_TAG="${TASK43_RSD_LIGHTCONE_PK0_COV_TAG:-x25_fkpP010000_fnlcov0_b1cov2p604_sigmas7p566}"
SMOKE_TAG="smoke_task43realspace_p0blockv2"

measure_phase() {
  local index="$1"
  local phase
  phase="$(printf 'ph%03d' "${index}")"
  local extra=()
  if (( index == 0 )); then
    extra=(--window-method smooth)
  fi
  python -u codes/task43/task43_measure_rsd_lightcone_pk0_jaxpower.py \
    --manifest "${MANIFEST}" --phase "${phase}" --tag "${MEASURE_TAG}" \
    --meshsize 256 --mesh-pad 400 --kmin 0.001 --kmax 0.3001 --dk 0.002 \
    "${extra[@]}"
}

case "${MODE}" in
  smoke)
    python -m pytest -q codes/task43/test_task43_rsd_lightcone_pk0.py
    python -u codes/task43/task43_measure_rsd_lightcone_pk0_jaxpower.py \
      --manifest "${MANIFEST}" --phase ph000 --tag "${SMOKE_TAG}" \
      --meshsize 32 --mesh-pad 200 --kmin 0.001 --kmax 0.0501 --dk 0.01 \
      --max-data 2000 --max-random 8000 --rescale-subsample --window-method smooth
    python -u codes/task43/task43_make_rsd_lightcone_pk0_covariance_jaxpower.py \
      --manifest "${MANIFEST}" --phase ph000 --tag "${SMOKE_TAG}" \
      --meshsize 32 --mesh-pad 200 --kmin 0.001 --kmax 0.0501 --dk 0.01 \
      --max-data 2000 --max-random 8000
    ;;
  measure_phase)
    INDEX="${2:?provide phase index 0..24}"
    if (( INDEX < 0 || INDEX > 24 )); then
      echo "phase index must be in [0, 24], got ${INDEX}" >&2
      exit 2
    fi
    measure_phase "${INDEX}"
    ;;
  measure_all)
    for INDEX in $(seq 0 24); do
      measure_phase "${INDEX}"
    done
    ;;
  covariance)
    python -u codes/task43/task43_make_rsd_lightcone_pk0_covariance_jaxpower.py \
      --manifest "${MANIFEST}" --phase ph000 --tag "${COV_TAG}" \
      --meshsize 128 --mesh-pad 400 --kmin 0.001 --kmax 0.3001 --dk 0.002 \
      --max-data 50000 --max-random 100000
    ;;
  summarize)
    python -u codes/task43/task43_summarize_rsd_lightcone_pk0.py
    ;;
  fit)
    python -u codes/task43/task43_fit_rsd_lightcone_pk0_vs_xi0_smin50.py \
      --threads "${THREADS}" --nwalkers 64 --nsteps 30000 --burnin 5000
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
