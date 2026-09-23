#!/usr/bin/env bash
set -euo pipefail

# CPU-only Task43 lightcone P(k) runner for login nodes.
# Usage examples:
#   bash codes/task43/run_task43_pk_cpu_login.sh smoke
#   bash codes/task43/run_task43_pk_cpu_login.sh measure_phase 0
#   bash codes/task43/run_task43_pk_cpu_login.sh window_phase 0
#   bash codes/task43/run_task43_pk_cpu_login.sh cov 0
#   bash codes/task43/run_task43_pk_cpu_login.sh summarize
#   bash codes/task43/run_task43_pk_cpu_login.sh summarize_desi
#   bash codes/task43/run_task43_pk_cpu_login.sh fit_free
#   bash codes/task43/run_task43_pk_cpu_login.sh fit_free_desi

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONPATH="${DESI_CLUSTERING_ROOT}:${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""

TASK43_PK_THREADS="${TASK43_PK_THREADS:-8}"
if [[ "${TASK43_PK_THREADS}" -gt 8 ]]; then
  echo "TASK43_PK_THREADS=${TASK43_PK_THREADS} exceeds the Task43 P0 rerun limit of 8" >&2
  exit 2
fi

export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${TASK43_PK_THREADS} inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "${PROJECT_ROOT}"

MODE="${1:-smoke}"
TAG="${TASK43_PK_TAG:-mmin1p4e13_x25_fkpP010000}"
P0="${TASK43_PK_P0:-10000}"
MEASURE_MESH="${TASK43_PK_MEASURE_MESH:-256}"
COV_MESH="${TASK43_PK_COV_MESH:-128}"
MESH_PAD="${TASK43_PK_MESH_PAD:-400}"
KMIN_GRID="${TASK43_PK_KMIN_GRID:-0.001}"
KMAX_GRID="${TASK43_PK_KMAX_GRID:-0.3001}"
DK_GRID="${TASK43_PK_DK_GRID:-0.002}"
WINDOW_METHOD="${TASK43_PK_WINDOW_METHOD:-smooth}"
FIT_EVALS="${TASK43_PK_FIT_EVALS:-120000}"
FIT_WALKERS="${TASK43_PK_FIT_WALKERS:-18}"

case "${MODE}" in
  smoke)
    python -u codes/task43/task43_measure_pk_jaxpower.py \
      --index 0 --tag smoke_window --p0 "${P0}" --meshsize 32 --mesh-pad 200 \
      --kmin 0.001 --kmax 0.0501 --dk 0.01 \
      --max-data 2000 --max-random 8000 --rescale-subsample \
      --window-method smooth --overwrite
    python -u codes/task43/task43_make_pk_covariance_jaxpower.py \
      --index 0 --tag smoke_window --p0 "${P0}" --meshsize 32 --mesh-pad 200 \
      --kmin 0.001 --kmax 0.0501 --dk 0.01 \
      --max-data 2000 --max-random 8000 --overwrite
    python -u codes/task43/task43_summarize_pk_lightcone.py \
      --tag smoke_window --p0 "${P0}" --output-tag smoke_window_cov --overwrite
    python -u codes/task43/task43_fit_pk_lightcone.py \
      --payload outputs/task43_outputs/pk_lightcone/summary/task43_pk_lightcone_smoke_window_cov_payload.npz \
      --label smoke_window_cov_free_sn0 --target-evals 720 --nwalkers 12 --overwrite
    ;;
  measure_phase)
    INDEX="${2:?provide phase index 0..24}"
    python -u codes/task43/task43_measure_pk_jaxpower.py \
      --index "${INDEX}" --tag "${TAG}" --p0 "${P0}" --meshsize "${MEASURE_MESH}" --mesh-pad "${MESH_PAD}" \
      --kmin "${KMIN_GRID}" --kmax "${KMAX_GRID}" --dk "${DK_GRID}" --overwrite
    ;;
  window_phase)
    INDEX="${2:-0}"
    python -u codes/task43/task43_measure_pk_jaxpower.py \
      --index "${INDEX}" --tag "${TAG}" --p0 "${P0}" --meshsize "${MEASURE_MESH}" --mesh-pad "${MESH_PAD}" \
      --kmin "${KMIN_GRID}" --kmax "${KMAX_GRID}" --dk "${DK_GRID}" \
      --window-method "${WINDOW_METHOD}" --overwrite
    ;;
  measure_all)
    for INDEX in $(seq 0 24); do
      bash "$0" measure_phase "${INDEX}"
    done
    ;;
  cov)
    INDEX="${2:-0}"
    python -u codes/task43/task43_make_pk_covariance_jaxpower.py \
      --index "${INDEX}" --tag "${TAG}" --p0 "${P0}" --meshsize "${COV_MESH}" --mesh-pad "${MESH_PAD}" \
      --kmin "${KMIN_GRID}" --kmax "${KMAX_GRID}" --dk "${DK_GRID}" --overwrite
    ;;
  summarize)
    python -u codes/task43/task43_summarize_pk_lightcone.py \
      --tag "${TAG}" --p0 "${P0}" --output-tag "${TAG}" --overwrite
    ;;
  summarize_desi)
    python -u codes/task43/task43_summarize_pk_lightcone.py \
      --tag "${TAG}" --p0 "${P0}" --output-tag "${TAG}_desi_rebin" \
      --fit-bin-policy desi_png --overwrite
    ;;
  fit_free)
    python -u codes/task43/task43_fit_pk_lightcone.py \
      --payload "outputs/task43_outputs/pk_lightcone/summary/task43_pk_lightcone_${TAG}_payload.npz" \
      --label "${TAG}_free_sn0" --sn0-policy free \
      --target-evals "${FIT_EVALS}" --nwalkers "${FIT_WALKERS}" --overwrite
    ;;
  fit_free_desi)
    python -u codes/task43/task43_fit_pk_lightcone.py \
      --payload "outputs/task43_outputs/pk_lightcone/summary/task43_pk_lightcone_${TAG}_desi_rebin_payload.npz" \
      --label "${TAG}_desi_rebin_free_sn0" --sn0-policy free \
      --target-evals "${FIT_EVALS}" --nwalkers "${FIT_WALKERS}" --overwrite
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
