#!/usr/bin/env bash
set -euo pipefail

# Five-realization fixed-amplitude EZmock calibration trial on a login node.
# The Python driver hard-codes Nreal=5 and FIX_AMPLITUDE=T for calibration.
# Production covariance mocks must use a separate FIX_AMPLITUDE=F workflow.

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
DESILIKE_PYTHON="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"

LABEL="${TASK43_EZMOCK_LABEL:-trial001_c1p00_e0p60_b0p22_v200}"
RHO_C="${TASK43_EZMOCK_RHO_C:-1.0}"
RHO_EXP="${TASK43_EZMOCK_RHO_EXP:-0.6}"
PDF_BASE="${TASK43_EZMOCK_PDF_BASE:-0.22}"
SIGMA_V="${TASK43_EZMOCK_SIGMA_V:-200.0}"
NGRID="${TASK43_EZMOCK_NGRID:-256}"
NTRACER="${TASK43_EZMOCK_NTRACER:-1297050}"
MESH_SIZE="${TASK43_EZMOCK_MESH_SIZE:-400}"
THREADS="${TASK43_EZMOCK_THREADS:-8}"
SEED_BASE="${TASK43_EZMOCK_SEED_BASE:-430100}"
NREAL="${TASK43_EZMOCK_NREAL:-5}"
RAND_GENERATOR="${TASK43_EZMOCK_RAND_GENERATOR:-1}"
PK_INTERP_LOG="${TASK43_EZMOCK_PK_INTERP_LOG:-T}"
INVERT_PHASE="${TASK43_EZMOCK_INVERT_PHASE:-F}"
BAO_ENHANCE="${TASK43_EZMOCK_BAO_ENHANCE:-0.0}"
ATTACH_PARTICLE="${TASK43_EZMOCK_ATTACH_PARTICLE:-F}"
CLASSIFICATION="${TASK43_EZMOCK_CLASSIFICATION:-manual_calibration}"
XI_PLOT_MAX="${TASK43_EZMOCK_XI_PLOT_MAX:-250}"
COMPARISON_LABEL="${TASK43_EZMOCK_COMPARISON_LABEL:-Manual calibration}"
SKIP_PLOT="${TASK43_EZMOCK_SKIP_PLOT:-0}"

if (( THREADS < 1 || THREADS > 8 )); then
  echo "TASK43_EZMOCK_THREADS must be in [1, 8], got ${THREADS}" >&2
  exit 2
fi
if (( NREAL < 1 || NREAL > 25 )); then
  echo "TASK43_EZMOCK_NREAL must be in [1, 25], got ${NREAL}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
"${DESILIKE_PYTHON}" -u codes/task43/task43_make_ezmock_abacus_c000_linear_pk.py

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONPATH="${DESI_CLUSTERING_ROOT}:${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS} inter_op_parallelism_threads=1"
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python -u codes/task43/task43_run_ezmock_rawbox_trial.py \
  --label "${LABEL}" \
  --rho-c "${RHO_C}" \
  --rho-exp "${RHO_EXP}" \
  --pdf-base "${PDF_BASE}" \
  --sigma-v "${SIGMA_V}" \
  --ngrid "${NGRID}" \
  --ntracer "${NTRACER}" \
  --meshsize "${MESH_SIZE}" \
  --threads "${THREADS}" \
  --seed-base "${SEED_BASE}" \
  --nreal "${NREAL}" \
  --rand-generator "${RAND_GENERATOR}" \
  --pk-interp-log "${PK_INTERP_LOG}" \
  --invert-phase "${INVERT_PHASE}" \
  --bao-enhance "${BAO_ENHANCE}" \
  --attach-particle "${ATTACH_PARTICLE}" \
  --classification "${CLASSIFICATION}"

SUMMARY="${PROJECT_ROOT}/outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13/${LABEL}/summary/${LABEL}_x${NREAL}_fixedamp_mean_2pcf_pk.npz"
if [[ "${SKIP_PLOT}" != "1" ]]; then
  python -u codes/task43/task43_plot_ezmock_rawbox_mean_clustering.py \
    --ezmock-input "${SUMMARY}" \
    --xi-plot-max "${XI_PLOT_MAX}" \
    --comparison-label "${COMPARISON_LABEL}" \
    --output-prefix "${PROJECT_ROOT}/plots/task43/ezmock_rawbox/task43_ezmock_rawbox_z0p725_mmin1p4e13_x25_mean_2pcf_pk"
fi
