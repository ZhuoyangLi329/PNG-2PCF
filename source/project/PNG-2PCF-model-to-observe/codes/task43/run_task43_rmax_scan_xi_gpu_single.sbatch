#!/bin/bash -l
#SBATCH -J t43_rmax_xi_one
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1700M
#SBATCH -t 01:30:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rmax_scan/xi_single_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rmax_scan/xi_single_%j.err

set -euo pipefail

PROJECT_DIR="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE_DIR="${PROJECT_DIR}/codes/task43"
: "${TASK43_INDEX:?set TASK43_INDEX to a manifest row in [0,24]}"
: "${TASK43_MANIFEST:=${PROJECT_DIR}/outputs/task43_outputs/rmax_scan/manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl}"
: "${TASK43_FKP_SUMMARY:=${PROJECT_DIR}/outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz}"

if (( TASK43_INDEX < 0 || TASK43_INDEX > 24 )); then
  echo "[error] TASK43_INDEX must be in [0,24], got ${TASK43_INDEX}" >&2
  exit 2
fi

mkdir -p "${PROJECT_DIR}/codes/logs/task43/rmax_scan"
echo "[info] host=$(hostname) start=$(date -u) job=${SLURM_JOB_ID:-unknown} index=${TASK43_INDEX}"
nvidia-smi -L 2>/dev/null || true

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_NUM_THREADS=1

cd "${PROJECT_DIR}"
python -u "${CODE_DIR}/task43_measure_xi_cucount.py" \
  --manifest "${TASK43_MANIFEST}" \
  --index "${TASK43_INDEX}" \
  --s-edge-min 50 \
  --s-edge-max 550 \
  --s-edge-step 10 \
  --fkp-summary "${TASK43_FKP_SUMMARY}" \
  --p0 10000 \
  --output-tag fkpP010000 \
  --backend jax \
  --nthreads 1 \
  --cosmology DESI

echo "[info] end=$(date -u)"
