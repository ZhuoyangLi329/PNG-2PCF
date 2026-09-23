#!/bin/bash -l
#SBATCH -J t43_rmax_xi
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1700M
#SBATCH -t 00:35:00
#SBATCH --array=0-24%1
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rmax_scan/xi_array_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rmax_scan/xi_array_%A_%a.err

set -euo pipefail

PROJECT_DIR="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE_DIR="${PROJECT_DIR}/codes/task43"
: "${TASK43_MANIFEST:=${PROJECT_DIR}/outputs/task43_outputs/rmax_scan/manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl}"
: "${TASK43_FKP_SUMMARY:=${PROJECT_DIR}/outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz}"

mkdir -p "${PROJECT_DIR}/codes/logs/task43/rmax_scan"
echo "[info] host=$(hostname) start=$(date -u) job=${SLURM_JOB_ID:-unknown} index=${SLURM_ARRAY_TASK_ID}"
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
  --index "${SLURM_ARRAY_TASK_ID}" \
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
