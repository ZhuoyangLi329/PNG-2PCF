#!/bin/bash -l
#SBATCH -J t44_png_xi_cu
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task44/pngbase_xi_cucount_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task44/pngbase_xi_cucount_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE_DIR="${PROJECT_ROOT}/codes/task44"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_INDEX}" in
  0) TAG="c300" ;;
  1) TAG="c302" ;;
  *) echo "[error] array index must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "${PROJECT_ROOT}/codes/logs/task44"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"
echo "[info] host=$(hostname) job=${SLURM_JOB_ID:-unknown} array=${TASK_INDEX} tag=${TAG} start=$(date -u)"
nvidia-smi -L
srun --exact -n 1 -c 32 --cpu-bind=cores \
  python -u "${CODE_DIR}/task44_validate_pngbase_hodmap_xi_engines.py" \
  cucount --tag "${TAG}"
echo "[info] end=$(date -u)"
