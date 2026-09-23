#!/bin/bash -l
#SBATCH -J t43_rsd_xi
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1900M
#SBATCH -t 01:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/lightcone_xi_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/lightcone_xi_%j.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_x25.jsonl"
TASK_PHASE="${1:-ph000}"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"
srun --exact -n 1 -c 8 --cpu-bind=cores python -u codes/task43/task43_measure_rsd_lightcone_xi.py \
  --manifest "${MANIFEST}" \
  --phase "${TASK_PHASE}" \
  --nmu 40
