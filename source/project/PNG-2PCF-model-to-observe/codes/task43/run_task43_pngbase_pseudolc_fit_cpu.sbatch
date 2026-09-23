#!/bin/bash -l
#SBATCH -J t43_pnglc_fit
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH --array=0-1
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/pngbase_pseudolc/fit_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/pngbase_pseudolc/fit_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
COSMOLOGIES=(c000 c302)
INDEX="${SLURM_ARRAY_TASK_ID:?array index required}"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${PROJECT_ROOT}"

srun --exact -n 1 -c 8 --cpu-bind=cores python -u codes/task43/task43_run_pngbase_pseudolc_joint.py \
  --cosmology "${COSMOLOGIES[INDEX]}" --threads 8 --nwalkers 64 --nsteps 30000 --burnin 5000

