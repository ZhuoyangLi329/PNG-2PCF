#!/bin/bash -l
#SBATCH -J t43_pnglc_pk
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-7
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/pngbase_pseudolc/pk_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/pngbase_pseudolc/pk_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
COSMOLOGIES=(c000 c000 c000 c000 c302 c302 c302 c302)
SPACES=(real real rsd rsd real real rsd rsd)
PHASES=(ph000 ph001 ph000 ph001 ph000 ph001 ph000 ph001)
INDEX="${SLURM_ARRAY_TASK_ID:?array index required}"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/pngbase_pseudolc_fnl0_fnl100/manifests/task43_pngbase_pseudolc_${COSMOLOGIES[INDEX]}_${SPACES[INDEX]}_x25.jsonl"
TAG="pngbase_pseudolc/${SPACES[INDEX]}/${COSMOLOGIES[INDEX]}_fkpP010000"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${PROJECT_ROOT}"

srun --exact -n 1 -c 8 --cpu-bind=cores python -u codes/task43/task43_measure_rsd_lightcone_p02_jaxpower.py \
  --manifest "${MANIFEST}" --phase "${PHASES[INDEX]}" --tag "${TAG}" \
  --meshsize 256 --kmin 0.001 --kmax 0.3001 --dk 0.002
