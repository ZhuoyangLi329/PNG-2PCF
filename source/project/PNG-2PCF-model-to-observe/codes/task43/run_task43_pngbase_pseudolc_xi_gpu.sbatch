#!/bin/bash -l
#SBATCH -J t43_pnglc_xi
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --array=0-7%2
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/pngbase_pseudolc/xi_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/pngbase_pseudolc/xi_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
COSMOLOGIES=(c000 c000 c000 c000 c302 c302 c302 c302)
SPACES=(real real rsd rsd real real rsd rsd)
PHASES=(ph000 ph001 ph000 ph001 ph000 ph001 ph000 ph001)
INDEX="${SLURM_ARRAY_TASK_ID:?array index required}"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/pngbase_pseudolc_fnl0_fnl100/manifests/task43_pngbase_pseudolc_${COSMOLOGIES[INDEX]}_${SPACES[INDEX]}_x25.jsonl"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${PROJECT_ROOT}"

srun --exact -n 1 -c 8 --cpu-bind=cores python -u codes/task43/task43_measure_rsd_lightcone_xi.py \
  --manifest "${MANIFEST}" --phase "${PHASES[INDEX]}" --nmu 40 --ells 0,2
