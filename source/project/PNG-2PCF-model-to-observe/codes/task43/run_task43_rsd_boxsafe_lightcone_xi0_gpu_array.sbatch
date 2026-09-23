#!/bin/bash -l
#SBATCH -J t43_bs_xi0
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1900M
#SBATCH -t 00:15:00
#SBATCH --array=0-24%2
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/boxsafe_xi0_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/boxsafe_xi0_%A_%a.err

# box-safe 0.4<zobs<0.8 lightcone xi0 GPU array 测量
# 与 run_task43_rsd_wide_lightcone_xi0_gpu_array.sbatch 相同结构，仅换 manifest；
# ph000 已由 smoke 作业(57204192, ells 0,2)完成，会自动 [skip]

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl"
printf -v PHASE 'ph%03d' "${SLURM_ARRAY_TASK_ID}"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"
srun --exact -n 1 -c 8 --cpu-bind=cores python -u codes/task43/task43_measure_rsd_lightcone_xi.py \
  --manifest "${MANIFEST}" \
  --phase "${PHASE}" \
  --nmu 40 --ells 0
