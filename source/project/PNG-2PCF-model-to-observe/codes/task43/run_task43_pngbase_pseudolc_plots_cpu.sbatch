#!/bin/bash -l
#SBATCH -J t43_pnglc_plot
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/pngbase_pseudolc/plot_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/pngbase_pseudolc/plot_%j.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${PROJECT_ROOT}"

python -u codes/task43/task43_plot_pngbase_pseudolc_joint.py --cosmology c000
python -u codes/task43/task43_plot_pngbase_pseudolc_joint.py --cosmology c302
python -u codes/task43/task43_plot_pngbase_paired_response.py
python -u codes/task43/task43_audit_pngbase_pseudolc_experiment.py
