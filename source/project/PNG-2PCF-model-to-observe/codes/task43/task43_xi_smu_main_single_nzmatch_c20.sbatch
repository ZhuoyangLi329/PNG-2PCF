#!/bin/bash -l
#SBATCH -J t43_xi_nz20
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem=4G
#SBATCH -t 00:07:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/logs/xi_main_single_nzmatch_c20_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/logs/xi_main_single_nzmatch_c20_%j.err
set -euo pipefail
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
ROOT=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60
MANIFEST=$ROOT/manifests/task43_ezmock_rsd_covariance_x1000_fixampF_nzmatch_b0.30_z0.60.jsonl
INDEX=${1:?index required}
cd "$PROJECT_ROOT"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="$PROJECT_ROOT/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=20 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" XLA_PYTHON_CLIENT_PREALLOCATE=false
/usr/bin/time -v python -u codes/task43/task43_measure_ezmock_rsd_lightcone_xi_fcfc.py --manifest "$MANIFEST" --index "$INDEX" --threads 20
