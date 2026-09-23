#!/bin/bash -l
#SBATCH -J t43_xi_smu_x10
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --array=0-9%2
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/xi_smu_x10_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/xi_smu_x10_%A_%a.err
set -euo pipefail
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
cd "$PROJECT_ROOT"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="$PROJECT_ROOT/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MANIFEST=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/manifests/task43_ezmock_rsd_covariance_x10_fixampF_common50.jsonl
/usr/bin/time -v python -u codes/task43/task43_measure_ezmock_rsd_lightcone_xi_fcfc.py --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --threads 64
