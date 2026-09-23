#!/bin/bash -l
#SBATCH -J t43_rsd_mcmc281
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/mcmc_preliminary_281_v2/mcmc_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/mcmc_preliminary_281_v2/mcmc_%j.err
set -eo pipefail
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
OUT=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/mcmc_preliminary_281_v2
cd "$PROJECT_ROOT"
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="$PROJECT_ROOT/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" XLA_PYTHON_CLIENT_PREALLOCATE=false
/usr/bin/time -v python -u "$PROJECT_ROOT/codes/task43/task43_rsd_ezmock281_inference.py" \
  --stage run --nmock 281 --nwalkers 48 --nsteps 8000 --burnin 2000 --ensembles 4 --threads 8 --workers 1
