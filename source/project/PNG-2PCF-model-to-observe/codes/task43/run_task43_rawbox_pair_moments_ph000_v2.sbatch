#!/bin/bash -l
#SBATCH -J t43_pair_p0
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/rawbox_pair_moments_ph000_v2_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/rawbox_pair_moments_ph000_v2_%j.err

set -euo pipefail

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

project=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
export PYTHONPATH="${project}/codes/task43${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${project}"
python -u codes/task43/task43_measure_rawbox_pair_moments_v2.py \
  --phase ph000 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --nanchors 1024 \
  --nblocks 8 \
  --seed 20260907 \
  --rmin 0 \
  --rmax 500 \
  --dr 5 \
  --inner-edge 20 \
  --nmu 12
