#!/bin/bash -l
#SBATCH -J t43_rr_smu_fb
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/rr_smu_fallback_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/rr_smu_fallback_%j.err
set -euo pipefail
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
cd "$PROJECT_ROOT"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="$PROJECT_ROOT/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RR=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/fcfc_pairs/RR_common50_zobs0p4_0p8_s30_350_ds10_mu120.bin
META=${RR%.bin}.json
if [[ -s "$RR" && -s "$META" ]] && grep -q '"status": "done"' "$META"; then
  echo "[skip] validated s-mu RR already exists: $RR"
  exit 0
fi
python -u codes/task43/task43_build_ezmock_rsd_common_rr_smu_fcfc.py --threads 64
