#!/bin/bash -l
#SBATCH -J t43_wide_up
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/wide_upstream_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/wide_upstream_%j.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_wide_zobs0p4_1p1_x25.jsonl"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${PROJECT_ROOT}"

run_phase() {
  local phase="$1"
  echo "[phase-start] ${phase} $(date -u +%FT%TZ)"
  python -u codes/task43/task43_build_rsd_lightcone_catalog.py --manifest "${MANIFEST}" --phase "${phase}"
  python -u codes/task43/task43_build_rsd_lightcone_random.py --manifest "${MANIFEST}" --phase "${phase}"
  echo "[phase-done] ${phase} $(date -u +%FT%TZ)"
}
export -f run_phase
export PROJECT_ROOT MANIFEST PYTHONUNBUFFERED OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS

for index in $(seq 0 24); do
  printf -v phase 'ph%03d' "${index}"
  while (( $(jobs -pr | wc -l) >= 4 )); do
    wait -n
  done
  srun --exclusive -N 1 -n 1 -c 2 --cpu-bind=cores bash -c 'run_phase "$1"' _ "${phase}" &
done
wait

python -u codes/task43/task43_audit_rsd_wide_lightcone_upstream.py --manifest "${MANIFEST}"
echo "[all-done] $(date -u +%FT%TZ)"
