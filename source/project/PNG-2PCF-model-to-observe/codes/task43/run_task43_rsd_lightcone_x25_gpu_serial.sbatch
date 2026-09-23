#!/bin/bash -l
#SBATCH -J t43_rsd_x22
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1900M
#SBATCH -t 01:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/lightcone_x25_serial_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/lightcone_x25_serial_%j.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_x25.jsonl"
PILOT_NPZ="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/lightcone/pilot/task43_rsd_lightcone_pilot_x3.npz"
PILOT_JSON="${PILOT_NPZ%.npz}.json"
PILOT_PDF="${PROJECT_ROOT}/plots/task43/rsd_validation/task43_rsd_lightcone_pilot_x3.pdf"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"
if [[ -s "${PILOT_NPZ}" && -s "${PILOT_JSON}" && -s "${PILOT_PDF}" ]]; then
  python - "${PILOT_JSON}" <<'PY'
import json
import sys
status = json.load(open(sys.argv[1], encoding="utf-8")).get("status")
if status != "pass":
    raise SystemExit(f"existing x3 pilot did not pass: status={status}")
print("[skip] existing validated x3 pilot")
PY
elif [[ -e "${PILOT_NPZ}" || -e "${PILOT_JSON}" || -e "${PILOT_PDF}" ]]; then
  echo "[error] partial x3 lightcone pilot exists" >&2
  exit 3
else
  srun --exact -n 1 -c 1 --cpu-bind=cores python -u codes/task43/task43_summarize_rsd_lightcone_pilot.py
fi

for index in $(seq 3 24); do
  printf -v phase 'ph%03d' "${index}"
  echo "[phase-start] ${phase}"
  srun --exact -n 1 -c 8 --cpu-bind=cores python -u codes/task43/task43_measure_rsd_lightcone_xi.py \
    --manifest "${MANIFEST}" --phase "${phase}" --nmu 40
  echo "[phase-done] ${phase}"
done
