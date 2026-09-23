#!/bin/bash -l
#SBATCH -J t43_xi_smu_1k
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --array=0-999%2
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/xi_smu_1k_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/xi_smu_1k_%A_%a.err
set -euo pipefail
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
cd "$PROJECT_ROOT"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="$PROJECT_ROOT/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MANIFEST=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/manifests/task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl
ROW=$(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
row = rows[int(sys.argv[2])]
print(row["lightcone_catalog_path"])
PY
)
for attempt in $(seq 1 60); do
  [[ -s "$ROW" ]] && break
  sleep 60
done
[[ -s "$ROW" ]] || { echo "missing lightcone after wait: $ROW" >&2; exit 3; }
/usr/bin/time -v python -u codes/task43/task43_measure_ezmock_rsd_lightcone_xi_fcfc.py --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --threads 64
