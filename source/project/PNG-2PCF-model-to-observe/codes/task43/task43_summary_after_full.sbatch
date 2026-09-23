#!/bin/bash -l
#SBATCH -J t43_rsd_covsum
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/summary_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/summary_%j.err
set -euo pipefail
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
cd "$PROJECT_ROOT"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="$PROJECT_ROOT/codes/task43:${PYTHONPATH:-}"
MANIFEST=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/manifests/task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl
for attempt in $(seq 1 720); do
  n=$(python - "$MANIFEST" <<'PY'
import json, os, sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(sum(os.path.isfile(r["pk_path"]) and os.path.isfile(r["xi_path"]) for r in rows))
PY
)
  [[ "$n" == "1000" ]] && break
  sleep 60
done
python -u codes/task43/task43_summarize_ezmock_rsd_covariance.py --manifest "$MANIFEST" --overwrite
