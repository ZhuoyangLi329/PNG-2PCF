#!/usr/bin/env bash
set -u

# Restart wrapper for the x1000 fixampF n(z)-matched production build+P0/P2 on
# the login node (8 cores via taskset -c 6-13, per user policy).  The login
# runner is per-row restartable: finished rows are skipped by the build/pk
# provenance gates, so re-running the same command resumes cleanly.
#
# Usage: nohup bash codes/task43/run_task43_ezmock_rsd_production_login_persistent.sh \
#          > /dev/null 2>&1 &
#        tail -f <BASE>/logs/login_runner_persistent.out

PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
BASE=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60
MANIFEST=$BASE/manifests/task43_ezmock_rsd_covariance_x1000_fixampF_nzmatch_b0.30_z0.60.jsonl
LOGJSONL=$BASE/logs/login_runner.jsonl
OUT=$BASE/logs/login_runner_persistent.out
MAX_ATTEMPTS=${1:-50}

mkdir -p "$BASE/logs"
cd "$PROJECT_ROOT"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="$PROJECT_ROOT/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" XLA_PYTHON_CLIENT_PREALLOCATE=false

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "$(date -u +%FT%TZ) persistent_attempt=$attempt start" >> "$OUT"
  taskset -c 6-13 python -u codes/task43/task43_run_ezmock_rsd_covariance_login.py \
    --manifest "$MANIFEST" --log "$LOGJSONL" --stage all --threads 8 >> "$OUT" 2>&1
  rc=$?
  nlines=$(wc -l < "$LOGJSONL")
  echo "$(date -u +%FT%TZ) persistent_attempt=$attempt rc=$rc runner_lines=$nlines" >> "$OUT"
  if [[ "$rc" -eq 0 ]]; then
    echo "$(date -u +%FT%TZ) persistent_complete" >> "$OUT"
    exit 0
  fi
  sleep 30
done

echo "$(date -u +%FT%TZ) persistent_exhausted attempts=$MAX_ATTEMPTS" >> "$OUT"
exit 1
