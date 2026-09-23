#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
cd "$PROJECT_ROOT"
STOP=${1:?stop index required}
LOG=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/fcfc_individual_submit_main_${STOP}.log
SCRIPT=$PROJECT_ROOT/codes/task43/task43_xi_smu_main_single.sbatch
for i in $(seq 0 "$STOP"); do
  if [[ "$i" -eq 636 ]]; then
    echo "$i SKIP_SMOKE" >> "$LOG"
    continue
  fi
  jid=$(sbatch --parsable --qos=debug --constraint=cpu --cpus-per-task=64 --mem=4G --time=00:03:00 "$SCRIPT" "$i" 2>&1) || {
    echo "$i SUBMIT_ERROR $jid" >> "$LOG"
    continue
  }
  echo "$i $jid" >> "$LOG"
done
echo "submitted through $STOP" >> "$LOG"
