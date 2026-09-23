#!/usr/bin/env bash
set -u
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
cd "$PROJECT_ROOT"
STOP=${1:-999}
CAP=${2:-180}
SCRIPT=$PROJECT_ROOT/codes/task43/task43_xi_smu_main_single.sbatch
LOG=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/fcfc_individual_rolling_main.log
MAP=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/fcfc_individual_main_jobmap.tsv
touch "$LOG" "$MAP"

is_submitted() {
  grep -q "^$1[[:space:]]" "$MAP" 2>/dev/null
}

while true; do
  active=$(squeue -u "${USER:-lzy}" -h -n t43_xi_main -o '%i' 2>/dev/null | wc -l)
  next=0
  while [[ "$active" -lt "$CAP" && "$next" -le "$STOP" ]]; do
    if [[ "$next" -eq 636 ]] || is_submitted "$next"; then
      next=$((next + 1))
      continue
    fi
    result=$(sbatch --parsable --qos=debug --constraint=cpu --cpus-per-task=64 --mem=4G --time=00:05:00 "$SCRIPT" "$next" 2>&1)
    if [[ "$result" =~ ^[0-9]+$ ]]; then
      echo -e "$next\t$result" >> "$MAP"
      echo "$(date -u +%FT%TZ) submit index=$next job=$result active_before=$active" >> "$LOG"
      active=$((active + 1))
      next=$((next + 1))
    else
      echo "$(date -u +%FT%TZ) submit_error index=$next result=$result" >> "$LOG"
      break
    fi
  done
  submitted=$(wc -l < "$MAP")
  if [[ "$submitted" -ge "$((STOP + 1 - 1))" ]]; then
    echo "$(date -u +%FT%TZ) submission loop reached stop=$STOP" >> "$LOG"
    exit 0
  fi
  sleep 60
done
