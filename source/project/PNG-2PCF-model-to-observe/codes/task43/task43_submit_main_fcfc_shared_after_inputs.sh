#!/usr/bin/env bash
set -u

# Submit one independent 3-minute FCFC job per main realization.  The
# debug QOS is capped at five submitted jobs per user on NERSC; shared
# permits 5000 submitted jobs and is therefore the intended QOS here.

PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
BASE=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50
SCRIPT=$PROJECT_ROOT/codes/task43/task43_xi_smu_main_single.sbatch
STOP=${1:-999}
LOG=$BASE/logs/fcfc_shared_submit_main.log
SUBMIT_LOG=$BASE/logs/fcfc_individual_submit_main_951.log
JOBMAP=$BASE/logs/fcfc_individual_main_jobmap.tsv
ROLLING_LOG=$BASE/logs/fcfc_individual_rolling_main.log

mkdir -p "$BASE/logs"
cd "$PROJECT_ROOT"
touch "$LOG"

echo "$(date -u +%FT%TZ) watcher_started stop=$STOP" >> "$LOG"

# Do not race the earlier debug-QOS loop.  It has a finite 0..951 pass;
# once its marker appears, only its accepted job ids are treated as seen.
while true; do
  old_done=0
  grep -q "submitted through 951" "$SUBMIT_LOG" 2>/dev/null && old_done=1
  ncat=$(find "$BASE/lightcone_catalogs" -maxdepth 1 -type f -name 'ezmock_m*_seed600*.npz' 2>/dev/null | wc -l)
  # If the old login-node loop is still alive, wait for its marker while
  # inputs are being produced.  Once all catalogs exist, proceed even if the
  # old loop lost its login node; accepted jobs are still filtered below.
  if [[ "$old_done" -eq 1 || "$ncat" -ge "$((STOP + 1))" ]]; then
    break
  fi
  echo "$(date -u +%FT%TZ) waiting_for_old_submit_loop" >> "$LOG"
  sleep 60
done

# Wait until the login-node EZmock/P0/P2 production has materialized every
# main lightcone catalog.  Submitting before this point would create avoidable
# missing-input failures for the last indices.
while true; do
  ncat=$(find "$BASE/lightcone_catalogs" -maxdepth 1 -type f -name 'ezmock_m*_seed600*.npz' 2>/dev/null | wc -l)
  if [[ "$ncat" -ge "$((STOP + 1))" ]]; then
    break
  fi
  echo "$(date -u +%FT%TZ) waiting_for_catalogs count=$ncat target=$((STOP + 1))" >> "$LOG"
  sleep 60
done

is_seen() {
  local idx=$1
  # Include this watcher's own accepted submissions so a login-node restart
  # can safely resume without duplicating hundreds of shared-QOS jobs.
  grep -qE "submit index=${idx}[[:space:]]+job=[0-9]+[[:space:]]+qos=shared" "$LOG" 2>/dev/null && return 0
  grep -qE "^${idx}[[:space:]]+[0-9]+$" "$JOBMAP" 2>/dev/null && return 0
  grep -qE "^${idx}[[:space:]]+[0-9]+$" "$SUBMIT_LOG" 2>/dev/null && return 0
  grep -qE "submit index=${idx}[[:space:]]+job=[0-9]+" "$ROLLING_LOG" 2>/dev/null && return 0
  return 1
}

for ((i=0; i<=STOP; i++)); do
  # Index 636 was already completed as the compute-node smoke test.
  if [[ "$i" -eq 636 ]] || is_seen "$i"; then
    continue
  fi
  jid=''
  for attempt in 1 2 3 4 5; do
    result=$(sbatch --parsable --qos=shared --constraint=cpu --cpus-per-task=64 --mem=4G --time=00:03:00 "$SCRIPT" "$i" 2>&1)
    if [[ "$result" =~ ^[0-9]+$ ]]; then
      jid=$result
      break
    fi
    echo "$(date -u +%FT%TZ) submit_retry index=$i attempt=$attempt result=$result" >> "$LOG"
    sleep 10
  done
  if [[ -n "$jid" ]]; then
    echo "$(date -u +%FT%TZ) submit index=$i job=$jid qos=shared" >> "$LOG"
  else
    echo "$(date -u +%FT%TZ) submit_failed index=$i" >> "$LOG"
  fi
done

echo "$(date -u +%FT%TZ) shared_submission_complete stop=$STOP" >> "$LOG"
