#!/usr/bin/env bash
set -u

# Streaming submitter for the nzmatch production FCFC jobs: a row's 3-minute
# shared-QOS job is submitted as soon as that row's lightcone metadata JSON
# exists (write_json is atomic and runs after atomic_savez of the catalog), no
# need to wait for the full x1000 build.  Duplicate submissions are safe: the xi
# script serialises each row with an exclusive flock and re-checks
# validate_existing() inside the lock.
#
# Usage: nohup bash codes/task43/task43_submit_xi_shared_nzmatch.sh > <out> 2>&1 &

PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
BASE=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60
SCRIPT=$PROJECT_ROOT/codes/task43/task43_xi_smu_main_single_nzmatch_c20.sbatch
STOP=${1:-999}
SEED_START=600001
JOBNAME=t43_xi_nz20
LOG=$BASE/logs/fcfc_shared_submit_nzmatch.log
JOBMAP=$BASE/logs/fcfc_shared_nzmatch_jobmap.tsv
POLL_SEC=120
RESUBMIT_AFTER_SEC=600
STALL_EXIT_SEC=10800

mkdir -p "$BASE/logs"
cd "$PROJECT_ROOT"
touch "$LOG" "$JOBMAP"

echo "$(date -u +%FT%TZ) streaming_watcher_started stop=$STOP" >> "$LOG"

catalog_ready() {
  local idx=$1
  local seed=$((SEED_START + idx))
  [[ -f "$(printf '%s/lightcone_catalogs/ezmock_m%04d_seed%d_zobs0p4_0p8.json' "$BASE" "$idx" "$seed")" ]]
}

xi_done() {
  local idx=$1
  local seed=$((SEED_START + idx))
  local json
  json=$(printf '%s/xi_fcfc/ezmock_m%04d_seed%d_xi02_s30_350_ds10.json' "$BASE" "$idx" "$seed")
  [[ -f "$json" ]] && grep -q '"status": "done"' "$json"
}

declare -A last_submit=()
declare -A last_job
while read -r idx jid epoch _rest; do
  [[ "$idx" =~ ^[0-9]+$ ]] || continue
  last_submit[$idx]=$epoch
  last_job[$idx]=$jid
done < "$JOBMAP"

last_progress=$(date +%s)
prev_ndone=-1
while true; do
  now=$(date +%s)
  ndone=0
  nready=0
  nsubmit=0
  for ((i=0; i<=STOP; i++)); do
    if xi_done "$i"; then
      ndone=$((ndone + 1))
      continue
    fi
    catalog_ready "$i" || continue
    nready=$((nready + 1))
    jid_prev=${last_job[$i]:-}
    if [[ -n "$jid_prev" ]] && squeue -h -j "$jid_prev" 2>/dev/null | grep -q .; then
      continue
    fi
    t=${last_submit[$i]:-0}
    [[ $((now - t)) -ge "$RESUBMIT_AFTER_SEC" ]] || continue
    jid=''
    for attempt in 1 2 3 4 5; do
      result=$(sbatch --parsable --qos=shared --constraint=cpu --cpus-per-task=20 --mem=4G --time=00:07:00 "$SCRIPT" "$i" 2>&1)
      if [[ "$result" =~ ^[0-9]+$ ]]; then
        jid=$result
        break
      fi
      echo "$(date -u +%FT%TZ) submit_retry index=$i attempt=$attempt result=$result" >> "$LOG"
      sleep 10
    done
    if [[ -n "$jid" ]]; then
      last_submit[$i]=$now
      last_job[$i]=$jid
      printf '%d\t%s\t%s\n' "$i" "$jid" "$now" >> "$JOBMAP"
      nsubmit=$((nsubmit + 1))
      echo "$(date -u +%FT%TZ) submit index=$i job=$jid qos=shared" >> "$LOG"
    else
      echo "$(date -u +%FT%TZ) submit_failed index=$i" >> "$LOG"
    fi
  done
  nq=$(squeue -u "$USER" -h -n "$JOBNAME" 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) scan done=$ndone ready=$nready submitted_now=$nsubmit queue=$nq" >> "$LOG"

  if [[ "$ndone" -ge "$((STOP + 1))" ]]; then
    echo "$(date -u +%FT%TZ) all_done stop=$STOP" >> "$LOG"
    exit 0
  fi
  if [[ "$ndone" -ne "$prev_ndone" || "$nsubmit" -gt 0 ]]; then
    last_progress=$now
  elif [[ $((now - last_progress)) -ge "$STALL_EXIT_SEC" ]]; then
    echo "$(date -u +%FT%TZ) stall_exit done=$ndone" >> "$LOG"
    exit 1
  fi
  prev_ndone=$ndone
  sleep "$POLL_SEC"
done
