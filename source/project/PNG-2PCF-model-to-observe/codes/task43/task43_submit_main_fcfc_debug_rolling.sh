#!/usr/bin/env bash
set -u

# Keep the NERSC debug QOS full without exceeding its account limits:
# MaxSubmitJobsPU=5, MaxJobsPU=2, MaxWall=30 minutes.  Every realization is
# an independent 3-minute FCFC job; completed xi0+xi2 products are skipped.

PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
BASE=$PROJECT_ROOT/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50
SCRIPT=$PROJECT_ROOT/codes/task43/task43_xi_smu_main_single.sbatch
STOP=${1:-999}
CAP=${2:-5}
SLEEP_SEC=${3:-15}
LOG=$BASE/logs/fcfc_debug_rolling_main.log
MAP=$BASE/logs/fcfc_debug_rolling_main_jobmap.tsv
RR_SHA=61e6eb2bf493d0a2b8042b065c2434b94b7ec6bc2b503e9dc9e6ea5e6274d574

mkdir -p "$BASE/logs"
cd "$PROJECT_ROOT"
touch "$LOG" "$MAP"

is_complete() {
  local idx=$1 phase seed npz meta
  printf -v phase '%04d' "$idx"
  seed=$((600001 + idx))
  npz=$BASE/xi_fcfc/ezmock_m${phase}_seed${seed}_xi02_s30_350_ds10.npz
  meta=$BASE/xi_fcfc/ezmock_m${phase}_seed${seed}_xi02_s30_350_ds10.json
  [[ -s "$npz" && -s "$meta" ]] || return 1
  grep -q '"status": "done"' "$meta" || return 1
  grep -q '"fix_amplitude": false' "$meta" || return 1
  grep -q '"ells":' "$meta" || return 1
  grep -q '"mu_bin_num": 120' "$meta" || return 1
  grep -q '"n_xi_bins": 32' "$meta" || return 1
  grep -q "\"common_rr_smu_sha256\": \"$RR_SHA\"" "$meta" || return 1
  return 0
}

build_inflight_index() {
  unset ACTIVE_JOB INFLIGHT_INDEX
  declare -gA ACTIVE_JOB=()
  declare -gA INFLIGHT_INDEX=()
  local output jid idx mapped_jid
  if ! output=$(squeue -u "${USER:-lzy}" -h -q debug -o '%A' 2>&1); then
    echo "$(date -u +%FT%TZ) squeue_error result=$output" >> "$LOG"
    return 1
  fi
  while read -r jid; do
    [[ "$jid" =~ ^[0-9]+$ ]] && ACTIVE_JOB[$jid]=1
  done <<< "$output"
  while read -r idx mapped_jid; do
    [[ "$idx" =~ ^[0-9]+$ && "$mapped_jid" =~ ^[0-9]+$ ]] || continue
    [[ -n "${ACTIVE_JOB[$mapped_jid]:-}" ]] && INFLIGHT_INDEX[$idx]=1
  done < "$MAP"
  return 0
}

active_count() {
  echo "${#ACTIVE_JOB[@]}"
}

next=0
pass=1
echo "$(date -u +%FT%TZ) rolling_started stop=$STOP cap=$CAP pass=$pass" >> "$LOG"

while true; do
  if ! build_inflight_index; then
    sleep "$SLEEP_SEC"
    continue
  fi
  active=$(active_count)

  while [[ "$active" -lt "$CAP" ]]; do
    while [[ "$next" -le "$STOP" ]]; do
      if is_complete "$next" || [[ -n "${INFLIGHT_INDEX[$next]:-}" ]]; then
        next=$((next + 1))
      else
        break
      fi
    done

    if [[ "$next" -gt "$STOP" ]]; then
      break
    fi

    result=$(sbatch --parsable --qos=debug --constraint=cpu --cpus-per-task=64 --mem=4G --time=00:05:00 "$SCRIPT" "$next" 2>&1)
    if [[ "$result" =~ ^[0-9]+$ ]]; then
      printf '%d\t%s\n' "$next" "$result" >> "$MAP"
      echo "$(date -u +%FT%TZ) submit index=$next job=$result active_before=$active pass=$pass" >> "$LOG"
      ACTIVE_JOB[$result]=1
      INFLIGHT_INDEX[$next]=1
      active=$((active + 1))
      next=$((next + 1))
    else
      echo "$(date -u +%FT%TZ) submit_error index=$next result=$result" >> "$LOG"
      break
    fi
  done

  if [[ "$next" -gt "$STOP" && "$active" -eq 0 ]]; then
    complete=0
    for ((i=0; i<=STOP; i++)); do
      is_complete "$i" && complete=$((complete + 1))
    done
    echo "$(date -u +%FT%TZ) pass_audit pass=$pass complete=$complete target=$((STOP + 1))" >> "$LOG"
    if [[ "$complete" -eq "$((STOP + 1))" ]]; then
      echo "$(date -u +%FT%TZ) rolling_complete stop=$STOP passes=$pass" >> "$LOG"
      exit 0
    fi
    pass=$((pass + 1))
    next=0
  fi

  sleep "$SLEEP_SEC"
done
