#!/usr/bin/env bash
set -euo pipefail

LOG=/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log/task43_autorun_watch.log
JOB_PK="${1:?missing pk job id}"
JOB_PCF="${2:?missing pcf job id}"

echo "[$(date '+%F %T %Z')] task43 watch start: jobs ${JOB_PK} ${JOB_PCF}" | tee -a "$LOG"
while squeue -h -j "${JOB_PK},${JOB_PCF}" | grep -q .; do
  echo "[$(date '+%F %T %Z')] waiting fnl0 pk/pcf jobs..." >> "$LOG"
  sleep 60
done

echo "[$(date '+%F %T %Z')] slurm jobs finished, checking fnl0 outputs" | tee -a "$LOG"
PKN=$(ls -1 /pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat 2>/dev/null | wc -l)
PCFN=$(ls -1 /pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat 2>/dev/null | wc -l)
echo "[$(date '+%F %T %Z')] fnl0 pk_count=$PKN pcf_count=$PCFN" | tee -a "$LOG"

if [[ "$PKN" -lt 98 || "$PCFN" -lt 98 ]]; then
  echo "[$(date '+%F %T %Z')] outputs incomplete, skip task43 run" | tee -a "$LOG"
  exit 2
fi

echo "[$(date '+%F %T %Z')] run task43 script" | tee -a "$LOG"
/global/homes/l/lzy/anaconda3/envs/desilike/bin/python -u /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log/task43_fnl0_pk_to_xi_check.py 2>&1 | tee /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log/task43_run.log

echo "[$(date '+%F %T %Z')] task43 run done" | tee -a "$LOG"
