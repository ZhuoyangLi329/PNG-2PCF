#!/usr/bin/env bash
set -euo pipefail

LOG=/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log/task4_autorun_watch.log
# 用法: run_task4_after_slurm.sh <pk_jobid> <pcf_jobid>
JOB_PK="${1:?missing pk job id}"
JOB_PCF="${2:?missing pcf job id}"

echo "[$(date '+%F %T %Z')] watch start: jobs ${JOB_PK} ${JOB_PCF}" | tee -a "$LOG"
while squeue -h -j "${JOB_PK},${JOB_PCF}" | grep -q .; do
  echo "[$(date '+%F %T %Z')] waiting slurm jobs..." >> "$LOG"
  sleep 60
done

echo "[$(date '+%F %T %Z')] slurm jobs finished, checking outputs" | tee -a "$LOG"
PKN=$(ls -1 /pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat 2>/dev/null | wc -l)
PCFN=$(ls -1 /pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat 2>/dev/null | wc -l)
echo "[$(date '+%F %T %Z')] pk_count=$PKN pcf_count=$PCFN" | tee -a "$LOG"

if [[ "$PKN" -lt 79 || "$PCFN" -lt 79 ]]; then
  echo "[$(date '+%F %T %Z')] outputs incomplete, skip task4 run" | tee -a "$LOG"
  exit 2
fi

echo "[$(date '+%F %T %Z')] run task4 script" | tee -a "$LOG"
/global/homes/l/lzy/anaconda3/envs/desilike/bin/python -u /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log/task4_boxlength_kcut_study.py 2>&1 | tee /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log/task4_rerun_1to80.log

echo "[$(date '+%F %T %Z')] task4 run done" | tee -a "$LOG"
