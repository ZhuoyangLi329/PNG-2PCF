#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission1_log"
PK_LOG="$LOG_DIR/recalc_pk_3gpc.log"
PCF_LOG="$LOG_DIR/recalc_2pcf_3gpc.log"
WATCH_LOG="$LOG_DIR/watch_3gpc_regen_then_stats.log"
STATUS_LOG="$LOG_DIR/mission1_repair_status.txt"

# 3Gpc 每个 tag 都是 N2..N99，因此总期望文件数=98+98=196
EXPECTED_NUM=196

echo "[$(date '+%F %T %Z')] watcher start" | tee -a "$WATCH_LOG"

echo "[$(date '+%F %T %Z')] waiting regen_rsd_pos_3gpc_masscut.py ..." | tee -a "$WATCH_LOG"
while pgrep -f "regen_rsd_pos_3gpc_masscut.py" >/dev/null; do
  sleep 60
  echo "[$(date '+%F %T %Z')] still waiting ..." >> "$WATCH_LOG"
done

echo "[$(date '+%F %T %Z')] regen finished, start pk" | tee -a "$WATCH_LOG"
python -u /pscratch/sd/l/lzy/3gpc-UNIT-fastpm/calc_powspec_3gpc.py 2>&1 | tee "$PK_LOG"

echo "[$(date '+%F %T %Z')] pk finished, start 2pcf" | tee -a "$WATCH_LOG"
python -u /pscratch/sd/l/lzy/3gpc-UNIT-fastpm/calc_2pcf_3gpc.py 2>&1 | tee "$PCF_LOG"

echo "[$(date '+%F %T %Z')] all done" | tee -a "$WATCH_LOG"

echo "[$(date '+%F %T %Z')] start averaged HMF plotting (all realizations)" | tee -a "$WATCH_LOG"
python -u /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission1_log/plot_mass_function_after_cut.py
echo "[$(date '+%F %T %Z')] averaged HMF plotting done" | tee -a "$WATCH_LOG"

# 只有在产物数量完整时才清理监测/运行日志，避免误删排错信息
PK_NUM=$(ls -1 /pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_*.dat 2>/dev/null | wc -l)
PCF_NUM=$(ls -1 /pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_*.dat 2>/dev/null | wc -l)

if [[ "$PK_NUM" -eq "$EXPECTED_NUM" && "$PCF_NUM" -eq "$EXPECTED_NUM" ]]; then
  echo "[$(date '+%F %T %Z')] outputs complete: pk=$PK_NUM, pcf=$PCF_NUM, start cleanup logs" | tee -a "$STATUS_LOG"

  # 删除过程日志：按你的要求，运行成功后清理监测日志
  rm -f "$LOG_DIR"/regen_3gpc_*.log
  rm -f "$LOG_DIR"/recalc_pk_3gpc.log "$LOG_DIR"/recalc_2pcf_3gpc.log
  rm -f "$LOG_DIR"/watch_3gpc_regen_then_stats.log "$LOG_DIR"/nohup_watch_3gpc.out

  echo "[$(date '+%F %T %Z')] cleanup done" | tee -a "$STATUS_LOG"
else
  echo "[$(date '+%F %T %Z')] skip cleanup: pk=$PK_NUM, pcf=$PCF_NUM, expected=$EXPECTED_NUM" | tee -a "$STATUS_LOG"
fi
