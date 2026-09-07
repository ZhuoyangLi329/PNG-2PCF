#!/bin/bash
#
# 一键提交 Mission 7 的 masscut 扫描 pipeline（export -> pk/pcf -> model validation）。
#
# 用法示例：
#   bash submit_pipeline.sh
#
# 切换到 full masscuts（包含 1e12..1e15）：
#   MASSCUTS_TSV=/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_masscut/masscuts_full.tsv bash submit_pipeline.sh
#
# 注意：
# - 这个脚本只负责 sbatch 提交，不会在登录节点做重计算。
# - 2PCF 对低 mass_min 可能极慢（halo 太多），建议先用 light 列表。

set -euo pipefail

ROOT="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_masscut"
SCRIPTS_DIR="${ROOT}/scripts"

MASSCUTS_TSV_DEFAULT="${ROOT}/masscuts_light.tsv"
MASSCUTS_TSV="${MASSCUTS_TSV:-$MASSCUTS_TSV_DEFAULT}"

if [ ! -f "$MASSCUTS_TSV" ]; then
  echo "[FATAL] MASSCUTS_TSV not found: $MASSCUTS_TSV" 1>&2
  exit 2
fi

# 统计有效行数（非注释，且至少 2 列）
N=$(awk '($0 !~ /^#/ && NF >= 2){c++} END{print c+0}' "$MASSCUTS_TSV")
if [ "$N" -le 0 ]; then
  echo "[FATAL] No valid masscuts in: $MASSCUTS_TSV" 1>&2
  exit 3
fi

ARRAY="0-$((N-1))"
echo "[INFO] MASSCUTS_TSV=$MASSCUTS_TSV"
echo "[INFO] array=$ARRAY (N=$N)"

export MASSCUTS_TSV

JOB_EXPORT=$(sbatch --parsable --array="$ARRAY" "${SCRIPTS_DIR}/run_export_pos_masscut_array.slurm")
echo "[SUBMIT] export_pos jobid=$JOB_EXPORT"

JOB_PK=$(sbatch --parsable --dependency="afterok:${JOB_EXPORT}" --array="$ARRAY" "${SCRIPTS_DIR}/run_pk_masscut_array.slurm")
echo "[SUBMIT] pk jobid=$JOB_PK (afterok export)"

JOB_PCF=$(sbatch --parsable --dependency="afterok:${JOB_EXPORT}" --array="$ARRAY" "${SCRIPTS_DIR}/run_2pcf_masscut_array.slurm")
echo "[SUBMIT] pcf jobid=$JOB_PCF (afterok export)"

JOB_MODEL=$(sbatch --parsable --dependency="afterok:${JOB_PK}:${JOB_PCF}" --array="$ARRAY" "${SCRIPTS_DIR}/run_model_validate_masscut_array.slurm")
echo "[SUBMIT] model_validate jobid=$JOB_MODEL (afterok pk+pcf)"

echo "[DONE] Submitted pipeline."

