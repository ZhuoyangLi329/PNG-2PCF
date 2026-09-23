#!/usr/bin/env bash
set -euo pipefail

# 最终汇总入口：只读取已经通过 gate 的 chain/covariance/Fisher 产物，
# 在登录节点单进程运行并严格限制数值库线程。
ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
PY="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
CPUSET="${TASK43_RMIN_CPUSET:-6-13}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
cd "${ROOT}"
NCPU=$(taskset -c "${CPUSET}" nproc)
if (( NCPU < 1 || NCPU > 8 )); then
  echo "[fatal] CPU affinity 必须包含 1..8 核：${CPUSET} -> ${NCPU}" >&2
  exit 2
fi
taskset -c "${CPUSET}" "${PY}" -u codes/task43/task43_summarize_rmin_scan.py
taskset -c "${CPUSET}" "${PY}" -u codes/task43/task43_audit_rmin_scan_hygiene.py
