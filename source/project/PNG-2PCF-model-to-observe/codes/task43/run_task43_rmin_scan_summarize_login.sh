#!/usr/bin/env bash
set -euo pipefail

# 执行逻辑大纲：先审计 25 个 phase 和旧区间 bridge，再生成唯一的
# 25-phase mean xi summary，最后检查 25x32/32-bin shape 与坐标。
ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
PY="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
SCAN_ROOT="${ROOT}/outputs/task43_outputs/rmin_scan"
MANIFEST="${SCAN_ROOT}/manifests/task43_rmin_scan_mmin1p4e13_x25.jsonl"
FKP="${ROOT}/plots/outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
PREFIX="${SCAN_ROOT}/summary/task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000"
CPUSET="${TASK43_RMIN_CPUSET:-6-13}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${ROOT}"
taskset -c "${CPUSET}" "${PY}" codes/task43/task43_audit_rmin_scan_measurements.py \
  --manifest "${MANIFEST}" --indices 0-24 --require-all

if [[ ! -e "${PREFIX}.npz" && ! -e "${PREFIX}.json" ]]; then
  taskset -c "${CPUSET}" "${PY}" codes/task43/task43_summarize_xi.py \
    --manifest "${MANIFEST}" --output-prefix "${PREFIX}" --p0 10000 \
    --fkp-summary "${FKP}" --require-all
elif [[ ! -s "${PREFIX}.npz" || ! -s "${PREFIX}.json" ]]; then
  echo "[fatal] summary pair 不完整：${PREFIX}.{npz,json}" >&2
  exit 2
fi

taskset -c "${CPUSET}" "${PY}" - "${PREFIX}.npz" <<'PY'
import sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
with np.load(path, allow_pickle=False) as data:
    edges = np.asarray(data["s_edges"], dtype="f8")
    centers = np.asarray(data["s"], dtype="f8")
    xi = np.asarray(data["xi0"], dtype="f8")
    xi_all = np.asarray(data["xi0_all"], dtype="f8")
    nreal = int(np.asarray(data["nreal"]).item())
expected_edges = np.arange(30.0, 351.0, 10.0)
if nreal != 25 or xi.shape != (32,) or xi_all.shape != (25, 32):
    raise RuntimeError(f"summary shape 错误：nreal={nreal}, xi={xi.shape}, xi_all={xi_all.shape}")
if not np.array_equal(edges, expected_edges) or not np.array_equal(centers, 0.5 * (edges[:-1] + edges[1:])):
    raise RuntimeError("summary separation 坐标错误")
if not np.all(np.isfinite(xi_all)):
    raise RuntimeError("summary 含非有限 xi")
print(f"[done] summary={path} nreal=25 nbins=32")
PY
