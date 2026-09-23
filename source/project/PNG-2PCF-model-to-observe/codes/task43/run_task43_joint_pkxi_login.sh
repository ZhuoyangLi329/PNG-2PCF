#!/usr/bin/env bash
# Task43 joint P(k)+xi0 covariance/fit/plot 的登录节点入口（CPU <= 8 核）。
#
# 用法：
#   bash codes/task43/run_task43_joint_pkxi_login.sh run64      # jaxpower 细网格协方差 (mesh64, kmin=0.001)
#   bash codes/task43/run_task43_joint_pkxi_login.sh assemble64 # 组装 45x45 联合协方差 + bridge gates
#   bash codes/task43/run_task43_joint_pkxi_login.sh run128     # mesh128 A/B（仅在 pp bridge gate 失败时需要）
#   bash codes/task43/run_task43_joint_pkxi_login.sh assemble128
#   bash codes/task43/run_task43_joint_pkxi_login.sh smoke      # 短链拟合 smoke
#   bash codes/task43/run_task43_joint_pkxi_login.sh fit        # 五条 64x20000 长链
#   bash codes/task43/run_task43_joint_pkxi_login.sh plot       # 三 contour 主图 + 审计

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
TASK="task43_joint_pkxi"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""

JOINT_THREADS="${JOINT_THREADS:-8}"
if [[ "${JOINT_THREADS}" -gt 8 ]]; then
  echo "JOINT_THREADS=${JOINT_THREADS} exceeds the Task43 login-node limit of 8" >&2
  exit 2
fi
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${JOINT_THREADS} inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "${PROJECT_ROOT}"
PY="$(command -v python)"
BUILDER="codes/task43/${TASK}_build_cov.py"
FITTER="codes/task43/${TASK}_fit.py"
PLOTTER="codes/task43/task43_plot_joint_pk_xi_contours.py"
FINE_DIR="outputs/task43_outputs/joint_pkxi_s50_350/covariance/fine"
FINE64="${FINE_DIR}/task43_joint_finecov_ph000_mesh64_kmin0p001.npz"
FINE128="${FINE_DIR}/task43_joint_finecov_ph000_mesh128_kmin0p001.npz"

case "${1:-}" in
  run64)
    taskset -c 6-13 "${PY}" -u "${BUILDER}" run --meshsize 64
    ;;
  assemble64)
    taskset -c 6-13 "${PY}" -u "${BUILDER}" assemble --fine-npz "${FINE64}"
    ;;
  run128)
    taskset -c 6-13 "${PY}" -u "${BUILDER}" run --meshsize 128
    ;;
  assemble128)
    taskset -c 6-13 "${PY}" -u "${BUILDER}" assemble --fine-npz "${FINE128}"
    ;;
  smoke)
    taskset -c 6-13 "${PY}" -u "${FITTER}" --smoke
    ;;
  fit)
    taskset -c 6-13 "${PY}" -u "${FITTER}"
    ;;
  plot)
    taskset -c 6-13 "${PY}" -u "${PLOTTER}"
    ;;
  *)
    echo "unknown stage: ${1:-}" >&2
    exit 2
    ;;
esac
