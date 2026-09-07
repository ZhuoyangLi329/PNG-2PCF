#!/usr/bin/env bash
set -euo pipefail

# Task4.1 LCp50 P(k) vs 2PCF r=50--350 长链 emcee 登录节点入口。
#
# 执行逻辑：
# 1. 把本脚本固定到最多 8 个 CPU cores；
# 2. 激活历史 Task45 使用的现成 ``desilike`` conda 环境，不安装任何包；
# 3. 把所有数值库限制为单线程；
# 4. 运行可 checkpoint/resume 的 Python MCMC 入口。
#
# 用法：
#   bash codes/task4/run_task45_fnl50_pk_xi_r50_emcee_login.sh --overwrite
#   bash codes/task4/run_task45_fnl50_pk_xi_r50_emcee_login.sh
#   bash codes/task4/run_task45_fnl50_pk_xi_r50_emcee_login.sh --plot-only

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
: "${TASK45_CPUSET:=6-13}"
: "${TASK45_NWORKERS:=8}"

cpu_count="$(
  python - "${TASK45_CPUSET}" <<'PY'
import sys
count = 0
for part in sys.argv[1].split(','):
    part = part.strip()
    if not part:
        continue
    if '-' in part:
        lo, hi = map(int, part.split('-', 1))
        count += hi - lo + 1
    else:
        int(part)
        count += 1
print(count)
PY
)"
if (( cpu_count > 8 )); then
  echo "[error] TASK45_CPUSET=${TASK45_CPUSET} exposes ${cpu_count} CPUs; use <=8" >&2
  exit 2
fi
if (( TASK45_NWORKERS < 1 || TASK45_NWORKERS > 8 )); then
  echo "[error] TASK45_NWORKERS=${TASK45_NWORKERS}; use 1..8" >&2
  exit 2
fi

if [[ "${TASK45_CPU_PINNED:-0}" != "1" ]] && command -v taskset >/dev/null 2>&1; then
  export TASK45_CPU_PINNED=1
  exec taskset -c "${TASK45_CPUSET}" bash "$0" "$@"
fi

set +u
source /global/homes/l/lzy/anaconda3/etc/profile.d/conda.sh
conda activate desilike
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JAX_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python -u codes/task4/task45_fnl50_pk_xi_r50_emcee.py \
  --nworkers "${TASK45_NWORKERS}" \
  "$@"
