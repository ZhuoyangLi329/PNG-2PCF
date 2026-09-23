#!/usr/bin/env bash
# Login-node runner for the remaining Task 4.3.2 formal-GIC windows.
# Four two-thread lanes keep the aggregate CPU affinity at exactly eight CPUs.

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_x25.jsonl"
MPL_CACHE="/tmp/task43_rsd_formal_gic_mplcache"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main >/dev/null 2>&1
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPL_CACHE}"

mkdir -p "${MPL_CACHE}"
cd "${PROJECT_ROOT}"
taskset -c 6 python -c 'import matplotlib' >/dev/null 2>&1

run_lane() {
  local first_index="$1"
  local affinity="$2"
  local index phase
  for ((index=first_index; index<=24; index+=4)); do
    printf -v phase 'ph%03d' "${index}"
    taskset -c "${affinity}" python -u codes/task43/task43_build_rsd_formal_gic_window.py \
      --manifest "${MANIFEST}" \
      --phase "${phase}" \
      --nsub 200000 \
      --seed-base 430340 \
      --nthreads 2
  done
}

pids=()
run_lane 4 6-7 & pids+=("$!")
run_lane 5 8-9 & pids+=("$!")
run_lane 6 10-11 & pids+=("$!")
run_lane 7 12-13 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
