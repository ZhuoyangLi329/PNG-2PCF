#!/bin/bash
# Boxsafe RSD quadrupole-increment test (xi0 -> xi0+xi2), Task 4.3.2.
# Login-node entry: stages data / deconv / fit, 8 CPUs, PDF only later.
set -eo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODES="${PROJECT_ROOT}/codes/task43"
LOG_DIR="${PROJECT_ROOT}/codes/logs/task43/rsd_ell2_increment"
mkdir -p "${LOG_DIR}"

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH="${CODES}:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"

stage="${1:-help}"
case "${stage}" in
  data)
    taskset -c 6-11 python -u "${CODES}/task43_rsd_boxsafe_build_ell02_products.py" data 2>&1 | tee "${LOG_DIR}/data.log"
    ;;
  deconv)
    taskset -c 6-11 python -u "${CODES}/task43_rsd_boxsafe_build_ell02_products.py" deconv 2>&1 | tee "${LOG_DIR}/deconv.log"
    ;;
  fit)
    taskset -c 6-11 python -u "${CODES}/task43_rsd_boxsafe_ell2_increment.py" 2>&1 | tee "${LOG_DIR}/fit.log"
    ;;
  all)
    "$0" data
    "$0" deconv
    "$0" fit
    ;;
  *)
    echo "usage: $0 {data|deconv|fit|all}" >&2
    exit 1
    ;;
esac
