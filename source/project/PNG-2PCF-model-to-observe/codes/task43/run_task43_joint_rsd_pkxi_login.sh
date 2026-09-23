#!/bin/bash
# Joint P0(k)+xi0(s) fNL experiment for the boxsafe RSD lightcone (Task 4.3.2).
# Login-node entry: stages assemble / smoke / fit / plot, 8 CPUs, PDF only.
set -eo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODES="${PROJECT_ROOT}/codes/task43"
LOG_DIR="${PROJECT_ROOT}/codes/logs/task43/joint_rsd_pkxi"
mkdir -p "${LOG_DIR}"

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH="${CODES}:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"

stage="${1:-help}"
case "${stage}" in
  assemble)
    taskset -c 6-11 python -u "${CODES}/task43_joint_rsd_pkxi_build_cov.py" 2>&1 | tee "${LOG_DIR}/assemble.log"
    ;;
  smoke)
    taskset -c 6-11 python -u "${CODES}/task43_joint_rsd_pkxi_fit.py" --smoke 2>&1 | tee "${LOG_DIR}/smoke.log"
    ;;
  fit)
    taskset -c 6-11 python -u "${CODES}/task43_joint_rsd_pkxi_fit.py" 2>&1 | tee "${LOG_DIR}/fit.log"
    ;;
  plot)
    taskset -c 6-11 python -u "${CODES}/task43_plot_joint_rsd_pkxi_contours.py" 2>&1 | tee "${LOG_DIR}/plot.log"
    ;;
  all)
    "$0" assemble
    "$0" smoke
    "$0" fit
    "$0" plot
    ;;
  *)
    echo "usage: $0 {assemble|smoke|fit|plot|all}" >&2
    exit 1
    ;;
esac
