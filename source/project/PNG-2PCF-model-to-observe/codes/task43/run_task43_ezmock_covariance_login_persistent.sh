#!/usr/bin/env bash
set -uo pipefail

# Durable, single-instance wrapper around the resumable Task43 login-node
# production runner.  Every science output is still validated by the Python
# row runner; this wrapper only supplies locking and bounded restart handling.
#
# Usage:
#   run_task43_ezmock_covariance_login_persistent.sh \
#     STAGE CPU_LIST THREADS [START] [STOP]
#
# Examples:
#   ... lightcone_xi 6-15 10 0 1000
#   ... pk           16-17 2 0 1000

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/task43_outputs/ezmock_lightcone_covariance_x1000_s50_350_common50"

STAGE="${1:?missing STAGE (lightcone_xi or pk)}"
CPU_LIST="${2:?missing CPU_LIST}"
THREADS="${3:?missing THREADS}"
START="${4:-0}"
STOP="${5:-1000}"
MAX_RESTARTS="${TASK43_EZCOV_MAX_RESTARTS:-20}"
RESTART_DELAY="${TASK43_EZCOV_RESTART_DELAY:-30}"

case "${STAGE}" in
  lightcone_xi)
    WAIT_FOR_LIGHTCONE=0
    ;;
  pk)
    WAIT_FOR_LIGHTCONE=1
    ;;
  *)
    echo "STAGE must be lightcone_xi or pk, got ${STAGE}" >&2
    exit 2
    ;;
esac

if ! [[ "${THREADS}" =~ ^[0-9]+$ ]] || (( THREADS < 1 || THREADS > 12 )); then
  echo "THREADS must be an integer in [1,12], got ${THREADS}" >&2
  exit 2
fi
if ! [[ "${START}" =~ ^[0-9]+$ && "${STOP}" =~ ^[0-9]+$ ]] || (( START < 0 || STOP > 1000 || START >= STOP )); then
  echo "require 0 <= START < STOP <= 1000, got START=${START} STOP=${STOP}" >&2
  exit 2
fi

RUNTIME_DIR="${OUTPUT_ROOT}/logs/persistent"
mkdir -p "${RUNTIME_DIR}"
LOCK_PATH="${RUNTIME_DIR}/${STAGE}.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "[$(date -u +%FT%TZ)] another ${STAGE} supervisor already holds ${LOCK_PATH}" >&2
  exit 3
fi

cd "${PROJECT_ROOT}"
attempt=0
while (( attempt <= MAX_RESTARTS )); do
  attempt=$((attempt + 1))
  echo "[$(date -u +%FT%TZ)] launch stage=${STAGE} attempt=${attempt} host=$(hostname -s) cpu=${CPU_LIST} threads=${THREADS} rows=${START}:${STOP}"
  taskset -c "${CPU_LIST}" env \
    TASK43_EZCOV_THREADS="${THREADS}" \
    TASK43_EZCOV_SUPERVISED="1" \
    ./codes/task43/run_task43_ezmock_covariance_login.sh \
    "${START}" "${STOP}" "${STAGE}" "${WAIT_FOR_LIGHTCONE}"
  status=$?
  if (( status == 0 )); then
    echo "[$(date -u +%FT%TZ)] complete stage=${STAGE} rows=${START}:${STOP}"
    exit 0
  fi
  if (( attempt > MAX_RESTARTS )); then
    echo "[$(date -u +%FT%TZ)] failed stage=${STAGE} status=${status}; restart limit exhausted" >&2
    exit "${status}"
  fi
  echo "[$(date -u +%FT%TZ)] stage=${STAGE} exited status=${status}; retry in ${RESTART_DELAY}s" >&2
  sleep "${RESTART_DELAY}"
done
