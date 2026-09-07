#!/bin/bash

set -euo pipefail

PK100_DIR=/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_remeasure_g1024_k5
PK0_DIR=/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_remeasure_g1024_k5_fnl0
SCRIPT=/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_k5.py
LOG=/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_k5_runtime.log

count_files() {
    local dir="$1"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -maxdepth 1 -name 'pk_rsd_N*.dat' | wc -l
    else
        echo 0
    fi
}

while true; do
    n100=$(count_files "${PK100_DIR}")
    n0=$(count_files "${PK0_DIR}")
    echo "[WAIT] $(date -u '+%F %T') fnl100=${n100} fnl0=${n0}" | tee -a "${LOG}"
    if [[ "${n100}" -ge 50 && "${n0}" -ge 50 ]]; then
        break
    fi
    sleep 120
done

source /global/homes/l/lzy/anaconda3/etc/profile.d/conda.sh
conda activate desilike
python "${SCRIPT}" 2>&1 | tee -a "${LOG}"
