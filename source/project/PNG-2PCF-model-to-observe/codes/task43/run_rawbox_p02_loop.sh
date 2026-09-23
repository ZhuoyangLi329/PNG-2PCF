#!/bin/bash
# Serial x25 rawbox P0+P2 measurement (Task 4.3.2 Phase 2), 6 cores, detached.
set -eo pipefail
cd /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/task43:/pscratch/sd/l/lzy/desi-clustering:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MANIFEST=outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_x25.jsonl
for i in $(seq 0 24); do
  p=$(printf "ph%03d" "$i")
  taskset -c 6-11 python -u codes/task43/task43_measure_rsd_rawbox_p02_jaxpower.py --manifest "$MANIFEST" --phase "$p" --threads 6
done
