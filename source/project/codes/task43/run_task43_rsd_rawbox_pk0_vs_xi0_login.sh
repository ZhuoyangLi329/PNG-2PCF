#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
THREADS="${TASK43_RSD_PK0_THREADS:-8}"
if (( THREADS < 1 || THREADS > 8 )); then
  echo "TASK43_RSD_PK0_THREADS must be in [1, 8], got ${THREADS}" >&2
  exit 2
fi
cd "${PROJECT_ROOT}"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python -u codes/task43/task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py \
  --threads "${THREADS}" --pk-kmin-edge 0.003 --nsteps 30000 --burnin 5000 \
  --output-prefix \
  outputs/task43_outputs/rsd_validation/rawbox/comparison/task43_rsd_rawbox_x25_pk0_kmin0p003_vs_xi0_smin50_l0only_longchain \
  --plot \
  plots/task43/rsd_validation/task43_rsd_rawbox_x25_pk0_kmin0p003_vs_xi0_smin50_l0only_longchain.pdf
