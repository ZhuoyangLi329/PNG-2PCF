#!/bin/bash -l
#SBATCH -J t45_pk_kmax
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task4/task45_pk_kmax_scan_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task4/task45_pk_kmax_scan_%j.err

set -euo pipefail

PROJECT_DIR="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
TASK45="${PROJECT_DIR}/codes/task4/task45_quijote_ultranest.py"
PLOT_OVERLAY="${PROJECT_DIR}/codes/task4/task45_plot_free_sn0_kmax_overlay.py"
LOG_DIR="${PROJECT_DIR}/codes/logs/task4"
IMPORTANT_DIR="${PROJECT_DIR}/plots/task4/important_4p1_4p2"
OVERLAY_PDF="${PROJECT_DIR}/plots/task4/quijote_ultranest_free_sn0/task45_quijote_ultranest_free_sn0_alltags_constraints_with_pk_kmax_scan.pdf"
IMPORTANT_LINK="${IMPORTANT_DIR}/4p1_diagnostic_task45_quijote_ultranest_free_sn0_pk_kmax_scan.pdf"

mkdir -p "${LOG_DIR}/desilike_config" "${LOG_DIR}/mplconfig" "${LOG_DIR}/xdg_cache" "${IMPORTANT_DIR}"

source ~/.bashrc >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
conda activate desilike

export PYTHONUNBUFFERED=1
export DESILIKE_CONFIG_DIR="${LOG_DIR}/desilike_config"
export MPLCONFIGDIR="${LOG_DIR}/mplconfig"
export XDG_CACHE_HOME="${LOG_DIR}/xdg_cache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JAX_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

cd "${PROJECT_DIR}"

echo "[info] start=$(date -u)"
echo "[info] task45=${TASK45}"

for spec in "0.06 free_sn0_kmax0p06" "0.10 free_sn0_kmax0p10"; do
  read -r KMAX LABEL <<< "${spec}"
  echo "[info] run P(k)-only free-sn0 kmax_fit=${KMAX} label=${LABEL}"
  python -u "${TASK45}" \
    --tags fid,LCp50,LCp100 \
    --cases pk_binavg \
    --free-sn0 \
    --output-label "${LABEL}" \
    --sn0-prior=-1,1 \
    --kmax-fit "${KMAX}" \
    --min-live 120 \
    --dlogz 0.5 \
    --min-ess 300 \
    --max-ncalls 20000 \
    --overwrite \
    --quiet
done

python -u "${PLOT_OVERLAY}"
ln -sfn "../quijote_ultranest_free_sn0/$(basename "${OVERLAY_PDF}")" "${IMPORTANT_LINK}"

echo "[info] overlay_pdf=${OVERLAY_PDF}"
echo "[info] important_link=${IMPORTANT_LINK}"
echo "[info] end=$(date -u)"
