#!/bin/bash -l
#SBATCH -J t43_wide_xprep
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/wide_xi0_prepare_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/wide_xi0_prepare_%j.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
WIDE_ROOT="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/lightcone_wide_zobs0p4_1p1"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_wide_zobs0p4_1p1_x25.jsonl"
SUMMARY_PREFIX="${WIDE_ROOT}/summary/task43_rsd_wide_lightcone_x25_mean_xi0_s30_350_ds10"
SUMMARY_PDF="${PROJECT_ROOT}/plots/task43/rsd_validation/task43_rsd_wide_zobs0p4_1p1_x25_mean_xi0.pdf"
WINDOW_ROOT="${WIDE_ROOT}/formal_gic_windows"
MEAN_WINDOW="${WINDOW_ROOT}/task43_rsd_wide_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz"
DATA="${WIDE_ROOT}/covariance_inputs/task43_rsd_covinput_AbacusSummit_base_c000_ph000_zobs0p4_1p1_data.npz"
RANDOM_CAT="${WIDE_ROOT}/covariance_inputs/task43_rsd_covinput_AbacusSummit_base_c000_ph000_zobs0p4_1p1_random_x25.npz"
RR_PREFIX="${WIDE_ROOT}/covariance/task43_rsd_wide_ph000_rr_smu_nran300k_s30_350_ds10_nmu40_seed430420"

mkdir -p "${PROJECT_ROOT}/.task43_home"
export HOME="${PROJECT_ROOT}/.task43_home"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8 inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

python -u codes/task43/task43_summarize_rsd_lightcone_x25.py \
  --manifest "${MANIFEST}" --ells 0 --output-prefix "${SUMMARY_PREFIX}" --plot "${SUMMARY_PDF}"

python -u codes/task43/task43_prepare_rsd_covariance_inputs.py --manifest "${MANIFEST}" --phase ph000

for index in $(seq 0 24); do
  printf -v phase 'ph%03d' "${index}"
  python -u codes/task43/task43_build_rsd_formal_gic_window.py \
    --manifest "${MANIFEST}" --phase "${phase}" --nsub 200000 --seed-base 430340 --nthreads 8
done
python -u codes/task43/task43_average_rsd_formal_gic_windows.py \
  --nsub 200000 --seed-base 430340 --phase-window-root "${WINDOW_ROOT}" --output "${MEAN_WINDOW}"

if [[ -s "${RR_PREFIX}.npz" && -s "${RR_PREFIX}.json" ]]; then
  echo "[skip] validated-candidate RR files already exist ${RR_PREFIX}"
elif [[ -e "${RR_PREFIX}.npz" || -e "${RR_PREFIX}.json" ]]; then
  echo "[error] partial RR product ${RR_PREFIX}" >&2
  exit 3
else
  python -u codes/task44/task44_build_lrg2_rr_smu_window.py \
    --sample lrgall --data "${DATA}" --random "${RANDOM_CAT}" --output-prefix "${RR_PREFIX}" \
    --max-random 300000 --seed 430420 --s-min 30 --s-max 350 --s-step 10 --nmu 40 --nthreads 8
fi

echo "[done] wide xi0 preparation $(date -u +%FT%TZ)"
