#!/bin/bash -l
#SBATCH -J t43_wide_xcov
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1900M
#SBATCH -t 00:10:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/wide_xi0_covariance_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/wide_xi0_covariance_%j.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
WIDE_ROOT="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/lightcone_wide_zobs0p4_1p1"
DATA="${WIDE_ROOT}/covariance_inputs/task43_rsd_covinput_AbacusSummit_base_c000_ph000_zobs0p4_1p1_data.npz"
RANDOM_CAT="${WIDE_ROOT}/covariance_inputs/task43_rsd_covinput_AbacusSummit_base_c000_ph000_zobs0p4_1p1_random_x25.npz"
ZEFF="${WIDE_ROOT}/fkp/task43_rsd_fkp_AbacusSummit_base_c000_ph000_mmin1p4e13_zobs0p4_1p1_dz0p01.npz"
XI="${WIDE_ROOT}/xi/task43_rsd_xi0_AbacusSummit_base_c000_ph000_mmin1p4e13_zobs0p4_1p1_x25_s30_350_ds10.npz"
RAW_PREFIX="${WIDE_ROOT}/covariance/task43_rsd_wide_ph000_jaxpower_raw_xi0_rsdpoles024_win02468_mesh64_p1_b1cov2p604_sigmas7p566_nran100k_ndata50k_seed20260817_kbox_fkpP010000_s30_350_ds10"

for path in "${DATA}" "${RANDOM_CAT}" "${ZEFF}" "${XI}"; do
  [[ -s "${path}" ]] || { echo "[error] missing ${path}" >&2; exit 3; }
done
if [[ -s "${RAW_PREFIX}.npz" && -s "${RAW_PREFIX}.json" ]]; then
  echo "[skip] existing ${RAW_PREFIX}.npz"
  exit 0
fi
if [[ -e "${RAW_PREFIX}.npz" || -e "${RAW_PREFIX}.json" ]]; then
  echo "[error] partial covariance ${RAW_PREFIX}" >&2
  exit 4
fi

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${PROJECT_ROOT}"

srun --exact -n 1 -c 8 --cpu-bind=cores python -u codes/task44/task44_make_lrg2_jaxpower_covariance.py \
  --sample lrgall --data "${DATA}" --random "${RANDOM_CAT}" --zeff-path "${ZEFF}" \
  --xi-path "${XI}" --output-prefix "${RAW_PREFIX}" --p0 10000 \
  --max-data 50000 --max-random 100000 --subsample-seed 20260817 \
  --meshsize 64 --mesh-pad 400 --window-s-max 3600 --window-ds 2 --window-basis bessel \
  --window-ells 0,2,4,6,8 --theory-ells 0,2,4 --interpolate-window \
  --covariance-flags smooth,fftlog --s-edge-min 30 --s-edge-max 350 --s-edge-step 10 \
  --kmin 0.0031415926535897933 --kmax 3.0001 --dk 0.002 --los local \
  --resampler cic --interlacing 1 --output-compression none \
  --b1-cov 2.6037 --fnl-cov 0 --p-fixed 1.0 --sigma-s-cov 7.566

echo "[done] ${RAW_PREFIX}.npz"
