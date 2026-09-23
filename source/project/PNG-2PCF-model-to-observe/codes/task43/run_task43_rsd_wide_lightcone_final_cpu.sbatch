#!/bin/bash -l
#SBATCH -J t43_wide_final
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/wide_final_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/wide_final_%j.err

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
WIDE_ROOT="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/lightcone_wide_zobs0p4_1p1"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_wide_zobs0p4_1p1_x25.jsonl"
XI_SUMMARY="${WIDE_ROOT}/summary/task43_rsd_wide_lightcone_x25_mean_xi0_s30_350_ds10.npz"
WINDOW_ROOT="${WIDE_ROOT}/formal_gic_windows"
MEAN_WINDOW="${WINDOW_ROOT}/task43_rsd_wide_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz"
RAW_COV="${WIDE_ROOT}/covariance/task43_rsd_wide_ph000_jaxpower_raw_xi0_rsdpoles024_win02468_mesh64_p1_b1cov2p604_sigmas7p566_nran100k_ndata50k_seed20260817_kbox_fkpP010000_s30_350_ds10.npz"
RR_PREFIX="${WIDE_ROOT}/covariance/task43_rsd_wide_ph000_rr_smu_nran300k_s30_350_ds10_nmu40_seed430420"
DECONV_PREFIX="${WIDE_ROOT}/covariance/task43_rsd_wide_ph000_jaxpower_rrdeconv_ell0_rsdpoles024_win02468_mesh64_p1_b1cov2p604_sigmas7p566_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10"
XI_FIT_PREFIX="${WIDE_ROOT}/closure/task43_rsd_wide_lightcone_x25_xi0_smin50_formalgic_lorentzian"
XI_FIT_PDF="${PROJECT_ROOT}/plots/task43/rsd_validation/task43_rsd_wide_zobs0p4_1p1_x25_xi0_smin50_formalgic_lorentzian.pdf"
PK_PAYLOAD="${WIDE_ROOT}/pk_summary/task43_rsd_wide_lightcone_pk0_x25_kmin0p003271970_kmax0p10_l0only_16bin.npz"
COMPARE_PREFIX="${WIDE_ROOT}/comparison/task43_rsd_wide_lightcone_x25_pk0_vs_xi0_smin50_l0only_longchain"
COMPARE_FIT_PDF="${PROJECT_ROOT}/plots/task43/rsd_validation/task43_rsd_wide_zobs0p4_1p1_x25_pk0_vs_xi0_smin50_l0only_longchain.pdf"
COMPARE_CONTOUR_PDF="${PROJECT_ROOT}/plots/task43/rsd_validation/task43_rsd_wide_zobs0p4_1p1_x25_pk0_vs_xi0_smin50_l0only_contours.pdf"
COMPARE_CONTOUR_JSON="${COMPARE_CONTOUR_PDF%.pdf}.json"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8 inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

for path in "${XI_SUMMARY}" "${MEAN_WINDOW}" "${RAW_COV}" "${RR_PREFIX}.npz" "${RR_PREFIX}.json" "${PK_PAYLOAD}"; do
  [[ -s "${path}" ]] || { echo "[error] missing ${path}" >&2; exit 3; }
done

if [[ -s "${DECONV_PREFIX}.npz" && -s "${DECONV_PREFIX}.json" ]]; then
  echo "[skip] existing RR-deconvolved xi0 covariance"
elif [[ -e "${DECONV_PREFIX}.npz" || -e "${DECONV_PREFIX}.json" ]]; then
  echo "[error] partial RR-deconvolved covariance" >&2
  exit 4
else
  python -u codes/task43/task43_apply_rr_smu_deconvolution_to_covariance.py \
    --covariance-path "${RAW_COV}" --rr-smu-path "${RR_PREFIX}.npz" \
    --output-prefix "${DECONV_PREFIX}" --xi-summary "${XI_SUMMARY}" \
    --rr-window-kind RR --rr-fkp-norm-json "${RR_PREFIX}.json" \
    --ells 0 --ellsin 0 --resolution 1
fi

if [[ -s "${XI_FIT_PREFIX}.npz" && -s "${XI_FIT_PREFIX}.json" && -s "${XI_FIT_PDF}" ]]; then
  echo "[skip] existing wide xi0 fit"
elif [[ -e "${XI_FIT_PREFIX}.npz" || -e "${XI_FIT_PREFIX}.json" || -e "${XI_FIT_PDF}" ]]; then
  echo "[error] partial wide xi0 fit" >&2
  exit 5
else
  python -u codes/task43/task43_fit_rsd_wide_lightcone_xi0_smin50.py \
    --manifest "${MANIFEST}" --summary-path "${XI_SUMMARY}" \
    --covariance-path "${DECONV_PREFIX}.npz" --mean-window-path "${MEAN_WINDOW}" \
    --phase-window-root "${WINDOW_ROOT}" --output-prefix "${XI_FIT_PREFIX}" --plot "${XI_FIT_PDF}" \
    --threads 8 --nsub-window 200000 --window-seed-base 430340 \
    --nmu 64 --sigma-grid-step 0.05 --nwalkers 48 --nsteps 8000 --burnin 2000 --seed 430450
fi

if [[ -s "${COMPARE_PREFIX}.npz" && -s "${COMPARE_PREFIX}.json" && -s "${COMPARE_FIT_PDF}" && -s "${COMPARE_CONTOUR_PDF}" && -s "${COMPARE_CONTOUR_JSON}" ]]; then
  echo "[skip] existing wide P0-vs-xi0 comparison"
elif [[ -e "${COMPARE_PREFIX}.npz" || -e "${COMPARE_PREFIX}.json" || -e "${COMPARE_FIT_PDF}" || -e "${COMPARE_CONTOUR_PDF}" || -e "${COMPARE_CONTOUR_JSON}" ]]; then
  echo "[error] partial wide P0-vs-xi0 comparison" >&2
  exit 6
else
  python -u codes/task43/task43_fit_rsd_lightcone_pk0_vs_xi0_smin50.py \
    --threads 8 --pk-payload "${PK_PAYLOAD}" --xi-prefix "${XI_FIT_PREFIX}" \
    --sigma-grid-step 0.05 --nmu 96 --nwalkers 64 --nsteps 30000 --burnin 5000 --seed 430650 \
    --output-prefix "${COMPARE_PREFIX}" --fit-pdf "${COMPARE_FIT_PDF}" --contour-pdf "${COMPARE_CONTOUR_PDF}"
fi

echo "[done] wide monopole-only final fits $(date -u +%FT%TZ)"
