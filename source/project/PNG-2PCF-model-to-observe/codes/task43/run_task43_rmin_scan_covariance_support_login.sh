#!/usr/bin/env bash
set -euo pipefail

# 执行逻辑大纲：保持 window、mesh、RR 与科学参数不变，只把 covariance
# 积分上限从 k=3.0001 延伸到 4.0001；随后生成匹配 Fisher 并比较低-r bins。
ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE="${ROOT}/codes/task43"
SCAN="${ROOT}/outputs/task43_outputs/rmin_scan"
MOVED="${ROOT}/plots/outputs/task43_outputs"
COV="${SCAN}/covariance"
MANIFEST="${SCAN}/manifests/task43_rmin_scan_mmin1p4e13_x25.jsonl"
FKP="${MOVED}/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
XI="${SCAN}/summary/task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000.npz"
RR="${COV}/task43_rr_smu_ph000_nran100k_seed20260702_s30_350_ds10_nmu20.npz"
RAW_PREFIX="${COV}/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_4000_dk002_p1p0_s30_350_ds10"
FINAL_PREFIX="${COV}/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_4000_dk002_p1p0_s30_350_ds10"
OP="${SCAN}/operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000_s30_350_ds10.npz"
PK="${MOVED}/pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
CPUSET="${TASK43_RMIN_CPUSET:-6-13}"
RR_NORM="4.89250917556916e-10"

for path in "${MANIFEST}" "${FKP}" "${XI}" "${RR}" "${OP}" "${PK}"; do
  [[ -s "${path}" ]] || { echo "[fatal] 缺少输入：${path}" >&2; exit 2; }
done
NCPU=$(taskset -c "${CPUSET}" nproc)
if (( NCPU < 1 || NCPU > 8 )); then echo "[fatal] CPU affinity 超限" >&2; exit 2; fi

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "${ROOT}"
run_python() { taskset -c "${CPUSET}" python -u "$@"; }

if [[ ! -e "${RAW_PREFIX}.npz" && ! -e "${RAW_PREFIX}.json" ]]; then
  run_python "${CODE}/task43_diagnose_jaxpower_lightcone_covariance.py" \
    --manifest "${MANIFEST}" --fkp-summary "${FKP}" --xi-summary "${XI}" \
    --output-prefix "${RAW_PREFIX}" --phase-index 0 --p0 10000 \
    --max-random 100000 --max-data 50000 --subsample-seed 20260702 \
    --meshsize 64 --mesh-pad 400 --window-s-max 3600 --window-ds 2 \
    --window-basis bessel --window-ells 0 --interpolate-window \
    --interpolate-window-ncoords 8192 --interpolate-window-log10-min -2 \
    --interpolate-window-log10-max 8 --covariance-flags smooth,fftlog \
    --s-edge-min 30 --s-edge-max 350 --s-edge-step 10 \
    --kmin 0.0001 --kmax 4.0001 --dk 0.002 --los local --resampler cic \
    --interlacing 1 --b1-cov 2.5 --fnl-cov 0 --p-fixed 1.0 --sn0-fixed 0 \
    --cosmology abacus_c000 --floor-fraction 1.0e-10
elif [[ ! -s "${RAW_PREFIX}.npz" || ! -s "${RAW_PREFIX}.json" ]]; then
  echo "[fatal] kmax4 raw pair 不完整" >&2; exit 2
fi

if [[ ! -e "${FINAL_PREFIX}.npz" && ! -e "${FINAL_PREFIX}.json" ]]; then
  run_python "${CODE}/task43_apply_rr_smu_deconvolution_to_covariance.py" \
    --covariance-path "${RAW_PREFIX}.npz" --rr-smu-path "${RR}" \
    --output-prefix "${FINAL_PREFIX}" --xi-summary "${XI}" \
    --rr-window-kind RR --rr-fkp-norm "${RR_NORM}" --ells 0 --resolution 1 \
    --floor-fraction 1.0e-10
elif [[ ! -s "${FINAL_PREFIX}.npz" || ! -s "${FINAL_PREFIX}.json" ]]; then
  echo "[fatal] kmax4 final pair 不完整" >&2; exit 2
fi

run_python "${CODE}/task43_rmin_scan_fisher.py" --xi "${XI}" \
  --covariance "${FINAL_PREFIX}.npz" --operator "${OP}" --pk-payload "${PK}" \
  --output-prefix "${SCAN}/fisher/task43_rmin_scan_fisher_information_kmax4"
run_python "${CODE}/task43_compare_rmin_covariance_support.py"
echo "[done] rmin-scan covariance kmax=3/4 support A/B 通过"
