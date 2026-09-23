#!/usr/bin/env bash
set -euo pipefail

# 执行逻辑大纲：
# 1. 用 ph000 matching random 构造 32x20 RR(s,mu)。
# 2. 用冻结的 jaxpower 设置构造 32x32 raw Gaussian covariance。
# 3. 做 ell=0 RR deconvolution，并运行独立的旧区间 bridge 审计。
# 所有步骤串行、可恢复，并用 taskset 把登录节点总占用限制到 8 核以内。
ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE="${ROOT}/codes/task43"
SCAN="${ROOT}/outputs/task43_outputs/rmin_scan"
COV="${SCAN}/covariance"
MOVED="${ROOT}/plots/outputs/task43_outputs"
MANIFEST="${SCAN}/manifests/task43_rmin_scan_mmin1p4e13_x25.jsonl"
FKP="${MOVED}/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
XI="${SCAN}/summary/task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000.npz"
CPUSET="${TASK43_RMIN_CPUSET:-6-13}"
RR_PREFIX="${COV}/task43_rr_smu_ph000_nran100k_seed20260702_s30_350_ds10_nmu20"
RAW_PREFIX="${COV}/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"
FINAL_PREFIX="${COV}/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"
OLD_RR="${MOVED}/summary/task43_rr_smu_window_smoke_ph000_nran100k_s50_350_ds10_nmu20_midpoint.npz"
RR_NORM="4.89250917556916e-10"

for path in "${MANIFEST}" "${FKP}" "${XI}" "${OLD_RR}"; do
  [[ -s "${path}" ]] || { echo "[fatal] 缺少输入：${path}" >&2; exit 2; }
done
NCPU=$(taskset -c "${CPUSET}" nproc)
if (( NCPU < 1 || NCPU > 8 )); then
  echo "[fatal] CPU affinity 必须包含 1..8 核：${CPUSET} -> ${NCPU}" >&2
  exit 2
fi

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "${COV}" "${SCAN}/audits"
cd "${ROOT}"

run_python() { taskset -c "${CPUSET}" python -u "$@"; }
pair_state() {
  local prefix="$1"
  if [[ -s "${prefix}.npz" && -s "${prefix}.json" ]]; then echo complete
  elif [[ ! -e "${prefix}.npz" && ! -e "${prefix}.json" ]]; then echo missing
  else echo partial; fi
}

state=$(pair_state "${RR_PREFIX}")
[[ "${state}" != partial ]] || { echo "[fatal] RR pair 不完整" >&2; exit 2; }
if [[ "${state}" == missing ]]; then
  run_python "${CODE}/task43_build_rmax_scan_rr_smu.py" \
    --manifest "${MANIFEST}" --fkp-summary "${FKP}" --output-prefix "${RR_PREFIX}" \
    --phase-index 0 --max-random 100000 --seed 20260702 --p0 10000 \
    --s-min 30 --s-max 350 --s-step 10 --nmu 20 --nthreads "${NCPU}" --old-rr "${OLD_RR}"
fi

state=$(pair_state "${RAW_PREFIX}")
[[ "${state}" != partial ]] || { echo "[fatal] raw covariance pair 不完整" >&2; exit 2; }
if [[ "${state}" == missing ]]; then
  run_python "${CODE}/task43_diagnose_jaxpower_lightcone_covariance.py" \
    --manifest "${MANIFEST}" --fkp-summary "${FKP}" --xi-summary "${XI}" \
    --output-prefix "${RAW_PREFIX}" --phase-index 0 --p0 10000 \
    --max-random 100000 --max-data 50000 --subsample-seed 20260702 \
    --meshsize 64 --mesh-pad 400 --window-s-max 3600 --window-ds 2 \
    --window-basis bessel --window-ells 0 --interpolate-window \
    --interpolate-window-ncoords 8192 --interpolate-window-log10-min -2 \
    --interpolate-window-log10-max 8 --covariance-flags smooth,fftlog \
    --s-edge-min 30 --s-edge-max 350 --s-edge-step 10 \
    --kmin 0.0001 --kmax 3.0001 --dk 0.002 --los local \
    --resampler cic --interlacing 1 --b1-cov 2.5 --fnl-cov 0 \
    --p-fixed 1.0 --sn0-fixed 0 --cosmology abacus_c000 --floor-fraction 1.0e-10
fi

state=$(pair_state "${FINAL_PREFIX}")
[[ "${state}" != partial ]] || { echo "[fatal] final covariance pair 不完整" >&2; exit 2; }
if [[ "${state}" == missing ]]; then
  run_python "${CODE}/task43_apply_rr_smu_deconvolution_to_covariance.py" \
    --covariance-path "${RAW_PREFIX}.npz" --rr-smu-path "${RR_PREFIX}.npz" \
    --output-prefix "${FINAL_PREFIX}" --xi-summary "${XI}" \
    --rr-window-kind RR --rr-fkp-norm "${RR_NORM}" --ells 0 --resolution 1 \
    --floor-fraction 1.0e-10
fi

run_python "${CODE}/task43_audit_rmin_scan_covariance.py" \
  --rr "${RR_PREFIX}.npz" --raw "${RAW_PREFIX}.npz" --final "${FINAL_PREFIX}.npz"
echo "[done] rmin-scan 32-bin jaxpower covariance 通过全部 gate"
