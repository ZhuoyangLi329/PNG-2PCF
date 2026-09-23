#!/bin/bash
# CPU-only Task4.3.2 lightcone diagnostic postprocessing; never submit to Slurm.

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/lightcone"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_x25.jsonl"
RAW_PREFIX="${OUTPUT_ROOT}/covariance/task43_rsd_ph000_jaxpower_raw_rsdpoles024_win02468_mesh64_p1_b1cov2p604_sigmas7p566_nran100k_ndata50k_seed20260816_kbox_fkpP010000_s30_350_ds10"
RR_PREFIX="${OUTPUT_ROOT}/covariance/task43_rsd_ph000_rr_smu_nran300k_s30_350_ds10_nmu40_seed430320"
DECONV_PREFIX="${OUTPUT_ROOT}/covariance/task43_rsd_ph000_jaxpower_rrdeconv_ell02_rsdpoles024_win02468_mesh64_p1_b1cov2p604_sigmas7p566_nran100k_ndata50k_seed20260816_rrnran300k_rrseed430320_kbox_fkpP010000_s30_350_ds10"
SUMMARY="${OUTPUT_ROOT}/summary/task43_rsd_lightcone_x25_mean_xi02_s30_350_ds10.npz"
SUMMARY_JSON="${SUMMARY%.npz}.json"
SUMMARY_PDF="${PROJECT_ROOT}/plots/task43/rsd_validation/task43_rsd_lightcone_x25_mean_xi02.pdf"
MEAN_WINDOW="${OUTPUT_ROOT}/formal_gic_windows/task43_rsd_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz"
FIT_PREFIX="${OUTPUT_ROOT}/closure/task43_rsd_lightcone_x25_jaxpower_rrdeconv_rrnran300k_fulldiscrete_lorentzian"
FIT_JSON="${FIT_PREFIX}.json"
FIT_NPZ="${FIT_PREFIX}.npz"
FIT_PDF="${PROJECT_ROOT}/plots/task43/rsd_validation/task43_rsd_lightcone_x25_jaxpower_rrdeconv_rrnran300k_fulldiscrete_lorentzian.pdf"
FINAL_AUDIT="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/audits/task43_rsd_validation_final_audit_rrnran300k_verified.json"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

if [[ ! -s "${SUMMARY}" || ! -s "${SUMMARY_JSON}" || ! -s "${SUMMARY_PDF}" ]]; then
  if [[ -e "${SUMMARY}" || -e "${SUMMARY_JSON}" || -e "${SUMMARY_PDF}" ]]; then
    echo "[error] partial x25 lightcone summary exists" >&2
    exit 3
  fi
  taskset -c 6 python -u codes/task43/task43_summarize_rsd_lightcone_x25.py
else
  echo "[skip] existing validated-summary candidate ${SUMMARY}"
fi

if [[ -s "${MEAN_WINDOW}" && -s "${MEAN_WINDOW%.npz}.json" ]]; then
  echo "[skip] existing x25 formal-GIC window ${MEAN_WINDOW}"
elif [[ -e "${MEAN_WINDOW}" || -e "${MEAN_WINDOW%.npz}.json" ]]; then
  echo "[error] partial x25 mean formal-GIC window exists" >&2
  exit 7
else
  for index in $(seq 0 24); do
    printf -v phase 'ph%03d' "${index}"
    taskset -c 6-13 python -u codes/task43/task43_build_rsd_formal_gic_window.py \
      --manifest "${MANIFEST}" --phase "${phase}" \
      --nsub 200000 --seed-base 430340 --nthreads 8
  done
  taskset -c 6 python -u codes/task43/task43_average_rsd_formal_gic_windows.py
fi

for path in "${RAW_PREFIX}.npz" "${RAW_PREFIX}.json" "${RR_PREFIX}.npz" "${RR_PREFIX}.json"; do
  [[ -s "${path}" ]] || { echo "[error] missing ${path}" >&2; exit 4; }
done

if [[ ! -s "${DECONV_PREFIX}.npz" || ! -s "${DECONV_PREFIX}.json" ]]; then
  if [[ -e "${DECONV_PREFIX}.npz" || -e "${DECONV_PREFIX}.json" ]]; then
    echo "[error] partial RR-deconvolved covariance exists" >&2
    exit 5
  fi
  taskset -c 6-13 python -u codes/task43/task43_apply_rr_smu_deconvolution_to_covariance.py \
    --covariance-path "${RAW_PREFIX}.npz" \
    --rr-smu-path "${RR_PREFIX}.npz" \
    --output-prefix "${DECONV_PREFIX}" \
    --xi-summary "${SUMMARY}" \
    --rr-window-kind RR --rr-fkp-norm-json "${RR_PREFIX}.json" \
    --ells 0,2 --ellsin 0,2 --resolution 1
else
  echo "[skip] existing RR-deconvolved covariance ${DECONV_PREFIX}.npz"
fi

python - "${DECONV_PREFIX}.npz" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as payload:
    edges = np.asarray(payload["s_edges"], dtype="f8")
    ells = tuple(int(value) for value in np.asarray(payload["ells"]).ravel())
    covariance = np.asarray(payload["covariance_single_realization"], dtype="f8")
expected = np.arange(30.0, 351.0, 10.0)
if not np.array_equal(edges, expected) or ells != (0, 2) or covariance.shape != (64, 64):
    raise RuntimeError(f"bad deconvolution contract: edges={edges}, ells={ells}, shape={covariance.shape}")
eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
if eigenvalues[0] <= 0.0:
    raise RuntimeError(f"non-positive covariance eigenvalue {eigenvalues[0]}")
print(f"[covariance] shape={covariance.shape} condition={np.linalg.cond(covariance):.6e}")
PY

if [[ ! -s "${FIT_JSON}" || ! -s "${FIT_NPZ}" || ! -s "${FIT_PDF}" ]]; then
  if [[ -e "${FIT_JSON}" || -e "${FIT_NPZ}" || -e "${FIT_PDF}" ]]; then
    echo "[error] partial lightcone diagnostic closure exists" >&2
    exit 8
  fi
  set +e
  taskset -c 6-13 python -u codes/task43/task43_fit_rsd_lightcone_x25.py \
    --summary-path "${SUMMARY}" --covariance-path "${DECONV_PREFIX}.npz" \
    --mean-window-path "${MEAN_WINDOW}" --output-prefix "${FIT_PREFIX}" --plot "${FIT_PDF}" \
    --threads 8 --nmu 64 --sigma-grid-step 0.05 \
    --nwalkers 48 --nsteps 8000 --burnin 2000 --seed 430350
  fit_status=$?
  set -e
  if [[ "${fit_status}" -ne 0 && "${fit_status}" -ne 2 ]]; then
    exit "${fit_status}"
  fi
  [[ -s "${FIT_JSON}" && -s "${FIT_NPZ}" && -s "${FIT_PDF}" ]] || {
    echo "[error] fit exited ${fit_status} without complete immutable outputs" >&2
    exit 9
  }
else
  echo "[skip] existing lightcone diagnostic ${FIT_JSON}"
fi

if [[ ! -s "${FINAL_AUDIT}" ]]; then
  [[ ! -e "${FINAL_AUDIT}" ]] || { echo "[error] partial final audit exists" >&2; exit 10; }
  taskset -c 6 python -u codes/task43/task43_finalize_rsd_validation.py --output "${FINAL_AUDIT}"
else
  echo "[skip] existing final audit ${FINAL_AUDIT}"
fi
