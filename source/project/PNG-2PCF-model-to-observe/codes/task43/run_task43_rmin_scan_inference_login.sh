#!/usr/bin/env bash
set -euo pipefail

# 执行逻辑大纲：编译唯一 32-bin radial-RIC operator，桥接旧 30-bin basis，
# 运行 xi/P(k) Fisher 分解，再对 rmin=30/40/50 运行三条同设置长链。
ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
PY="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
SCAN="${ROOT}/outputs/task43_outputs/rmin_scan"
MOVED="${ROOT}/plots/outputs/task43_outputs"
XI="${SCAN}/summary/task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000.npz"
COV="${SCAN}/covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10.npz"
FKP="${MOVED}/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
KERNEL="${MOVED}/ric_singleterm/kernels/task43_ric_factorized_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712.npz"
PK="${MOVED}/pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
W2="${MOVED}/summary/task43_formal_gic_window_mmin1p4e13_x25_ph000_fkpP010000_L2000_nsub200000_seed20260703.npz"
OLD_OP="${MOVED}/ric_singleterm/operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
OPDIR="${SCAN}/operators"
OP="${OPDIR}/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000_s30_350_ds10.npz"
OPAUDIT="${OPDIR}/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000_s30_350_ds10_audit.json"
FITROOT="${SCAN}/fits"
CPUSET="${TASK43_RMIN_CPUSET:-6-13}"
RMINS="${TASK43_RMINS:-30 40 50}"

for path in "${PY}" "${XI}" "${COV}" "${FKP}" "${KERNEL}" "${PK}" "${W2}" "${OLD_OP}"; do
  [[ -s "${path}" ]] || { echo "[fatal] 缺少输入：${path}" >&2; exit 2; }
done
NCPU=$(taskset -c "${CPUSET}" nproc)
if (( NCPU < 1 || NCPU > 8 )); then
  echo "[fatal] CPU affinity 必须包含 1..8 核：${CPUSET} -> ${NCPU}" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${ROOT}"
run_python() { taskset -c "${CPUSET}" "${PY}" -u "$@"; }
mkdir -p "${OPDIR}" "${FITROOT}" "${SCAN}/fisher"

if [[ ! -e "${OP}" && ! -e "${OPAUDIT}" ]]; then
  run_python codes/task43/task43_make_ric_factorized_operator.py \
    --kernel "${KERNEL}" --xi-path "${XI}" --pk-payload "${PK}" --w2-path "${W2}" \
    --boxsize 2000 --kmax 5 --ndense 60000 --cosmology abacus_c000 \
    --component-batch 256 --k-batch 512 --output-dir "${OPDIR}" --output-tag s30_350_ds10
elif [[ ! -s "${OP}" || ! -s "${OPAUDIT}" ]]; then
  echo "[fatal] operator pair 不完整" >&2
  exit 2
fi

# 新 operator 的后 30 行必须数值复现旧 50--350 operator，P(k) response 也必须相同。
run_python - "${OP}" "${OLD_OP}" <<'PY'
import sys
from pathlib import Path
import numpy as np

new_path, old_path = map(Path, sys.argv[1:])
with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
    if not np.array_equal(new['target_s_edges'], np.arange(30.0, 351.0, 10.0)):
        raise RuntimeError('新 operator edge 错误')
    bridges = {}
    for key in ('xi_basis_pk_dd', 'xi_basis_alpha_pk_dd', 'xi_basis_alpha2_pk_dd'):
        left, right = np.asarray(new[key], dtype='f8')[2:], np.asarray(old[key], dtype='f8')
        bridges[key] = float(np.linalg.norm(left - right) / max(np.linalg.norm(right), np.finfo('f8').tiny))
    left, right = np.asarray(new['pk_ric_matrix'], dtype='f8'), np.asarray(old['pk_ric_matrix'], dtype='f8')
    bridges['pk_ric_matrix'] = float(np.linalg.norm(left - right) / max(np.linalg.norm(right), np.finfo('f8').tiny))
if any((not np.isfinite(value)) or value > 1e-10 for value in bridges.values()):
    raise RuntimeError(f'operator bridge 失败：{bridges}')
print(f'[bridge:operator] {bridges}')
PY

run_python codes/task43/task43_rmin_scan_fisher.py --xi "${XI}" --covariance "${COV}" --operator "${OP}" --pk-payload "${PK}"

for rmin in ${RMINS}; do
  case "${rmin}" in 30|40|50) ;; *) echo "[fatal] 不支持 rmin=${rmin}" >&2; exit 2 ;; esac
  OUT="${FITROOT}/rmin${rmin}"
  SUMMARY="${OUT}/task43_minimal_closure_mcmc_summary.json"
  SAMPLES="${OUT}/task43_mcmc_radial_singleterm_samples.npz"
  if [[ ! -e "${SUMMARY}" && ! -e "${SAMPLES}" ]]; then
    run_python codes/task43/task43_fit_minimal_closure.py \
      --xi-path "${XI}" --output-dir "${OUT}" --fit-target mean \
      --covariance-mode npz --covariance-path "${COV}" --covariance-key covariance_single_realization \
      --rmin "${rmin}" --rmax 350 --models radial_singleterm --radial-ric-operator "${OP}" \
      --fkp-summary "${FKP}" --p0 10000 --output-tag rmin_scan \
      --cosmology abacus_c000 --theory-boxsize 2000 --p-fixed 1.0 --png-order full \
      --sn0-fixed 0 --kmax 5 --ndense 60000 --xi-kernel shell-averaged \
      --nwalkers 64 --nsteps 20000 --burnin 5000 --seed 20260803
  elif [[ ! -s "${SUMMARY}" || ! -s "${SAMPLES}" ]]; then
    echo "[fatal] rmin=${rmin} fit pair 不完整" >&2
    exit 2
  fi
  run_python - "${SUMMARY}" "${SAMPLES}" "${rmin}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

summary_path, samples_path = map(Path, sys.argv[1:3])
rmin = int(sys.argv[3])
summary = json.loads(summary_path.read_text(encoding='utf-8'))
fit = summary.get('fit_range', {})
expected_n = (350 - rmin) // 10
if summary.get('status') != 'done' or int(fit.get('nbins', -1)) != expected_n or int(fit.get('rmin', -1)) != rmin:
    raise RuntimeError(f'fit contract 错误：{fit}')
if summary.get('covariance', {}).get('key') != 'covariance_single_realization':
    raise RuntimeError('fit 没有使用 single-realization covariance')
model = summary['models'][0]
if model.get('parameter_names') != ['fnl_loc', 'b1'] or model.get('model') != 'radial_singleterm':
    raise RuntimeError('模型/参数口径错误')
with np.load(samples_path, allow_pickle=False) as data:
    samples = np.asarray(data['samples'], dtype='f8')
    logp = np.asarray(data['log_prob'], dtype='f8')
if samples.shape != (960000, 2) or logp.shape != (960000,) or not np.all(np.isfinite(samples)):
    raise RuntimeError(f'chain shape/finite 错误：{samples.shape}, {logp.shape}')
print(f'[fit:validated] rmin={rmin} nbins={expected_n} samples={samples.shape}')
PY
done
echo "[done] rmin-scan Fisher 与三条长链均已生成并通过基础 gate"
