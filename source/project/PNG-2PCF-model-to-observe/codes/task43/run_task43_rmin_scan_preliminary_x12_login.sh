#!/usr/bin/env bash
set -euo pipefail

# 执行逻辑大纲：
# 1. 冻结 ph000--ph(N-1)，生成独立 xN mean，不吸收后台后来完成的 phase。
# 2. 用与最终流程相同的 jaxpower/RR-deconvolution 设置生成 preliminary
#    single-lightcone covariance，并桥接旧 50--350 子块。
# 3. 编译同一 radial single-term RIC operator，运行完整 Fisher 分解。
# 4. 对 rmin=30/40/50 各跑 64x10000、burn=2000 的 preliminary 链，
#    输出隔离 JSON/CSV/PDF；这不是最终 x25 结果，也不覆盖正式目录。
ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE="${ROOT}/codes/task43"
SCAN="${ROOT}/outputs/task43_outputs/rmin_scan"
NREAL="${TASK43_PRELIM_NREAL:-12}"
if ! [[ "${NREAL}" =~ ^[0-9]+$ ]] || (( NREAL < 1 || NREAL > 25 )); then
  echo "[fatal] TASK43_PRELIM_NREAL 必须是 1..25：${NREAL}" >&2
  exit 2
fi
LAST_INDEX=$((NREAL - 1))
TAG="preliminary_x${NREAL}"
SHORT="prelimx${NREAL}"
PRE="${SCAN}/${TAG}"
MOVED="${ROOT}/plots/outputs/task43_outputs"
PY_DESI="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
CPUSET="${TASK43_RMIN_CPUSET:-6-13}"

MANIFEST="${PRE}/manifests/task43_rmin_scan_mmin1p4e13_x${NREAL}.jsonl"
XI_PREFIX="${PRE}/summary/task43_mean_xi_mmin1p4e13_x${NREAL}_s30_350_ds10_fkpP010000"
XI="${XI_PREFIX}.npz"
FKP="${MOVED}/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
RR="${SCAN}/covariance/task43_rr_smu_ph000_nran100k_seed20260702_s30_350_ds10_nmu20.npz"
RAW_PREFIX="${PRE}/covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_${SHORT}_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"
FINAL_PREFIX="${PRE}/covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_${SHORT}_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"
OLD_FINAL="${MOVED}/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
KERNEL="${MOVED}/ric_singleterm/kernels/task43_ric_factorized_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712.npz"
OLD_OP="${MOVED}/ric_singleterm/operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
PK="${MOVED}/pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
W2="${MOVED}/summary/task43_formal_gic_window_mmin1p4e13_x25_ph000_fkpP010000_L2000_nsub200000_seed20260703.npz"
OP="${PRE}/operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000_s30_350_ds10_${SHORT}.npz"
OP_AUDIT="${PRE}/operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000_s30_350_ds10_${SHORT}_audit.json"
FISHER_PREFIX="${PRE}/fisher/task43_rmin_scan_fisher_information_${SHORT}"
FITROOT="${PRE}/fits"
AUDITROOT="${PRE}/audits"
PLOT="${ROOT}/plots/task43/rmin_scan/${TAG}/task43_jaxpower_2pcf_rmin_scan_${TAG}_b1_fnl.pdf"
RR_NORM="4.89250917556916e-10"

NCPU=$(taskset -c "${CPUSET}" nproc)
if (( NCPU < 1 || NCPU > 8 )); then
  echo "[fatal] CPU affinity 必须包含 1..8 核：${CPUSET} -> ${NCPU}" >&2
  exit 2
fi
for path in "${PY_DESI}" "${FKP}" "${RR}" "${OLD_FINAL}" "${KERNEL}" "${OLD_OP}" "${PK}" "${W2}"; do
  [[ -s "${path}" ]] || { echo "[fatal] 缺少 preliminary 输入：${path}" >&2; exit 2; }
done

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PRE}"/{manifests,summary,covariance,operators,fisher,fits,audits} "$(dirname "${PLOT}")"
cd "${ROOT}"
run_desi() { taskset -c "${CPUSET}" "${PY_DESI}" -u "$@"; }
pair_state() {
  local left="$1" right="$2"
  if [[ -s "${left}" && -s "${right}" ]]; then echo complete
  elif [[ ! -e "${left}" && ! -e "${right}" ]]; then echo missing
  else echo partial; fi
}

if [[ ! -e "${MANIFEST}" && ! -e "${MANIFEST%.jsonl}.json" ]]; then
  run_desi "${CODE}/task43_make_rmin_scan_subset_manifest.py" --nreal "${NREAL}" --output "${MANIFEST}"
elif [[ ! -s "${MANIFEST}" || ! -s "${MANIFEST%.jsonl}.json" ]]; then
  echo "[fatal] x${NREAL} manifest pair 不完整" >&2; exit 2
fi

if [[ ! -e "${XI_PREFIX}.npz" && ! -e "${XI_PREFIX}.json" ]]; then
  run_desi "${CODE}/task43_summarize_xi.py" --manifest "${MANIFEST}" \
    --output-prefix "${XI_PREFIX}" --p0 10000 --fkp-summary "${FKP}" --require-all
elif [[ ! -s "${XI_PREFIX}.npz" || ! -s "${XI_PREFIX}.json" ]]; then
  echo "[fatal] x${NREAL} xi summary pair 不完整" >&2; exit 2
fi
run_desi "${CODE}/task43_audit_rmin_scan_measurements.py" --indices "0-${LAST_INDEX}" \
  --output "${AUDITROOT}/task43_rmin_scan_measurement_bridge_${TAG}.json"

# jaxpower covariance 入口依赖 cosmodesi 主环境；把 source 和两个 covariance
# 命令封装在子 shell，避免其 Python-3.12 PYTHONPATH 污染后续 desilike
# Python-3.11 的 NumPy C-extension。线程与 CPU affinity 仍保持统一 contract。
(
  set +u
  source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
  set -u
  run_cov() { taskset -c "${CPUSET}" python -u "$@"; }

  state=$(pair_state "${RAW_PREFIX}.npz" "${RAW_PREFIX}.json")
  [[ "${state}" != partial ]] || { echo "[fatal] x${NREAL} raw covariance pair 不完整" >&2; exit 2; }
  if [[ "${state}" == missing ]]; then
    run_cov "${CODE}/task43_diagnose_jaxpower_lightcone_covariance.py" \
    --manifest "${MANIFEST}" --fkp-summary "${FKP}" --xi-summary "${XI}" \
    --output-prefix "${RAW_PREFIX}" --phase-index 0 --p0 10000 \
    --max-random 100000 --max-data 50000 --subsample-seed 20260702 \
    --meshsize 64 --mesh-pad 400 --window-s-max 3600 --window-ds 2 \
    --window-basis bessel --window-ells 0 --interpolate-window \
    --interpolate-window-ncoords 8192 --interpolate-window-log10-min -2 \
    --interpolate-window-log10-max 8 --covariance-flags smooth,fftlog \
    --s-edge-min 30 --s-edge-max 350 --s-edge-step 10 \
    --kmin 0.0001 --kmax 3.0001 --dk 0.002 --los local --resampler cic \
    --interlacing 1 --b1-cov 2.5 --fnl-cov 0 --p-fixed 1.0 --sn0-fixed 0 \
      --cosmology abacus_c000 --floor-fraction 1.0e-10
  fi

  state=$(pair_state "${FINAL_PREFIX}.npz" "${FINAL_PREFIX}.json")
  [[ "${state}" != partial ]] || { echo "[fatal] x${NREAL} final covariance pair 不完整" >&2; exit 2; }
  if [[ "${state}" == missing ]]; then
    run_cov "${CODE}/task43_apply_rr_smu_deconvolution_to_covariance.py" \
    --covariance-path "${RAW_PREFIX}.npz" --rr-smu-path "${RR}" \
    --output-prefix "${FINAL_PREFIX}" --xi-summary "${XI}" \
    --rr-window-kind RR --rr-fkp-norm "${RR_NORM}" --ells 0 --resolution 1 \
      --floor-fraction 1.0e-10
  fi
)

# preliminary covariance 只允许 nreal=N 的 scatter metadata；其矩阵与旧
# 50--350 子块仍必须机器精度桥接，并且不得依赖 eigenvalue flooring。
run_desi - "${RAW_PREFIX}" "${FINAL_PREFIX}" "${OLD_FINAL}" "${AUDITROOT}/task43_rmin_scan_covariance_${TAG}.json" "${NREAL}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
import numpy as np

raw_prefix, final_prefix, old_path, output = map(Path, sys.argv[1:5])
nreal = int(sys.argv[5])
with np.load(final_prefix.with_suffix('.npz'), allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
    edges = np.asarray(new['s_edges'], dtype='f8')
    cov = np.asarray(new['covariance_single_realization'], dtype='f8')
    old_cov = np.asarray(old['covariance_single_realization'], dtype='f8')
    meta = json.loads(str(np.asarray(new['meta_json']).item()))
if not np.array_equal(edges, np.arange(30.0, 351.0, 10.0)) or cov.shape != (32, 32):
    raise RuntimeError(f'x{nreal} covariance 不是 32-bin contract')
eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
bridge = float(np.linalg.norm(cov[2:, 2:] - old_cov) / np.linalg.norm(old_cov))
raw_meta = json.loads(raw_prefix.with_suffix('.json').read_text())
final_meta = json.loads(final_prefix.with_suffix('.json').read_text())
raw_floor = int(raw_meta.get('spd', {}).get('n_floored', -1))
final_floor = int(final_meta.get('spd', {}).get('covariance_single_realization', {}).get('n_floored', -1))
scatter = meta.get('scatter_comparison') or final_meta.get('scatter_comparison')
if eig[0] <= 0 or bridge > 1e-8 or raw_floor != 0 or final_floor != 0:
    raise RuntimeError(f'x{nreal} covariance gate 失败: eigmin={eig[0]}, bridge={bridge}, floors={raw_floor}/{final_floor}')
if not isinstance(scatter, dict) or scatter.get('available') is not True or int(scatter.get('nreal', -1)) != nreal:
    raise RuntimeError(f'x{nreal} scatter metadata 错误: {scatter}')
payload={'status':'pass','task':f'task43_rmin_scan_covariance_preliminary_x{nreal}','nreal':nreal,
         'eig_min':float(eig[0]),'condition_number':float(np.linalg.cond(cov)),
         'old_50_350_bridge_relative_frobenius':bridge,'flooring':{'raw':raw_floor,'final':final_floor},
         'scatter_comparison':scatter}
output.parent.mkdir(parents=True, exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f'.{output.name}.',suffix='.tmp',dir=output.parent)
with os.fdopen(fd,'w') as stream:
    json.dump(payload,stream,indent=2,sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
os.replace(tmp,output)
print(json.dumps(payload,indent=2,sort_keys=True))
PY

if [[ ! -e "${OP}" && ! -e "${OP_AUDIT}" ]]; then
  run_desi "${CODE}/task43_make_ric_factorized_operator.py" \
    --kernel "${KERNEL}" --xi-path "${XI}" --pk-payload "${PK}" --w2-path "${W2}" \
    --boxsize 2000 --kmax 5 --ndense 60000 --cosmology abacus_c000 \
    --component-batch 256 --k-batch 512 --output-dir "${PRE}/operators" \
    --output-tag "s30_350_ds10_${SHORT}"
elif [[ ! -s "${OP}" || ! -s "${OP_AUDIT}" ]]; then
  echo "[fatal] x${NREAL} operator pair 不完整" >&2; exit 2
fi

run_desi - "${OP}" "${OLD_OP}" "${NREAL}" <<'PY'
import sys
from pathlib import Path
import numpy as np
new_path, old_path = map(Path, sys.argv[1:3])
nreal = int(sys.argv[3])
with np.load(new_path,allow_pickle=False) as new, np.load(old_path,allow_pickle=False) as old:
    bridges={}
    for key in ('xi_basis_pk_dd','xi_basis_alpha_pk_dd','xi_basis_alpha2_pk_dd'):
        left,right=np.asarray(new[key],dtype='f8')[2:],np.asarray(old[key],dtype='f8')
        bridges[key]=float(np.linalg.norm(left-right)/max(np.linalg.norm(right),np.finfo('f8').tiny))
if any(value > 1e-10 or not np.isfinite(value) for value in bridges.values()):
    raise RuntimeError(f'x{nreal} operator bridge 失败: {bridges}')
print(f'[bridge:x{nreal}-operator] {bridges}')
PY

run_desi "${CODE}/task43_rmin_scan_fisher.py" --xi "${XI}" \
  --covariance "${FINAL_PREFIX}.npz" --operator "${OP}" --pk-payload "${PK}" \
  --output-prefix "${FISHER_PREFIX}"

for rmin in 30 40 50; do
  OUT="${FITROOT}/rmin${rmin}"
  SUMMARY="${OUT}/task43_minimal_closure_mcmc_summary.json"
  SAMPLES="${OUT}/task43_mcmc_radial_singleterm_samples.npz"
  if [[ ! -e "${SUMMARY}" && ! -e "${SAMPLES}" ]]; then
    run_desi "${CODE}/task43_fit_minimal_closure.py" \
      --xi-path "${XI}" --output-dir "${OUT}" --fit-target mean \
      --covariance-mode npz --covariance-path "${FINAL_PREFIX}.npz" \
      --covariance-key covariance_single_realization --rmin "${rmin}" --rmax 350 \
      --models radial_singleterm --radial-ric-operator "${OP}" \
      --fkp-summary "${FKP}" --p0 10000 --output-tag "${TAG}" \
      --cosmology abacus_c000 --theory-boxsize 2000 --p-fixed 1.0 --png-order full \
      --sn0-fixed 0 --kmax 5 --ndense 60000 --xi-kernel shell-averaged \
      --nwalkers 64 --nsteps 10000 --burnin 2000 --seed 20260804
  elif [[ ! -s "${SUMMARY}" || ! -s "${SAMPLES}" ]]; then
    echo "[fatal] x${NREAL} rmin=${rmin} fit pair 不完整" >&2; exit 2
  fi
  run_desi - "${SAMPLES}" "${rmin}" "${NREAL}" <<'PY'
import sys
from pathlib import Path
import numpy as np
path=Path(sys.argv[1]); rmin=int(sys.argv[2]); nreal=int(sys.argv[3])
with np.load(path,allow_pickle=False) as data:
    samples=np.asarray(data['samples'],dtype='f8'); logp=np.asarray(data['log_prob'],dtype='f8')
if samples.shape != (512000,2) or logp.shape != (512000,) or not np.all(np.isfinite(samples)):
    raise RuntimeError(f'x{nreal} rmin={rmin} chain contract 错误: {samples.shape}/{logp.shape}')
print(f'[fit:x{nreal}] rmin={rmin} samples={samples.shape}')
PY
done

run_desi "${CODE}/task43_summarize_rmin_scan.py" --xi "${XI}" \
  --covariance "${FINAL_PREFIX}.npz" --fisher "${FISHER_PREFIX}.json" \
  --fit-root "${FITROOT}" --expected-nreal "${NREAL}" \
  --output-json "${AUDITROOT}/task43_jaxpower_2pcf_rmin_scan_${TAG}.json" \
  --output-csv "${AUDITROOT}/task43_jaxpower_2pcf_rmin_scan_${TAG}.csv" \
  --output-pdf "${PLOT}"
echo "[done] preliminary x${NREAL} rmin scan 已完成；仅供初测，不替代最终 x25"
