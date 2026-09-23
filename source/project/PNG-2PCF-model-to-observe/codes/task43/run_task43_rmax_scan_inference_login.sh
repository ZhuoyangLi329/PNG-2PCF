#!/usr/bin/env bash
# Compile one 50-bin radial single-term operator and run the five matched
# Task43 rmax chains.  This is CPU-only and never submits Slurm jobs.

set -euo pipefail

ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
PY="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
SCAN_ROOT="${ROOT}/outputs/task43_outputs/rmax_scan"
XI="${SCAN_ROOT}/summary/task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000.npz"
COV="${SCAN_ROOT}/covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_550_ds10.npz"
FKP="${ROOT}/outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
KERNEL="${ROOT}/outputs/task43_outputs/ric_singleterm/kernels/task43_ric_factorized_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712.npz"
PK_PAYLOAD="${ROOT}/outputs/task43_outputs/pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
W2="${ROOT}/outputs/task43_outputs/summary/task43_formal_gic_window_mmin1p4e13_x25_ph000_fkpP010000_L2000_nsub200000_seed20260703.npz"
OLD_OPERATOR="${ROOT}/outputs/task43_outputs/ric_singleterm/operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
OPDIR="${SCAN_ROOT}/operators"
OP_PREFIX="${OPDIR}/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000_s50_550_ds10"
OPERATOR="${OP_PREFIX}.npz"
OP_AUDIT="${OP_PREFIX}_audit.json"
FIT_ROOT="${SCAN_ROOT}/fits"
CPUSET="${TASK43_RMAX_CPUSET:-6-13}"
RMAXS="${TASK43_RMAXS:-350 400 450 500 550}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${ROOT}"
for path in "${PY}" "${XI}" "${COV}" "${FKP}" "${KERNEL}" "${PK_PAYLOAD}" "${W2}" "${OLD_OPERATOR}"; do
  [[ -s "${path}" ]] || { echo "[fatal] missing or empty input: ${path}" >&2; exit 2; }
done
taskset -c "${CPUSET}" true || { echo "[fatal] unavailable CPU set: ${CPUSET}" >&2; exit 2; }
NCPU=$("${PY}" - "${CPUSET}" <<'PY'
import sys
cpus = set()
for item in sys.argv[1].split(','):
    if '-' in item:
        lo, hi = map(int, item.split('-', 1)); cpus.update(range(lo, hi + 1))
    else:
        cpus.add(int(item))
print(len(cpus))
PY
)
if (( NCPU < 1 || NCPU > 8 )); then
  echo "[fatal] login-node CPU set must contain 1..8 CPUs; got ${NCPU}" >&2
  exit 2
fi

run_python() {
  taskset -c "${CPUSET}" "${PY}" -u "$@"
}

# Fail before operator/MCMC if the common vector and covariance are not the
# exact 50-bin objects promised by the experiment contract.
run_python - "${XI}" "${COV}" <<'PY'
import sys
from pathlib import Path
import numpy as np

xi_path, cov_path = map(Path, sys.argv[1:])
edges = np.arange(50.0, 560.0, 10.0)
centers = 0.5 * (edges[:-1] + edges[1:])
with np.load(xi_path, allow_pickle=False) as data:
    if not np.array_equal(data['s_edges'], edges) or not np.array_equal(data['s'], centers):
        raise RuntimeError('xi coordinates violate the 50-bin scan contract')
    if np.asarray(data['xi0_all']).shape != (25, 50) or int(data['nreal']) != 25:
        raise RuntimeError('xi summary must contain 25x50 realizations')
with np.load(cov_path, allow_pickle=False) as data:
    if not np.array_equal(data['s_edges'], edges) or not np.array_equal(data['s'], centers):
        raise RuntimeError('covariance coordinates do not match xi')
    cov = np.asarray(data['covariance_single_realization'], dtype='f8')
if cov.shape != (50, 50) or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=0, atol=1e-15):
    raise RuntimeError('fit covariance must be finite symmetric 50x50')
eig = np.linalg.eigvalsh(cov)
if eig[0] <= 0 or np.linalg.cond(cov) >= 1e12:
    raise RuntimeError(f'fit covariance is unusable: eigmin={eig[0]}, cond={np.linalg.cond(cov)}')
print(f'[preflight] xi=25x50 covariance=50x50 eigmin={eig[0]:.6e} cond={np.linalg.cond(cov):.6g}')
PY

mkdir -p "${OPDIR}" "${FIT_ROOT}"
if [[ -e "${OPERATOR}" || -L "${OPERATOR}" || -e "${OP_AUDIT}" || -L "${OP_AUDIT}" ]]; then
  if [[ ! -s "${OPERATOR}" || ! -s "${OP_AUDIT}" ]]; then
    echo "[fatal] incomplete operator pair: ${OPERATOR}, ${OP_AUDIT}" >&2
    exit 2
  fi
  echo "[skip] complete 50-bin operator pair exists"
else
  run_python codes/task43/task43_make_ric_factorized_operator.py \
    --kernel "${KERNEL}" \
    --xi-path "${XI}" \
    --pk-payload "${PK_PAYLOAD}" \
    --w2-path "${W2}" \
    --boxsize 2000 \
    --kmax 5 \
    --ndense 60000 \
    --cosmology abacus_c000 \
    --component-batch 256 \
    --k-batch 512 \
    --output-dir "${OPDIR}" \
    --output-tag s50_550_ds10
fi

# Bridge all three radial-response bases before any chain is launched.
run_python - "${OPERATOR}" "${OP_AUDIT}" "${OLD_OPERATOR}" \
  "${KERNEL}" "${XI}" "${PK_PAYLOAD}" "${W2}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

new_path, audit_path, old_path, kernel_path, xi_path, pk_path, w2_path = map(Path, sys.argv[1:])
keys = ('xi_basis_pk_dd', 'xi_basis_alpha_pk_dd', 'xi_basis_alpha2_pk_dd')
with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
    edges = np.asarray(new['target_s_edges'], dtype='f8')
    if not np.array_equal(edges, np.arange(50.0, 560.0, 10.0)):
        raise RuntimeError('operator target edges violate the 50-bin contract')
    bridge = {}
    for key in keys:
        left = np.asarray(new[key], dtype='f8')
        right = np.asarray(old[key], dtype='f8')
        if (
            left.shape != (50,)
            or right.shape != (30,)
            or not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
        ):
            raise RuntimeError(f'invalid operator basis {key}: {left.shape}, {right.shape}')
        denom = max(float(np.linalg.norm(right)), np.finfo('f8').tiny)
        bridge[key] = float(np.linalg.norm(left[:30] - right) / denom)
    meta = json.loads(str(np.asarray(new['meta_json']).item()))
audit = json.loads(audit_path.read_text(encoding='utf-8'))
if meta.get('status') != 'done' or audit.get('status') != 'done':
    raise RuntimeError('operator audit status is not done')
if meta.get('output_tag') != 's50_550_ds10' or meta.get('target_2pcf') != {'s_edge_min': 50.0, 's_edge_max': 550.0, 'nbins': 50}:
    raise RuntimeError('operator output tag/target metadata mismatch')
def same(left, right):
    value = Path(left)
    if not value.is_absolute(): value = Path.cwd() / value
    return value.resolve() == Path(right).resolve()
inputs = meta.get('inputs', {})
if not same(inputs.get('kernel', ''), kernel_path) or not same(inputs.get('xi', ''), xi_path) or not same(inputs.get('pk_payload', ''), pk_path):
    raise RuntimeError(f'operator input provenance mismatch: {inputs}')
theory = meta.get('theory', {})
expected_theory = {
    'boxsize': 2000.0, 'kmax': 5.0, 'ndense': 60000,
    'cosmology': 'abacus_c000', 'p_fixed': 1.0,
    'png_order': 'full including fNL^2',
}
for key, expected in expected_theory.items():
    if theory.get(key) != expected:
        raise RuntimeError(f'operator theory {key}={theory.get(key)!r}, expected {expected!r}')
w2 = meta.get('tests', {}).get('global_limit', {}).get('comparison_to_current_w2', {})
if w2.get('available') is not True or not same(w2.get('path', ''), w2_path):
    raise RuntimeError(f'operator W2 audit provenance mismatch: {w2}')
paths = meta.get('paths', {})
if not same(paths.get('operator_npz', ''), new_path) or not same(paths.get('audit_json', ''), audit_path):
    raise RuntimeError(f'operator output-path metadata mismatch: {paths}')
if audit.get('inputs') != meta.get('inputs') or audit.get('theory') != meta.get('theory'):
    raise RuntimeError('operator NPZ/JSON science metadata do not match')
if any((not np.isfinite(value)) or value > 1e-10 for value in bridge.values()):
    raise RuntimeError(f'operator first-30 bridge failed: {bridge}')
print(f'[bridge:operator] {bridge}')
PY

validate_fit() {
  local rmax="$1"
  local outdir="${FIT_ROOT}/smax${rmax}"
  run_python - "${outdir}/task43_minimal_closure_mcmc_summary.json" \
    "${outdir}/task43_mcmc_radial_singleterm_samples.npz" \
    "${XI}" "${COV}" "${OPERATOR}" "${FKP}" "${rmax}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

summary_path, samples_path, xi_path, cov_path, operator_path, fkp_path = map(Path, sys.argv[1:7])
rmax = int(sys.argv[7])
summary = json.loads(summary_path.read_text(encoding='utf-8'))
if summary.get('status') != 'done' or summary.get('fit_target') != 'mean':
    raise RuntimeError('fit summary status/target mismatch')
fit = summary.get('fit_range', {})
expected_nbins = (rmax - 50) // 10
expected_s = np.arange(55.0, float(rmax), 10.0)
if (
    float(fit.get('rmin', -1)) != 50.0
    or float(fit.get('rmax', -1)) != float(rmax)
    or int(fit.get('nbins', -1)) != expected_nbins
    or int(fit.get('nreal', -1)) != 25
    or int(fit.get('data_vector_size', -1)) != expected_nbins
    or not np.array_equal(np.asarray(fit.get('s_centers', []), dtype='f8'), expected_s)
):
    raise RuntimeError(f'fit range mismatch for rmax={rmax}: {fit}')
cov = summary.get('covariance', {})
recorded_cov = Path(cov.get('path', ''))
if not recorded_cov.is_absolute(): recorded_cov = Path.cwd() / recorded_cov
if cov.get('mode') != 'npz' or cov.get('key') != 'covariance_single_realization' or recorded_cov.resolve() != cov_path.resolve():
    raise RuntimeError(f'fit covariance provenance mismatch: {cov}')
if Path(summary.get('xi_path', '')).resolve() != xi_path.resolve():
    raise RuntimeError('fit xi provenance mismatch')
theory = summary.get('theory', {})
expected_theory = {
    'kmax': 5.0, 'ndense': 60000, 'boxsize': 2000.0, 'p_fixed': 1.0,
    'png_order': 'full', 'xi_kernel': 'shell-averaged', 'sn0_fixed': 0.0,
    'sn0_policy': 'fixed', 'cosmology': 'abacus_c000',
}
for key, expected_value in expected_theory.items():
    if theory.get(key) != expected_value:
        raise RuntimeError(f'fit theory {key}={theory.get(key)!r}, expected {expected_value!r}')
if theory.get('models') != ['radial_singleterm']:
    raise RuntimeError(f'fit theory model list mismatch: {theory.get("models")}')
weighting = theory.get('weighting', {})
recorded_fkp = Path(weighting.get('fkp_summary_path', ''))
if not recorded_fkp.is_absolute(): recorded_fkp = Path.cwd() / recorded_fkp
if weighting.get('p0') != 10000.0 or weighting.get('output_tag') != 'rmax_scan' or recorded_fkp.resolve() != fkp_path.resolve():
    raise RuntimeError(f'fit weighting provenance mismatch: {weighting}')
models = summary.get('models', [])
if len(models) != 1 or models[0].get('model') != 'radial_singleterm':
    raise RuntimeError('fit must contain only radial_singleterm')
model = models[0]
if model.get('parameter_names') != ['fnl_loc', 'b1']:
    raise RuntimeError(f'fit free-parameter list mismatch: {model.get("parameter_names")}')
model_data = model.get('data', {})
if (
    model_data.get('fit_target') != 'mean'
    or int(model_data.get('nreal', -1)) != 1
    or int(model_data.get('nbins', -1)) != expected_nbins
    or int(model_data.get('data_vector_size', -1)) != expected_nbins
):
    raise RuntimeError(f'fit model data metadata mismatch: {model_data}')
mcmc = model.get('mcmc', {})
expected = {'nwalkers': 64, 'nsteps': 20000, 'burnin': 5000, 'seed': 20260720, 'nsamples': 960000, 'ndim': 2}
if any(int(mcmc.get(key, -1)) != value for key, value in expected.items()):
    raise RuntimeError(f'MCMC settings mismatch: {mcmc}')
ric_path = Path(model.get('radial_singleterm', {}).get('operator', {}).get('path', ''))
if not ric_path.is_absolute(): ric_path = Path.cwd() / ric_path
if ric_path.resolve() != operator_path.resolve():
    raise RuntimeError('fit radial operator provenance mismatch')
ric = model.get('radial_singleterm', {})
ric_operator = ric.get('operator', {})
if ric.get('extra_global_sigma_w2') is not False or ric_operator.get('selected_indices') != list(range(expected_nbins)) or not np.array_equal(np.asarray(ric_operator.get('selected_s', []), dtype='f8'), expected_s):
    raise RuntimeError(f'fit radial operator selection mismatch: {ric}')
with np.load(samples_path, allow_pickle=False) as data:
    samples = np.asarray(data['samples'], dtype='f8')
    logp = np.asarray(data['log_prob'], dtype='f8')
    pred = np.asarray(data['prediction_map'], dtype='f8')
    residual = np.asarray(data['residual_map'], dtype='f8')
if samples.shape != (960000, 2) or logp.shape != (960000,) or pred.shape != (expected_nbins,) or residual.shape != (expected_nbins,):
    raise RuntimeError(f'fit output shapes mismatch: {samples.shape}, {logp.shape}, {pred.shape}, {residual.shape}')
if not all(np.all(np.isfinite(value)) for value in (samples, logp, pred, residual)):
    raise RuntimeError('fit output contains non-finite values')
if not np.array_equal(pred, np.asarray(model_data.get('prediction_map', []), dtype='f8')) or not np.array_equal(residual, np.asarray(model_data.get('residual_map', []), dtype='f8')):
    raise RuntimeError('fit summary and samples NPZ prediction/residual are not an exact pair')
names = model['parameter_names']
imax = int(np.argmax(logp))
for index, name in enumerate(names):
    if not np.isclose(float(samples[imax, index]), float(model['map'][name]), rtol=0.0, atol=0.0):
        raise RuntimeError(f'fit summary MAP {name} does not match samples/log_prob')
    q16, q50, q84 = np.percentile(samples[:, index], [16.0, 50.0, 84.0])
    for label, value in (('q16', q16), ('q50', q50), ('q84', q84)):
        if not np.isclose(float(model[name][label]), float(value), rtol=0.0, atol=1e-12):
            raise RuntimeError(f'fit summary {name}.{label} does not match raw samples')
print(f'[validate:fit] rmax={rmax} nbins={expected_nbins} samples={samples.shape}')
PY
}

for rmax in ${RMAXS}; do
  case "${rmax}" in
    350|400|450|500|550) ;;
    *) echo "[fatal] TASK43_RMAXS contains unsupported value: ${rmax}" >&2; exit 2 ;;
  esac
  outdir="${FIT_ROOT}/smax${rmax}"
  summary="${outdir}/task43_minimal_closure_mcmc_summary.json"
  samples="${outdir}/task43_mcmc_radial_singleterm_samples.npz"
  if [[ -e "${summary}" || -L "${summary}" || -e "${samples}" || -L "${samples}" ]]; then
    if [[ ! -s "${summary}" || ! -s "${samples}" ]]; then
      echo "[fatal] incomplete fit pair for rmax=${rmax}: ${summary}, ${samples}" >&2
      exit 2
    fi
    echo "[skip] complete fit pair exists for rmax=${rmax}; validating"
  else
    echo "[run] matched long chain rmax=${rmax}"
    run_python codes/task43/task43_fit_minimal_closure.py \
      --xi-path "${XI}" \
      --output-dir "${outdir}" \
      --fit-target mean \
      --covariance-mode npz \
      --covariance-path "${COV}" \
      --covariance-key covariance_single_realization \
      --rmin 50 \
      --rmax "${rmax}" \
      --models radial_singleterm \
      --radial-ric-operator "${OPERATOR}" \
      --fkp-summary "${FKP}" \
      --p0 10000 \
      --output-tag rmax_scan \
      --cosmology abacus_c000 \
      --theory-boxsize 2000 \
      --p-fixed 1.0 \
      --png-order full \
      --sn0-fixed 0 \
      --kmax 5 \
      --ndense 60000 \
      --xi-kernel shell-averaged \
      --nwalkers 64 \
      --nsteps 20000 \
      --burnin 5000 \
      --seed 20260720
  fi
  validate_fit "${rmax}"
done

echo "[done] operator bridge and requested matched chains passed output gates"
